"""Adapter unit tests plus opt-in isolated PostgreSQL integration tests.

Set FUNES_TEST_POSTGRES_DSN to a disposable PostgreSQL database with CREATE
SCHEMA privilege. Each test owns a random schema, never public/application data.
No connection string or source text is printed by the suite.
"""
from __future__ import annotations

import gzip
import json
import os
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest import mock

from service.postgres import (
    Connection,
    PostgresNotReady,
    PostgresStore,
    Row,
    _bind_sql,
    _translate_sql,
    _row_factory,
    decode_pg_text,
    encode_pg_text,
)
from service.server import Store, technical_index_text


class AdapterTests(unittest.TestCase):
    def test_row_supports_name_index_slice_iteration_and_dict(self):
        row = Row(("id", "raw_text"), (7, "original"))
        self.assertEqual(row[0], row["id"])
        self.assertEqual(row[:1], (7,))
        self.assertEqual(tuple(row), (7, "original"))
        self.assertEqual(dict(row), {"id": 7, "raw_text": "original"})

    def test_row_factory_expands_only_null_and_decodes_once(self):
        cursor = mock.Mock(description=[mock.Mock(name="col") for _ in range(2)])
        cursor.description[0].name = "raw_text"
        cursor.description[1].name = "retrieval_text"
        row = _row_factory(cursor)([encode_pg_text("a\x00\ue0000"), None])
        self.assertEqual(row["retrieval_text"], "a\x00\ue0000")
        self.assertEqual(_row_factory(cursor)(["raw", ""])["retrieval_text"], "")

    def test_qmark_conversion_respects_literals_comments_and_percent(self):
        sql = "SELECT ?, '?', 'a''?b', \"?\", 5 % 2, '80%' /* ? /* ? */ */ -- ?\n, $tag$?%$tag$"
        actual = _bind_sql(sql, parameters=True)
        self.assertEqual(actual.count("%s"), 1)
        self.assertIn("5 %% 2", actual)
        self.assertIn("'a''?b'", actual)
        self.assertIn("$tag$?%%$tag$", actual)
        self.assertEqual(_bind_sql(sql, parameters=False), sql)

    def test_ignore_replace_and_returning_are_explicit(self):
        sql, returns_id = _translate_sql("INSERT OR IGNORE INTO reindex_controls(generation) VALUES(?)", parameters=True)
        self.assertIn("ON CONFLICT DO NOTHING", sql)
        self.assertFalse(returns_id)
        sql, returns_id = _translate_sql("INSERT INTO memories(source_identity) VALUES(?)", parameters=True)
        self.assertTrue(returns_id)
        self.assertTrue(sql.endswith("RETURNING id"))
        sql, _ = _translate_sql("INSERT OR REPLACE INTO translation_cache(query,rewritten,created_at,translation_hash,translation_version,translation_status) VALUES(?,?,?,?,?,?)", parameters=True)
        self.assertIn("ON CONFLICT (query) DO UPDATE", sql)
        with self.assertRaises(ValueError):
            _translate_sql("INSERT OR REPLACE INTO memories(id) VALUES(?)", parameters=True)

    def test_transaction_context_does_not_close_or_replay_connection(self):
        raw = mock.MagicMock()
        raw.closed = False
        connection = Connection(raw, require_ready=False)
        with connection:
            connection.execute("UPDATE memories SET raw_text=?", ("one",))
        raw.transaction.return_value.__enter__.assert_called_once()
        raw.transaction.return_value.__exit__.assert_called_once_with(None, None, None)
        raw.close.assert_not_called()
        raw.closed = True
        with self.assertRaises(PostgresNotReady):
            connection.execute("SELECT 1")

    def test_text_codec_preserves_nul_and_literal_marker_collisions(self):
        samples = ("", "普通中文", "a\x00b", "\ue000", "\ue000e", "\ue0000", "\x00\ue0000\ue000e\x00", "𝌆\x00previous_response_id")
        for value in samples:
            with self.subTest(value=repr(value)):
                self.assertNotIn("\x00", encode_pg_text(value))
                self.assertEqual(decode_pg_text(encode_pg_text(value)), value)

    def test_missing_dsn_fails_closed_without_creating_sqlite(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"FUNES_POSTGRES_DSN": ""}):
            with self.assertRaises(PostgresNotReady):
                PostgresStore(directory)
            self.assertFalse((Path(directory) / "funes.sqlite3").exists())


@unittest.skipUnless(os.getenv("FUNES_TEST_POSTGRES_DSN"), "FUNES_TEST_POSTGRES_DSN not configured")
class PostgresIntegrationTests(unittest.TestCase):
    def setUp(self):
        import psycopg
        from psycopg.conninfo import conninfo_to_dict, make_conninfo
        from psycopg import sql

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.schema = "funes_test_" + uuid.uuid4().hex
        self.base_dsn = os.environ["FUNES_TEST_POSTGRES_DSN"]
        self.admin = psycopg.connect(self.base_dsn, autocommit=True)
        self.addCleanup(self.admin.close)
        self.admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
        self.addCleanup(lambda: self.admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema))))
        options = conninfo_to_dict(self.base_dsn).get("options", "")
        self.dsn = make_conninfo(self.base_dsn, options=f"{options} -csearch_path={self.schema},public".strip())
        self.store = PostgresStore(self.temporary.name, self.dsn, initialize=True)
        self.addCleanup(self.store.close)
        self.store.finalize_migration(expected_count=0)

    def runtime(self):
        store = PostgresStore(self.temporary.name, self.dsn)
        self.addCleanup(store.close)
        return store

    @staticmethod
    def doc(identity="source", **changes):
        return {
            "source_identity": identity,
            "source_version": "v1",
            "raw_text": "配置 previous_response_id and Northflank",
            "source_agent": "codex",
            "source_type": "memory",
            "updated_at": "2026-09-22T10:00:00+00:00",
            **changes,
        }

    def test_ingest_lastrowid_dedupe_stale_metadata_and_generations_match_sqlite(self):
        sqlite = Store(str(Path(self.temporary.name) / "sqlite"))
        self.addCleanup(sqlite.close)
        documents = [
            self.doc(project="alpha", metadata={"private": "first"}, embedding_generation=7),
            self.doc(project="alpha", metadata={"private": "first"}, embedding_generation=7),
            self.doc(project="beta", updated_at="2026-09-22T11:00:00+00:00"),
            self.doc(source_version="older", raw_text="stale version", updated_at="2026-09-20T11:00:00+00:00"),
            self.doc(source_version="v2", raw_text="updated raw", updated_at="2026-09-22T12:00:00+00:00"),
        ]
        for document in documents:
            self.assertEqual(sqlite.ingest([document]), self.store.ingest([document]))
            expected, actual = sqlite.get("source"), self.store.get("source")
            expected.pop("ingested_at")
            actual.pop("ingested_at")
            self.assertEqual(actual, expected)
        self.assertIsNone(self.store.get("not-a-number.jsonl"))
        self.assertIsNone(self.store.get("9" * 100))
        self.assertEqual(self.store.get(1)["source_identity"], "source")

    def test_nul_text_identity_marker_snapshot_and_reopen_are_lossless(self):
        raw = "原文\x00previous_response_id\ue000e\ue0000，保持原样"
        identity = "session\x00marker\ue0000"
        self.store.ingest([self.doc(identity, raw_text=raw, retrieval_text=raw, source_path="/tmp/\ue000e\x00.md")])
        self.assertEqual(self.store.get(identity)["raw_text"], raw)
        self.assertEqual(self.store.get(identity)["retrieval_text"], raw)
        self.assertEqual(self.store.existing_identities([identity]), [identity])
        self.assertEqual(self.runtime().get(identity)["source_path"], "/tmp/\ue000e\x00.md")
        self.assertEqual(self.store.search(identity, allow_broad_scan=False)[0]["source_identity"], identity)
        path = Path(self.temporary.name) / "source.jsonl.gz"
        self.store.snapshot(path)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        self.assertEqual(records[0]["raw_text"], raw)
        self.assertEqual(records[0]["source_identity"], identity)
        self.assertNotIn("search_vector", records[0])

    def test_commit_rollback_read_isolation_and_reuse(self):
        runtime = self.runtime()
        runtime.ingest([self.doc()])
        with self.assertRaisesRegex(ValueError, "raw_text"):
            runtime.ingest([self.doc("rolled-back"), {"source_identity": "invalid"}])
        self.assertIsNone(runtime.get("rolled-back"))
        self.assertFalse(runtime.conn.raw.closed)
        with runtime.lock, runtime.conn:
            runtime.conn.execute("UPDATE memories SET raw_text=? WHERE source_identity=?", ("not committed", "source"))
            self.assertNotEqual(runtime.get("source")["raw_text"], "not committed")
        self.assertEqual(runtime.get("source")["raw_text"], "not committed")
        runtime.ingest([self.doc("after-rollback")])
        self.assertEqual(runtime.count(), 2)

    def test_reindex_controls_translation_upsert_and_new_embedding_generation(self):
        self.store.ingest([self.doc()])
        self.store.translation_put("查询", "first", translation_hash="a")
        self.store.translation_put("查询", "second", translation_hash="b")
        self.assertEqual(self.store.translation_get("查询"), "second")
        control = self.store.next_reindex_control("all")
        self.assertTrue(self.store.record_reindex_control(control))
        self.assertFalse(self.store.record_reindex_control(control))
        self.assertEqual(self.store.latest_reindex_generation(), 1)
        self.store.drain_reindex_controls(batch_size=1)
        before = self.store.get("source")
        self.assertEqual(before["embedding_generation"], 1)
        self.assertEqual(before["retrieval_generation"], 1)
        self.assertEqual(before["native_generation"], 1)
        self.assertEqual(before["translation_status"], "pending_provider")
        self.store.ingest([self.doc("new-after-control")])
        self.assertEqual(self.store.get("new-after-control")["embedding_generation"], 1)
        self.assertEqual(self.store.compact_reindex_controls()["kept"], 1)

    def test_native_null_transitions_counts_revisions_and_ignored_conflicts(self):
        profile = {"fingerprint": "profile"}
        self.store.native_index_checkpoint(profile, "memory")
        self.store.ingest([
            self.doc("pending"),
            self.doc("waiting", native_index_status="waiting_durability"),
            self.doc("low", content_type="progress"),
            self.doc("held", native_index_status="held_invalid", native_index_profile="profile", native_index_memory="memory"),
        ])
        state = self.store.native_index_checkpoint(profile, "memory")
        self.assertEqual((state["eligible"], state["indexed"], state["held"], state["invalid"], state["revision"]), (3, 0, 1, 1, 3))
        self.assertEqual([item["source_identity"] for item in self.store.canonical_index_candidates(10, "profile", "memory")], ["pending"])
        pending = self.store.get("pending")
        update = {**pending, "native_index_status": "indexed", "native_index_profile": "profile", "native_index_memory": "memory"}
        self.store.update_native_index([update])
        self.store.update_native_index([update])
        state = self.store.native_index_checkpoint(profile, "memory")
        self.assertEqual((state["indexed"], state["revision"]), (1, 4))
        # BEFORE INSERT triggers also run on ignored conflicts: count deltas
        # must be AFTER INSERT, or this would corrupt the native checkpoint.
        with self.store.lock, self.store.conn:
            self.store.conn.execute("INSERT OR IGNORE INTO memories(source_identity,raw_text,retrieval_text,content_hash,ingested_at,updated_at) VALUES(?,?,?,?,?,?)", ("pending", "x", "x", "h", "now", "now"))
        self.assertEqual(self.store.native_index_checkpoint(profile, "memory"), state)
        self.store.update_native_index([{**update, "native_index_status": None, "native_index_profile": None, "native_index_memory": None}])
        self.assertEqual(self.store.native_index_checkpoint(profile, "memory")["indexed"], 0)
        with self.store.lock, self.store.conn:
            self.store.conn.execute("DELETE FROM memories WHERE source_identity=?", ("held",))
        state = self.store.native_index_checkpoint(profile, "memory")
        self.assertEqual((state["eligible"], state["held"], state["invalid"]), (2, 0, 0))

    def test_v2_catalog_has_only_btrees_and_no_full_text_objects(self):
        rows = self.admin.execute("SELECT data_type,is_generated,is_nullable,column_name FROM information_schema.columns WHERE table_schema=%s AND table_name='memories'", (self.schema,)).fetchall()
        self.assertFalse(any(row[0] == "tsvector" or row[1] != "NEVER" for row in rows))
        self.assertEqual(next(row[2] for row in rows if row[3] == "retrieval_text"), "YES")
        methods = self.store.conn.execute("SELECT a.amname FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid JOIN pg_am a ON a.oid=c.relam WHERE i.indrelid='memories'::regclass").fetchall()
        self.assertTrue(methods)
        self.assertEqual({row[0] for row in methods}, {"btree"})
        functions = self.admin.execute("SELECT proname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname=%s", (self.schema,)).fetchall()
        self.assertFalse(any("search" in row[0] or "lexical" in row[0] for row in functions))
        self.assertFalse(self.store.fts_ready())
        self.assertEqual(self.store.reindex(), 0)
        self.runtime().verify_schema()

    def test_exact_identity_filters_use_index_and_never_content_scan(self):
        self.store.ingest([
            self.doc("target", raw_text="中文previous_response_id chatcmpl-*", project="wanted", timestamp="2026-09-22"),
            self.doc("other", project="other", timestamp="2026-09-21"),
        ])
        hits = self.store.search("target", filters={"project": "wanted", "since": "2026-09-22"}, allow_broad_scan=False)
        self.assertEqual([row["source_identity"] for row in hits], ["target"])
        for filters in ({"project": "missing"}, {"since": "2026-09-23"}, {"until": "2026-09-21"}, {"source_agent": "pi"}, {"source_missing": True}):
            self.assertEqual(self.store.search("target", filters=filters), [])
        self.assertEqual(self.store.search("previous_response_id"), [])
        self.assertEqual(self.store.search("1")[0]["source_identity"], "target")
        statements = []
        original = self.store._read_connection
        def traced():
            reader = original()
            reader.set_trace_callback(statements.append)
            return reader
        with mock.patch.object(self.store, "_read_connection", side_effect=traced):
            self.store.search("previous_response_id", limit=1)
        self.assertFalse(any(" LIKE " in sql or "tsquery" in sql or "search_vector" in sql for sql in statements))
        self.assertTrue(any("WHERE source_identity=?" in sql for sql in statements))
        with self.store.conn.raw.transaction():
            self.store.conn.raw.execute("SET LOCAL enable_seqscan=off")
            plan = self.store.conn.execute("EXPLAIN SELECT id FROM memories WHERE source_identity=?", ("target",)).fetchall()
        self.assertIn("memories_source_identity_key", " ".join(row[0] for row in plan))

    def test_translation_requeue_matches_logical_shadow_and_guards_stale_writes(self):
        raw = "原文 previous_response_id chatcmpl-*\x00\ue000e"
        for identity, shadow in (("equal", raw), ("translated", "English shadow"), ("empty", "")):
            with self.subTest(shadow=identity):
                self.store.ingest([self.doc(identity, raw_text=raw, retrieval_text=shadow,
                                           translation_status="ok", retrieval_generation=2)])
                if shadow == "":
                    self.store.conn.execute(
                        "UPDATE memories SET retrieval_text='' WHERE source_identity=?", (identity,))
                selected = self.store.get(identity)
                for field, stale in (("source_identity", "missing"), ("source_version", "old"),
                                     ("content_hash", "old"), ("retrieval_text", "old"),
                                     ("translation_status", "old"), ("retrieval_generation", 1)):
                    self.store.mark_translations_pending([{**selected, field: stale}])
                    self.assertEqual(self.store.get(identity), selected, field)
                # Use the committed row, not untrusted caller raw_text, for identifiers.
                self.store.mark_translations_pending([{**selected, "raw_text": "wrong_raw_identifier"}])
                updated = self.store.get(identity)
                self.assertEqual(updated["raw_text"], raw)
                self.assertEqual(updated["retrieval_text"], raw)
                self.assertEqual(updated["translation_status"], "pending_provider")
                self.assertEqual(updated["retrieval_generation"], 2)
                physical = self.admin.execute(
                    f'SELECT retrieval_text,search_identifiers FROM "{self.schema}".memories '
                    'WHERE source_identity=%s', (identity,)).fetchone()
                self.assertIsNone(physical[0])
                self.assertEqual(decode_pg_text(physical[1]), technical_index_text(raw, raw))
                self.store.mark_translations_pending([selected])
                self.assertEqual(self.store.get(identity), updated)

    def test_null_shadow_is_physically_compact_and_logically_lossless(self):
        raw = "原文 previous_response_id chatcmpl-*\x00\ue000e"
        self.store.ingest([
            self.doc("same", raw_text=raw, retrieval_text=raw),
            self.doc("different", raw_text=raw, retrieval_text="English shadow"),
            self.doc("empty", raw_text=raw, retrieval_text=""),
        ])
        # The shared ingest policy normalizes empty input; a legacy imported
        # empty shadow must nevertheless stay empty, not expand as NULL.
        self.store.conn.execute("UPDATE memories SET retrieval_text='' WHERE source_identity='empty'")
        physical = self.admin.execute(f'SELECT source_identity,retrieval_text FROM "{self.schema}".memories ORDER BY id').fetchall()
        self.assertEqual(physical, [("same", None), ("different", "English shadow"), ("empty", "")])
        revision = self.store.native_index_state_record()["revision"]
        self.assertEqual(self.store.ingest([self.doc("same", raw_text=raw, retrieval_text=raw)])["deduped"], 1)
        self.assertEqual(self.store.native_index_state_record()["revision"], revision)
        for ident, expected in (("same", raw), ("different", "English shadow"), ("empty", "")):
            self.assertEqual(self.store.get(ident)["retrieval_text"], expected)
        self.assertEqual(self.store.get_many(["same"])[0]["retrieval_text"], raw)
        batches = list(self.store.iter_documents(batch_size=1))
        self.assertEqual(batches[0][0]["retrieval_text"], raw)
        self.assertEqual(self.store.conn.execute(
            "SELECT search_identifiers FROM memories WHERE source_identity=?", ("same",)
        ).fetchone()[0], technical_index_text(raw, raw))
        # Equal physical normalization must not churn native revisions.
        self.store.conn.execute("UPDATE memories SET retrieval_text=raw_text WHERE source_identity=?", ("same",))
        self.assertEqual(self.store.native_index_state_record()["revision"], revision)

    def test_large_unicode_source_is_not_subject_to_pg_tsvector_limits(self):
        raw = "𝌆 previous_response_id " * 60000
        self.store.ingest([self.doc("large", raw_text=raw, retrieval_text=raw)])
        self.assertEqual(self.runtime().get("large")["raw_text"], raw)
        self.assertEqual(self.store.get("large")["retrieval_text"], raw)

    def test_v1_marker_is_rejected_without_automatic_schema_rewrite(self):
        self.store.conn.execute("UPDATE funes_schema_state SET schema_version=1 WHERE id=1")
        with self.assertRaises(PostgresNotReady):
            self.runtime()
        row = self.admin.execute(f'SELECT schema_version FROM "{self.schema}".funes_schema_state').fetchone()
        self.assertEqual(row[0], 1)

    def test_snapshot_and_iterator_stream_and_checkpoint_survives_reopen(self):
        self.store.ingest([self.doc(str(i)) for i in range(7)])
        profile = {"fingerprint": "fp"}
        state = self.store.native_index_checkpoint(profile, "mem")
        self.store.set_native_optimize_checkpoint({"provider": "voyage", "model": "voyage-4-lite", "dimensions": 1024, "schema_version": 2, "index_layout_version": 3, "memory": "mem", "fingerprint": "fp", "index_fingerprint": state["index_fingerprint"], "status": "optimized", "optimized_at": "2026-09-22T12:00:00+00:00", "revision": state["revision"]})
        self.store.set_sync(last_sync="source-sync", restored_at="restore-at", snapshot_path="private-path")
        self.store.record_reindex_control({"generation": 9, "scope": "all", "created_at": "time"})
        self.store.apply_pending_reindex_controls(batch_size=2)
        native_state = self.store.native_index_state_record()
        optimize = self.store.native_optimize_checkpoint()
        state_before = self.store.sync_status()
        runtime = self.runtime()
        self.assertEqual(runtime.sync_status(), state_before)
        self.assertEqual(runtime.native_optimize_checkpoint(), optimize)
        self.assertEqual(runtime.native_index_state_record(), native_state)
        batches = list(runtime.iter_documents(batch_size=2))
        self.assertEqual([len(batch) for batch in batches], [2, 2, 2, 1])
        self.assertEqual(runtime.conn.raw.info.transaction_status.name, "IDLE")
        path = Path(self.temporary.name) / "snapshot.jsonl"
        runtime.snapshot(path)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len([row for row in records if row["_funes_record"] == "memory"]), 7)
        self.assertEqual(next(row for row in records if row["_funes_record"] == "reindex_control")["row_cursor"], 2)
        self.assertEqual(records[-1], native_state)

    def test_runtime_readiness_and_reconnect_fail_closed_without_row_scans(self):
        runtime = self.runtime()
        # A hot restart must not need permission to scan source rows at all.
        with mock.patch.object(PostgresStore, "_rebuild_native_checkpoint_state_locked", side_effect=AssertionError("startup scan")), mock.patch.object(PostgresStore, "count", side_effect=AssertionError("startup count")):
            reopened = self.runtime()
            reopened.verify_schema()
        with self.store.lock, self.store.conn:
            self.store.conn.execute("UPDATE funes_schema_state SET migration_ready=FALSE WHERE id=1")
        with self.assertRaises(PostgresNotReady):
            PostgresStore(self.temporary.name, self.dsn)
        with self.assertRaises(PostgresNotReady):
            runtime.get("anything")
        with self.assertRaises(PostgresNotReady):
            runtime.ingest([self.doc()])
        with self.assertRaises(PostgresNotReady):
            runtime.reconnect()
        self.assertFalse((Path(self.temporary.name) / "funes.sqlite3").exists())
        self.store.finalize_migration(expected_count=0)
        runtime.reconnect()
        runtime.ingest([self.doc()])
        self.assertEqual(runtime.count(), 1)

    def test_finalization_keeps_revision_repairs_identity_sequence_and_fails_counts(self):
        self.store.ingest([self.doc()])
        self.store.set_sync(native_index_revision=1001)
        with self.store.lock, self.store.conn:
            self.store.conn.execute("UPDATE memories SET id=50 WHERE source_identity=?", ("source",))
            self.store.conn.execute("UPDATE funes_schema_state SET migration_ready=FALSE WHERE id=1")
        with self.assertRaises(PostgresNotReady):
            self.store.finalize_migration(expected_count=2)
        with self.assertRaises(PostgresNotReady):
            PostgresStore(self.temporary.name, self.dsn)
        self.store.finalize_migration(expected_count=1, source="test")
        self.assertEqual(self.store.native_index_state_record()["revision"], 1001)
        result = self.store.ingest([self.doc("next")])
        self.assertEqual(result["items"][0]["id"], 51)

    def test_bulk_restore_keeps_source_live_and_restores_native_state(self):
        document = self.doc(native_index_status="held_secret", native_index_profile="profile", native_index_memory="memory")
        self.store.begin_bulk_restore()
        self.store.restore_documents([document], batch_size=1)
        self.assertFalse(self.store.fts_ready())
        self.assertEqual(self.store.search("source")[0]["source_identity"], "source")
        self.store.finish_bulk_restore(rebuild_fts=False)
        self.assertEqual(self.store.native_index_checkpoint({"fingerprint": "profile"}, "memory")["held"], 1)
        with self.assertRaises(RuntimeError):
            self.store.finish_bulk_restore()

    def test_offline_bulk_copy_defers_indexes_and_triggers_then_finalizes(self):
        with self.assertRaises(PostgresNotReady):
            self.runtime().prepare_bulk_migration()
        with self.assertRaises(PostgresNotReady):
            self.store.prepare_bulk_migration()
        with self.store.lock, self.store.conn:
            self.store.conn.execute("UPDATE funes_schema_state SET migration_ready=FALSE WHERE id=1")
        self.store.prepare_bulk_migration()
        self.assertIsNone(self.store.conn.execute("SELECT to_regclass('memories_search_gin')").fetchone()[0])
        with self.assertRaises(PostgresNotReady):
            PostgresStore(self.temporary.name, self.dsn)
        raw = "原文\x00copy previous_response_id\ue0000"
        with self.store.conn.raw.cursor() as cursor:
            with cursor.copy("COPY memories(id,source_identity,raw_text,retrieval_text,content_hash,ingested_at,updated_at,native_index_status,native_index_profile,native_index_memory) FROM STDIN") as copy:
                values = [77, "copy-id", raw, None, "hash", "time", "time", "held_secret", "profile", "memory"]
                copy.write_row([encode_pg_text(value) if isinstance(value, str) else value for value in values])
        self.store.set_sync(native_index_revision=922, native_checkpoint_profile="profile", native_checkpoint_memory="memory")
        self.assertEqual(self.store.conn.execute("SELECT native_index_revision FROM sync_state WHERE id=1").fetchone()[0], 922)
        self.store.finalize_migration(expected_count=1, source="copy-test")
        runtime = self.runtime()
        self.assertEqual(runtime.get("copy-id")["raw_text"], raw)
        self.assertEqual(runtime.count(), 1)
        checkpoint = runtime.native_index_checkpoint({"fingerprint": "profile"}, "memory")
        self.assertEqual((checkpoint["revision"], checkpoint["eligible"], checkpoint["held"]), (922, 1, 1))
        self.assertEqual(runtime.canonical_index_candidates(10, "profile", "memory"), [])
        self.assertEqual(runtime.search("copy-id")[0]["source_identity"], "copy-id")
        created = runtime.ingest([self.doc("after-copy")])
        self.assertEqual(created["items"][0]["id"], 78)
        self.assertEqual(runtime.count(), 2)
        self.assertEqual(runtime.native_index_checkpoint({"fingerprint": "profile"}, "memory")["revision"], 923)

    def test_two_store_writers_serialize_before_read_merge(self):
        other = self.runtime()
        barrier = threading.Barrier(2)
        results, errors = [], []
        def worker(store):
            try:
                barrier.wait(timeout=5)
                results.append(store.ingest([self.doc()]))
            except Exception as error:
                errors.append(type(error).__name__)
        threads = [threading.Thread(target=worker, args=(store,)) for store in (self.store, other)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(sum(result["created"] for result in results), 1)
        self.assertEqual(sum(result["deduped"] for result in results), 1)
        self.assertEqual(self.store.count(), 1)


@unittest.skipUnless(os.getenv("FUNES_TEST_POSTGRES_DSN"), "FUNES_TEST_POSTGRES_DSN not configured")
class InheritedStoreContractTests(unittest.TestCase):
    def test_existing_merge_metadata_generation_and_checkpoint_contracts(self):
        """Run original, unchanged Store tests against isolated PG databases.

        Intentional backend-specific differences (SQLite DDL, BM25 ordering,
        PRAGMAs) are not asserted as shared source business semantics.
        """
        import psycopg
        from psycopg import sql
        from psycopg.conninfo import make_conninfo
        from service.tests.test_service import ServiceTests
        names = (
            "dedupe_and_update",
            "native_index_failure_counts_are_allowlisted",
            "same_revision_updates_only_derived_fields_in_place",
            "same_revision_updates_source_metadata_without_resetting_derived_state",
            "equal_timestamp_metadata_deltas_restore_deterministically",
            "source_metadata_fields_use_independent_high_water_clocks",
            "canonical_clock_only_payload_persists_provenance",
            "same_revision_native_status_update_preserves_raw_and_retrieval",
            "restore_does_not_regress_terminal_native_state",
            "restore_does_not_regress_new_profile_terminal_checkpoint",
            "multiple_chunks_same_source_path",
            "numeric_source_identity_precedes_sqlite_id",
            "existing_identities_is_ordered_chunked_and_selects_no_raw_payload",
            "snapshot_roundtrip_preserves_retrieval_and_native_state",
            "canonical_checkpoint_rebuilds_profile_mismatch_including_session",
            "legacy_codex_automation_output_is_retained_but_not_indexed",
            "native_state_record_can_project_pre_durable_status_update",
            "held_invalid_is_terminal_and_does_not_regress",
            "native_index_fingerprint_is_stable_across_restore_row_order",
            "native_optimize_checkpoint_restore_uses_monotonic_revision",
            "native_optimize_layout_version_snapshot_roundtrip",
            "native_optimize_old_marker_cannot_regress_layout_version",
            "reindex_generations_reset_only_derived_state",
            "reindex_row_cursor_is_bounded_and_survives_restart",
        )
        base = os.environ["FUNES_TEST_POSTGRES_DSN"]
        paths, stores = {}, []
        with psycopg.connect(base, autocommit=True) as admin:
            def factory(path):
                fresh = path not in paths
                if fresh:
                    schema = "funes_contract_" + uuid.uuid4().hex
                    admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
                    paths[path] = schema
                store = PostgresStore(path, make_conninfo(base, options="-csearch_path=" + paths[path] + ",public"), initialize=fresh)
                if fresh:
                    store.finalize_migration(expected_count=0)
                stores.append(store)
                return store
            try:
                with mock.patch("service.tests.test_service.Store", factory):
                    for name in names:
                        with self.subTest(contract=name):
                            result = unittest.TestResult()
                            ServiceTests("test_" + name).run(result)
                            self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
            finally:
                for store in stores:
                    store.close()
                for schema in paths.values():
                    admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


if __name__ == "__main__":
    unittest.main()
