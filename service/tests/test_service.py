import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
from unittest import mock
from http.client import HTTPConnection
from pathlib import Path
from http.server import ThreadingHTTPServer

from service.server import QUERY_PROMPT_VERSION, QUERY_RETRIEVAL_PROMPT, RETRIEVAL_PROMPT, App, Store, Translator, ingest_documents, make_handler, persist_translation_documents, prepare_ingest_documents


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = {k: os.environ.get(k) for k in ("FUNES_DATA_DIR", "FUNES_AUTH_TOKEN", "FUNES_API_TOKEN", "FUNES_STORAGE_KEY", "FUNES_STORAGE_REPO", "FUNES_SNAPSHOT_FILE", "FUNES_RESTORE_MANIFEST_FILE", "HF_TOKEN", "FUNES_REQUIRE_DURABLE_ACK", "FUNES_ALLOW_EMPTY_REMOTE", "FUNES_MAX_BODY_BYTES", "FUNES_RETRIEVAL_LANGUAGE_MODE", "TRANSLATION_BASE_URL", "TRANSLATION_API_KEY", "TRANSLATION_MODEL", "TRANSLATION_MAX_PER_INGEST", "TRANSLATION_QUERY_MAX_TOKENS", "TRANSLATION_RECONCILE_INTERVAL", "RETURN_RETRIEVAL_TEXT")}
        os.environ["FUNES_DATA_DIR"] = self.tmp.name
        os.environ["FUNES_AUTH_TOKEN"] = "test-token"
        for k in ("FUNES_API_TOKEN", "FUNES_STORAGE_KEY", "FUNES_STORAGE_REPO", "FUNES_SNAPSHOT_FILE", "FUNES_RESTORE_MANIFEST_FILE", "HF_TOKEN", "FUNES_REQUIRE_DURABLE_ACK", "FUNES_ALLOW_EMPTY_REMOTE", "FUNES_MAX_BODY_BYTES", "FUNES_RETRIEVAL_LANGUAGE_MODE", "TRANSLATION_BASE_URL", "TRANSLATION_API_KEY", "TRANSLATION_MODEL", "TRANSLATION_MAX_PER_INGEST", "TRANSLATION_QUERY_MAX_TOKENS", "TRANSLATION_RECONCILE_INTERVAL", "RETURN_RETRIEVAL_TEXT"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    @staticmethod
    def trace_read_connections(store, statements):
        original = store._read_connection

        def traced():
            connection = original()
            connection.set_trace_callback(statements.append)
            return connection

        return mock.patch.object(store, "_read_connection", side_effect=traced)

    def test_dedupe_and_update(self):
        store = Store(self.tmp.name)
        first = {"source_path": "a.md", "source_version": "1", "raw_text": "hello world", "project": "p"}
        self.assertEqual(store.ingest([first])["created"], 1)
        before_retry = store.get("a.md")
        self.assertEqual(store.ingest([first])["deduped"], 1)
        after_retry = store.get("a.md")
        self.assertEqual(
            after_retry["_source_metadata_clocks"],
            before_retry["_source_metadata_clocks"],
        )
        self.assertEqual(after_retry["updated_at"], before_retry["updated_at"])
        changed = dict(first, source_version="2", raw_text="hello revised")
        self.assertEqual(store.ingest([changed])["updated"], 1)
        self.assertEqual(store.count(), 1)
        self.assertEqual(store.get("a.md")["raw_text"], "hello revised")
        store.close()

    def test_native_index_failure_counts_are_allowlisted(self):
        store = Store(self.tmp.name)
        rows = [
            {"source_identity": f"failure-{index}", "raw_text": "private raw"}
            for index in range(6)
        ]
        store.ingest(rows)
        failures = [
            "TimeoutExpired",
            "native_exit",
            "invalid_report",
            "native_stale",
            "durability_pending",
            "provider said private raw",
        ]
        with store.lock, store.conn:
            for row, failure in zip(rows[:-2], failures[:-2], strict=True):
                store.conn.execute(
                    "UPDATE memories SET native_index_error=? WHERE source_identity=?",
                    (failure, row["source_identity"]),
                )
            store.conn.execute(
                "UPDATE memories SET native_index_error=? WHERE source_identity=?",
                (failures[-1], rows[-1]["source_identity"]),
            )
        waiting = store.get(rows[-2]["source_identity"])
        store.update_native_index(
            [
                {
                    "source_identity": waiting["source_identity"],
                    "source_version": waiting["source_version"],
                    "content_hash": waiting["content_hash"],
                    "native_index_version": None,
                    "native_index_status": "waiting_durability",
                    "native_index_profile": None,
                    "native_index_memory": None,
                    "native_indexed_at": None,
                    "native_index_error": "durability_pending",
                    "native_generation": waiting["native_generation"],
                }
            ]
        )
        with store.lock:
            pending = store.conn.execute(
                "SELECT native_index_pending FROM memories WHERE source_identity=?",
                (waiting["source_identity"],),
            ).fetchone()[0]
        try:
            counts = store.native_index_failure_counts()
        finally:
            store.close()
        self.assertEqual(pending, 0)
        self.assertEqual(
            counts,
            {
                "timeout": 1,
                "native_exit": 1,
                "invalid_report": 1,
                "stale": 1,
                "durability_pending": 1,
                "other": 1,
            },
        )
        self.assertNotIn("private raw", json.dumps(counts))

    def test_legacy_retrieval_only_fts_migrates_to_raw_primary_once(self):
        store = Store(self.tmp.name)
        store.ingest(
            [{
                "source_identity": "legacy-raw",
                "raw_text": "raw-only-needle",
                "retrieval_text": "unrelated shadow",
            }]
        )
        with store.lock, store.conn:
            for trigger in ("memories_ai", "memories_ad", "memories_au"):
                store.conn.execute(f"DROP TRIGGER {trigger}")
            store.conn.execute("DROP TABLE memories_fts")
            store.conn.execute(
                """CREATE VIRTUAL TABLE memories_fts USING fts5(
                retrieval_text, content='memories', content_rowid='id', tokenize='unicode61'
                )"""
            )
            store.conn.execute(
                """CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, retrieval_text) VALUES (new.id, new.retrieval_text);
                END"""
            )
            store.conn.execute(
                """CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, retrieval_text)
                VALUES('delete', old.id, old.retrieval_text);
                END"""
            )
            store.conn.execute(
                """CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, retrieval_text)
                VALUES('delete', old.id, old.retrieval_text);
                INSERT INTO memories_fts(rowid, retrieval_text) VALUES (new.id, new.retrieval_text);
                END"""
            )
            store.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        store.close()

        migrated = Store(self.tmp.name)
        try:
            columns = [
                row[1]
                for row in migrated.conn.execute("PRAGMA table_info(memories_fts)")
            ]
            self.assertEqual(
                columns, ["raw_text", "retrieval_text", "search_identifiers"]
            )
            self.assertEqual(
                migrated.search("raw-only-needle")[0]["source_identity"],
                "legacy-raw",
            )
        finally:
            migrated.close()

        reopened = Store(self.tmp.name)
        try:
            self.assertEqual(
                [row[1] for row in reopened.conn.execute("PRAGMA table_info(memories_fts)")],
                ["raw_text", "retrieval_text", "search_identifiers"],
            )
        finally:
            reopened.close()

    def test_bm25_prefers_raw_match_over_retrieval_shadow_match(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": "raw-hit",
                    "raw_text": "primaryneedle in original text",
                    "retrieval_text": "unrelated shadow",
                },
                {
                    "source_identity": "shadow-hit",
                    "raw_text": "unrelated original",
                    "retrieval_text": "primaryneedle in derived shadow",
                },
            ]
        )
        try:
            self.assertEqual(
                [item["source_identity"] for item in store.search("primaryneedle")],
                ["raw-hit", "shadow-hit"],
            )
        finally:
            store.close()

    def test_search_can_disable_broad_scans_without_losing_indexed_hits(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": "raw-hit",
                    "raw_text": "primaryneedle appears only in raw source",
                    "retrieval_text": "unrelated shadow",
                },
                {
                    "source_identity": "identifier-hit",
                    "raw_text": "讨论通过Tailscale连接办公室网络",
                },
            ]
        )
        statements = []
        try:
            with self.trace_read_connections(store, statements):
                self.assertEqual(
                    store.search("primaryneedle", allow_broad_scan=False)[0][
                        "source_identity"
                    ],
                    "raw-hit",
                )
                self.assertEqual(
                    store.search(
                        "之前的 Tailscale 延迟", allow_broad_scan=False
                    )[0]["source_identity"],
                    "identifier-hit",
                )
                self.assertEqual(
                    store.search('没有匹配 "', allow_broad_scan=False), []
                )
        finally:
            store.close()
        self.assertFalse(any(" LIKE " in sql.upper() for sql in statements))

    def test_translation_defaults_to_background_only(self):
        store = Store(self.tmp.name)
        try:
            self.assertEqual(Translator(store).max_per_ingest, 0)
        finally:
            store.close()

    def test_prepare_ingest_reads_reindex_generations_once_per_batch(self):
        app = App()
        try:
            with mock.patch.object(
                app.store,
                "latest_reindex_generation",
                wraps=app.store.latest_reindex_generation,
            ) as latest, mock.patch.object(
                app.store,
                "latest_embedding_generation",
                wraps=app.store.latest_embedding_generation,
            ) as latest_embedding:
                prepared = prepare_ingest_documents(
                    app,
                    [
                        {"source_identity": "batch-one", "source_type": "session", "raw_text": "one"},
                        {"source_identity": "batch-two", "source_type": "session", "raw_text": "two"},
                    ],
                )
            self.assertEqual(latest.call_count, 1)
            self.assertEqual(latest_embedding.call_count, 1)
            self.assertEqual(
                {item["retrieval_generation"] for item in prepared},
                {app.store.latest_reindex_generation()},
            )
        finally:
            app.close()

    def test_metadata_refresh_does_not_consume_pending_reindex_generation(self):
        store = Store(self.tmp.name)
        raw = "原始正文"
        store.ingest(
            [{
                "source_identity": "metadata-before-drain",
                "source_version": "v1",
                "raw_text": raw,
                "retrieval_text": "old retrieval shadow",
                "updated_at": "2026-09-14T00:00:00Z",
                "translation_hash": "old-translation",
                "translation_status": "ok",
                "native_index_version": "old-native",
                "native_index_status": "indexed",
                "native_index_profile": "profile-v1",
                "native_index_memory": "memory-v1",
                "source_missing": False,
            }]
        )
        store.record_reindex_control(
            {
                "generation": 1,
                "scope": "all",
                "created_at": "2026-09-14T01:00:00Z",
            }
        )
        app = mock.Mock(store=store)
        app.syncer.upload.return_value = {"durable": True}
        app.translator.pending_document.side_effect = AssertionError(
            "unchanged derived values must not schedule provider work"
        )
        result = ingest_documents(
            app,
            [{
                "source_identity": "metadata-before-drain",
                "source_version": "v1",
                "raw_text": raw,
                "updated_at": "2026-09-14T02:00:00Z",
                "device_id": "new-device",
                "source_missing": False,
            }],
        )
        before_drain = store.get("metadata-before-drain")
        self.assertEqual(result["items"][0]["status"], "metadata_updated")
        self.assertEqual(before_drain["retrieval_generation"], 0)
        self.assertEqual(before_drain["native_generation"], 0)
        self.assertEqual(before_drain["embedding_generation"], 0)
        self.assertEqual(before_drain["retrieval_text"], "old retrieval shadow")
        self.assertEqual(before_drain["native_index_status"], "indexed")
        app.translator.pending_document.assert_not_called()

        drained = store.drain_reindex_controls(10)
        after_drain = store.get("metadata-before-drain")
        self.assertEqual(drained["updated"], 1)
        self.assertEqual(after_drain["retrieval_generation"], 1)
        self.assertEqual(after_drain["native_generation"], 1)
        self.assertEqual(after_drain["embedding_generation"], 1)
        self.assertEqual(after_drain["retrieval_text"], raw)
        self.assertEqual(after_drain["translation_status"], "pending_provider")
        self.assertIsNone(after_drain["native_index_status"])
        store.close()

    def test_same_revision_updates_only_derived_fields_in_place(self):
        store = Store(self.tmp.name)
        raw = "原始正文"
        original = {
            "source_identity": "derived-only",
            "source_version": "v1",
            "raw_text": raw,
            "retrieval_text": raw,
            "translation_hash": "same-hash",
            "translation_version": "v1",
            "translation_status": "pending_provider",
        }
        self.assertEqual(store.ingest([original])["created"], 1)
        reconciled = dict(
            original,
            retrieval_text=raw + " english retrieval",
            translation_status="ok",
        )
        result = store.ingest([reconciled])
        item = store.get("derived-only")
        self.assertEqual(result["updated"], 1)
        self.assertEqual(result["items"][0]["status"], "derived_updated")
        self.assertEqual(store.count(), 1)
        self.assertEqual(item["raw_text"], raw)
        self.assertEqual(item["retrieval_text"], raw + " english retrieval")
        self.assertEqual(item["translation_status"], "ok")
        # An older/raw pending delta can restore later without regressing the
        # successfully reconciled shadow.
        self.assertEqual(store.ingest([original])["deduped"], 1)
        self.assertEqual(store.get("derived-only")["translation_status"], "ok")
        store.close()

    def test_same_revision_updates_source_metadata_without_resetting_derived_state(self):
        store = Store(self.tmp.name)
        raw = "stable source bytes"
        metadata_fields = (
            "device_id",
            "project",
            "repo",
            "worktree",
            "source_agent",
            "source_type",
            "session_id",
            "message_id",
            "role",
            "timestamp",
            "source_path",
            "agent_type",
            "parent_session_id",
            "agent_id",
        )
        original = {
            "source_identity": "metadata-only",
            "source_version": "v1",
            "raw_text": raw,
            "retrieval_text": "stable retrieval shadow",
            "updated_at": "2026-09-14T01:00:00Z",
            "content_type": "memory",
            "source_missing": False,
            "translation_hash": "translation-hash",
            "translation_version": "translation-v1",
            "translation_status": "ok",
            "retrieval_updated_at": "2026-09-14T01:01:00Z",
            "native_index_version": "native-v1",
            "native_index_status": "indexed",
            "native_index_profile": "profile-v1",
            "native_index_memory": "memory-v1",
            "native_indexed_at": "2026-09-14T01:02:00Z",
            "native_index_error": "retained-checkpoint-detail",
            "retrieval_generation": 4,
            "native_generation": 5,
            "embedding_generation": 6,
            "metadata": {"label": "old", "nested": {"revision": 1}},
            **{name: f"old-{name}" for name in metadata_fields},
        }
        self.assertEqual(store.ingest([original])["created"], 1)
        with store.lock, store.conn:
            store.conn.execute(
                "UPDATE memories SET native_index_pending=0 WHERE source_identity=?",
                (original["source_identity"],),
            )
        before = store.get(original["source_identity"])
        derived_fields = (
            "raw_text",
            "retrieval_text",
            "content_hash",
            "source_version",
            "translation_hash",
            "translation_version",
            "translation_status",
            "retrieval_updated_at",
            "native_index_version",
            "native_index_status",
            "native_index_profile",
            "native_index_memory",
            "native_indexed_at",
            "native_index_error",
            "retrieval_generation",
            "native_generation",
            "embedding_generation",
            "content_type",
            "source_missing",
        )

        refreshed = {
            "source_identity": original["source_identity"],
            "source_version": original["source_version"],
            "raw_text": raw,
            "updated_at": "2026-09-14T02:00:00Z",
            "metadata": {"label": "new", "nested": {"revision": 2}},
            **{name: f"new-{name}" for name in metadata_fields},
        }
        result = store.ingest([refreshed])
        current = store.get(original["source_identity"])
        self.assertEqual(result["updated"], 1)
        self.assertEqual(result["items"][0]["status"], "metadata_updated")
        self.assertEqual(
            {name: current[name] for name in metadata_fields},
            {name: refreshed[name] for name in metadata_fields},
        )
        self.assertEqual(current["metadata"], refreshed["metadata"])
        self.assertEqual(current["updated_at"], refreshed["updated_at"])
        self.assertEqual(
            {name: current[name] for name in derived_fields},
            {name: before[name] for name in derived_fields},
        )
        self.assertEqual(store.pending_translations(10), [])
        with store.lock:
            self.assertEqual(
                store.conn.execute(
                    "SELECT native_index_pending FROM memories WHERE source_identity=?",
                    (original["source_identity"],),
                ).fetchone()[0],
                0,
            )

        # A legacy generation-zero delta may still replay later. Its older
        # source timestamp must not roll current source metadata backward.
        stale = {
            **refreshed,
            "updated_at": "2026-09-14T00:00:00Z",
            "metadata": {"label": "stale"},
            **{name: f"stale-{name}" for name in metadata_fields},
        }
        stale_result = store.ingest([stale])
        current = store.get(original["source_identity"])
        self.assertEqual(stale_result["items"][0]["status"], "deduped")
        self.assertEqual(current["metadata"], refreshed["metadata"])
        self.assertEqual(current["updated_at"], refreshed["updated_at"])
        self.assertEqual(
            {name: current[name] for name in metadata_fields},
            {name: refreshed[name] for name in metadata_fields},
        )

        # A partial device-only refresh changes no other source metadata and
        # does not invalidate the native/embedding checkpoint.
        device_refresh = {
            "source_identity": original["source_identity"],
            "source_version": original["source_version"],
            "raw_text": raw,
            "updated_at": "2026-09-14T03:00:00Z",
            "device_id": "newest-device",
            "source_missing": False,
        }
        app = mock.Mock()
        app.store = store
        app.translator.pending_document.side_effect = AssertionError(
            "metadata-only ingest must not schedule provider work"
        )
        app.syncer.upload.return_value = {"durable": True}
        device_result = ingest_documents(app, [device_refresh])
        current = store.get(original["source_identity"])
        self.assertEqual(device_result["items"][0]["status"], "metadata_updated")
        app.translator.pending_document.assert_not_called()
        self.assertEqual(current["device_id"], "newest-device")
        self.assertEqual(current["project"], refreshed["project"])
        self.assertEqual(current["metadata"], refreshed["metadata"])
        self.assertEqual(
            {name: current[name] for name in derived_fields},
            {name: before[name] for name in derived_fields},
        )

        # The canonical sync payload may carry the same source_missing value in
        # metadata instead. A failed metadata-delta upload must still leave the
        # already-durable native checkpoint usable for the retry.
        app.syncer.upload.return_value = {
            "uploaded": False,
            "durable": False,
            "reason": "offline",
        }
        failed_refresh = {
            "source_identity": original["source_identity"],
            "source_version": original["source_version"],
            "raw_text": raw,
            "updated_at": "2026-09-14T04:00:00Z",
            "device_id": "retry-device",
            "metadata": {**current["metadata"], "source_missing": False},
        }
        failed_result = ingest_documents(app, [failed_refresh])
        current = store.get(original["source_identity"])
        self.assertEqual(failed_result["items"][0]["status"], "metadata_updated")
        self.assertFalse(failed_result["durable"])
        self.assertEqual(failed_result["error"], "durability_pending")
        self.assertEqual(current["native_index_status"], "indexed")
        self.assertEqual(current["native_index_profile"], "profile-v1")
        self.assertEqual(current["embedding_generation"], 6)
        self.assertEqual(store.pending_translations(10), [])
        app.translator.pending_document.assert_not_called()
        with store.lock:
            self.assertEqual(
                store.conn.execute(
                    "SELECT native_index_pending FROM memories WHERE source_identity=?",
                    (original["source_identity"],),
                ).fetchone()[0],
                0,
            )
        store.close()

    def test_equal_timestamp_metadata_deltas_restore_deterministically(self):
        from service.server import SnapshotSync

        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        raw = "same immutable source"
        fields = (
            "device_id",
            "project",
            "repo",
            "worktree",
            "source_agent",
            "source_type",
            "session_id",
            "message_id",
            "role",
            "timestamp",
            "source_path",
            "agent_type",
            "parent_session_id",
            "agent_id",
        )
        base = {
            "source_identity": "metadata-tie",
            "source_version": "v1",
            "raw_text": raw,
            "retrieval_text": "retained shadow",
            "updated_at": "2026-09-14T00:00:00Z",
            "translation_status": "ok",
            "native_index_status": "indexed",
            "native_index_profile": "profile-v1",
            "native_index_memory": "memory-v1",
            "metadata": {"label": "base"},
            **{name: "base" for name in fields},
            # Both replacements sort below their old values. A row-wide clock
            # would incorrectly keep these values after the other partial delta
            # advances updated_at.
            "device_id": "old-device",
            "project": "old-project",
        }

        def candidate(field, value, label):
            return {
                "source_identity": base["source_identity"],
                "source_version": base["source_version"],
                "raw_text": raw,
                "updated_at": "2026-09-14T01:00:00Z",
                "source_missing": False,
                "metadata": {"label": label, "nested": {"value": label}},
                field: value,
            }

        # This is the minimal non-commutative counterexample for a whole-state
        # fingerprint: each equal-time delta changes a different source field.
        deltas = [
            {**candidate("device_id", "dev-0", "device"), "agent_id": None},
            {**candidate("project", "proj-0", "project"), "agent_id": "b"},
        ]
        artifact_dir = Path(self.tmp.name) / "metadata-clock-artifacts"
        artifact_dir.mkdir()
        artifact_names = []
        api = mock.Mock()
        api.repo_info.return_value = mock.Mock(sha="head-1")
        api.list_repo_tree.return_value = []
        api.file_exists.return_value = False

        def capture_upload(**kwargs):
            name = kwargs["path_in_repo"]
            target_path = artifact_dir / name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(
                Path(kwargs["path_or_fileobj"]).read_bytes()
            )
            artifact_names.append(name)
            return mock.Mock(oid=f"head-{len(artifact_names) + 1}")

        api.upload_file.side_effect = capture_upload
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            for delta in deltas:
                # Build the remote delta through the production canonical path.
                # The private clock map must survive get_many -> encryption.
                branch_dir = tempfile.TemporaryDirectory()
                branch = Store(branch_dir.name)
                try:
                    branch.ingest([base])
                    self.assertEqual(
                        branch.ingest([delta])["items"][0]["status"],
                        "metadata_updated",
                    )
                    durable = branch.get_many([base["source_identity"]])
                    self.assertIn("_source_metadata_clocks", durable[0])
                    self.assertTrue(SnapshotSync(branch).upload(durable)["durable"])
                finally:
                    branch.close()
                    branch_dir.cleanup()

        def replay(order):
            target_dir = tempfile.TemporaryDirectory()
            target = Store(target_dir.name)
            target.ingest([base])
            try:
                with mock.patch(
                    "huggingface_hub.hf_hub_download",
                    side_effect=AssertionError("prefetched deltas must be used"),
                ):
                    for position, index in enumerate(order):
                        reader = SnapshotSync(target)
                        reader._restore_prefetch_root = artifact_dir
                        reader._restore_file(artifact_names[index])
                        if position == 0:
                            # Reopen SQLite between artifacts: correctness must
                            # come from persisted clocks, not in-memory state.
                            target.close()
                            target = Store(target_dir.name)
                return target.get(base["source_identity"])
            finally:
                target.close()
                target_dir.cleanup()

        forward = replay((0, 1))
        reverse = replay((1, 0))
        self.assertEqual(
            {name: forward[name] for name in (*fields, "metadata", "updated_at")},
            {name: reverse[name] for name in (*fields, "metadata", "updated_at")},
        )
        self.assertIn(forward["metadata"], [delta["metadata"] for delta in deltas])
        self.assertEqual(
            (forward["device_id"], forward["project"]),
            ("dev-0", "proj-0"),
        )
        self.assertEqual(forward["agent_id"], "b")
        self.assertEqual(forward["raw_text"], raw)
        self.assertEqual(forward["retrieval_text"], "retained shadow")
        self.assertEqual(forward["native_index_status"], "indexed")

    def test_source_metadata_fields_use_independent_high_water_clocks(self):
        raw = "same immutable source"
        base = {
            "source_identity": "metadata-independent-clocks",
            "source_version": "v1",
            "raw_text": raw,
            "retrieval_text": "retained shadow",
            "updated_at": "2026-09-14T00:00:00Z",
            "device_id": "old-device",
            "project": "old-project",
            "translation_status": "ok",
            "native_index_status": "indexed",
        }
        device_t2 = {
            "source_identity": base["source_identity"],
            "source_version": "v1",
            "raw_text": raw,
            "updated_at": "2026-09-14T02:00:00Z",
            "device_id": "dev-0",
            "source_missing": False,
        }
        project_t1 = {
            "source_identity": base["source_identity"],
            "source_version": "v1",
            "raw_text": raw,
            "updated_at": "2026-09-14T01:00:00Z",
            "project": "proj-0",
            "source_missing": False,
        }

        results = []
        for order in ((device_t2, project_t1), (project_t1, device_t2)):
            directory = tempfile.TemporaryDirectory()
            store = Store(directory.name)
            try:
                store.ingest([base])
                statuses = [store.ingest([delta])["items"][0]["status"] for delta in order]
                current = store.get(base["source_identity"])
                self.assertEqual(statuses, ["metadata_updated", "metadata_updated"])
                self.assertEqual(current["updated_at"], device_t2["updated_at"])
                self.assertEqual(current["retrieval_text"], "retained shadow")
                self.assertEqual(current["native_index_status"], "indexed")
                results.append((current["device_id"], current["project"]))

                stale = dict(
                    device_t2,
                    updated_at="2026-09-14T01:30:00Z",
                    device_id="zzzz-device",
                )
                stale_result = store.ingest([stale])
                self.assertEqual(stale_result["items"][0]["status"], "deduped")
                self.assertEqual(store.get(base["source_identity"])["device_id"], "dev-0")
            finally:
                store.close()
                directory.cleanup()
        self.assertEqual(results, [("dev-0", "proj-0"), ("dev-0", "proj-0")])

    def test_canonical_clock_only_payload_persists_provenance(self):
        store = Store(self.tmp.name)
        raw = "same immutable source"
        identity = "metadata-clock-only"
        store.ingest(
            [{
                "source_identity": identity,
                "source_version": "v1",
                "raw_text": raw,
                "retrieval_text": "retained shadow",
                "updated_at": "2026-09-14T00:00:00Z",
                "project": "same-project",
                "native_index_status": "indexed",
            }]
        )
        canonical = store.get(identity)
        canonical["updated_at"] = "2026-09-14T02:00:00Z"
        canonical["_source_metadata_clocks"] = {
            **canonical["_source_metadata_clocks"],
            "project": canonical["updated_at"],
        }
        result = store.ingest([canonical])
        current = store.get(identity)
        self.assertEqual(result["items"][0]["status"], "metadata_updated")
        self.assertEqual(
            current["_source_metadata_clocks"]["project"],
            canonical["updated_at"],
        )
        self.assertEqual(current["retrieval_text"], "retained shadow")
        self.assertEqual(current["native_index_status"], "indexed")

        stale = {
            "source_identity": identity,
            "source_version": "v1",
            "raw_text": raw,
            "updated_at": "2026-09-14T01:00:00Z",
            "project": "zzzz-project",
        }
        self.assertEqual(store.ingest([stale])["items"][0]["status"], "deduped")
        self.assertEqual(store.get(identity)["project"], "same-project")
        store.close()

    def test_source_metadata_clock_migration_accepts_legacy_database(self):
        identity = "legacy-metadata-clock"
        store = Store(self.tmp.name)
        store.ingest(
            [{
                "source_identity": identity,
                "source_version": "v1",
                "raw_text": "same immutable source",
                "updated_at": "2026-09-14T00:00:00Z",
                "device_id": "old-device",
            }]
        )
        store.close()
        connection = sqlite3.connect(Path(self.tmp.name) / "funes.sqlite3")
        connection.execute(
            "ALTER TABLE memories DROP COLUMN source_metadata_clock_json"
        )
        connection.commit()
        connection.close()

        store = Store(self.tmp.name)
        result = store.ingest(
            [{
                "source_identity": identity,
                "source_version": "v1",
                "raw_text": "same immutable source",
                "updated_at": "2026-09-14T01:00:00Z",
                "device_id": "dev-0",
            }]
        )
        current = store.get(identity)
        self.assertEqual(result["items"][0]["status"], "metadata_updated")
        self.assertEqual(current["device_id"], "dev-0")
        self.assertEqual(
            current["_source_metadata_clocks"]["device_id"],
            "2026-09-14T01:00:00Z",
        )
        store.close()

    def test_same_revision_native_status_update_preserves_raw_and_retrieval(self):
        store = Store(self.tmp.name)
        raw = "原始正文"
        store.ingest(
            [{
                "source_identity": "native-state",
                "source_version": "v1",
                "raw_text": raw,
                "retrieval_text": "english retrieval shadow",
                "translation_status": "ok",
                "retrieval_updated_at": "2026-09-13T01:00:00Z",
            }]
        )
        result = store.ingest(
            [{
                "source_identity": "native-state",
                "source_version": "v1",
                "raw_text": raw,
                "native_index_version": "canonical-v1",
                "native_index_status": "indexed",
                "native_indexed_at": "2026-09-13T02:00:00Z",
            }]
        )
        item = store.get("native-state")
        self.assertEqual(result["items"][0]["status"], "derived_updated")
        self.assertEqual(item["raw_text"], raw)
        self.assertEqual(item["retrieval_text"], "english retrieval shadow")
        self.assertEqual(item["translation_status"], "ok")
        self.assertEqual(item["native_index_status"], "indexed")
        changed = store.ingest(
            [{
                "source_identity": "native-state",
                "source_version": "v1",
                "raw_text": raw,
                "source_missing": True,
            }]
        )
        item = store.get("native-state")
        self.assertEqual(changed["items"][0]["status"], "derived_updated")
        self.assertTrue(item["source_missing"])
        self.assertIsNone(item["native_index_status"])
        store.close()

    def test_restore_does_not_regress_terminal_native_state(self):
        store = Store(self.tmp.name)
        base = {
            "source_identity": "monotonic-native",
            "source_version": "raw-v1",
            "raw_text": "原始正文",
            "retrieval_text": "english retrieval shadow",
            "translation_hash": "translation-hash",
            "translation_version": "translation-version",
            "translation_status": "ok",
            "retrieval_updated_at": "2026-09-13T01:00:00Z",
            "native_index_version": None,
            "native_index_status": None,
            "native_indexed_at": None,
            "native_index_error": None,
        }
        store.ingest([base])
        indexed = {
            **base,
            "native_index_version": "canonical-v1",
            "native_index_status": "indexed",
            "native_indexed_at": "2026-09-13T02:00:00Z",
        }
        store.ingest([indexed])
        store.restore_documents([base])
        item = store.get("monotonic-native")
        self.assertEqual(item["native_index_status"], "indexed")
        self.assertEqual(item["native_index_version"], "canonical-v1")
        held = {
            **indexed,
            "native_index_status": "held_secret",
            "native_indexed_at": None,
        }
        store.ingest([held])
        store.restore_documents(
            [{
                **base,
                "native_index_status": "retry",
                "native_index_error": "native_exit",
            }]
        )
        item = store.get("monotonic-native")
        self.assertEqual(item["native_index_status"], "held_secret")
        self.assertIsNone(item["native_index_error"])
        store.close()

    def test_restore_does_not_regress_new_profile_terminal_checkpoint(self):
        store = Store(self.tmp.name)
        base = {
            "source_identity": "profile-monotonic",
            "source_version": "raw-v1",
            "raw_text": "durable raw",
            "translation_status": "skipped_raw_mode",
        }
        old_profile = {
            **base,
            "native_index_version": "old-version",
            "native_index_status": "indexed",
            "native_index_profile": "old-profile",
            "native_indexed_at": "2026-09-13T01:00:00Z",
        }
        new_profile = {
            **base,
            "native_index_version": "new-version",
            "native_index_status": "indexed",
            "native_index_profile": "new-profile",
            "native_indexed_at": "2026-09-13T02:00:00.123456+00:00",
        }
        store.ingest([old_profile])
        store.ingest([new_profile])
        store.restore_documents([old_profile])
        try:
            item = store.get("profile-monotonic")
            self.assertEqual(item["native_index_version"], "new-version")
            self.assertEqual(item["native_index_profile"], "new-profile")
            self.assertEqual(
                item["native_indexed_at"], "2026-09-13T02:00:00.123456+00:00"
            )
        finally:
            store.close()

    def test_multiple_chunks_same_source_path(self):
        store = Store(self.tmp.name)
        result = store.ingest([
            {"source_path": "a.md", "chunk_id": "0", "raw_text": "first chunk"},
            {"source_path": "a.md", "chunk_id": "1", "raw_text": "second chunk"},
        ])
        self.assertEqual(result["created"], 2)
        self.assertEqual(store.count(), 2)
        self.assertEqual({r["source_identity"] for r in store.sources()}, {"a.md#chunk:0", "a.md#chunk:1"})
        store.close()

    def test_numeric_source_identity_precedes_sqlite_id(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {"source_identity": "first", "raw_text": "row one"},
                {"source_identity": "1", "raw_text": "numeric identity"},
            ]
        )
        self.assertEqual(store.get("1")["raw_text"], "numeric identity")
        store.close()

    def test_get_and_search_do_not_wait_for_writer_python_lock(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": "read-target",
                    "raw_text": "independent-reader-marker",
                },
                {"source_identity": "write-target", "raw_text": "writer row"},
            ]
        )
        writer_ready = threading.Event()
        release_writer = threading.Event()
        read_done = threading.Event()
        result = {}

        def hold_uncommitted_write():
            with store.lock:
                store.conn.execute("BEGIN IMMEDIATE")
                try:
                    store.conn.execute(
                        "UPDATE memories SET project='in-flight' WHERE source_identity='write-target'"
                    )
                    writer_ready.set()
                    release_writer.wait(5)
                finally:
                    store.conn.rollback()

        def read_while_writer_is_active():
            result["get"] = store.get("read-target")
            result["search"] = store.search("independent-reader-marker")
            read_done.set()

        writer = threading.Thread(target=hold_uncommitted_write)
        reader = threading.Thread(target=read_while_writer_is_active)
        writer.start()
        self.assertTrue(writer_ready.wait(2))
        reader.start()
        try:
            self.assertTrue(
                read_done.wait(2),
                "read-only get/search waited behind the writer Python lock",
            )
        finally:
            release_writer.set()
            writer.join(5)
            reader.join(5)
            store.close()
        self.assertEqual(result["get"]["raw_text"], "independent-reader-marker")
        self.assertEqual(
            result["search"][0]["source_identity"], "read-target"
        )

    def test_existing_identities_is_ordered_chunked_and_selects_no_raw_payload(self):
        store = Store(self.tmp.name)
        store.ingest(
            [{"source_identity": "present", "raw_text": "raw-secret-must-not-load"}]
        )
        statements = []
        store.conn.set_trace_callback(statements.append)
        try:
            identities = ["missing-first", "present", "present"] + [
                f"missing-{index}" for index in range(500)
            ]
            self.assertEqual(store.existing_identities(identities), ["present"])
        finally:
            store.conn.set_trace_callback(None)
            store.close()
        selects = [statement for statement in statements if statement.startswith("SELECT")]
        self.assertEqual(len(selects), 2)
        self.assertTrue(all(statement.startswith("SELECT source_identity FROM") for statement in selects))
        self.assertTrue(all("raw_text" not in statement for statement in selects))

    def _server(self):
        app = App()
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), app.close()))
        return server

    def _request(self, server, method, path, payload=None, token="test-token"):
        conn = HTTPConnection(*server.server_address)
        headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
        conn.request(method, path, json.dumps(payload or {}).encode(), headers)
        response = conn.getresponse()
        data = json.loads(response.read())
        conn.close()
        return response.status, data

    def _request_bytes(self, server, body, headers=None):
        conn = HTTPConnection(*server.server_address)
        request_headers = {"Authorization": "Bearer test-token", "Content-Type": "application/json"}
        request_headers.update(headers or {})
        conn.request("POST", "/ingest", body, request_headers)
        response = conn.getresponse()
        data = json.loads(response.read())
        conn.close()
        return response.status, data

    def test_gzip_ingest_rejects_invalid_and_oversized_payloads(self):
        server = self._server()
        valid = gzip.compress(json.dumps({"documents": [{"source_identity": "gzip-valid", "raw_text": "compressed"}]}).encode())
        status, result = self._request_bytes(server, valid, {"Content-Encoding": "gzip"})
        self.assertEqual(status, 200)
        self.assertEqual(result["created"], 1)

        status, result = self._request_bytes(server, b"not-a-gzip-stream", {"Content-Encoding": "gzip"})
        self.assertEqual(status, 400)
        self.assertEqual(result["error"], "invalid gzip request body")
        status, result = self._request_bytes(server, valid[:-1], {"Content-Encoding": "gzip"})
        self.assertEqual(status, 400)
        self.assertEqual(result["error"], "invalid gzip request body")
        corrupt_deflate = bytes.fromhex("1f8b0800000000000003ffff0000000000000000")
        status, result = self._request_bytes(server, corrupt_deflate, {"Content-Encoding": "gzip"})
        self.assertEqual(status, 400)
        self.assertEqual(result["error"], "invalid gzip request body")
        status, result = self._request_bytes(server, valid, {"Content-Encoding": "br"})
        self.assertEqual(status, 400)
        self.assertEqual(result["error"], "unsupported Content-Encoding")
        status, _ = self._request(server, "POST", "/get", {"id": "gzip-invalid"})
        self.assertEqual(status, 404)

        os.environ["FUNES_MAX_BODY_BYTES"] = "128"
        oversized = gzip.compress(json.dumps({"documents": [{"source_identity": "gzip-oversized", "raw_text": "x" * 1024}]}).encode())
        status, result = self._request_bytes(server, oversized, {"Content-Encoding": "gzip"})
        self.assertEqual(status, 400)
        self.assertEqual(result["error"], "request too large")
        status, _ = self._request(server, "POST", "/get", {"id": "gzip-oversized"})
        self.assertEqual(status, 404)

    def test_auth_search_get(self):
        server = self._server()
        status, _ = self._request(server, "POST", "/ingest", {"raw_text": "alpha beta", "source_path": "x"}, token="bad")
        self.assertEqual(status, 401)
        status, result = self._request(server, "POST", "/ingest", {"raw_text": "alpha beta", "source_path": "x"})
        self.assertEqual(status, 200)
        ident = result["items"][0]["id"]
        status, found = self._request(server, "POST", "/search", {"query": "alpha"})
        self.assertEqual(status, 200)
        self.assertEqual(found["results"][0]["raw_text"], "alpha beta")
        self.assertNotIn("_source_metadata_clocks", found["results"][0])
        status, item = self._request(server, "POST", "/get", {"id": ident})
        self.assertEqual(status, 200)
        self.assertEqual(item["source_path"], "x")
        self.assertNotIn("_source_metadata_clocks", item)

    def test_sources_check_requires_auth_and_returns_only_present_and_missing(self):
        server = self._server()
        secret = "raw-secret-must-not-leak"
        status, _ = self._request(
            server,
            "POST",
            "/ingest",
            {"source_identity": "present", "raw_text": secret},
        )
        self.assertEqual(status, 200)
        payload = {"source_identities": ["missing", "present", "missing", "present"]}
        status, body = self._request(
            server, "POST", "/sources/check", payload, token="bad"
        )
        self.assertEqual(status, 401)
        status, body = self._request(server, "POST", "/sources/check", payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {"ok": True, "present": ["present"], "missing": ["missing"]},
        )
        self.assertNotIn(secret, json.dumps(body))

    def test_sources_check_validates_bounds_and_restore_readiness(self):
        server = self._server()
        status, body = self._request(
            server, "POST", "/sources/check", {"source_identities": "invalid"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "source_identities must be a list")
        status, body = self._request(
            server,
            "POST",
            "/sources/check",
            {"source_identities": [f"id-{index}" for index in range(5001)]},
        )
        self.assertEqual(status, 400)
        self.assertIn("at most 5000", body["error"])

        app = App()
        app.syncer.restoring = True
        unavailable = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        thread = threading.Thread(target=unavailable.serve_forever, daemon=True)
        thread.start()
        try:
            status, body = self._request(
                unavailable,
                "POST",
                "/sources/check",
                {"source_identities": ["id"]},
            )
        finally:
            app.syncer.restoring = False
            unavailable.shutdown()
            unavailable.server_close()
            thread.join(timeout=2)
            app.close()
        self.assertEqual(status, 503)
        self.assertEqual(body, {"ok": False, "error": "restore_in_progress"})

    def test_compatibility_search_fuses_expanded_raw_and_rewrite_rankings(self):
        calls = []

        class Translator:
            def rewrite_query(self, query):
                calls.append(("rewrite", query))
                return "provider query"

        class Store:
            def search(self, query, limit, filters):
                calls.append(("search", query, limit, filters))
                if query == "raw query":
                    return [
                        {"source_identity": "raw-only", "raw_text": "raw BM25", "retrieval_text": "raw shadow"},
                        {"source_identity": "both", "raw_text": "both raw", "retrieval_text": "both shadow"},
                    ]
                return [
                    {"source_identity": "provider-only", "raw_text": "provider raw", "retrieval_text": "provider shadow"},
                    {"source_identity": "both", "raw_text": "both raw", "retrieval_text": "both shadow"},
                ]

        app = type("CompatibilityApp", (), {})()
        app.translator = Translator()
        app.store = Store()
        app.syncer = type("Syncer", (), {"restoring": False})()
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, body = self._request(
                server,
                "POST",
                "/search",
                {"query": "raw query", "limit": 3, "project": "demo"},
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(status, 200)
        self.assertEqual(
            [item["source_identity"] for item in body["results"]],
            ["both", "raw-only", "provider-only"],
        )
        self.assertTrue(all("retrieval_text" not in item for item in body["results"]))
        self.assertEqual(
            calls,
            [
                ("rewrite", "raw query"),
                ("search", "raw query", 9, {"project": "demo"}),
                ("search", "provider query", 9, {"project": "demo"}),
            ],
        )

    def test_chinese_fallback_and_normalization_cache(self):
        store = Store(self.tmp.name)
        tr = Translator(store)
        self.assertEqual(tr.rewrite("  中文   查询 "), "中文 查询")
        # Configure an unreachable endpoint: retries must fall back without raising.
        os.environ.update(FUNES_RETRIEVAL_LANGUAGE_MODE="auto", TRANSLATION_BASE_URL="http://127.0.0.1:1", TRANSLATION_API_KEY="x", TRANSLATION_MODEL="m")
        tr = Translator(store)
        self.assertEqual(tr.rewrite("中文 查询"), "中文 查询")
        store.translation_put(tr._cache_key("中文 查询"), "cached words")
        self.assertEqual(tr.rewrite("中文 查询"), "cached words")
        store.ingest([{"source_path": "zh", "raw_text": "中文检索内容"}])
        self.assertEqual(store.search("中文")[0]["raw_text"], "中文检索内容")
        store.close()

    def test_chinese_query_uses_technical_fts_before_character_scan(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": "target",
                    "source_agent": "pi",
                    "role": "user",
                    "raw_text": "MacBook Pro通过Tailscale连接办公室Mac mini时高延迟。",
                },
                {
                    "source_identity": "wrong-agent",
                    "source_agent": "codex",
                    "role": "user",
                    "raw_text": "Pi Tailscale 延迟",
                },
                {
                    "source_identity": "wrong-role",
                    "source_agent": "pi",
                    "role": "assistant",
                    "raw_text": "Pi Tailscale 延迟",
                },
            ]
        )
        statements = []
        with self.trace_read_connections(store, statements):
            results = store.search(
                "之前 Pi 里讨论过的 Tailscale 延迟",
                filters={"source_agent": "pi", "role": "user"},
            )
        self.assertEqual([item["source_identity"] for item in results], ["target"])
        traced = "\n".join(statements).upper()
        self.assertIn("MATCH '\"TAILSCALE\"'", traced)
        self.assertNotIn(" LIKE ", traced)

        statements.clear()
        with self.trace_read_connections(store, statements):
            malformed_results = store.search(
                '之前 Tailscale "',
                filters={"source_agent": "pi", "role": "user"},
            )
        self.assertEqual(
            [item["source_identity"] for item in malformed_results], ["target"]
        )
        malformed_trace = "\n".join(statements).upper()
        self.assertIn("MATCH '\"TAILSCALE\"'", malformed_trace)
        self.assertNotIn(" LIKE ", malformed_trace)
        store.close()

    def test_chinese_technical_fts_skips_like_when_one_hit_covers_all_terms(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": "fast-target",
                    "source_agent": "pi",
                    "project": "fast",
                    "raw_text": "Pi Tailscale latency analysis",
                },
                {
                    "source_identity": "wrong-project",
                    "source_agent": "pi",
                    "project": "other",
                    "raw_text": "Pi Tailscale unrelated",
                },
            ]
        )
        statements = []
        with self.trace_read_connections(store, statements):
            results = store.search(
                "之前 Pi Tailscale 的延迟",
                filters={"project": "fast"},
            )
        self.assertEqual([item["source_identity"] for item in results], ["fast-target"])
        self.assertFalse(any(" LIKE " in statement.upper() for statement in statements))
        store.close()

    def test_chinese_technical_like_recovers_strong_mixed_token(self):
        store = Store(self.tmp.name)
        documents = [
            {
                "source_identity": "mixed-target",
                "source_agent": "pi",
                "role": "user",
                "project": "mixed",
                "raw_text": "讨论通过Tailscale连接办公室网络",
                "updated_at": "2020-01-01T00:00:00Z",
            }
        ]
        documents.extend(
            {
                "source_identity": f"weak-distractor-{index}",
                "source_agent": "pi",
                "role": "user",
                "project": "mixed",
                "raw_text": "Mac unrelated notes",
                "updated_at": f"2026-01-01T00:00:0{index}Z",
            }
            for index in range(6)
        )
        store.ingest(documents)

        results = store.search(
            "讨论 Mac Tailscale",
            limit=1,
            filters={"source_agent": "pi", "role": "user"},
        )

        self.assertEqual(
            [item["source_identity"] for item in results], ["mixed-target"]
        )
        store.close()

    def test_chinese_ngrams_rerank_older_technical_candidate(self):
        store = Store(self.tmp.name)
        documents = [
            {
                "source_identity": "context-loss-target",
                "source_agent": "codex",
                "role": "user",
                "raw_text": "CPA 部署后第二轮 previous_response_id 导致上下文丢失。",
                "updated_at": "2020-01-01T00:00:00Z",
            }
        ]
        documents.extend(
            {
                "source_identity": f"generic-cpa-{index}",
                "source_agent": "codex",
                "role": "user",
                "raw_text": "CPA unrelated deployment note",
                "updated_at": f"2026-01-01T00:00:{index:02d}Z",
            }
            for index in range(20)
        )
        store.ingest(documents)

        results = store.search(
            "CPA 第二轮为什么丢上下文？",
            limit=1,
            filters={"source_agent": "codex", "role": "user"},
        )

        self.assertEqual(
            [item["source_identity"] for item in results],
            ["context-loss-target"],
        )
        store.close()

    def test_single_cjk_character_technical_query_respects_limit(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": f"cpa-note-{index}",
                    "source_agent": "codex",
                    "raw_text": "CPA note",
                }
                for index in range(40)
            ]
        )

        results = store.search(
            "查 CPA",
            limit=1,
            filters={"source_agent": "codex"},
        )

        self.assertEqual(len(results), 1)
        store.close()

    def test_translation_cache_key_is_hashed_and_permanent_error_opens_circuit(self):
        store = Store(self.tmp.name)
        os.environ.update(
            FUNES_RETRIEVAL_LANGUAGE_MODE="auto",
            TRANSLATION_BASE_URL="https://provider.example/v1",
            TRANSLATION_API_KEY="test-key",
            TRANSLATION_MODEL="test-model",
        )
        tr = Translator(store)
        cache_key = tr._cache_key("中文 查询")
        self.assertEqual(len(cache_key), 64)
        self.assertNotIn("中文", cache_key)
        failure = urllib.error.HTTPError(
            "https://provider.example/v1/chat/completions", 402, "payment", {}, None
        )
        with mock.patch("service.server.urllib.request.urlopen", side_effect=failure) as request:
            self.assertEqual(tr.rewrite("中文 查询"), "中文 查询")
            self.assertEqual(tr.rewrite("另一个中文查询"), "另一个中文查询")
        self.assertEqual(request.call_count, 1)
        store.close()

    def test_query_rewrite_rejects_hallucinations_and_does_not_cache_them(self):
        store = Store(self.tmp.name)
        os.environ.update(
            FUNES_RETRIEVAL_LANGUAGE_MODE="auto",
            TRANSLATION_BASE_URL="https://provider.example/v1",
            TRANSLATION_API_KEY="test-key",
            TRANSLATION_MODEL="test-model",
        )
        tr = Translator(store)

        class Response:
            def __init__(self, content):
                self.content = content

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": self.content}}]}).encode()

        invalid = (
            ("Funes MCP 怎么配置？", "Funes MCP use 1.0 with 1024 tokens and temperature 0.9"),
            ("Funes MCP 怎么检索？", "Funes MCP " + "detailed answer " * 40),
            ("请检索最近 30 天的 CPA 记录", "recent CPA records"),
            (
                "CPA previous_response_id 为什么丢上下文？",
                "CPA context loss",
            ),
        )
        try:
            for raw, hallucination in invalid:
                with self.subTest(raw=raw), mock.patch(
                    "service.server.urllib.request.urlopen",
                    return_value=Response(hallucination),
                ) as request:
                    self.assertEqual(tr.rewrite_query(raw), raw)
                    self.assertIsNone(
                        store.translation_get(tr._cache_key(raw, QUERY_PROMPT_VERSION))
                    )
                    self.assertEqual(request.call_count, 1)
        finally:
            store.close()

    def test_query_rewrite_uses_query_prompt_token_limit_and_versioned_cache(self):
        store = Store(self.tmp.name)
        os.environ.update(
            FUNES_RETRIEVAL_LANGUAGE_MODE="auto",
            TRANSLATION_BASE_URL="https://provider.example/v1",
            TRANSLATION_API_KEY="test-key",
            TRANSLATION_MODEL="test-model",
            TRANSLATION_QUERY_MAX_TOKENS="96",
        )
        tr = Translator(store)
        raw_query = "Funes MCP CPA previous_response_id 请问之前为什么会丢上下文，应该如何检索？"

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"Funes MCP CPA previous_response_id retrieval configuration"}}]}'

        try:
            with mock.patch(
                "service.server.urllib.request.urlopen", return_value=Response()
            ) as request:
                self.assertEqual(
                    tr.rewrite_query(raw_query),
                    "Funes MCP CPA previous_response_id retrieval configuration",
                )
                self.assertEqual(
                    tr.rewrite_query(raw_query),
                    "Funes MCP CPA previous_response_id retrieval configuration",
                )
            self.assertEqual(request.call_count, 1)
            payload = json.loads(request.call_args.args[0].data)
            self.assertEqual(payload["max_tokens"], 96)
            self.assertEqual(payload["messages"][0]["content"], QUERY_RETRIEVAL_PROMPT)
            self.assertNotEqual(
                tr._cache_key(raw_query),
                tr._cache_key(raw_query, QUERY_PROMPT_VERSION),
            )
        finally:
            store.close()

    def test_document_normalization_keeps_original_prompt_and_english_only_shadow(self):
        store = Store(self.tmp.name)
        os.environ.update(
            FUNES_RETRIEVAL_LANGUAGE_MODE="auto",
            TRANSLATION_BASE_URL="https://provider.example/v1",
            TRANSLATION_API_KEY="test-key",
            TRANSLATION_MODEL="test-model",
        )
        tr = Translator(store)

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"English CPA retrieval text"}}]}'

        try:
            with mock.patch(
                "service.server.urllib.request.urlopen", return_value=Response()
            ) as request:
                normalized = tr.normalize_document("中文 CPA 正文")
            self.assertEqual(normalized[0], "English CPA retrieval text")
            self.assertEqual(normalized[3], "ok")
            payload = json.loads(request.call_args.args[0].data)
            self.assertEqual(payload["messages"][0]["content"], RETRIEVAL_PROMPT)
            self.assertNotIn("max_tokens", payload)
        finally:
            store.close()

    def test_records_and_api_token_alias(self):
        os.environ.pop("FUNES_AUTH_TOKEN")
        os.environ["FUNES_API_TOKEN"] = "api-token"
        server = self._server()
        status, result = self._request(server, "POST", "/ingest", {"records": [{"source_path": "r", "raw_text": "record"}],}, token="api-token")
        self.assertEqual(status, 200)
        self.assertEqual(result["created"], 1)

    def test_snapshot_roundtrip(self):
        source = Store(self.tmp.name)
        source.ingest([{"source_path": "s", "raw_text": "durable text", "role": "user", "metadata": {"project": "demo"}}])
        source.translation_put("cache-key", "cached retrieval text")
        snap = Path(self.tmp.name) / "snapshot.jsonl.gz"
        source.snapshot(snap)
        second_dir = tempfile.TemporaryDirectory()
        target = Store(second_dir.name)
        self.assertEqual(target.restore(snap), 1)
        self.assertEqual(target.restore(snap), 0)
        target.reindex()
        self.assertEqual(target.search("durable")[0]["raw_text"], "durable text")
        self.assertEqual(target.translation_get("cache-key"), "cached retrieval text")
        source.close(); target.close(); second_dir.cleanup()

    def test_snapshot_streams_memory_cursor_without_fetchall(self):
        source = Store(self.tmp.name)
        source.ingest(
            [
                {"source_identity": f"stream-{index}", "raw_text": f"row {index}"}
                for index in range(3)
            ]
        )
        connection = source.conn
        iterated = False

        class StreamingCursor:
            def __init__(self, cursor):
                self.cursor = cursor

            def __iter__(self):
                nonlocal iterated
                iterated = True
                return iter(self.cursor)

            def fetchall(self):
                raise AssertionError("snapshot must not materialize the memories cursor")

        class ConnectionProbe:
            def execute(self, sql, *args):
                cursor = connection.execute(sql, *args)
                if "SELECT * FROM memories ORDER BY id" in " ".join(sql.split()):
                    return StreamingCursor(cursor)
                return cursor

            def __getattr__(self, name):
                return getattr(connection, name)

        snapshot = Path(self.tmp.name) / "streamed.jsonl.gz"
        source.conn = ConnectionProbe()
        try:
            source.snapshot(snapshot)
        finally:
            source.conn = connection
        with gzip.open(snapshot, "rt", encoding="utf-8") as stream:
            memories = [
                json.loads(line)["source_identity"]
                for line in stream
                if '"_funes_record": "memory"' in line
            ]
        self.assertTrue(iterated)
        self.assertEqual(memories, ["stream-0", "stream-1", "stream-2"])
        source.close()

    def test_snapshot_roundtrip_preserves_retrieval_and_native_state(self):
        source = Store(self.tmp.name)
        source.ingest(
            [{
                "source_identity": "stateful",
                "source_version": "raw-v1",
                "raw_text": "原文",
                "retrieval_text": "retrieval shadow",
                "translation_status": "ok",
                "retrieval_updated_at": "2026-09-13T01:00:00Z",
                "native_index_version": "canonical-v1",
                "native_index_status": "indexed",
                "native_index_profile": "profile-v1",
                "native_index_memory": "memory-v1",
                "native_indexed_at": "2026-09-13T02:00:00Z",
                "native_index_error": None,
            }]
        )
        profile = {"fingerprint": "profile-v1"}
        before = source.native_index_checkpoint(profile, "memory-v1")
        snapshot = Path(self.tmp.name) / "state.jsonl.gz"
        source.snapshot(snapshot)
        second_dir = tempfile.TemporaryDirectory()
        target = Store(second_dir.name)
        self.assertEqual(target.restore(snapshot), 1)
        item = target.get("stateful")
        self.assertEqual(item["retrieval_updated_at"], "2026-09-13T01:00:00Z")
        self.assertEqual(item["native_index_version"], "canonical-v1")
        self.assertEqual(item["native_index_status"], "indexed")
        self.assertEqual(item["native_index_profile"], "profile-v1")
        self.assertEqual(item["native_index_memory"], "memory-v1")
        self.assertEqual(item["native_indexed_at"], "2026-09-13T02:00:00Z")
        with gzip.open(snapshot, "rt", encoding="utf-8") as stream:
            state_lines = [
                json.loads(line)
                for line in stream
                if '"_funes_record": "native_index_state"' in line
            ]
        self.assertEqual(len(state_lines), 1)
        self.assertEqual(state_lines[0]["state_version"], 2)
        self.assertNotIn("raw_text", state_lines[0])
        after = target.native_index_checkpoint(profile, "memory-v1")
        self.assertEqual(after["revision"], before["revision"])
        self.assertEqual(after["index_fingerprint"], before["index_fingerprint"])
        source.close(); target.close(); second_dir.cleanup()

    def test_restore_builds_fts_exactly_once(self):
        source = Store(self.tmp.name)
        source.ingest([{"source_identity": "one", "raw_text": "single build"}])
        snapshot = Path(self.tmp.name) / "single-build.jsonl.gz"
        source.snapshot(snapshot)
        directory = tempfile.TemporaryDirectory()
        target = Store(directory.name)
        statements = []
        target.conn.set_trace_callback(statements.append)
        try:
            self.assertEqual(target.restore(snapshot), 1)
        finally:
            target.conn.set_trace_callback(None)
        rebuilds = [
            sql
            for sql in statements
            if "INSERTINTOMEMORIES_FTS(MEMORIES_FTS)VALUES('REBUILD')"
            in sql.upper().replace(" ", "")
        ]
        self.assertEqual(len(rebuilds), 1, statements)
        self.assertEqual(target.search("single")[0]["source_identity"], "one")
        source.close(); target.close(); directory.cleanup()

    def test_bulk_restore_drops_and_rebuilds_secondary_indexes_preserves_unique(self):
        store = Store(self.tmp.name)
        store.ingest([
            {
                "source_identity": "doc-1",
                "source_agent": "codex",
                "role": "user",
                "source_type": "conversation",
                "raw_text": "sample text",
            }
        ])

        def get_indexes():
            return {
                row["name"]
                for row in store.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memories'"
                )
            }

        initial_indexes = get_indexes()
        expected_secondaries = {
            "memories_source_agent_role_idx",
            "memories_source_agent_type_idx",
            "memories_translation_pending_idx",
            "memories_canonical_pending_idx",
        }
        for name in expected_secondaries:
            self.assertIn(name, initial_indexes)
        self.assertIn("sqlite_autoindex_memories_1", initial_indexes)

        # 1. begin_bulk_restore drops 4 secondary indexes, preserves UNIQUE
        store.begin_bulk_restore()
        bulk_indexes = get_indexes()
        self.assertIn("sqlite_autoindex_memories_1", bulk_indexes)
        for name in expected_secondaries:
            self.assertNotIn(name, bulk_indexes)
        # Pragmas applied
        self.assertEqual(store.conn.execute("PRAGMA synchronous").fetchone()[0], 1)
        self.assertEqual(store.conn.execute("PRAGMA cache_size").fetchone()[0], -64000)
        self.assertEqual(store.conn.execute("PRAGMA mmap_size").fetchone()[0], 268435456)

        # 2. Nested restore does not repeat drops or prematurely rebuild
        store.begin_bulk_restore()
        self.assertEqual(store._bulk_restore_depth, 2)
        nested_indexes = get_indexes()
        for name in expected_secondaries:
            self.assertNotIn(name, nested_indexes)

        store.finish_bulk_restore()
        self.assertEqual(store._bulk_restore_depth, 1)
        after_nested_finish = get_indexes()
        for name in expected_secondaries:
            self.assertNotIn(name, after_nested_finish)

        # 3. Outermost finish_bulk_restore restores all secondary indexes and pragmas
        store.finish_bulk_restore()
        self.assertEqual(store._bulk_restore_depth, 0)
        final_indexes = get_indexes()
        for name in expected_secondaries:
            self.assertIn(name, final_indexes)
        self.assertIn("sqlite_autoindex_memories_1", final_indexes)
        # Pragmas restored
        self.assertEqual(store.conn.execute("PRAGMA synchronous").fetchone()[0], 2)
        self.assertEqual(store.conn.execute("PRAGMA cache_size").fetchone()[0], -2000)
        self.assertEqual(store.conn.execute("PRAGMA mmap_size").fetchone()[0], 0)
        store.close()

    def test_bulk_restore_exception_in_restore_recovers_indexes_and_pragmas(self):
        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "doc-err", "raw_text": "failure test"}])
        expected_secondaries = {
            "memories_source_agent_role_idx",
            "memories_source_agent_type_idx",
            "memories_translation_pending_idx",
            "memories_canonical_pending_idx",
        }

        store.begin_bulk_restore()
        try:
            try:
                # Simulate an error during restore
                raise RuntimeError("restore aborted halfway")
            finally:
                store.finish_bulk_restore()
        except RuntimeError:
            pass

        current_indexes = {
            row["name"]
            for row in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memories'"
            )
        }
        for name in expected_secondaries:
            self.assertIn(name, current_indexes)
        self.assertIn("sqlite_autoindex_memories_1", current_indexes)
        self.assertEqual(store.conn.execute("PRAGMA synchronous").fetchone()[0], 2)
        self.assertEqual(store.conn.execute("PRAGMA cache_size").fetchone()[0], -2000)
        self.assertEqual(store.conn.execute("PRAGMA mmap_size").fetchone()[0], 0)
        store.close()

    def test_bulk_restore_exception_in_fts_rebuild_still_recovers_indexes_and_pragmas(self):
        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "doc-fts-err", "raw_text": "fts error test"}])
        expected_secondaries = {
            "memories_source_agent_role_idx",
            "memories_source_agent_type_idx",
            "memories_translation_pending_idx",
            "memories_canonical_pending_idx",
        }

        store.begin_bulk_restore()
        with mock.patch.object(
            store,
            "_rebuild_native_checkpoint_state_locked",
            side_effect=sqlite3.OperationalError("simulated rebuild corruption"),
        ):
            with self.assertRaises(sqlite3.OperationalError):
                store.finish_bulk_restore()

        current_indexes = {
            row["name"]
            for row in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memories'"
            )
        }
        for name in expected_secondaries:
            self.assertIn(name, current_indexes)
        self.assertIn("sqlite_autoindex_memories_1", current_indexes)
        self.assertEqual(store.conn.execute("PRAGMA synchronous").fetchone()[0], 2)
        self.assertEqual(store.conn.execute("PRAGMA cache_size").fetchone()[0], -2000)
        self.assertEqual(store.conn.execute("PRAGMA mmap_size").fetchone()[0], 0)
        store.close()

    def test_bulk_restore_custom_pragmas_honored_and_restored(self):
        store = Store(self.tmp.name)
        with mock.patch.dict(
            os.environ,
            {
                "FUNES_BULK_RESTORE_SYNCHRONOUS": "OFF",
                "FUNES_BULK_RESTORE_CACHE_SIZE": "-32000",
                "FUNES_BULK_RESTORE_MMAP_SIZE": "134217728",
            },
        ):
            store.begin_bulk_restore()
            self.assertEqual(store.conn.execute("PRAGMA synchronous").fetchone()[0], 0)
            self.assertEqual(store.conn.execute("PRAGMA cache_size").fetchone()[0], -32000)
            self.assertEqual(store.conn.execute("PRAGMA mmap_size").fetchone()[0], 134217728)
            store.finish_bulk_restore()

        self.assertEqual(store.conn.execute("PRAGMA synchronous").fetchone()[0], 2)
        self.assertEqual(store.conn.execute("PRAGMA cache_size").fetchone()[0], -2000)
        self.assertEqual(store.conn.execute("PRAGMA mmap_size").fetchone()[0], 0)
        store.close()

    def test_snapshot_large_dataset_streaming_preserves_byte_and_record_semantics(self):
        source = Store(self.tmp.name)
        docs = [
            {
                "source_identity": f"bulk-stream-{index}",
                "source_version": f"v-{index}",
                "raw_text": f"Raw payload number {index} with unicode 测试 and technical terms func_{index}()",
                "role": "user" if index % 2 == 0 else "assistant",
                "source_agent": "codex" if index % 3 == 0 else "claude",
                "source_type": "conversation",
                "metadata": {"batch": index // 100, "tag": f"item-{index}"},
            }
            for index in range(500)
        ]
        source.ingest(docs)
        for i in range(20):
            source.translation_put(f"query-{i}", f"rewritten-{i}")
        for i in range(5):
            source.record_reindex_control({"generation": i + 1, "scope": "all", "created_at": "2026-09-18T00:00:00Z"})

        connection = source.conn
        iterated = False

        class StreamingCursor:
            def __init__(self, cursor):
                self.cursor = cursor

            def __iter__(self):
                nonlocal iterated
                iterated = True
                return iter(self.cursor)

            def fetchall(self):
                raise AssertionError("snapshot must not materialize via fetchall")

        class ConnectionProbe:
            def execute(self, sql, *args):
                cursor = connection.execute(sql, *args)
                if "SELECT * FROM memories ORDER BY id" in " ".join(sql.split()):
                    return StreamingCursor(cursor)
                return cursor

            def __getattr__(self, name):
                return getattr(connection, name)

        snapshot_path = Path(self.tmp.name) / "large-streamed.jsonl.gz"
        source.conn = ConnectionProbe()
        try:
            source.snapshot(snapshot_path)
        finally:
            source.conn = connection

        self.assertTrue(iterated)

        # Verify exact line by line content and ordering semantics
        memory_records = []
        translation_records = []
        control_records = []
        with gzip.open(snapshot_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                rec = json.loads(line)
                rec_type = rec.get("_funes_record")
                if rec_type == "memory":
                    memory_records.append(rec)
                elif rec_type == "translation_cache":
                    translation_records.append(rec)
                elif rec_type == "reindex_control":
                    control_records.append(rec)

        self.assertEqual(len(memory_records), 500)
        self.assertEqual(len(translation_records), 20)
        self.assertEqual(len(control_records), 5)
        ids = [r["id"] for r in memory_records]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual([r["source_identity"] for r in memory_records[:3]], ["bulk-stream-0", "bulk-stream-1", "bulk-stream-2"])

        target_dir = tempfile.TemporaryDirectory()
        target = Store(target_dir.name)
        restored = target.restore(snapshot_path, apply_controls=False)
        self.assertEqual(restored, 500)
        self.assertEqual(target.count(), 500)
        item = target.get("bulk-stream-42")
        self.assertIsNotNone(item)
        self.assertIn("func_42()", item["raw_text"])
        self.assertEqual(target.translation_get("query-5"), "rewritten-5")

        source.close()
        target.close()
        target_dir.cleanup()

    def test_canonical_checkpoint_rebuilds_profile_mismatch_including_session(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": "memory-old-profile",
                    "source_type": "memory",
                    "raw_text": "memory raw",
                    "translation_status": "skipped_raw_mode",
                    "native_index_status": "indexed",
                    "native_index_profile": "old-profile",
                    "native_index_memory": "old-memory",
                },
                {
                    "source_identity": "session-old-profile",
                    "source_type": "session",
                    "raw_text": "session raw",
                    "translation_status": "skipped_native_session",
                    "native_index_status": "indexed",
                    "native_index_profile": "old-profile",
                    "native_index_memory": "old-memory",
                },
                {
                    "source_identity": "already-current",
                    "source_type": "memory",
                    "raw_text": "current raw",
                    "translation_status": "skipped_raw_mode",
                    "native_index_status": "indexed",
                    "native_index_profile": "new-profile",
                    "native_index_memory": "new-memory",
                },
                {
                    "source_identity": "low-value",
                    "source_type": "memory",
                    "content_type": "tool_result",
                    "raw_text": "tool output",
                    "translation_status": "skipped_low_value",
                    "native_index_status": "indexed",
                    "native_index_profile": "old-profile",
                    "native_index_memory": "old-memory",
                },
            ]
        )
        profile = {
            "provider": "voyage",
            "model": "voyage-4-lite",
            "dimensions": 1024,
            "schema_version": 2,
            "fingerprint": "new-profile",
        }
        try:
            self.assertEqual(
                {
                    item["source_identity"]
                    for item in store.canonical_index_candidates(
                        10, "new-profile", "new-memory"
                    )
                },
                {"memory-old-profile", "session-old-profile"},
            )
            checkpoint = store.native_index_checkpoint(profile, "new-memory")
            self.assertEqual({key: checkpoint[key] for key in profile}, profile)
            self.assertEqual(checkpoint["eligible"], 3)
            self.assertEqual(checkpoint["indexed"], 1)
            self.assertEqual(checkpoint["pending"], 2)
            self.assertFalse(checkpoint["complete"])
            self.assertEqual(checkpoint["memory"], "new-memory")
        finally:
            store.close()

    def test_legacy_codex_automation_output_is_retained_but_not_indexed(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": "legacy-automation-run",
                    "source_agent": "codex",
                    "source_type": "memory",
                    "source_path": "~/.codex/automations/daily/runs/run.jsonl",
                    "content_type": "memory",
                    "raw_text": "retained automation output",
                },
                {
                    "source_identity": "automation-instructions",
                    "source_agent": "codex",
                    "source_type": "memory",
                    "source_path": "~/.codex/automations/daily/automation.toml",
                    "content_type": "memory",
                    "raw_text": "retained automation instructions",
                },
            ]
        )
        try:
            legacy = store.get("legacy-automation-run")
            self.assertEqual(legacy["raw_text"], "retained automation output")
            self.assertEqual(legacy["content_type"], "progress")
            self.assertEqual(
                {
                    item["source_identity"]
                    for item in store.canonical_index_candidates(
                        10, "profile", "memory-a"
                    )
                },
                {"automation-instructions"},
            )
        finally:
            store.close()

    def test_native_pending_paths_are_indexed_and_checkpoint_is_constant_work(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": "indexed",
                    "raw_text": "indexed raw",
                    "native_index_status": "indexed",
                    "native_index_profile": "profile",
                    "native_index_memory": "memory-a",
                },
                {
                    "source_identity": "retry-first",
                    "raw_text": "retry raw",
                    "updated_at": "2020-01-01T00:00:00Z",
                    "native_index_status": "retry",
                },
                {
                    "source_identity": "fresh-second",
                    "raw_text": "fresh raw",
                    "updated_at": "2021-01-01T00:00:00Z",
                    "translation_status": "pending_provider",
                },
            ]
        )
        profile = {
            "provider": "voyage",
            "model": "voyage-4-lite",
            "dimensions": 1024,
            "schema_version": 2,
            "fingerprint": "profile",
        }
        self.assertEqual(
            store.canonical_index_candidates(1, "profile", "memory-a")[0][
                "source_identity"
            ],
            "fresh-second",
        )
        candidate_plan = " ".join(
            str(value)
            for row in store.conn.execute(
                """EXPLAIN QUERY PLAN SELECT id FROM memories
                WHERE native_index_pending=1
                ORDER BY CASE WHEN native_index_status='retry' THEN 1 ELSE 0 END,
                COALESCE(retrieval_updated_at, updated_at), id LIMIT 1"""
            )
            for value in row
        )
        translation_plan = " ".join(
            str(value)
            for row in store.conn.execute(
                """EXPLAIN QUERY PLAN SELECT id FROM memories
                WHERE translation_status='pending_provider'
                AND COALESCE(native_index_status, '') != 'waiting_durability'
                ORDER BY updated_at, id LIMIT 1"""
            )
            for value in row
        )
        self.assertIn("memories_canonical_pending_idx", candidate_plan)
        self.assertIn("memories_translation_pending_idx", translation_plan)

        statements = []
        store.conn.set_trace_callback(statements.append)
        checkpoint = store.native_index_checkpoint(profile, "memory-a")
        store.conn.set_trace_callback(None)
        self.assertEqual(checkpoint["eligible"], 3)
        self.assertEqual(checkpoint["indexed"], 1)
        self.assertEqual(checkpoint["pending"], 2)
        self.assertGreater(checkpoint["revision"], 0)
        self.assertFalse(
            any("FROM MEMORIES" in sql.upper() for sql in statements), statements
        )
        store.close()

    def test_native_state_record_can_project_pre_durable_status_update(self):
        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "row", "raw_text": "raw"}])
        item = store.canonical_index_candidates(1, "profile", "memory-a")[0]
        before = store.native_index_checkpoint(
            {"fingerprint": "profile"}, "memory-a"
        )
        update = {
            "source_identity": item["source_identity"],
            "source_version": item["source_version"],
            "content_hash": item["content_hash"],
            "native_generation": item["native_generation"],
            "native_index_version": "canonical-v1",
            "native_index_status": "indexed",
            "native_index_profile": "profile",
            "native_index_memory": "memory-a",
            "native_indexed_at": "2026-09-13T02:00:00Z",
        }
        projected = store.native_index_state_record([update])
        self.assertEqual(projected["revision"], before["revision"] + 1)
        self.assertEqual(projected["indexed"], 1)
        self.assertEqual(
            store.native_index_checkpoint(
                {"fingerprint": "profile"}, "memory-a"
            )["indexed"],
            0,
        )
        self.assertEqual(store.update_native_index([update]), 1)
        actual = store.native_index_checkpoint(
            {"fingerprint": "profile"}, "memory-a"
        )
        self.assertEqual(actual["revision"], projected["revision"])
        self.assertEqual(actual["index_fingerprint"], projected["index_fingerprint"])
        store.close()

    def test_native_state_migration_backfills_once_and_is_restart_safe(self):
        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "pending", "raw_text": "raw"}])
        store.close()
        connection = sqlite3.connect(Path(self.tmp.name) / "funes.sqlite3")
        with connection:
            connection.execute(
                "UPDATE sync_state SET native_checkpoint_state_version=0"
            )
            connection.execute("UPDATE memories SET native_index_pending=0")
            connection.execute("DROP INDEX memories_canonical_pending_idx")
            connection.execute("DROP INDEX memories_translation_pending_idx")
            for trigger in (
                "memories_native_ai",
                "memories_native_ad",
                "memories_native_au",
            ):
                connection.execute(f"DROP TRIGGER {trigger}")
        connection.close()

        migrated = Store(self.tmp.name)
        self.assertEqual(
            migrated.conn.execute(
                "SELECT native_checkpoint_state_version FROM sync_state WHERE id=1"
            ).fetchone()[0],
            2,
        )
        self.assertEqual(
            migrated.conn.execute(
                "SELECT native_index_pending FROM memories WHERE source_identity='pending'"
            ).fetchone()[0],
            1,
        )
        migrated.close()
        reopened = Store(self.tmp.name)
        self.assertEqual(reopened.conn.total_changes, 0)
        reopened.close()

    def test_embedding_generation_migration_is_additive_and_defaults_to_zero(self):
        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "legacy", "raw_text": "raw"}])
        store.close()
        connection = sqlite3.connect(Path(self.tmp.name) / "funes.sqlite3")
        with connection:
            connection.execute("DROP TRIGGER memories_native_au")
            connection.execute("ALTER TABLE memories DROP COLUMN embedding_generation")
        connection.close()

        migrated = Store(self.tmp.name)
        try:
            columns = {
                row[1] for row in migrated.conn.execute("PRAGMA table_info(memories)")
            }
            self.assertIn("embedding_generation", columns)
            self.assertEqual(migrated.get("legacy")["embedding_generation"], 0)
        finally:
            migrated.close()

    def test_native_state_v2_retires_legacy_automation_output_without_pending(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": "legacy-automation-run",
                    "source_agent": "codex",
                    "source_type": "memory",
                    "source_path": "~/.codex/automations/daily/runs/run.jsonl",
                    "content_type": "memory",
                    "raw_text": "retained automation output",
                }
            ]
        )
        with store.lock, store.conn:
            # Recreate the durable shape written before automation outputs
            # were removed from discovery and classified as low-value.
            store.conn.execute(
                """UPDATE memories SET content_type='memory',
                native_index_status='indexed',native_index_version='v1',
                native_index_profile='profile',native_index_memory='memory-a',
                native_indexed_at='2026-09-15T00:00:00Z'"""
            )
            store.conn.execute(
                """UPDATE sync_state SET native_checkpoint_profile='profile',
                native_checkpoint_memory='memory-a',
                native_checkpoint_state_version=1"""
            )
        store.close()

        migrated = Store(self.tmp.name)
        try:
            self.assertEqual(
                migrated.get("legacy-automation-run")["content_type"], "progress"
            )
            self.assertEqual(
                migrated.canonical_index_candidates(10, "profile", "memory-a"),
                [],
            )
            checkpoint = migrated.native_index_checkpoint(
                {"fingerprint": "profile"}, "memory-a"
            )
            self.assertEqual(
                migrated.conn.execute(
                    """SELECT native_checkpoint_state_version
                    FROM sync_state WHERE id=1"""
                ).fetchone()[0],
                2,
            )
            self.assertEqual(checkpoint["eligible"], 0)
            self.assertEqual(checkpoint["indexed"], 0)
            self.assertEqual(checkpoint["pending"], 0)
        finally:
            migrated.close()

    def test_held_invalid_is_terminal_and_does_not_regress(self):
        store = Store(self.tmp.name)
        terminal = {
            "source_identity": "invalid",
            "source_version": "v1",
            "raw_text": "deterministically invalid canonical row",
            "native_index_version": "canonical-v1",
            "native_index_status": "held_invalid",
            "native_index_profile": "profile",
            "native_index_memory": "memory-a",
            "native_indexed_at": "2026-09-13T02:00:00Z",
        }
        store.ingest([terminal])
        self.assertEqual(
            store.canonical_index_candidates(10, "profile", "memory-a"), []
        )
        checkpoint = store.native_index_checkpoint(
            {"fingerprint": "profile"}, "memory-a"
        )
        self.assertEqual(checkpoint["held"], 1)
        self.assertEqual(checkpoint["invalid"], 1)
        self.assertTrue(checkpoint["complete"])
        store.ingest(
            [{
                **terminal,
                "native_index_status": "retry",
                "native_indexed_at": "2026-09-13T01:00:00Z",
            }]
        )
        self.assertEqual(store.get("invalid")["native_index_status"], "held_invalid")
        store.ingest(
            [{
                **terminal,
                "native_index_status": "indexed",
                "native_indexed_at": None,
            }]
        )
        self.assertEqual(store.get("invalid")["native_index_status"], "held_invalid")
        store.close()

    def test_native_index_fingerprint_is_stable_across_restore_row_order(self):
        profile = {
            "provider": "voyage",
            "model": "voyage-4-lite",
            "dimensions": 1024,
            "schema_version": 2,
            "fingerprint": "stable-profile",
        }
        documents = [
            {
                "source_identity": identity,
                "raw_text": f"raw {identity}",
                "translation_status": "skipped_raw_mode",
                "native_index_version": f"version-{identity}",
                "native_index_status": "indexed",
                "native_index_profile": "stable-profile",
                "native_indexed_at": "2026-09-13T02:00:00Z",
            }
            for identity in ("a", "b")
        ]
        first_dir = tempfile.TemporaryDirectory()
        second_dir = tempfile.TemporaryDirectory()
        first = Store(first_dir.name)
        second = Store(second_dir.name)
        try:
            first.ingest(documents)
            second.ingest(list(reversed(documents)))
            self.assertEqual(
                first.native_index_checkpoint(profile)["index_fingerprint"],
                second.native_index_checkpoint(profile)["index_fingerprint"],
            )
        finally:
            first.close()
            second.close()
            first_dir.cleanup()
            second_dir.cleanup()

    def test_native_optimize_checkpoint_restore_uses_monotonic_revision(self):
        store = Store(self.tmp.name)
        profile = {
            "provider": "voyage",
            "model": "voyage-4-lite",
            "dimensions": 1024,
            "schema_version": 2,
            "fingerprint": "profile",
            "index_fingerprint": "index",
            "memory": "memory-a",
        }
        pending = {
            **profile,
            "status": "pending",
            "optimized_at": "2026-09-13T02:00:00Z",
            "revision": 2,
        }
        stale_optimized = {
            **profile,
            "status": "optimized",
            "optimized_at": "2026-09-13T03:00:00Z",
            "revision": 1,
        }
        current_optimized = {
            **profile,
            "status": "optimized",
            "optimized_at": "2026-09-13T04:00:00Z",
            "revision": 3,
        }
        try:
            self.assertTrue(store.set_native_optimize_checkpoint(pending))
            self.assertFalse(store.set_native_optimize_checkpoint(stale_optimized))
            self.assertEqual(store.native_optimize_checkpoint()["status"], "pending")
            self.assertTrue(store.set_native_optimize_checkpoint(current_optimized))
            same_revision_pending = {
                **profile,
                "status": "pending",
                "optimized_at": "2026-09-13T05:00:00Z",
                "revision": 3,
            }
            self.assertFalse(
                store.set_native_optimize_checkpoint(same_revision_pending)
            )
            self.assertEqual(
                store.native_optimize_checkpoint()["status"], "optimized"
            )
            other_memory = {
                **same_revision_pending,
                "memory": "memory-b",
                "optimized_at": "2026-09-13T06:00:00Z",
            }
            self.assertTrue(store.set_native_optimize_checkpoint(other_memory))
            self.assertEqual(
                store.native_optimize_checkpoint()["memory"], "memory-b"
            )
            store.restore_documents(
                [{**pending, "_funes_record": "native_optimize_checkpoint"}]
            )
            checkpoint = store.native_optimize_checkpoint()
            self.assertEqual(checkpoint["status"], "pending")
            self.assertEqual(checkpoint["memory"], "memory-b")
            self.assertEqual(checkpoint["revision"], 3)
        finally:
            store.close()

    def test_native_optimize_layout_version_migrates_legacy_sync_state(self):
        store = Store(self.tmp.name)
        marker = {
            "fingerprint": "legacy-profile",
            "status": "optimized",
            "optimized_at": "2026-09-13T02:00:00Z",
            "revision": 1,
        }
        self.assertTrue(store.set_native_optimize_checkpoint(marker))
        store.close()

        connection = sqlite3.connect(Path(self.tmp.name) / "funes.sqlite3")
        with connection:
            connection.execute(
                "ALTER TABLE sync_state DROP COLUMN native_optimize_layout_version"
            )
        connection.close()

        migrated = Store(self.tmp.name)
        try:
            columns = {
                row[1]: row
                for row in migrated.conn.execute("PRAGMA table_info(sync_state)")
            }
            self.assertIn("native_optimize_layout_version", columns)
            self.assertEqual(columns["native_optimize_layout_version"][3], 1)
            self.assertEqual(columns["native_optimize_layout_version"][4], "0")
            checkpoint = migrated.native_optimize_checkpoint()
            self.assertEqual(checkpoint["fingerprint"], "legacy-profile")
            self.assertEqual(checkpoint["index_layout_version"], 0)
        finally:
            migrated.close()

    def test_native_optimize_layout_version_snapshot_roundtrip(self):
        source = Store(self.tmp.name)
        marker = {
            "fingerprint": "profile-v2",
            "index_fingerprint": "index-v2",
            "index_layout_version": 2,
            "status": "optimized",
            "optimized_at": "2026-09-13T02:00:00Z",
            "revision": 4,
        }
        snapshot = Path(self.tmp.name) / "layout-version.jsonl.gz"
        target_dir = tempfile.TemporaryDirectory()
        target = Store(target_dir.name)
        try:
            self.assertTrue(source.set_native_optimize_checkpoint(marker))
            source.snapshot(snapshot)
            self.assertEqual(target.restore(snapshot), 0)
            self.assertEqual(
                target.native_optimize_checkpoint()["index_layout_version"], 2
            )
            with gzip.open(snapshot, "rt", encoding="utf-8") as stream:
                optimize = next(
                    json.loads(line)
                    for line in stream
                    if '"_funes_record": "native_optimize_checkpoint"' in line
                )
            self.assertEqual(optimize["index_layout_version"], 2)
        finally:
            source.close()
            target.close()
            target_dir.cleanup()

    def test_native_optimize_old_marker_cannot_regress_layout_version(self):
        store = Store(self.tmp.name)
        current = {
            "fingerprint": "profile",
            "index_layout_version": 2,
            "status": "optimized",
            "optimized_at": "2026-09-13T02:00:00Z",
            "revision": 1,
        }
        old_snapshot_marker = {
            "fingerprint": "profile",
            "status": "optimized",
            "optimized_at": "2026-09-13T03:00:00Z",
            "revision": 2,
        }
        legacy_dir = tempfile.TemporaryDirectory()
        legacy = Store(legacy_dir.name)
        try:
            self.assertTrue(store.set_native_optimize_checkpoint(current))
            self.assertEqual(
                store.native_optimize_checkpoint()["index_layout_version"], 2
            )
            self.assertFalse(
                store.set_native_optimize_checkpoint(old_snapshot_marker)
            )
            self.assertEqual(
                store.native_optimize_checkpoint()["index_layout_version"], 2
            )
            self.assertEqual(store.native_optimize_checkpoint()["revision"], 1)

            legacy.restore_documents(
                [{**old_snapshot_marker, "_funes_record": "native_optimize_checkpoint"}]
            )
            self.assertEqual(
                legacy.native_optimize_checkpoint()["index_layout_version"], 0
            )
        finally:
            store.close()
            legacy.close()
            legacy_dir.cleanup()

    def test_http_never_returns_retrieval_text_even_when_legacy_flag_is_set(self):
        os.environ["RETURN_RETRIEVAL_TEXT"] = "true"
        server = self._server()
        status, _ = self._request(
            server,
            "POST",
            "/ingest",
            {"source_identity": "raw-only", "raw_text": "原文", "retrieval_text": "english shadow"},
        )
        self.assertEqual(status, 200)
        status, item = self._request(server, "POST", "/get", {"id": "raw-only"})
        self.assertEqual(status, 200)
        self.assertEqual(item["raw_text"], "原文")
        self.assertNotIn("retrieval_text", item)

    def test_ingest_does_not_ack_before_durable_snapshot(self):
        os.environ["FUNES_REQUIRE_DURABLE_ACK"] = "true"
        server = self._server()
        status, result = self._request(server, "POST", "/ingest", {"source_path": "pending", "raw_text": "must persist"})
        self.assertEqual(status, 503)
        self.assertEqual(result["error"], "durability_pending")
        self.assertFalse(result["sync"]["durable"])

    def test_reindex_generations_reset_only_derived_state(self):
        store = Store(self.tmp.name)
        original = {
            "source_identity": "provider-row",
            "source_version": "raw-v1",
            "raw_text": "原始中文",
            "retrieval_text": "english shadow",
            "translation_hash": "old-hash",
            "translation_version": "old-version",
            "translation_status": "ok",
            "native_index_version": "native-v1",
            "native_index_status": "indexed",
            "metadata": {"keep": "unchanged"},
        }
        store.ingest(
            [
                original,
                {
                    "source_identity": "english-row",
                    "raw_text": "plain english",
                    "retrieval_text": "plain english",
                    "translation_status": "skipped_non_cjk",
                    "native_index_version": "native-en",
                    "native_index_status": "indexed",
                },
                {
                    "source_identity": "old-raw-mode-row",
                    "raw_text": "需要按当前模式重新判断",
                    "retrieval_text": "旧 raw mode shadow",
                    "translation_status": "skipped_raw_mode",
                    "native_index_status": "indexed",
                },
                {
                    "source_identity": "waiting-row",
                    "raw_text": "等待持久化",
                    "retrieval_text": "old waiting shadow",
                    "translation_status": "ok",
                    "native_index_status": "waiting_durability",
                    "native_index_error": "durability_pending",
                },
            ]
        )
        store.record_reindex_control(
            {"generation": 2, "scope": "retrieval_text", "created_at": "2026-09-13T00:00:02Z"}
        )
        store.drain_reindex_controls(1)

        provider = store.get("provider-row")
        self.assertEqual(provider["raw_text"], original["raw_text"])
        self.assertEqual(provider["source_version"], original["source_version"])
        self.assertEqual(provider["metadata"], original["metadata"])
        self.assertEqual(provider["retrieval_text"], original["raw_text"])
        self.assertEqual(provider["translation_status"], "pending_provider")
        self.assertIsNone(provider["native_index_status"])
        self.assertEqual(provider["retrieval_generation"], 2)
        self.assertEqual(provider["native_generation"], 2)
        self.assertEqual(provider["embedding_generation"], 0)
        english = store.get("english-row")
        self.assertEqual(english["translation_status"], "pending_provider")
        self.assertIsNone(english["native_index_status"])
        self.assertEqual(
            store.get("old-raw-mode-row")["translation_status"], "pending_provider"
        )
        waiting = store.get("waiting-row")
        self.assertEqual(waiting["translation_status"], "pending_provider")
        self.assertEqual(waiting["native_index_status"], "waiting_durability")
        self.assertEqual(
            {item["source_identity"] for item in store.pending_translations(10)},
            {"provider-row", "english-row", "old-raw-mode-row"},
        )

        # A pre-control derived delta must not overwrite generation 2.
        store.ingest([original])
        self.assertEqual(store.get("provider-row")["translation_status"], "pending_provider")

        # An older all-control arriving after generation 2 still clears the
        # independent native generation of canonical-eligible English rows.
        provider = store.get("provider-row")
        self.assertEqual(
            store.update_native_index(
                [
                    {
                        "source_identity": provider["source_identity"],
                        "source_version": provider["source_version"],
                        "content_hash": provider["content_hash"],
                        "native_generation": provider["native_generation"],
                        "native_index_version": "indexed-after-retrieval",
                        "native_index_status": "indexed",
                        "native_index_profile": "profile",
                        "native_index_memory": "memory",
                        "native_indexed_at": "2026-09-13T00:00:03Z",
                    }
                ]
            ),
            1,
        )
        store.record_reindex_control(
            {"generation": 1, "scope": "all", "created_at": "2026-09-13T00:00:01Z"}
        )
        store.drain_reindex_controls(1)
        self.assertIsNone(store.get("english-row")["native_index_status"])
        self.assertIsNone(store.get("provider-row")["native_index_status"])
        self.assertEqual(store.get("waiting-row")["native_index_status"], "waiting_durability")
        self.assertEqual(store.get("provider-row")["embedding_generation"], 1)

        # Rows first seen after an all-control inherit that embedding epoch,
        # while legacy/incoming generation zero cannot regress an existing row.
        store.ingest([{"source_identity": "new-after-all", "raw_text": "new"}])
        self.assertEqual(store.get("new-after-all")["embedding_generation"], 1)
        store.ingest([{**original, "embedding_generation": 0}])
        self.assertEqual(store.get("provider-row")["embedding_generation"], 1)
        store.close()

    def test_reindex_row_cursor_is_bounded_and_survives_restart(self):
        directory = tempfile.TemporaryDirectory()
        store = Store(directory.name)
        store.ingest(
            [
                {
                    "source_identity": f"row-{index}",
                    "raw_text": f"raw {index}",
                    "retrieval_text": f"old shadow {index}",
                    "translation_status": "skipped_non_cjk",
                    "native_index_status": "indexed",
                }
                for index in range(5)
            ]
        )
        control = {
            "generation": 1,
            "scope": "retrieval_text",
            "created_at": "2026-09-13T00:00:01Z",
        }
        store.record_reindex_control(control)
        first = store.apply_pending_reindex_controls(2)
        state = store.conn.execute(
            "SELECT row_cursor, applied_at FROM reindex_controls WHERE generation=1"
        ).fetchone()
        self.assertEqual(first["scanned"], 2)
        self.assertEqual(first["applied"], 0)
        self.assertEqual(state["row_cursor"], 2)
        self.assertIsNone(state["applied_at"])
        self.assertEqual(store.get("row-1")["translation_status"], "pending_provider")
        self.assertEqual(store.get("row-2")["translation_status"], "skipped_non_cjk")
        store.close()

        restored = Store(directory.name)
        restored.record_reindex_control(control)
        resumed = restored.conn.execute(
            "SELECT row_cursor FROM reindex_controls WHERE generation=1"
        ).fetchone()
        self.assertEqual(resumed["row_cursor"], 2)
        second = restored.apply_pending_reindex_controls(2)
        final = restored.apply_pending_reindex_controls(2)
        state = restored.conn.execute(
            "SELECT row_cursor, applied_at FROM reindex_controls WHERE generation=1"
        ).fetchone()
        self.assertEqual(second["scanned"], 2)
        self.assertEqual(second["applied"], 0)
        self.assertEqual(final["scanned"], 1)
        self.assertEqual(final["applied"], 1)
        self.assertEqual(state["row_cursor"], 5)
        self.assertIsNotNone(state["applied_at"])
        self.assertEqual(
            {restored.get(f"row-{index}")["translation_status"] for index in range(5)},
            {"pending_provider"},
        )
        restored.close()
        directory.cleanup()

    def test_reindex_rechecks_raw_with_current_translation_configuration(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": "old-skipped",
                    "raw_text": "原始内容",
                    "retrieval_text": "old shadow",
                    "translation_hash": "old-model-hash",
                    "translation_status": "skipped_non_cjk",
                }
            ]
        )
        store.record_reindex_control(
            {"generation": 1, "scope": "retrieval_text", "created_at": "2026-09-13T00:00:01Z"}
        )
        store.drain_reindex_controls(10)
        self.assertEqual(store.get("old-skipped")["translation_status"], "pending_provider")

        class Syncer:
            def upload(self, _documents):
                return {"durable": True}

        with mock.patch.dict(
            os.environ,
            {"FUNES_RETRIEVAL_LANGUAGE_MODE": "raw", "TRANSLATION_MODEL": "current-model"},
        ):
            app = mock.Mock()
            app.store = store
            app.translator = Translator(store)
            app.syncer = Syncer()
            app.translation_lock = threading.Lock()
            result = persist_translation_documents(app, store.pending_translations(10))
        item = store.get("old-skipped")
        self.assertTrue(result["durable"])
        self.assertEqual(item["translation_status"], "skipped_raw_mode")
        self.assertNotEqual(item["translation_hash"], "old-model-hash")
        store.close()

    def test_full_hub_restore_rewinds_partial_cursor_before_drain(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": f"hub-row-{index}",
                    "source_version": "v1",
                    "raw_text": f"old raw {index}",
                    "retrieval_text": f"old shadow {index}",
                    "translation_status": "ok",
                }
                for index in range(4)
            ]
        )
        control = {
            "generation": 1,
            "scope": "retrieval_text",
            "created_at": "2026-09-13T00:00:01Z",
        }
        store.record_reindex_control(control)
        first = store.apply_pending_reindex_controls(2)
        self.assertEqual(first["scanned"], 2)
        self.assertEqual(first["applied"], 0)

        syncer = SnapshotSync(store)
        syncer.repo = "owner/private"
        syncer.token = "test-token"
        syncer.restore_batch = 2

        def restore_cursor_prefix_revision(_filename):
            result = store.ingest(
                [
                    {
                        "source_identity": "hub-row-0",
                        "source_version": "v2",
                        "raw_text": "new raw revision before cursor",
                        "retrieval_text": "stale generation-zero shadow",
                        "translation_status": "skipped_non_cjk",
                        "retrieval_generation": 0,
                        "native_generation": 0,
                        "updated_at": "9999-01-01T00:00:00Z",
                    }
                ]
            )
            return result["updated"]

        with mock.patch.object(syncer, "_repo_files", return_value=["delta.enc"]), mock.patch.object(
            syncer, "_restore_file", side_effect=restore_cursor_prefix_revision
        ):
            self.assertEqual(syncer.restore(), 1)

        restored = store.get("hub-row-0")
        state = store.conn.execute(
            "SELECT row_cursor, applied_at FROM reindex_controls WHERE generation=1"
        ).fetchone()
        self.assertEqual(restored["raw_text"], "new raw revision before cursor")
        self.assertEqual(restored["retrieval_text"], "new raw revision before cursor")
        self.assertEqual(restored["translation_status"], "pending_provider")
        self.assertEqual(restored["retrieval_generation"], 1)
        self.assertEqual(state["row_cursor"], 4)
        self.assertIsNotNone(state["applied_at"])
        store.close()

    def test_full_hub_restore_prefetches_remote_files_before_replay(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        syncer = SnapshotSync(store)
        syncer.repo = "owner/private"
        syncer.token = "test-token"
        syncer.restore_download_workers = 16
        files = ["funes-snapshot.jsonl.gz.enc", "funes-delta-a.jsonl.gz.enc"]
        events = []

        def prefetch(**kwargs):
            events.append(("prefetch", kwargs))
            return str(Path(self.tmp.name) / "remote")

        def restore_file(filename):
            events.append(("restore", filename))
            return 1

        with mock.patch.object(syncer, "_repo_files", return_value=files), mock.patch(
            "huggingface_hub.snapshot_download", side_effect=prefetch
        ), mock.patch.object(syncer, "_restore_file", side_effect=restore_file):
            self.assertEqual(syncer.restore(), 2)

        self.assertEqual([event[0] for event in events], ["prefetch", "restore", "restore"])
        prefetch_kwargs = events[0][1]
        self.assertEqual(prefetch_kwargs["repo_id"], "owner/private")
        self.assertEqual(prefetch_kwargs["repo_type"], "dataset")
        self.assertEqual(prefetch_kwargs["allow_patterns"], files)
        self.assertEqual(prefetch_kwargs["max_workers"], 16)
        self.assertEqual(prefetch_kwargs["token"], "test-token")
        self.assertIsNone(syncer._restore_prefetch_root)
        store.close()

    def test_full_hub_restore_falls_back_when_prefetch_fails(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        syncer = SnapshotSync(store)
        syncer.repo = "owner/private"
        syncer.token = "test-token"
        files = ["funes-snapshot.jsonl.gz.enc", "funes-delta-a.jsonl.gz.enc"]
        with mock.patch.object(syncer, "_repo_files", return_value=files), mock.patch(
            "huggingface_hub.snapshot_download", side_effect=OSError("offline")
        ), mock.patch.object(syncer, "_restore_file", return_value=1) as restore_file:
            self.assertEqual(syncer.restore(), 2)
        self.assertEqual(
            restore_file.call_args_list,
            [mock.call(files[0]), mock.call(files[1])],
        )
        self.assertIsNone(syncer._restore_prefetch_root)
        store.close()

    def test_hub_restore_without_manifest_replays_legacy_history(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        syncer = SnapshotSync(store)
        syncer.repo = "owner/private"
        syncer.token = "test-token"
        files = [
            "funes-delta-b.jsonl.gz.enc",
            "funes-snapshot-old.jsonl.gz.enc",
            "funes-reindex-0002.jsonl.gz.enc",
            "funes-snapshot.jsonl.gz.enc",
            "funes-delta-a.jsonl.gz.enc",
        ]
        api = mock.Mock()
        api.repo_info.return_value.sha = "head-1"
        api.list_repo_tree.return_value = [mock.Mock(path=name) for name in files]
        module = mock.Mock(HfApi=mock.Mock(return_value=api))
        with mock.patch.dict("sys.modules", {"huggingface_hub": module}):
            selected = syncer._repo_files()

        self.assertEqual(
            selected,
            [
                "funes-snapshot-old.jsonl.gz.enc",
                "funes-snapshot.jsonl.gz.enc",
                "funes-delta-a.jsonl.gz.enc",
                "funes-delta-b.jsonl.gz.enc",
                "funes-reindex-0002.jsonl.gz.enc",
            ],
        )
        store.close()

    def test_hub_restore_manifest_selects_only_active_history(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        syncer = SnapshotSync(store)
        syncer.repo = "owner/private"
        syncer.token = "test-token"
        active = {
            "version": 1,
            "snapshot": "funes-snapshot-current.jsonl.gz.enc",
            "deltas": ["funes-delta-later.jsonl.gz.enc"],
            "controls": ["funes-reindex-0003.jsonl.gz.enc"],
        }
        manifest_path = Path(self.tmp.name) / syncer.manifest_filename
        manifest_path.write_text(json.dumps(active), encoding="utf-8")
        repo_files = [
            syncer.manifest_filename,
            "funes-snapshot-old.jsonl.gz.enc",
            "funes-delta-old.jsonl.gz.enc",
            active["snapshot"],
            *active["deltas"],
            *active["controls"],
        ]
        api = mock.Mock()
        api.repo_info.return_value.sha = "head-1"
        api.list_repo_tree.return_value = [mock.Mock(path=name) for name in repo_files]
        module = mock.Mock(
            HfApi=mock.Mock(return_value=api),
            hf_hub_download=mock.Mock(return_value=str(manifest_path)),
        )
        with mock.patch.dict("sys.modules", {"huggingface_hub": module}):
            selected = syncer._repo_files()

        self.assertEqual(
            selected,
            [active["snapshot"], *active["deltas"], *active["controls"]],
        )
        self.assertEqual(api.list_repo_tree.call_args.kwargs["revision"], "head-1")
        self.assertEqual(
            module.hf_hub_download.call_args.kwargs["revision"], "head-1"
        )
        store.close()

    def test_present_restore_manifest_fails_closed_when_malformed(self):
        from service.server import SnapshotSync

        invalid_manifests = [
            {
                "version": 2,
                "snapshot": "funes-snapshot-current.jsonl.gz.enc",
                "deltas": [],
                "controls": [],
            },
            {
                "version": 1,
                "snapshot": "../funes-snapshot-current.jsonl.gz.enc",
                "deltas": [],
                "controls": [],
            },
            {
                "version": 1,
                "snapshot": "funes-snapshot-current.jsonl.gz.enc",
                "deltas": ["wrong-prefix.jsonl.gz.enc"],
                "controls": [],
            },
            {
                "version": 1,
                "snapshot": "funes-snapshot-current.jsonl.gz",
                "deltas": [],
                "controls": [],
            },
        ]
        store = Store(self.tmp.name)
        try:
            for index, value in enumerate(invalid_manifests):
                with self.subTest(index=index):
                    syncer = SnapshotSync(store)
                    syncer.repo = "owner/private"
                    syncer.token = "test-token"
                    manifest_path = Path(self.tmp.name) / f"invalid-{index}.json"
                    manifest_path.write_text(json.dumps(value), encoding="utf-8")
                    api = mock.Mock()
                    api.repo_info.return_value.sha = "head-1"
                    api.list_repo_tree.return_value = [
                        mock.Mock(path=syncer.manifest_filename),
                        mock.Mock(path="funes-snapshot-current.jsonl.gz.enc"),
                    ]
                    module = mock.Mock(
                        HfApi=mock.Mock(return_value=api),
                        hf_hub_download=mock.Mock(return_value=str(manifest_path)),
                    )
                    with mock.patch.dict(
                        "sys.modules", {"huggingface_hub": module}
                    ), mock.patch.object(syncer, "_restore_file") as restore_file:
                        self.assertEqual(syncer.restore(), -1)
                    self.assertTrue(syncer.restore_failed)
                    restore_file.assert_not_called()
        finally:
            store.close()

    def test_present_restore_manifest_fails_closed_on_missing_reference(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        syncer = SnapshotSync(store)
        syncer.repo = "owner/private"
        syncer.token = "test-token"
        manifest = {
            "version": 1,
            "snapshot": "funes-snapshot-current.jsonl.gz.enc",
            "deltas": ["funes-delta-missing.jsonl.gz.enc"],
            "controls": [],
        }
        manifest_path = Path(self.tmp.name) / syncer.manifest_filename
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        api = mock.Mock()
        api.repo_info.return_value.sha = "head-1"
        api.list_repo_tree.return_value = [
            mock.Mock(path=syncer.manifest_filename),
            mock.Mock(path=manifest["snapshot"]),
        ]
        module = mock.Mock(
            HfApi=mock.Mock(return_value=api),
            hf_hub_download=mock.Mock(return_value=str(manifest_path)),
        )
        with mock.patch.dict(
            "sys.modules", {"huggingface_hub": module}
        ), mock.patch.object(syncer, "_restore_file") as restore_file:
            self.assertEqual(syncer.restore(), -1)
        self.assertTrue(syncer.restore_failed)
        restore_file.assert_not_called()
        store.close()

    def test_restore_compacts_history_and_skips_satisfied_updates(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": f"compact-row-{index}",
                    "raw_text": f"raw {index}",
                    "retrieval_text": f"old shadow {index}",
                    "translation_status": "ok",
                    "native_index_status": "indexed",
                }
                for index in range(20)
            ]
        )
        for generation, scope in enumerate(
            ("all", "retrieval_text", "retrieval_text", "all", "retrieval_text"),
            start=1,
        ):
            store.record_reindex_control(
                {
                    "generation": generation,
                    "scope": scope,
                    "created_at": f"2026-09-13T00:00:0{generation}Z",
                }
            )

        compacted = store.compact_reindex_controls(replay=True)
        first = store.drain_reindex_controls(7)
        controls = store.conn.execute(
            "SELECT generation, scope FROM reindex_controls ORDER BY generation"
        ).fetchall()
        self.assertEqual(compacted, {"kept": 2, "deleted": 3})
        self.assertEqual(
            [(row["generation"], row["scope"]) for row in controls],
            [(4, "all"), (5, "retrieval_text")],
        )
        self.assertEqual(first["scanned"], 40)
        # The newest retrieval control advances retrieval/native state first;
        # the retained older all-control then advances the independent
        # embedding epoch for every row.
        self.assertEqual(first["updated"], 40)
        self.assertEqual(
            {
                store.get(f"compact-row-{index}")["embedding_generation"]
                for index in range(20)
            },
            {4},
        )

        store.compact_reindex_controls(replay=True)
        replay = store.drain_reindex_controls(7)
        self.assertEqual(replay["scanned"], 40)
        self.assertEqual(replay["updated"], 0)
        store.close()

    def test_reindex_http_returns_202_only_after_durable_queue(self):
        class Syncer:
            restoring = False
            restore_failed = False

            def __init__(self, durable):
                self.durable = durable
                self.controls = []

            def upload_reindex_control(self, control):
                self.controls.append(dict(control))
                return {"durable": self.durable, "reason": "not_durable"}

        def request(durable):
            directory = tempfile.TemporaryDirectory()
            app = mock.Mock()
            app.store = Store(directory.name)
            app.syncer = Syncer(durable)
            app.reindex_lock = threading.Lock()
            app.translation_lock = threading.Lock()
            app.reindex_wake = threading.Event()
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                status, body = self._request(
                    server, "POST", "/reindex", {"scope": "retrieval_text"}
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                app.store.close()
                directory.cleanup()
            return status, body, app.syncer.controls

        status, body, controls = request(True)
        self.assertEqual(status, 202)
        self.assertEqual(body, {"queued": True, "durable": True, "scope": "retrieval_text", "generation": 1})
        self.assertEqual(controls[0]["_funes_record"], "reindex_control")
        self.assertNotIn("raw_text", json.dumps(body))
        status, body, _ = request(False)
        self.assertEqual(status, 503)
        self.assertFalse(body["queued"])

    def test_queue_reindex_holds_upload_lock_until_control_is_local(self):
        from service.server import SnapshotSync, queue_reindex

        store = Store(self.tmp.name)
        syncer = SnapshotSync(store)
        app = mock.Mock(
            store=store,
            syncer=syncer,
            reindex_lock=threading.Lock(),
            reindex_wake=threading.Event(),
        )
        remote_committed = threading.Event()
        competitor_attempted = threading.Event()
        observed_local_counts = []
        queue_results = []
        errors = []
        original_record = store.record_reindex_control

        def upload_control(_control):
            with syncer.upload_lock:
                remote_committed.set()
                if not competitor_attempted.wait(1):
                    raise AssertionError("snapshot competitor did not start")
                return {"uploaded": True, "durable": True}

        def record_while_locked(control):
            if not syncer.upload_lock._is_owned():
                raise AssertionError("control record escaped the upload lock")
            return original_record(control)

        def queue_worker():
            try:
                queue_results.append(queue_reindex(app, "all"))
            except BaseException as exc:
                errors.append(exc)

        def snapshot_competitor():
            if not remote_committed.wait(1):
                errors.append(AssertionError("remote control was not committed"))
                return
            competitor_attempted.set()
            with syncer.upload_lock:
                observed_local_counts.append(
                    store.conn.execute("SELECT count(*) FROM reindex_controls").fetchone()[0]
                )

        with mock.patch.object(
            syncer, "upload_reindex_control", side_effect=upload_control
        ), mock.patch.object(
            store, "record_reindex_control", side_effect=record_while_locked
        ):
            competitor = threading.Thread(target=snapshot_competitor)
            queue = threading.Thread(target=queue_worker)
            competitor.start()
            queue.start()
            queue.join(timeout=2)
            competitor.join(timeout=2)

        self.assertFalse(queue.is_alive())
        self.assertFalse(competitor.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(queue_results[0]["durable"])
        self.assertEqual(observed_local_counts, [1])
        store.close()

    def test_queue_reindex_remote_failure_does_not_create_local_control(self):
        from service.server import SnapshotSync, queue_reindex

        store = Store(self.tmp.name)
        syncer = SnapshotSync(store)
        app = mock.Mock(
            store=store,
            syncer=syncer,
            reindex_lock=threading.Lock(),
            reindex_wake=threading.Event(),
        )
        with mock.patch.object(
            syncer,
            "upload_reindex_control",
            return_value={"uploaded": False, "durable": False, "reason": "offline"},
        ):
            result = queue_reindex(app, "all")

        self.assertFalse(result["durable"])
        self.assertEqual(
            store.conn.execute("SELECT count(*) FROM reindex_controls").fetchone()[0],
            0,
        )
        store.close()

    def test_reindex_control_is_encrypted_before_local_durable_ack(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        syncer = SnapshotSync(store)
        control = store.next_reindex_control("all")
        result = syncer.upload_reindex_control(control)
        encrypted = next((Path(self.tmp.name) / "reindex-queue").glob("*.enc"))
        self.assertTrue(result["durable"])
        self.assertTrue(encrypted.read_bytes().startswith(b"FUNES-SOURCE-V1\0"))
        self.assertNotIn(b"reindex_control", encrypted.read_bytes())
        store.close()

    def test_configured_remote_without_hf_token_is_not_durable(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "remote", "raw_text": "must reach hub"}])
        os.environ["FUNES_STORAGE_REPO"] = "owner/private"
        result = SnapshotSync(store).upload(store.get_many(["remote"]))
        self.assertFalse(result["durable"])
        self.assertEqual(result["reason"], "HF storage not configured")
        store.close()

    def test_full_snapshot_and_manifest_are_one_atomic_commit(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "compact", "raw_text": "current"}])
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        api = mock.Mock()
        api.repo_info.return_value.sha = "head-1"
        api.list_repo_tree.return_value = []
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            syncer = SnapshotSync(store)
            result = syncer.upload()

        self.assertTrue(result["durable"])
        api.upload_file.assert_not_called()
        api.create_commit.assert_called_once()
        kwargs = api.create_commit.call_args.kwargs
        self.assertEqual(
            set(kwargs),
            {
                "repo_id",
                "repo_type",
                "operations",
                "commit_message",
                "parent_commit",
            },
        )
        self.assertEqual(kwargs["repo_id"], "owner/private")
        self.assertEqual(kwargs["repo_type"], "dataset")
        self.assertEqual(kwargs["parent_commit"], "head-1")
        operations = {operation.path_in_repo: operation for operation in kwargs["operations"]}
        snapshot_name = "funes-snapshot.jsonl.gz.enc"
        self.assertEqual(set(operations), {snapshot_name, syncer.manifest_filename})
        manifest = json.loads(operations[syncer.manifest_filename].path_or_fileobj)
        self.assertEqual(
            manifest,
            {
                "version": 1,
                "snapshot": snapshot_name,
                "deltas": [],
                "controls": [],
            },
        )
        self.assertEqual(kwargs["commit_message"], "funes encrypted source snapshot")
        store.close()

    def test_unrestored_active_manifest_cannot_compact(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "partial", "raw_text": "partial state"}])
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        syncer = SnapshotSync(store)
        manifest = {
            "version": 1,
            "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": ["funes-delta-not-restored.jsonl.gz.enc"],
            "controls": [],
        }
        manifest_path = Path(self.tmp.name) / syncer.manifest_filename
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        api = mock.Mock()
        api.repo_info.return_value.sha = "head-active"
        api.list_repo_tree.return_value = [
            mock.Mock(path=syncer.manifest_filename, blob_id="manifest-active"),
            mock.Mock(path=manifest["snapshot"], blob_id="snapshot-active"),
            mock.Mock(path=manifest["deltas"][0], blob_id="delta-active"),
        ]
        original_snapshot = store.snapshot
        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch(
            "huggingface_hub.hf_hub_download", return_value=str(manifest_path)
        ), mock.patch.object(store, "snapshot", wraps=original_snapshot) as snapshot:
            result = syncer.upload()

        self.assertFalse(result["durable"])
        snapshot.assert_not_called()
        api.create_commit.assert_not_called()
        store.close()

    def test_unrestored_legacy_history_cannot_create_first_manifest(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        syncer = SnapshotSync(store)
        api = mock.Mock()
        api.repo_info.return_value.sha = "head-legacy"
        api.list_repo_tree.return_value = [
            mock.Mock(
                path="funes-snapshot-old.jsonl.gz.enc", blob_id="snapshot-old"
            ),
            mock.Mock(path="funes-delta-old.jsonl.gz.enc", blob_id="delta-old"),
        ]
        original_snapshot = store.snapshot
        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch.object(
            store, "snapshot", wraps=original_snapshot
        ) as snapshot:
            result = syncer.upload()

        self.assertFalse(result["durable"])
        snapshot.assert_not_called()
        api.create_commit.assert_not_called()
        store.close()

    def test_compaction_retries_with_only_concurrent_manifest_suffix(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "compact", "raw_text": "current"}])
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        syncer = SnapshotSync(store)
        syncer.covered_revision = "head-1"
        snapshot_name = "funes-snapshot.jsonl.gz.enc"
        old_delta = "funes-delta-old.jsonl.gz.enc"
        new_delta = "funes-delta-concurrent.jsonl.gz.enc"
        old_control = "funes-reindex-0001-old.jsonl.gz.enc"
        new_control = "funes-reindex-0002-new.jsonl.gz.enc"
        base_manifest = {
            "version": 1,
            "snapshot": snapshot_name,
            "deltas": [old_delta],
            "controls": [old_control],
        }
        concurrent_manifest = {
            **base_manifest,
            "deltas": [old_delta, new_delta],
            "controls": [old_control, new_control],
        }
        base_path = Path(self.tmp.name) / "compact-head-1.json"
        concurrent_path = Path(self.tmp.name) / "compact-head-2.json"
        base_path.write_text(json.dumps(base_manifest), encoding="utf-8")
        concurrent_path.write_text(json.dumps(concurrent_manifest), encoding="utf-8")
        api = mock.Mock()
        api.repo_info.side_effect = [mock.Mock(sha="head-1"), mock.Mock(sha="head-2")]
        api.list_repo_tree.side_effect = [
            [
                mock.Mock(path=syncer.manifest_filename, blob_id="manifest-1"),
                mock.Mock(path=snapshot_name, blob_id="snapshot-1"),
                mock.Mock(path=old_delta, blob_id="delta-old"),
                mock.Mock(path=old_control, blob_id="control-old"),
            ],
            [
                mock.Mock(path=syncer.manifest_filename, blob_id="manifest-2"),
                mock.Mock(path=snapshot_name, blob_id="snapshot-1"),
                mock.Mock(path=old_delta, blob_id="delta-old"),
                mock.Mock(path=new_delta, blob_id="delta-new"),
                mock.Mock(path=old_control, blob_id="control-old"),
                mock.Mock(path=new_control, blob_id="control-new"),
            ],
        ]
        api.create_commit.side_effect = [
            RuntimeError("stale parent"),
            mock.Mock(oid="head-3"),
        ]
        snapshot_head_reads = []
        original_snapshot = store.snapshot

        def snapshot_after_head_read(path):
            snapshot_head_reads.append(api.repo_info.call_count)
            return original_snapshot(path)

        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch(
            "huggingface_hub.hf_hub_download",
            side_effect=[str(base_path), str(concurrent_path)],
        ), mock.patch.object(store, "snapshot", side_effect=snapshot_after_head_read):
            result = syncer.upload()

        self.assertTrue(result["durable"])
        self.assertEqual(snapshot_head_reads, [1])
        self.assertEqual(
            [call.kwargs["parent_commit"] for call in api.create_commit.call_args_list],
            ["head-1", "head-2"],
        )
        operations = {
            operation.path_in_repo: operation
            for operation in api.create_commit.call_args_list[-1].kwargs["operations"]
        }
        manifest = json.loads(operations[syncer.manifest_filename].path_or_fileobj)
        self.assertEqual(manifest["snapshot"], snapshot_name)
        self.assertEqual(manifest["deltas"], [new_delta])
        self.assertEqual(manifest["controls"], [new_control])
        self.assertEqual(syncer.covered_revision, "head-1")
        store.close()

    def test_compaction_fails_closed_when_concurrent_snapshot_wins(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        syncer = SnapshotSync(store)
        syncer.covered_revision = "head-1"
        snapshot_name = "funes-snapshot.jsonl.gz.enc"
        manifest = {
            "version": 1,
            "snapshot": snapshot_name,
            "deltas": [],
            "controls": [],
        }
        base_path = Path(self.tmp.name) / "winner-head-1.json"
        winner_path = Path(self.tmp.name) / "winner-head-2.json"
        base_path.write_text(json.dumps(manifest), encoding="utf-8")
        winner_path.write_text(json.dumps(manifest), encoding="utf-8")
        api = mock.Mock()
        api.repo_info.side_effect = [mock.Mock(sha="head-1"), mock.Mock(sha="head-2")]
        api.list_repo_tree.side_effect = [
            [
                mock.Mock(path=syncer.manifest_filename, blob_id="manifest-1"),
                mock.Mock(path=snapshot_name, blob_id="snapshot-1"),
            ],
            [
                mock.Mock(path=syncer.manifest_filename, blob_id="manifest-2"),
                mock.Mock(path=snapshot_name, blob_id="snapshot-2"),
            ],
        ]
        api.create_commit.side_effect = RuntimeError("stale parent")
        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch(
            "huggingface_hub.hf_hub_download",
            side_effect=[str(base_path), str(winner_path)],
        ):
            result = syncer.upload()

        self.assertFalse(result["durable"])
        self.assertEqual(api.create_commit.call_count, 1)
        store.close()

    def test_successful_restore_then_compaction_resets_base_manifest_entries(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        syncer = SnapshotSync(store)
        manifest = {
            "version": 1,
            "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": ["funes-delta-base.jsonl.gz.enc"],
            "controls": ["funes-reindex-0001-base.jsonl.gz.enc"],
        }
        manifest_path = Path(self.tmp.name) / syncer.manifest_filename
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        tree = [
            mock.Mock(path=syncer.manifest_filename, blob_id="manifest-base"),
            mock.Mock(path=manifest["snapshot"], blob_id="snapshot-base"),
            mock.Mock(path=manifest["deltas"][0], blob_id="delta-base"),
            mock.Mock(path=manifest["controls"][0], blob_id="control-base"),
        ]
        api = mock.Mock()
        api.repo_info.side_effect = [mock.Mock(sha="head-1"), mock.Mock(sha="head-1")]
        api.list_repo_tree.return_value = tree
        api.create_commit.return_value = mock.Mock(oid="head-2")
        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch(
            "huggingface_hub.hf_hub_download", return_value=str(manifest_path)
        ), mock.patch.object(syncer, "_restore_file", return_value=0), mock.patch.object(
            syncer, "_prefetch_restore_files", return_value=None
        ):
            self.assertEqual(syncer.restore(), 0)
            self.assertEqual(syncer.covered_revision, "head-1")
            result = syncer.upload()

        self.assertTrue(result["durable"])
        self.assertEqual(syncer.covered_revision, "head-2")
        operations = {
            operation.path_in_repo: operation
            for operation in api.create_commit.call_args.kwargs["operations"]
        }
        compacted = json.loads(
            operations[syncer.manifest_filename].path_or_fileobj
        )
        self.assertEqual(compacted["deltas"], [])
        self.assertEqual(compacted["controls"], [])
        store.close()

    def test_existing_delta_requires_manifest_membership_before_durable_ack(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "remote", "raw_text": "retry delta"}])
        docs = store.get_many(["remote"])
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        syncer = SnapshotSync(store)
        durable_docs = [*docs, store.native_index_state_record()]
        digest = hashlib.sha256(
            json.dumps(
                durable_docs,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:24]
        delta_name = syncer.delta_target(digest)
        snapshot_name = "funes-snapshot.jsonl.gz.enc"
        manifest = {
            "version": 1,
            "snapshot": snapshot_name,
            "deltas": [],
            "controls": [],
        }
        manifest_path = Path(self.tmp.name) / syncer.manifest_filename
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        api = mock.Mock()
        api.repo_info.return_value.sha = "head-1"
        api.list_repo_tree.return_value = [
            mock.Mock(path=syncer.manifest_filename),
            mock.Mock(path=snapshot_name),
            mock.Mock(path=delta_name),
        ]
        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch(
            "huggingface_hub.hf_hub_download", return_value=str(manifest_path)
        ):
            result = syncer.upload(docs)

        self.assertTrue(result["durable"])
        api.upload_file.assert_not_called()
        api.create_commit.assert_called_once()
        operations = {
            operation.path_in_repo: operation
            for operation in api.create_commit.call_args.kwargs["operations"]
        }
        self.assertEqual(
            api.create_commit.call_args.kwargs["parent_commit"], "head-1"
        )
        self.assertEqual(set(operations), {delta_name, syncer.manifest_filename})
        updated = json.loads(operations[syncer.manifest_filename].path_or_fileobj)
        self.assertEqual(updated["deltas"], [delta_name])
        store.close()

    def test_stale_delta_writer_confirms_new_membership_before_durable_ack(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "ours", "raw_text": "our delta"}])
        docs = store.get_many(["ours"])
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        syncer = SnapshotSync(store)
        snapshot_name = "funes-snapshot.jsonl.gz.enc"
        concurrent_delta = "funes-delta-concurrent.jsonl.gz.enc"
        durable_docs = [*docs, store.native_index_state_record()]
        digest = hashlib.sha256(
            json.dumps(
                durable_docs,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:24]
        delta_name = f"funes-delta-{digest}.jsonl.gz.enc"
        base_manifest = {
            "version": 1,
            "snapshot": snapshot_name,
            "deltas": [],
            "controls": [],
        }
        concurrent_manifest = {
            **base_manifest,
            "deltas": [concurrent_delta, delta_name],
        }
        base_path = Path(self.tmp.name) / "manifest-head-1.json"
        concurrent_path = Path(self.tmp.name) / "manifest-head-2.json"
        base_path.write_text(json.dumps(base_manifest), encoding="utf-8")
        concurrent_path.write_text(json.dumps(concurrent_manifest), encoding="utf-8")
        api = mock.Mock()
        api.repo_info.side_effect = [mock.Mock(sha="head-1"), mock.Mock(sha="head-2")]
        api.list_repo_tree.side_effect = [
            [
                mock.Mock(path=syncer.manifest_filename, blob_id="manifest-1"),
                mock.Mock(path=snapshot_name, blob_id="snapshot-1"),
            ],
            [
                mock.Mock(path=syncer.manifest_filename, blob_id="manifest-2"),
                mock.Mock(path=snapshot_name, blob_id="snapshot-1"),
                mock.Mock(path=concurrent_delta, blob_id="delta-other"),
                mock.Mock(path=delta_name, blob_id="delta-ours"),
            ],
        ]
        api.create_commit.side_effect = RuntimeError("stale parent")
        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch(
            "huggingface_hub.hf_hub_download",
            side_effect=[str(base_path), str(concurrent_path)],
        ):
            result = syncer.upload(docs)

        self.assertTrue(result["durable"])
        self.assertFalse(result["uploaded"])
        self.assertTrue(result["already_uploaded"])
        self.assertEqual(api.create_commit.call_count, 1)
        self.assertEqual(
            [call.kwargs["parent_commit"] for call in api.create_commit.call_args_list],
            ["head-1"],
        )
        self.assertEqual(concurrent_manifest["deltas"], [concurrent_delta, delta_name])
        store.close()

    def test_active_manifest_control_upload_is_one_atomic_commit(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        syncer = SnapshotSync(store)
        snapshot_name = "funes-snapshot.jsonl.gz.enc"
        manifest = {
            "version": 1,
            "snapshot": snapshot_name,
            "deltas": ["funes-delta-existing.jsonl.gz.enc"],
            "controls": [],
        }
        manifest_path = Path(self.tmp.name) / "control-manifest-head-1.json"
        concurrent_path = Path(self.tmp.name) / "control-manifest-head-2.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        api = mock.Mock()
        api.repo_info.side_effect = [mock.Mock(sha="head-1"), mock.Mock(sha="head-2")]
        control = store.next_reindex_control("all")
        control_digest = hashlib.sha256(
            json.dumps(
                control,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:16]
        control_name = (
            f"funes-reindex-{control['generation']:020d}-{control_digest}.jsonl.gz.enc"
        )
        concurrent_control = (
            "funes-reindex-00000000000000000001-aaaaaaaaaaaaaaaa.jsonl.gz.enc"
        )
        concurrent_manifest = {
            **manifest,
            "controls": [concurrent_control],
        }
        concurrent_path.write_text(json.dumps(concurrent_manifest), encoding="utf-8")
        api.list_repo_tree.side_effect = [
            [
                mock.Mock(path=syncer.manifest_filename, blob_id="manifest-1"),
                mock.Mock(path=snapshot_name, blob_id="snapshot-1"),
                mock.Mock(path=manifest["deltas"][0], blob_id="delta-1"),
                mock.Mock(path=control_name, blob_id="control-ours"),
            ],
            [
                mock.Mock(path=syncer.manifest_filename, blob_id="manifest-2"),
                mock.Mock(path=snapshot_name, blob_id="snapshot-1"),
                mock.Mock(path=manifest["deltas"][0], blob_id="delta-1"),
                mock.Mock(path=concurrent_control, blob_id="control-other"),
                mock.Mock(path=control_name, blob_id="control-ours"),
            ],
        ]
        api.create_commit.side_effect = [RuntimeError("stale parent"), None]
        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch(
            "huggingface_hub.hf_hub_download",
            side_effect=[str(manifest_path), str(concurrent_path)],
        ):
            result = syncer.upload_reindex_control(control)

        self.assertTrue(result["durable"])
        api.upload_file.assert_not_called()
        self.assertEqual(api.create_commit.call_count, 2)
        self.assertEqual(
            [call.kwargs["parent_commit"] for call in api.create_commit.call_args_list],
            ["head-1", "head-2"],
        )
        operations = {
            operation.path_in_repo: operation
            for operation in api.create_commit.call_args_list[-1].kwargs["operations"]
        }
        self.assertEqual(
            api.create_commit.call_args_list[-1].kwargs["parent_commit"], "head-2"
        )
        control_names = [
            name for name in operations if name.startswith("funes-reindex-")
        ]
        self.assertEqual(control_names, [control_name])
        self.assertEqual(set(operations), {control_names[0], syncer.manifest_filename})
        updated = json.loads(operations[syncer.manifest_filename].path_or_fileobj)
        self.assertEqual(updated["deltas"], manifest["deltas"])
        self.assertEqual(updated["controls"], [concurrent_control, *control_names])
        store.close()

    def test_local_delta_and_control_commits_advance_covered_revision(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "local", "raw_text": "local delta"}])
        docs = store.get_many(["local"])
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        syncer = SnapshotSync(store)
        syncer.covered_revision = "head-1"
        durable_docs = [*docs, store.native_index_state_record()]
        digest = hashlib.sha256(
            json.dumps(
                durable_docs,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:24]
        delta_name = f"funes-delta-{digest}.jsonl.gz.enc"
        snapshot_name = "funes-snapshot.jsonl.gz.enc"
        base_manifest = {
            "version": 1,
            "snapshot": snapshot_name,
            "deltas": [],
            "controls": [],
        }
        delta_manifest = {**base_manifest, "deltas": [delta_name]}
        base_path = Path(self.tmp.name) / "coverage-head-1.json"
        delta_path = Path(self.tmp.name) / "coverage-head-2.json"
        base_path.write_text(json.dumps(base_manifest), encoding="utf-8")
        delta_path.write_text(json.dumps(delta_manifest), encoding="utf-8")
        api = mock.Mock()
        api.repo_info.side_effect = [mock.Mock(sha="head-1"), mock.Mock(sha="head-2")]
        api.list_repo_tree.side_effect = [
            [
                mock.Mock(path=syncer.manifest_filename, blob_id="manifest-1"),
                mock.Mock(path=snapshot_name, blob_id="snapshot-1"),
            ],
            [
                mock.Mock(path=syncer.manifest_filename, blob_id="manifest-2"),
                mock.Mock(path=snapshot_name, blob_id="snapshot-1"),
                mock.Mock(path=delta_name, blob_id="delta-local"),
            ],
        ]
        api.create_commit.side_effect = [mock.Mock(oid="head-2"), mock.Mock(oid="head-3")]
        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch(
            "huggingface_hub.hf_hub_download",
            side_effect=[str(base_path), str(delta_path)],
        ):
            delta_result = syncer.upload(docs)
            self.assertEqual(syncer.covered_revision, "head-2")
            control_result = syncer.upload_reindex_control(
                store.next_reindex_control("all")
            )

        self.assertTrue(delta_result["durable"])
        self.assertTrue(control_result["durable"])
        self.assertEqual(syncer.covered_revision, "head-3")
        store.close()

    def test_existing_content_addressed_delta_is_durable_without_reupload(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "remote", "raw_text": "already durable"}])
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        api = mock.Mock()
        api.repo_info.return_value.sha = "head-1"
        api.list_repo_tree.return_value = []
        api.file_exists.return_value = True
        module = mock.Mock(HfApi=mock.Mock(return_value=api))
        syncer = SnapshotSync(store)
        with mock.patch.dict("sys.modules", {"huggingface_hub": module}):
            result = syncer.upload(store.get_many(["remote"]))
        self.assertTrue(result["durable"])
        self.assertTrue(result["already_uploaded"])
        api.upload_file.assert_not_called()
        self.assertFalse(list(Path(self.tmp.name).glob("funes-delta-*.jsonl.gz")))
        store.close()

    def test_encrypted_source_snapshot_roundtrip_hides_plaintext(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        marker = "sensitive bearer material must remain encrypted"
        store.ingest([{"source_path": "secret", "raw_text": marker}])
        os.environ["FUNES_STORAGE_KEY"] = "test-storage-key"
        syncer = SnapshotSync(store)
        plain = Path(self.tmp.name) / "source.jsonl.gz"
        encrypted = Path(self.tmp.name) / "source.jsonl.gz.enc"
        decrypted = Path(self.tmp.name) / "restored.jsonl.gz"
        store.snapshot(plain)
        syncer._encrypt_file(plain, encrypted)
        self.assertNotIn(marker.encode(), encrypted.read_bytes())
        syncer._decrypt_file(encrypted, decrypted)
        second_dir = tempfile.TemporaryDirectory()
        restored = Store(second_dir.name)
        self.assertEqual(restored.restore(decrypted), 1)
        self.assertEqual(restored.get("secret")["raw_text"], marker)
        restored.close(); second_dir.cleanup()
        store.close()

    def test_remote_source_upload_fails_closed_without_encryption_key(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        store.ingest([{"source_path": "secret", "raw_text": "private source"}])
        os.environ.update(FUNES_STORAGE_REPO="owner/private", HF_TOKEN="hf-test")
        os.environ.pop("FUNES_AUTH_TOKEN", None)
        os.environ.pop("FUNES_API_TOKEN", None)
        os.environ.pop("FUNES_STORAGE_KEY", None)
        result = SnapshotSync(store).upload()
        self.assertFalse(result["durable"])
        self.assertEqual(result["reason"], "RuntimeError")
        store.close()

    def test_restore_file_decrypts_inside_temporary_directory(self):
        from service.server import SnapshotSync
        source = Store(self.tmp.name)
        source.ingest([{"source_identity": "restore-one", "raw_text": "encrypted restore"}])
        os.environ["FUNES_STORAGE_KEY"] = "test-storage-key"
        writer = SnapshotSync(source)
        plaintext = Path(self.tmp.name) / "funes-delta-test.jsonl.gz"
        encrypted = Path(self.tmp.name) / "funes-delta-test.jsonl.gz.enc"
        writer._write_jsonl_gzip(source.get_many(["restore-one"]), plaintext)
        writer._encrypt_file(plaintext, encrypted)
        target_dir = tempfile.TemporaryDirectory()
        target = Store(target_dir.name)
        reader = SnapshotSync(target)
        prefetch = target.data_dir / "remote"
        prefetch.mkdir(parents=True)
        prefetched_encrypted = prefetch / encrypted.name
        prefetched_encrypted.write_bytes(encrypted.read_bytes())
        reader._restore_prefetch_root = prefetch
        with mock.patch(
            "huggingface_hub.hf_hub_download",
            side_effect=AssertionError("prefetched restore must not download twice"),
        ):
            self.assertEqual(reader._restore_file(encrypted.name), 1)
        self.assertEqual(target.get("restore-one")["raw_text"], "encrypted restore")
        source.close(); target.close(); target_dir.cleanup()

    def test_restore_failure_is_fail_closed(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        os.environ.update(FUNES_STORAGE_REPO="owner/private", HF_TOKEN="hf-test")
        syncer = SnapshotSync(store)
        with mock.patch.dict("sys.modules", {"huggingface_hub": mock.Mock(hf_hub_download=mock.Mock(side_effect=RuntimeError("offline")))}):
            self.assertEqual(syncer.restore(), -1)
        self.assertTrue(syncer.restore_failed)
        result = syncer.upload()
        self.assertFalse(result["durable"])
        self.assertEqual(result["reason"], "restore_failed")
        store.close()



    def test_delta_target_shards_deterministically_by_digest_prefix(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        syncer = SnapshotSync(store)
        digest = "4a1b2c3d4e5f60718293a4b5"
        self.assertEqual(
            syncer.delta_target(digest),
            "deltas/4a/funes-delta-4a1b2c3d4e5f60718293a4b5.jsonl.gz.enc",
        )

        with mock.patch.dict(os.environ, {"FUNES_DELTA_DIR": "archive/deltas", "FUNES_DELTA_PREFIX": "custom-delta-"}):
            custom_syncer = SnapshotSync(store)
            self.assertEqual(
                custom_syncer.delta_target(digest),
                "archive/deltas/4a/custom-delta-4a1b2c3d4e5f60718293a4b5.jsonl.gz.enc",
            )

        with mock.patch.dict(os.environ, {"FUNES_DELTA_DIR": ""}):
            unsharded_syncer = SnapshotSync(store)
            self.assertEqual(
                unsharded_syncer.delta_target(digest),
                "funes-delta-4a1b2c3d4e5f60718293a4b5.jsonl.gz.enc",
            )
        store.close()

    def test_sharded_delta_deduplication_recognizes_legacy_root_deltas(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "legacy-dedupe", "raw_text": "text"}])
        docs = store.get_many(["legacy-dedupe"])
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        syncer = SnapshotSync(store)
        durable_docs = [*docs, store.native_index_state_record()]
        digest = hashlib.sha256(
            json.dumps(
                durable_docs,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:24]
        legacy_delta = f"funes-delta-{digest}.jsonl.gz.enc"
        sharded_delta = syncer.delta_target(digest)
        self.assertNotEqual(legacy_delta, sharded_delta)

        manifest = {
            "version": 1,
            "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": [legacy_delta],
            "controls": [],
        }
        manifest_path = Path(self.tmp.name) / "manifest-legacy-dedupe.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        api = mock.Mock()
        api.repo_info.return_value = mock.Mock(sha="head-1")
        api.list_repo_tree.return_value = [
            mock.Mock(path=syncer.manifest_filename, blob_id="m-1"),
            mock.Mock(path="funes-snapshot.jsonl.gz.enc", blob_id="s-1"),
            mock.Mock(path=legacy_delta, blob_id="d-1"),
        ]
        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch(
            "huggingface_hub.hf_hub_download", return_value=str(manifest_path)
        ):
            result = syncer.upload(docs)

        self.assertTrue(result["durable"])
        self.assertTrue(result["already_uploaded"])
        self.assertFalse(result["uploaded"])
        api.create_commit.assert_not_called()
        api.upload_file.assert_not_called()
        store.close()

    def test_repo_files_orders_legacy_root_deltas_before_sharded_deltas_without_manifest(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        syncer = SnapshotSync(store)
        syncer.repo = "owner/private"
        syncer.token = "test-token"
        repo_files = [
            "funes-snapshot.jsonl.gz.enc",
            "deltas/0b/funes-delta-0b2.jsonl.gz.enc",
            "funes-delta-z.jsonl.gz.enc",
            "deltas/0a/funes-delta-0a1.jsonl.gz.enc",
            "funes-delta-a.jsonl.gz.enc",
            "funes-reindex-0001.jsonl.gz.enc",
            "deltas/invalid-not-three-parts.jsonl.gz.enc",
            "other_dir/0a/funes-delta-0a1.jsonl.gz.enc",
        ]
        api = mock.Mock()
        api.repo_info.return_value = mock.Mock(sha="head-1")
        api.list_repo_tree.return_value = [mock.Mock(path=p) for p in repo_files]
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            files = syncer._repo_files()

        self.assertEqual(
            files,
            [
                "funes-snapshot.jsonl.gz.enc",
                "funes-delta-a.jsonl.gz.enc",
                "funes-delta-z.jsonl.gz.enc",
                "deltas/0a/funes-delta-0a1.jsonl.gz.enc",
                "deltas/0b/funes-delta-0b2.jsonl.gz.enc",
                "funes-reindex-0001.jsonl.gz.enc",
            ],
        )
        store.close()

    def test_restore_roundtrip_mixed_snapshot_legacy_and_sharded_deltas(self):
        from service.server import SnapshotSync

        os.environ["FUNES_STORAGE_KEY"] = "test-storage-key"
        source_dir = tempfile.TemporaryDirectory()
        source = Store(source_dir.name)
        source.ingest([{"source_identity": "doc-snapshot", "raw_text": "from snapshot"}])
        source.ingest([{"source_identity": "doc-legacy", "raw_text": "from legacy delta"}])
        source.ingest([{"source_identity": "doc-sharded", "raw_text": "from sharded delta"}])

        syncer_writer = SnapshotSync(source)
        artifacts_dir = Path(self.tmp.name) / "mixed-restore-artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 1. Snapshot
        snapshot_plain = artifacts_dir / "funes-snapshot.jsonl.gz"
        snapshot_enc = artifacts_dir / "funes-snapshot.jsonl.gz.enc"
        syncer_writer._write_jsonl_gzip(source.get_many(["doc-snapshot"]), snapshot_plain)
        syncer_writer._encrypt_file(snapshot_plain, snapshot_enc)

        # 2. Legacy delta
        legacy_plain = artifacts_dir / "funes-delta-legacy.jsonl.gz"
        legacy_enc = artifacts_dir / "funes-delta-legacy.jsonl.gz.enc"
        syncer_writer._write_jsonl_gzip(source.get_many(["doc-legacy"]), legacy_plain)
        syncer_writer._encrypt_file(legacy_plain, legacy_enc)

        # 3. Sharded delta
        sharded_dir = artifacts_dir / "deltas" / "3f"
        sharded_dir.mkdir(parents=True, exist_ok=True)
        sharded_plain = sharded_dir / "funes-delta-3f123.jsonl.gz"
        sharded_enc = sharded_dir / "funes-delta-3f123.jsonl.gz.enc"
        syncer_writer._write_jsonl_gzip(source.get_many(["doc-sharded"]), sharded_plain)
        syncer_writer._encrypt_file(sharded_plain, sharded_enc)

        # Restore target
        target_dir = tempfile.TemporaryDirectory()
        target = Store(target_dir.name)
        syncer_reader = SnapshotSync(target)
        syncer_reader.repo = "owner/private"
        syncer_reader.token = "test-token"

        manifest = {
            "version": 1,
            "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": [
                "funes-delta-legacy.jsonl.gz.enc",
                "deltas/3f/funes-delta-3f123.jsonl.gz.enc",
            ],
            "controls": [],
        }
        manifest_path = artifacts_dir / syncer_reader.manifest_filename
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        api = mock.Mock()
        api.repo_info.return_value = mock.Mock(sha="head-1")
        api.list_repo_tree.return_value = [
            mock.Mock(path=syncer_reader.manifest_filename),
            mock.Mock(path=manifest["snapshot"]),
            mock.Mock(path=manifest["deltas"][0]),
            mock.Mock(path=manifest["deltas"][1]),
        ]

        def fake_download(**kwargs):
            return str(artifacts_dir / kwargs["filename"])

        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch(
            "huggingface_hub.hf_hub_download", side_effect=fake_download
        ):
            restored_count = syncer_reader.restore()

        self.assertEqual(restored_count, 3)
        self.assertEqual(target.get("doc-snapshot")["raw_text"], "from snapshot")
        self.assertEqual(target.get("doc-legacy")["raw_text"], "from legacy delta")
        self.assertEqual(target.get("doc-sharded")["raw_text"], "from sharded delta")

        source.close(); source_dir.cleanup()
        target.close(); target_dir.cleanup()

    def test_sharded_manifest_validation_rejects_traversal_and_malformed_deltas(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        syncer = SnapshotSync(store)
        repo_files = {
            "funes-snapshot.jsonl.gz.enc",
            "deltas/ab/funes-delta-ab12.jsonl.gz.enc",
        }
        # Valid sharded delta passes
        valid_manifest = {
            "version": 1,
            "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": ["deltas/ab/funes-delta-ab12.jsonl.gz.enc"],
            "controls": [],
        }
        validated = syncer._validate_restore_manifest(valid_manifest, repo_files)
        self.assertEqual(validated["deltas"], ["deltas/ab/funes-delta-ab12.jsonl.gz.enc"])

        # Path traversal fails
        with self.assertRaises(ValueError):
            syncer._validate_restore_manifest(
                {**valid_manifest, "deltas": ["../deltas/ab/funes-delta-ab12.jsonl.gz.enc"]},
                repo_files,
            )
        # Slashes traversal
        with self.assertRaises(ValueError):
            syncer._validate_restore_manifest(
                {**valid_manifest, "deltas": ["deltas/../funes-delta-ab12.jsonl.gz.enc"]},
                repo_files,
            )
        # Bad prefix under deltas/
        with self.assertRaises(ValueError):
            syncer._validate_restore_manifest(
                {**valid_manifest, "deltas": ["deltas/ab/wrong-prefix.jsonl.gz.enc"]},
                repo_files,
            )
        store.close()


    def test_compaction_retains_concurrent_sharded_deltas(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        syncer = SnapshotSync(store)
        syncer.covered_revision = "head-1"
        snapshot_name = "funes-snapshot.jsonl.gz.enc"
        base_delta = "funes-delta-base.jsonl.gz.enc"
        sharded_concurrent_delta = "deltas/ab/funes-delta-ab1234567890123456789012.jsonl.gz.enc"
        base_manifest = {
            "version": 1,
            "snapshot": snapshot_name,
            "deltas": [base_delta],
            "controls": [],
        }
        concurrent_manifest = {
            **base_manifest,
            "deltas": [base_delta, sharded_concurrent_delta],
        }
        base_path = Path(self.tmp.name) / "compact-shard-head-1.json"
        concurrent_path = Path(self.tmp.name) / "compact-shard-head-2.json"
        base_path.write_text(json.dumps(base_manifest), encoding="utf-8")
        concurrent_path.write_text(json.dumps(concurrent_manifest), encoding="utf-8")
        api = mock.Mock()
        api.repo_info.side_effect = [mock.Mock(sha="head-1"), mock.Mock(sha="head-2")]
        api.list_repo_tree.side_effect = [
            [
                mock.Mock(path=syncer.manifest_filename, blob_id="manifest-1"),
                mock.Mock(path=snapshot_name, blob_id="snapshot-1"),
                mock.Mock(path=base_delta, blob_id="delta-base"),
            ],
            [
                mock.Mock(path=syncer.manifest_filename, blob_id="manifest-2"),
                mock.Mock(path=snapshot_name, blob_id="snapshot-1"),
                mock.Mock(path=base_delta, blob_id="delta-base"),
                mock.Mock(path=sharded_concurrent_delta, blob_id="delta-sharded"),
            ],
        ]
        api.create_commit.side_effect = [
            RuntimeError("stale parent"),
            mock.Mock(oid="head-3"),
        ]
        with mock.patch("huggingface_hub.HfApi", return_value=api), mock.patch(
            "huggingface_hub.hf_hub_download",
            side_effect=[str(base_path), str(concurrent_path)],
        ):
            result = syncer.upload()

        self.assertTrue(result["durable"])
        operations = {
            operation.path_in_repo: operation
            for operation in api.create_commit.call_args_list[-1].kwargs["operations"]
        }
        manifest = json.loads(operations[syncer.manifest_filename].path_or_fileobj)
        self.assertEqual(manifest["snapshot"], snapshot_name)
        self.assertEqual(manifest["deltas"], [sharded_concurrent_delta])
        store.close()

    def test_unmanifested_existing_legacy_delta_is_durable_without_reupload(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "legacy-nomani", "raw_text": "text"}])
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        docs = store.get_many(["legacy-nomani"])
        syncer = SnapshotSync(store)
        durable_docs = [*docs, store.native_index_state_record()]
        digest = hashlib.sha256(
            json.dumps(
                durable_docs,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:24]
        legacy_delta = f"funes-delta-{digest}.jsonl.gz.enc"
        sharded_delta = syncer.delta_target(digest)

        api = mock.Mock()
        api.repo_info.return_value = mock.Mock(sha="head-1")
        api.list_repo_tree.return_value = []
        # Sharded target does not exist, but legacy target exists
        def fake_file_exists(repo_id, filename, repo_type, revision, token):
            return filename == legacy_delta

        api.file_exists.side_effect = fake_file_exists
        module = mock.Mock(HfApi=mock.Mock(return_value=api))
        with mock.patch.dict("sys.modules", {"huggingface_hub": module}):
            result = syncer.upload(docs)

        self.assertTrue(result["durable"])
        self.assertTrue(result["already_uploaded"])
        api.upload_file.assert_not_called()
        self.assertFalse(list(Path(self.tmp.name).glob("funes-delta-*.jsonl.gz")))
        store.close()


    def test_arbitrary_delta_dir_prefix_sharding_and_manifest_validation(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        with mock.patch.dict(
            os.environ,
            {
                "FUNES_DELTA_DIR": "archive/nested/deltas",
                "FUNES_DELTA_PREFIX": "funes-delta-",
            },
        ):
            syncer = SnapshotSync(store)
            self.assertEqual(syncer.delta_dir, "archive/nested/deltas")
            digest = "ab1234567890abcdef123456"
            target = syncer.delta_target(digest)
            self.assertEqual(
                target,
                "archive/nested/deltas/ab/funes-delta-ab1234567890abcdef123456.jsonl.gz.enc",
            )
            # Accepts configured multi-level directory
            self.assertTrue(syncer._is_delta_name(target))
            # Accepts canonical deltas/ fallback
            self.assertTrue(
                syncer._is_delta_name(
                    "deltas/ab/funes-delta-ab1234567890abcdef123456.jsonl.gz.enc"
                )
            )
            # Accepts legacy root delta fallback
            self.assertTrue(
                syncer._is_delta_name(
                    "funes-delta-ab1234567890abcdef123456.jsonl.gz.enc"
                )
            )
            # Fails on path traversal
            self.assertFalse(
                syncer._is_delta_name(
                    "archive/nested/deltas/../other/ab/funes-delta-ab.jsonl.gz.enc"
                )
            )
            # Fails on unmatched directory
            self.assertFalse(
                syncer._is_delta_name(
                    "other/nested/deltas/ab/funes-delta-ab.jsonl.gz.enc"
                )
            )
            # Fails on insufficient segments
            self.assertFalse(
                syncer._is_delta_name(
                    "archive/nested/deltas/funes-delta-ab.jsonl.gz.enc"
                )
            )

            # Manifest validation accepts multi-level sharded delta alongside legacy root
            manifest = {
                "version": 1,
                "snapshot": "funes-snapshot.jsonl.gz.enc",
                "deltas": [
                    "funes-delta-legacy.jsonl.gz.enc",
                    target,
                ],
                "controls": [],
            }
            repo_files = {
                syncer.manifest_filename,
                manifest["snapshot"],
                *manifest["deltas"],
            }
            validated = syncer._validate_restore_manifest(manifest, repo_files)
            self.assertEqual(validated["deltas"], manifest["deltas"])

            # _repo_files fallback orders legacy root deltas before multi-level sharded deltas
            syncer.repo = "owner/private"
            syncer.token = "test-token"
            api = mock.Mock()
            api.repo_info.return_value = mock.Mock(sha="head-1")
            api.list_repo_tree.return_value = [
                mock.Mock(path=manifest["snapshot"]),
                mock.Mock(path=target),
                mock.Mock(path="funes-delta-legacy.jsonl.gz.enc"),
            ]
            with mock.patch("huggingface_hub.HfApi", return_value=api):
                ordered = syncer._repo_files()
            self.assertEqual(
                ordered,
                [
                    "funes-snapshot.jsonl.gz.enc",
                    "funes-delta-legacy.jsonl.gz.enc",
                    target,
                ],
            )
        store.close()

if __name__ == "__main__":
    unittest.main()
