"""Lossless migration tests; live tests use a NEW unique schema, never production.

FUNES_TEST_POSTGRES_DSN must identify a disposable PostgreSQL test database.
Each test creates and drops only its own random ``migration_test_*`` schema.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from pathlib import Path
from unittest import mock

import pytest

from scripts.migrate_source_postgres import (
    LEDGER, TABLES, Migration, MigrationError, Snapshot, digest_rows, main, quoted,
)
from service.postgres import PostgresNotReady, PostgresStore, encode_pg_text
from service.server import Store


@pytest.fixture
def source_path(tmp_path):
    store = Store(str(tmp_path / "source"))
    store.ingest([
        {"source_identity": f"source-{i}", "source_version": f"version-{i}",
         "raw_text": f"raw {i} 中文 e\u0301", "retrieval_text": f"retrieval {i}",
         "source_agent": "codex", "source_path": f"/source/{i}",
         "content_type": "message", "metadata": {"nested": {"i": i}}}
        for i in range(1, 7)
    ])
    with store.conn:
        store.conn.execute("DELETE FROM memories WHERE id=2")
        store.conn.execute(
            "UPDATE memories SET raw_text=?,retrieval_text=?,metadata_json=?,"
            "source_metadata_clock_json=?,search_identifiers=?,native_index_pending=0,"
            "retrieval_generation=13,native_generation=11,embedding_generation=7,"
            "native_index_status='indexed',native_index_profile='profile',"
            "native_index_memory='memory' WHERE id=1",
            ("raw\0bytes\r\n\t\ue0000\ue000e 中文 e\u0301",
             "retrieval\0尾\ue000", '{ "k": "\\u0000", "n": 1 }',
             '{ "source_path": "clock 1" }', "NUL\0encoded\ue000"),
        )
        store.conn.execute("UPDATE memories SET raw_text=?,retrieval_text=? WHERE id=4",
                           ("equal\0\ue0000\r\n中文", "equal\0\ue0000\r\n中文"))
        store.conn.execute("UPDATE memories SET raw_text='nonempty',retrieval_text='' WHERE id=5")
        store.conn.execute("UPDATE memories SET raw_text='',retrieval_text='' WHERE id=6")
        # Include NUL + escape-marker collisions in a primary key as well.
        store.conn.execute("UPDATE memories SET source_identity=? WHERE id=3", ("key\0\ue0000",))
        for query in ("z", "query\0", "\ue000", "\ue0000", "\ue000e", "中文"):
            store.conn.execute(
                "INSERT INTO translation_cache(query,rewritten,created_at,translation_hash,translation_version,translation_status) "
                "VALUES(?,?,?,?,?,?)", (query, "answer\0\ue000\r\n", "cache-created", None, "v1", "cached"),
            )
        store.conn.execute(
            "INSERT INTO reindex_controls(generation,scope,created_at,row_cursor,applied_at) VALUES(8,'all','created',5,NULL)"
        )
        # Nonzero/opaque checkpoints prove the importer does not rebuild them.
        store.conn.execute(
            "UPDATE sync_state SET last_sync='2026-09-22T00:00:00Z',last_error=?,"
            "snapshot_path='snapshot.enc',restored_at='restored',native_checkpoint_profile='profile',"
            "native_checkpoint_memory='memory',native_index_revision=97,native_eligible_count=23,"
            "native_indexed_count=17,native_held_count=3,native_invalid_count=1,"
            "native_optimize_revision=89,native_optimize_provider='voyage',native_optimize_model='voyage-4-lite',"
            "native_optimize_dimensions=1024,native_optimize_schema_version=4,native_optimize_layout_version=3,"
            "native_optimize_memory='hf://fixture',native_optimize_fingerprint='opaque-fingerprint',"
            "native_optimize_index_fingerprint='opaque-index-fingerprint',native_optimize_status='optimized',"
            "native_optimized_at='opaque-time',fts_ready=0 WHERE id=1", ("error\0\ue000",),
        )
        # The earlier native-status UPDATE invokes SQLite's pending trigger.
        # Set this source field separately to prove COPY does not recalculate it.
        store.conn.execute("UPDATE memories SET native_index_pending=0 WHERE id=1")
    path = store.db_path
    store.close()
    assert not Path(str(path) + "-wal").exists()
    return path


@pytest.fixture
def pg_dsn():
    base = os.environ.get("FUNES_TEST_POSTGRES_DSN")
    if not base:
        pytest.skip("set FUNES_TEST_POSTGRES_DSN to a disposable PostgreSQL database")
    psycopg = pytest.importorskip("psycopg")
    from psycopg.conninfo import make_conninfo
    schema = "migration_test_" + uuid.uuid4().hex
    with psycopg.connect(base, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    dsn = make_conninfo(base, options=f"-c search_path={schema}")
    try:
        yield dsn
    finally:
        with psycopg.connect(base, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def source_report(source):
    return {table: digest_rows(table, source.columns[table], source.rows(table)) for table in TABLES}


def test_typed_digest_preserves_raw_bytes_and_null_boundaries():
    def digest(values):
        return digest_rows("test", ["a", "b"], [values])["sha256"]
    variants = [(None, ""), ("", None), ("ab", "c"), ("a", "bc"),
                ("x\0y", ""), ("xy", ""), (1, ""), ("1", ""),
                ("é", ""), ("e\u0301", "")]
    assert len({digest(values) for values in variants}) == len(variants)


def test_source_is_readonly_and_full_file_identity_is_pinned(source_path):
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    with Snapshot(source_path, before) as source:
        assert source.identity["sha256"] == before
        assert source.counts == {"memories": 5, "translation_cache": 6, "reindex_controls": 1, "sync_state": 1}
        report = source_report(source)
        assert report["memories"]["nul_fields"]["raw_text"] == 2
        assert report["memories"]["nul_fields"]["retrieval_text"] == 2
        with pytest.raises(sqlite3.OperationalError):
            source.conn.execute("DELETE FROM memories")
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before
    with pytest.raises(MigrationError, match="expected snapshot"):
        Snapshot(source_path, "0" * 64)


def test_live_wal_and_changed_source_are_refused(source_path):
    wal = Path(str(source_path) + "-wal")
    wal.write_bytes(b"not-an-immutable-backup")
    with pytest.raises(MigrationError, match="WAL/journal"):
        Snapshot(source_path)
    wal.unlink()
    with Snapshot(source_path) as source:
        with source_path.open("ab") as handle:
            handle.write(b"changed")
        with pytest.raises(MigrationError, match="snapshot changed"):
            source.assert_unchanged()


def test_dry_run_has_no_postgres_writes_and_no_raw_payload(source_path, capsys):
    with mock.patch("scripts.migrate_source_postgres.Migration", side_effect=AssertionError("must not connect")):
        assert main(["--source", str(source_path), "--dry-run"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out.splitlines()[-1])
    assert output["writes"] is False
    assert output["tables"]["memories"]["rows"] == 5
    assert "raw\\u0000bytes" not in captured.out


def test_cli_requires_deferred_ready_and_redacts_unknown_errors(source_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--source", str(source_path), "--initialize"])
    assert error.value.code == 2
    capsys.readouterr()
    secret = "postgres://user:secret@host/private-source-payload"
    with mock.patch("scripts.migrate_source_postgres.Migration", side_effect=RuntimeError(secret)):
        assert main(["--source", str(source_path), "--resume", "--defer-ready"]) == 1
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "readiness not confirmed" in captured.err
    assert "inspect the durable marker" in captured.err


def test_interrupted_copy_resumes_atomically_and_preserves_every_field(source_path, pg_dsn, tmp_path):
    with Snapshot(source_path) as source:
        expected = source_report(source)
        with Migration(source, pg_dsn) as migration:
            migration.initialize()
            assert migration.pg.execute("SELECT count(*) FROM pg_index WHERE indrelid='memories'::regclass "
                                        "AND NOT indisunique").fetchone() == (0,)
            assert migration.pg.execute("SELECT count(*) FROM pg_trigger WHERE tgrelid='memories'::regclass "
                                        "AND NOT tgisinternal AND tgenabled='D'").fetchone() == (2,)
            result = migration.copy(batch_rows=1, max_batches=1)
            assert result["phase"] == "paused"
            assert migration.pg.execute("SELECT count(*) FROM memories").fetchone() == (1,)
            assert migration.pg.execute("SELECT last_value,is_called FROM memories_id_seq").fetchone() == (1, False)
        with pytest.raises(PostgresNotReady, match="not finalized"):
            PostgresStore(str(tmp_path / "guard"), pg_dsn)
        with Migration(source, pg_dsn) as migration:
            def interrupt(*_):
                raise KeyboardInterrupt()
            with pytest.raises(KeyboardInterrupt):
                migration.copy(batch_rows=1, before_commit=interrupt)
            assert migration.pg.execute("SELECT count(*) FROM memories").fetchone() == (1,)
            checkpoint = migration.pg.execute(f"SELECT progress FROM {LEDGER} WHERE id=1").fetchone()[0]
            assert checkpoint["memories"]["rows"] == 1
            assert migration.pg.execute("SELECT count(*) FROM pg_trigger WHERE tgrelid='memories'::regclass "
                                        "AND NOT tgisinternal AND tgenabled='D'").fetchone() == (2,)
        with Migration(source, pg_dsn) as migration:
            result = migration.copy(batch_rows=2, batch_bytes=100)
            assert result["phase"] == "verified" and result["ready"] is False
            assert result["tables"] == expected
            assert migration.pg.execute("SELECT id,retrieval_text FROM memories WHERE id IN (4,5,6) ORDER BY id").fetchall() == [(4, None), (5, ""), (6, None)]
            assert migration.pg.execute("SELECT document_count FROM funes_schema_state").fetchone() == (5,)
            assert migration.pg.execute("SELECT native_index_revision,native_optimize_revision,native_eligible_count "
                                        "FROM sync_state").fetchone() == (97, 89, 23)
            assert migration.pg.execute("SELECT native_index_pending FROM memories WHERE id=1").fetchone() == (0,)
            assert migration.pg.execute("SELECT last_value,is_called FROM memories_id_seq").fetchone() == (6, True)
            with pytest.raises(MigrationError, match="tail-confirmed"):
                migration.finalize()
            result = migration.finalize(tail_confirmed=True)
            assert result["before_finalize_tables"] == expected
            assert result["tables"]["memories"] == expected["memories"]
            assert result["tables"]["sync_state"] != expected["sync_state"]
            assert result["rebuilt_fields"]["memories"] == ["native_index_pending"]
            assert migration.pg.execute("SELECT fts_ready,native_index_revision,native_optimize_revision,native_eligible_count "
                                        "FROM sync_state").fetchone() == (0, 97, 89, 5)
        store = PostgresStore(str(tmp_path / "ready"), pg_dsn)
        try:
            assert store.count() == 5
            assert store.get(1)["raw_text"] == "raw\0bytes\r\n\t\ue0000\ue000e 中文 e\u0301"
            assert store.get("key\0\ue0000")["id"] == 3
            store.ingest([{"source_identity": "after-migration", "raw_text": "new"}])
            assert store.get("after-migration")["id"] == 7
            assert store.count() == 6
        finally:
            store.close()


def test_corrupted_committed_prefix_refuses_resume(source_path, pg_dsn):
    with Snapshot(source_path) as source, Migration(source, pg_dsn) as migration:
        migration.initialize()
        migration.copy(batch_rows=1, max_batches=1)
        migration.pg.execute("UPDATE memories SET raw_text='corrupted' WHERE id=1")
        with pytest.raises(MigrationError, match="committed target differs"):
            migration.copy()
        assert migration.pg.execute("SELECT migration_ready FROM funes_schema_state").fetchone() == (False,)


def test_full_readback_detects_corruption_after_verified_import(source_path, pg_dsn):
    with Snapshot(source_path) as source, Migration(source, pg_dsn) as migration:
        migration.initialize()
        migration.copy()
        migration.pg.execute("UPDATE translation_cache SET rewritten='same-row-count-corruption'")
        with pytest.raises(MigrationError, match="digest/count mismatch: translation_cache"):
            migration.finalize(tail_confirmed=True)
        assert migration.pg.execute(f"SELECT phase FROM {LEDGER}").fetchone() == ("verified",)
        assert migration.pg.execute("SELECT migration_ready FROM funes_schema_state").fetchone() == (False,)


def test_different_source_and_schema_are_refused(source_path, pg_dsn, tmp_path):
    with Snapshot(source_path) as source, Migration(source, pg_dsn) as migration:
        migration.initialize()
        migration.copy(batch_rows=1, max_batches=1)
    other = tmp_path / "other.sqlite3"
    shutil.copyfile(source_path, other)
    with sqlite3.connect(other) as conn:
        conn.execute("UPDATE translation_cache SET rewritten='different immutable snapshot'")
    conn.close()
    with Snapshot(other) as source, Migration(source, pg_dsn) as migration:
        with pytest.raises(MigrationError, match="source identity/version differs"):
            migration.copy()
    with Snapshot(source_path) as source, Migration(source, pg_dsn) as migration:
        migration.pg.execute("ALTER TABLE reindex_controls ADD COLUMN unexpected TEXT")
        with pytest.raises(MigrationError, match="column schema differs: reindex_controls"):
            migration.copy()
        assert migration.pg.execute("SELECT migration_ready FROM funes_schema_state").fetchone() == (False,)


def test_nonempty_unowned_target_is_not_modified(source_path, pg_dsn, tmp_path):
    store = PostgresStore(str(tmp_path / "other-target"), pg_dsn, initialize=True)
    store.ingest([{"source_identity": "unrelated", "raw_text": "must retain"}])
    store.close()
    with Snapshot(source_path) as source, Migration(source, pg_dsn) as migration:
        with pytest.raises(MigrationError, match="already initialized/finalized"):
            migration.initialize()
        assert migration.pg.execute("SELECT raw_text FROM memories").fetchone() == ("must retain",)
        assert not migration._exists(LEDGER)


def test_initialization_crash_recovers_only_pristine_bootstrap(source_path, pg_dsn, tmp_path):
    store = PostgresStore(str(tmp_path / "bootstrap"), pg_dsn, initialize=True)
    store.prepare_bulk_migration()
    store.close()  # Simulates dying after bulk preparation but before ledger creation.
    with Snapshot(source_path) as source, Migration(source, pg_dsn) as migration:
        migration.initialize()
        assert migration.copy()["ready"] is False


def test_modified_bootstrap_checkpoint_is_not_overwritten(source_path, pg_dsn, tmp_path):
    store = PostgresStore(str(tmp_path / "bootstrap"), pg_dsn, initialize=True)
    store.conn.execute("UPDATE sync_state SET native_optimize_fingerprint='already-owned' WHERE id=1")
    store.close()
    with Snapshot(source_path) as source, Migration(source, pg_dsn) as migration:
        with pytest.raises(MigrationError, match="existing sync_state data"):
            migration.initialize()
        assert migration.pg.execute("SELECT native_optimize_fingerprint FROM sync_state").fetchone() == ("already-owned",)


def test_final_tail_requires_independent_exact_final_snapshot(source_path, pg_dsn, tmp_path):
    tail_dir = tmp_path / "tail"
    tail_dir.mkdir()
    shutil.copyfile(source_path, tail_dir / "funes.sqlite3")
    tail = Store(str(tail_dir))
    tail.ingest([{"source_identity": "incremental-only", "raw_text": "tail\0raw", "source_version": "tail-v1"}])
    tail_path = tail.db_path
    tail.close()
    with Snapshot(source_path) as source, Migration(source, pg_dsn) as migration:
        migration.initialize()
        migration.copy()
        with Snapshot(tail_path) as final:
            # Simulate the caller's bounded tail mechanism, not a script replay.
            # SQLite FTS startup rebuild updates identifiers on our deliberately
            # mutated fixture rows 1/4/5/6; include those and the new row 7.
            with migration.pg.transaction():
                columns = final.columns["memories"]
                assignments = ",".join(f"{quoted(name)}=EXCLUDED.{quoted(name)}" for name in columns if name != "id")
                for values in final.conn.execute("SELECT " + ",".join(map(quoted, columns))
                                                 + " FROM memories WHERE id IN (1,4,5,6,7)"):
                    migration.pg.execute("INSERT INTO memories (" + ",".join(map(quoted, columns)) + ") VALUES ("
                                         + ",".join(["%s"] * len(columns)) + ") ON CONFLICT(id) DO UPDATE SET " + assignments,
                                         tuple(encode_pg_text(value) if isinstance(value, str) else value for value in values))
                columns = final.columns["sync_state"]
                values = final.conn.execute("SELECT " + ",".join(map(quoted, columns)) + " FROM sync_state").fetchone()
                migration.pg.execute("UPDATE sync_state SET " + ",".join(f"{quoted(name)}=%s" for name in columns),
                                     tuple(encode_pg_text(value) if isinstance(value, str) else value for value in values))
            with pytest.raises(MigrationError, match="digest/count mismatch: memories"):
                migration.finalize(tail_confirmed=True)
            result = migration.finalize(tail_confirmed=True, tail_source=final)
            assert result["ready"] is True
            assert result["source"] == final.identity
            assert result["before_finalize_tables"] == source_report(final)
            assert result["tables"]["memories"]["rows"] == 6


@pytest.mark.parametrize("damage,expected_error", [
    ("UPDATE memories SET raw_text='wrong protected bytes' WHERE id=1", "digest/count mismatch: memories"),
    ("UPDATE sync_state SET native_index_revision=native_index_revision+1", "digest/count mismatch: sync_state"),
    ("UPDATE memories SET native_index_pending=1 WHERE id=1", "digest/count mismatch: memories"),
    ("DROP INDEX memories_canonical_pending_idx", "indexes are missing or invalid"),
])
def test_post_finalize_validation_failure_rolls_back_ready_and_source(source_path, pg_dsn, damage, expected_error):
    with Snapshot(source_path) as source, Migration(source, pg_dsn) as migration:
        migration.initialize()
        migration.copy()
        original_finalize = PostgresStore.finalize_migration

        def damaged_finalize(store, **kwargs):
            result = original_finalize(store, **kwargs)
            store.conn.execute(damage)
            return result

        with mock.patch.object(PostgresStore, "finalize_migration", damaged_finalize):
            with pytest.raises(MigrationError, match=expected_error):
                migration.finalize(tail_confirmed=True)
        assert migration.pg.execute("SELECT migration_ready,migration_source,document_count "
                                    "FROM funes_schema_state").fetchone() == (False, None, 5)
        assert migration.pg.execute(f"SELECT phase,final_verification FROM {LEDGER}").fetchone() == ("verified", None)
        with migration.pg.transaction():
            assert migration._verify_tables(source)["tables"] == source_report(source)
        migration.check_schema(require_runtime=True)
        # A failed final pass is not a terminal migration or a poisoned connection.
        assert migration.finalize(tail_confirmed=True)["ready"] is True


def test_schema_lock_blocks_concurrent_migrators(source_path, pg_dsn):
    with Snapshot(source_path) as source, Migration(source, pg_dsn):
        with pytest.raises(MigrationError, match="another source migration"):
            Migration(source, pg_dsn)


def test_empty_source_tables_finalize_without_consuming_identity(pg_dsn, tmp_path):
    store = Store(str(tmp_path / "empty-source"))
    path = store.db_path
    store.close()
    with Snapshot(path) as source, Migration(source, pg_dsn) as migration:
        migration.initialize()
        result = migration.copy(batch_rows=1)
        assert result["tables"]["memories"]["rows"] == 0
        assert result["tables"]["reindex_controls"]["rows"] == 0
        assert result["tables"]["translation_cache"]["rows"] == 0
        migration.finalize(tail_confirmed=True)
        assert migration.pg.execute("SELECT last_value,is_called FROM memories_id_seq").fetchone() == (1, False)
        assert migration.pg.execute("SELECT document_count FROM funes_schema_state").fetchone() == (0,)
