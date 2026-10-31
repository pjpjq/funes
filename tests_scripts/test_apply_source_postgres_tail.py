"""Tail application tests; live cases require a disposable PostgreSQL DSN.

Reuse the migration fixture's per-test random schema and cleanup. Never point
FUNES_TEST_POSTGRES_DSN at production. No test finalizes a source store.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from unittest import mock

import pytest

from scripts.apply_source_postgres_tail import (
    TAIL_LEDGER,
    apply_tail,
    check_snapshots,
    iter_changes,
    main,
    summarize_tail,
)
from scripts.migrate_source_postgres import (
    KEYS,
    LEDGER,
    TABLES,
    Migration,
    MigrationError,
    Snapshot,
    quoted,
)
from service.postgres import encode_pg_text
from service.tests.test_postgres_migration import (
    pg_dsn as pg_dsn,  # noqa: PLC0414 -- pytest fixture re-export
)
from service.tests.test_postgres_migration import (
    source_path as source_path,  # noqa: PLC0414 -- pytest fixture re-export
)
from service.tests.test_postgres_migration import source_report


@pytest.fixture
def baseline_path(source_path):
    with sqlite3.connect(source_path) as conn:
        conn.execute("INSERT INTO reindex_controls(generation,scope,created_at) VALUES(9,'all','old')")
    conn.close()
    return source_path


def clone_snapshot(source, destination):
    shutil.copyfile(source, destination)
    conn = sqlite3.connect(destination)
    # A fixture deliberately sets every opaque source field. Do not let SQLite
    # derived-field triggers recalculate what this test needs to preserve.
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        conn.execute(f"DROP TRIGGER {quoted(name)}")
    conn.commit()
    return conn


def clone_memory(conn, source_id, new_id, identity):
    columns = [row[1] for row in conn.execute("PRAGMA table_info(memories)")]
    values = list(conn.execute("SELECT " + ",".join(map(quoted, columns))
                               + " FROM memories WHERE id=?", (source_id,)).fetchone())
    values[columns.index("id")] = new_id
    values[columns.index("source_identity")] = identity
    conn.execute("INSERT INTO memories (" + ",".join(map(quoted, columns)) + ") VALUES ("
                 + ",".join("?" for _ in values) + ")", values)


def rewrite_every_nonkey_field(conn, table, key_value):
    schema = conn.execute(f"PRAGMA table_info({quoted(table)})").fetchall()
    updates, values = [], []
    for _, name, kind, notnull, _, _ in schema:
        if name == KEYS[table]:
            continue
        value = 73 if kind == "INTEGER" else f"tail:{name}\0\ue0000\ue000e 中文 e\u0301\r\n"
        if name in {"raw_text", "retrieval_text"}:
            value = "tail identical raw/shadow\0\ue0000 中文"
        if not notnull and name in {"last_sync", "native_index_error", "translation_hash"}:
            value = None
        updates.append(quoted(name) + "=?")
        values.append(value)
    conn.execute(f"UPDATE {quoted(table)} SET {','.join(updates)} WHERE {quoted(KEYS[table])}=?",
                 [*values, key_value])


@pytest.fixture
def desired_path(baseline_path, tmp_path):
    path = tmp_path / "final.sqlite3"
    conn = clone_snapshot(baseline_path, path)
    with conn:
        rewrite_every_nonkey_field(conn, "memories", 1)
        conn.execute("DELETE FROM memories WHERE id=3")
        conn.execute("UPDATE memories SET retrieval_text='new distinct shadow' WHERE id=4")
        clone_memory(conn, 5, 2, "inserted-at-old-hole\0\ue0000")
        clone_memory(conn, 6, 19, "inserted-at-high-id\0\ue000e")
        rewrite_every_nonkey_field(conn, "translation_cache", "query\0")
        conn.execute("DELETE FROM translation_cache WHERE query='z'")
        conn.execute("INSERT INTO translation_cache(query,rewritten,created_at) VALUES(?,?,?)",
                     ("new\0\ue0000", "answer\0\ue000e", "new-time"))
        rewrite_every_nonkey_field(conn, "reindex_controls", 8)
        conn.execute("DELETE FROM reindex_controls WHERE generation=9")
        conn.execute("INSERT INTO reindex_controls(generation,scope,created_at) VALUES(13,'tail','new')")
        rewrite_every_nonkey_field(conn, "sync_state", 1)
    conn.close()
    return path


def ready_baseline(baseline, dsn):
    migration = Migration(baseline, dsn)
    try:
        migration.initialize()
        migration.copy()
    except BaseException:
        migration.close()
        raise
    return migration


def target_report(migration, source):
    with migration.pg.transaction():
        return {table: migration._pg_digest(source, table) for table in TABLES}


def install_desired_row(pg, final, table, key):
    columns = final.columns[table]
    row = final.conn.execute("SELECT " + ",".join(map(quoted, columns))
                             + f" FROM {quoted(table)} WHERE {quoted(KEYS[table])}=?", (key,)).fetchone()
    values = [encode_pg_text(value) if isinstance(value, str) else value for value in row]
    assignments = ",".join(f"{quoted(name)}=EXCLUDED.{quoted(name)}"
                           for name in columns if name != KEYS[table])
    pg.execute(f"INSERT INTO {quoted(table)} ({','.join(map(quoted, columns))}) VALUES "
               f"({','.join('%s' for _ in columns)}) ON CONFLICT ({quoted(KEYS[table])}) "
               f"DO UPDATE SET {assignments}", values)


def test_offline_diff_covers_inserts_old_id_updates_deletes_and_all_tables(baseline_path, desired_path):
    with Snapshot(baseline_path) as baseline, Snapshot(desired_path) as desired:
        result = summarize_tail(baseline, desired)
        assert result["changes"] == {
            "memories": {"inserted": 2, "updated": 2, "deleted": 1},
            "translation_cache": {"inserted": 1, "updated": 1, "deleted": 1},
            "reindex_controls": {"inserted": 1, "updated": 1, "deleted": 1},
            "sync_state": {"inserted": 0, "updated": 1, "deleted": 0},
        }
        assert result["writes"] is result["final_write_boundary"] is False
        changes = list(iter_changes(baseline, desired, "memories"))
        assert [(old[0] if old else None, new[0] if new else None) for old, new in changes] == [
            (1, 1), (None, 2), (3, None), (4, 4), (None, 19)]


def test_cache_diff_uses_encoded_key_byte_order(baseline_path, desired_path):
    with Snapshot(baseline_path) as baseline, Snapshot(desired_path) as desired:
        changes = list(iter_changes(baseline, desired, "translation_cache"))
        keys = [(new or old)[0] for old, new in changes]
        assert keys == sorted(keys, key=lambda value: encode_pg_text(value).encode())
        assert set(keys) == {"z", "query\0", "new\0\ue0000"}


def test_dry_run_never_connects_or_leaks_payload(baseline_path, desired_path, capsys):
    with mock.patch("scripts.apply_source_postgres_tail.Migration", side_effect=AssertionError("must not connect")):
        assert main(["--source", str(baseline_path), "--tail-source", str(desired_path), "--dry-run"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out.splitlines()[-1])
    assert result["phase"] == "tail_dry_run"
    assert "tail identical" not in captured.out + captured.err


def test_cli_requires_deferred_ready_and_redacts_driver_failures(baseline_path, desired_path, capsys):
    args = ["--source", str(baseline_path), "--tail-source", str(desired_path), "--apply"]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    capsys.readouterr()
    secret = "postgres://user:password@server/private-raw-payload"
    with mock.patch("scripts.apply_source_postgres_tail.Migration", side_effect=RuntimeError(secret)):
        assert main([*args, "--defer-ready"]) == 1
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "readiness not confirmed" in captured.err


def test_schema_difference_and_wrong_sha_refused_before_connect(baseline_path, desired_path, capsys):
    with sqlite3.connect(desired_path) as conn:
        conn.execute("ALTER TABLE reindex_controls ADD COLUMN unexpected TEXT")
    conn.close()
    with Snapshot(baseline_path) as baseline, Snapshot(desired_path) as desired, \
            pytest.raises(MigrationError, match="schema differs"):
        check_snapshots(baseline, desired)
    with mock.patch("scripts.apply_source_postgres_tail.Migration", side_effect=AssertionError("must not connect")):
        assert main(["--source", str(baseline_path), "--tail-source", str(desired_path),
                     "--apply", "--defer-ready", "--tail-expected-sha256", "0" * 64]) == 1
    assert "expected snapshot" in capsys.readouterr().err


def test_apply_is_byte_exact_atomic_idempotent_and_keeps_baseline_ledger(baseline_path, desired_path, pg_dsn):
    with Snapshot(baseline_path) as baseline, Snapshot(desired_path) as desired, \
            ready_baseline(baseline, pg_dsn) as migration:
        ledger_before = migration.pg.execute(f"SELECT * FROM {LEDGER}").fetchone()
        sequence_before = migration.pg.execute("SELECT last_value,is_called FROM memories_id_seq").fetchone()
        result = apply_tail(migration, desired)
        assert result["ready"] is result["final_write_boundary"] is False
        assert result["tables"] == source_report(desired) == target_report(migration, desired)
        assert migration.pg.execute(f"SELECT * FROM {LEDGER}").fetchone() == ledger_before
        assert migration.pg.execute("SELECT migration_ready,migration_source,document_count "
                                    "FROM funes_schema_state").fetchone() == (False, None, 6)
        assert migration.pg.execute("SELECT last_value,is_called FROM memories_id_seq").fetchone() == sequence_before
        assert migration.pg.execute("SELECT id,retrieval_text FROM memories WHERE id IN(1,2,4,5,6,19) "
                                    "ORDER BY id").fetchall() == [(1, None), (2, ""), (4, "new distinct shadow"),
                                                               (5, ""), (6, None), (19, None)]
        assert migration.pg.execute("SELECT count(*) FROM pg_trigger WHERE tgrelid='memories'::regclass "
                                    "AND NOT tgisinternal AND tgenabled='D'").fetchone() == (2,)
        receipt = migration.pg.execute(f"SELECT baseline_identity,source_identity,verification FROM {TAIL_LEDGER}").fetchone()
        assert receipt == (baseline.identity, desired.identity,
                           {"source": desired.identity, "tables": source_report(desired)})
        second = apply_tail(migration, desired)
        assert second["changes"] == result["changes"]
        assert second["applied"] == {table: {"removed": 0, "written": 0} for table in TABLES}
        assert migration.pg.execute(f"SELECT * FROM {LEDGER}").fetchone() == ledger_before


def test_accepts_mixed_already_desired_whole_rows_not_only_pristine_baseline(baseline_path, desired_path, pg_dsn):
    with Snapshot(baseline_path) as baseline, Snapshot(desired_path) as desired, \
            ready_baseline(baseline, pg_dsn) as migration:
        install_desired_row(migration.pg, desired, "memories", 1)
        install_desired_row(migration.pg, desired, "memories", 19)
        migration.pg.execute("DELETE FROM memories WHERE id=3")
        install_desired_row(migration.pg, desired, "translation_cache", "query\0")
        result = apply_tail(migration, desired)
        assert result["tables"] == source_report(desired)
        assert result["applied"]["memories"] == {"removed": 1, "written": 2}


@pytest.mark.parametrize("damage,table", [
    ("UPDATE memories SET raw_text='unknown-third-value' WHERE id=1", "memories"),
    ("DELETE FROM memories WHERE id=1", "memories"),
    ("UPDATE memories SET metadata_json='unknown-delete-value' WHERE id=3", "memories"),
    ("UPDATE translation_cache SET rewritten='unknown' WHERE query='z'", "translation_cache"),
    ("UPDATE reindex_controls SET row_cursor=9000 WHERE generation=8", "reindex_controls"),
    ("UPDATE sync_state SET native_index_revision=9000", "sync_state"),
])
def test_unknown_affected_row_divergence_fails_closed_every_table(baseline_path, desired_path, pg_dsn, damage, table):
    with Snapshot(baseline_path) as baseline, Snapshot(desired_path) as desired, \
            ready_baseline(baseline, pg_dsn) as migration:
        migration.pg.execute(damage)
        before = target_report(migration, baseline)
        with pytest.raises(MigrationError, match="target row diverged.*" + table):
            apply_tail(migration, desired)
        assert target_report(migration, baseline) == before
        assert migration.pg.execute("SELECT migration_ready,document_count FROM funes_schema_state").fetchone() == (False, 5)
        assert not migration._exists(TAIL_LEDGER)


def test_unknown_insert_collision_is_not_overwritten(baseline_path, desired_path, pg_dsn):
    with Snapshot(baseline_path) as baseline, Snapshot(desired_path) as desired, \
            ready_baseline(baseline, pg_dsn) as migration:
        install_desired_row(migration.pg, desired, "memories", 19)
        migration.pg.execute("UPDATE memories SET source_version='third-party' WHERE id=19")
        before = target_report(migration, baseline)
        with pytest.raises(MigrationError, match="target row diverged"):
            apply_tail(migration, desired)
        assert target_report(migration, baseline) == before


def test_hybrid_old_and_desired_fields_are_not_treated_as_a_known_row(baseline_path, desired_path, pg_dsn):
    with Snapshot(baseline_path) as baseline, Snapshot(desired_path) as desired, \
            ready_baseline(baseline, pg_dsn) as migration:
        desired_version = desired.conn.execute("SELECT source_version FROM memories WHERE id=1").fetchone()[0]
        migration.pg.execute("UPDATE memories SET source_version=%s WHERE id=1",
                             (encode_pg_text(desired_version),))
        before = target_report(migration, baseline)
        with pytest.raises(MigrationError, match="target row diverged"):
            apply_tail(migration, desired)
        assert target_report(migration, baseline) == before


@pytest.mark.parametrize("damage", [
    "UPDATE memories SET raw_text='corrupt-unchanged-row' WHERE id=6",
    "INSERT INTO translation_cache(query,rewritten,created_at) VALUES('extra','unknown','now')",
])
def test_full_final_digest_catches_unaffected_divergence_and_rolls_back(baseline_path, desired_path, pg_dsn, damage):
    with Snapshot(baseline_path) as baseline, Snapshot(desired_path) as desired, \
            ready_baseline(baseline, pg_dsn) as migration:
        migration.pg.execute(damage)
        before = target_report(migration, baseline)
        with pytest.raises(MigrationError, match="full decoded row digest/count mismatch"):
            apply_tail(migration, desired)
        assert target_report(migration, baseline) == before
        assert not migration._exists(TAIL_LEDGER)
        assert migration.pg.execute("SELECT migration_ready,document_count FROM funes_schema_state").fetchone() == (False, 5)


def test_interrupt_rolls_back_all_tables_receipt_and_triggers_then_retry_works(baseline_path, desired_path, pg_dsn):
    with Snapshot(baseline_path) as baseline, Snapshot(desired_path) as desired, \
            ready_baseline(baseline, pg_dsn) as migration:
        for trigger in ("memories_native_state", "memories_native_pending"):
            migration.pg.execute(f"ALTER TABLE memories ENABLE TRIGGER {trigger}")
        before = target_report(migration, baseline)
        sequence_before = migration.pg.execute("SELECT last_value,is_called FROM memories_id_seq").fetchone()
        def interrupt():
            raise KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            apply_tail(migration, desired, before_commit=interrupt)
        assert target_report(migration, baseline) == before
        assert not migration._exists(TAIL_LEDGER)
        assert migration.pg.execute("SELECT last_value,is_called FROM memories_id_seq").fetchone() == sequence_before
        assert migration.pg.execute("SELECT count(*) FROM pg_trigger WHERE tgrelid='memories'::regclass "
                                    "AND NOT tgisinternal AND tgenabled='O'").fetchone() == (2,)
        assert apply_tail(migration, desired)["tables"] == source_report(desired)
        assert migration.pg.execute("SELECT count(*) FROM pg_trigger WHERE tgrelid='memories'::regclass "
                                    "AND NOT tgisinternal AND tgenabled='O'").fetchone() == (2,)


def test_snapshot_changed_before_commit_rolls_back(baseline_path, desired_path, pg_dsn):
    with Snapshot(baseline_path) as baseline, Snapshot(desired_path) as desired, \
            ready_baseline(baseline, pg_dsn) as migration:
        def change_source():
            with desired_path.open("ab") as stream:
                stream.write(b"changed snapshot")
        with pytest.raises(MigrationError, match="snapshot changed"):
            apply_tail(migration, desired, before_commit=change_source)
        assert target_report(migration, baseline) == source_report(baseline)
        assert not migration._exists(TAIL_LEDGER)


def test_unverified_or_ready_target_is_refused(baseline_path, desired_path, pg_dsn):
    with Snapshot(baseline_path) as baseline, Snapshot(desired_path) as desired, Migration(baseline, pg_dsn) as migration:
        migration.initialize()
        migration.copy(batch_rows=1, max_batches=1)
        with pytest.raises(MigrationError, match="fully verified"):
            apply_tail(migration, desired)
        migration.copy()
        migration.pg.execute("UPDATE funes_schema_state SET migration_ready=true")
        with pytest.raises(MigrationError, match="already ready"):
            apply_tail(migration, desired)
        assert not migration._exists(TAIL_LEDGER)


def test_wrong_baseline_identity_is_refused(baseline_path, desired_path, pg_dsn):
    with Snapshot(baseline_path) as baseline, ready_baseline(baseline, pg_dsn):
        pass
    with Snapshot(desired_path) as wrong, Migration(wrong, pg_dsn) as migration:
        with pytest.raises(MigrationError, match="source identity/version differs"):
            apply_tail(migration, wrong)
        assert not migration._exists(TAIL_LEDGER)


def test_unique_source_identity_swap_is_supported(baseline_path, tmp_path, pg_dsn):
    path = tmp_path / "swapped.sqlite3"
    conn = clone_snapshot(baseline_path, path)
    with conn:
        conn.execute("UPDATE memories SET source_identity='temporary' WHERE id=5")
        conn.execute("UPDATE memories SET source_identity='source-5' WHERE id=6")
        conn.execute("UPDATE memories SET source_identity='source-6' WHERE id=5")
    conn.close()
    with Snapshot(baseline_path) as baseline, Snapshot(path) as desired, ready_baseline(baseline, pg_dsn) as migration:
        result = apply_tail(migration, desired)
        assert result["changes"]["memories"] == {"inserted": 0, "updated": 2, "deleted": 0}
        assert result["tables"] == source_report(desired)


def test_noop_tail_does_not_rewrite_source_rows(baseline_path, pg_dsn):
    with Snapshot(baseline_path) as baseline, ready_baseline(baseline, pg_dsn) as migration:
        result = apply_tail(migration, baseline)
        assert result["applied"] == {table: {"removed": 0, "written": 0} for table in TABLES}
        assert result["changes"] == {table: {"inserted": 0, "updated": 0, "deleted": 0} for table in TABLES}


def test_thousand_row_tail_uses_bounded_bulk_statement_count(baseline_path, tmp_path, pg_dsn):
    path = tmp_path / "many.sqlite3"
    conn = clone_snapshot(baseline_path, path)
    with conn:
        for number in range(100, 1100):
            clone_memory(conn, 6, number, f"bulk-{number}")
    conn.close()
    with Snapshot(baseline_path) as baseline, Snapshot(path) as desired, ready_baseline(baseline, pg_dsn) as migration:
        class RecordedPG:
            def __init__(self, pg):
                self.raw = pg
                self.statements = []
            def execute(self, statement, *args, **kwargs):
                self.statements.append(statement)
                return self.raw.execute(statement, *args, **kwargs)
            def __getattr__(self, name):
                return getattr(self.raw, name)
        migration.pg = RecordedPG(migration.pg)
        result = apply_tail(migration, desired)
        assert result["changes"]["memories"]["inserted"] == 1000
        assert result["tables"] == source_report(desired)
        assert len(migration.pg.statements) < 100


def test_snapshot_files_are_not_modified(baseline_path, desired_path, pg_dsn):
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (baseline_path, desired_path)]
    with Snapshot(baseline_path) as baseline, Snapshot(desired_path) as desired, ready_baseline(baseline, pg_dsn) as migration:
        apply_tail(migration, desired)
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in (baseline_path, desired_path)] == before
