"""Split-role acceptance against a disposable PG, never the configured service DSN."""
import os
import threading
import uuid

import pytest

from service.roles import WriterLease


@pytest.fixture
def isolated_dsn():
    dsn = os.getenv("FUNES_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("FUNES_TEST_POSTGRES_DSN must identify disposable PostgreSQL")
    return dsn


def test_two_writer_sessions_have_exactly_one_owner(isolated_dsn):
    memory = "split-test/" + uuid.uuid4().hex
    leases = [WriterLease(isolated_dsn, memory) for _ in range(2)]
    barrier = threading.Barrier(2)
    outcomes = []

    def compete(lease):
        barrier.wait(timeout=5)
        try:
            lease.acquire(monitor=False)
            outcomes.append("owner")
        except RuntimeError as error:
            assert "lease" in str(error)
            outcomes.append("refused")

    threads = [threading.Thread(target=compete, args=(lease,)) for lease in leases]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()
        assert sorted(outcomes) == ["owner", "refused"]
        assert sum(lease.ready for lease in leases) == 1
    finally:
        for lease in leases:
            lease.close()
    replacement = WriterLease(isolated_dsn, memory)
    try:
        replacement.acquire(monitor=False)
        assert replacement.ready
    finally:
        replacement.close()


def test_terminated_session_revokes_writer_and_never_reacquires(isolated_dsn):
    import psycopg

    lost = threading.Event()
    lease = WriterLease(isolated_dsn, "split-test/" + uuid.uuid4().hex,
                        on_loss=lost.set, interval=0.05)
    try:
        lease.acquire(monitor=False)
        pid = lease._session.info.backend_pid
        with psycopg.connect(isolated_dsn, autocommit=True) as admin:
            assert admin.execute("SELECT pg_terminate_backend(%s)", (pid,)).fetchone()[0]
        assert not lease.check()
        assert lost.wait(1)
        assert not lease.ready
        with pytest.raises(RuntimeError, match="lease"):
            lease.require()
        with pytest.raises(RuntimeError, match="reacquired"):
            lease.acquire(monitor=False)
    finally:
        lease.close()


def test_lease_monitor_is_reaped_on_shutdown(isolated_dsn):
    lease = WriterLease(isolated_dsn, "split-test/" + uuid.uuid4().hex, interval=0.05)
    lease.acquire()
    thread = lease._thread
    assert thread.is_alive()
    lease.close()
    assert not thread.is_alive()
    assert not lease.ready


def test_query_uses_existing_data_without_mutating_checkpoint(isolated_dsn, tmp_path, monkeypatch):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from service.postgres import PostgresStore
    from service.server import App

    suffix = uuid.uuid4().hex
    schema, reader = "split_source_" + suffix, "split_reader_" + suffix
    scoped = make_conninfo(isolated_dsn, options="-csearch_path=" + schema + ",public")
    monkeypatch.setenv("FUNES_SERVICE_ROLE", "combined")
    with psycopg.connect(isolated_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        admin.execute(sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(reader)))
        try:
            bootstrap = PostgresStore(str(tmp_path / "bootstrap"), scoped, initialize=True)
            try:
                bootstrap.finalize_migration(expected_count=0)
                bootstrap.ingest([{
                    "source_identity": "existing-indexed-document",
                    "source_version": "unchanged-version",
                    "raw_text": "现有中文记忆，不重新生成 embedding",
                    "source_agent": "gemini", "source_type": "memory",
                }])
                with bootstrap.lock, bootstrap.conn:
                    bootstrap.conn.execute(
                        "UPDATE sync_state SET native_checkpoint_profile=?,native_checkpoint_memory=?,"
                        "native_index_revision=123,native_eligible_count=2,native_indexed_count=1,"
                        "native_held_count=0,native_invalid_count=0 WHERE id=1",
                        ("existing-profile", "split-test/existing-index"),
                    )
            finally:
                bootstrap.close()
            admin.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                sql.Identifier(schema), sql.Identifier(reader)))
            admin.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(
                sql.Identifier(schema), sql.Identifier(reader)))

            def snapshot():
                return {
                    table: admin.execute(sql.SQL("SELECT to_jsonb(t) FROM {}.{} t ORDER BY id").format(
                        sql.Identifier(schema), sql.Identifier(table))).fetchall()
                    for table in ("sync_state", "memories", "funes_schema_state")
                }

            before = snapshot()
            monkeypatch.setenv("FUNES_SERVICE_ROLE", "query")
            monkeypatch.setenv("FUNES_POSTGRES_DSN", make_conninfo(scoped, user=reader))
            monkeypatch.setenv("FUNES_DATA_DIR", str(tmp_path / "query"))
            app = App()
            try:
                assert app.reconcile_thread is None
                assert app.reindex_thread is None
                assert app.restore_thread is None
                assert app.writer_lease is None
                assert app.store.get("existing-indexed-document")["source_version"] == "unchanged-version"
                assert app.syncer.probe_ready()
                checkpoint = app.store.native_index_checkpoint(
                    {"fingerprint": "mismatched-reader-profile"}, "split-test/other-index")
                assert checkpoint["revision"] == 123
                assert checkpoint["indexed"] == 1
                assert checkpoint["pending"] == 1
                assert not checkpoint["checkpoint_current"]
                app.store.translation_put("中文查询", "translated query")
                assert app.store.translation_get("中文查询") == "translated query"
                assert app.store.conn.execute("SHOW default_transaction_read_only").fetchone()[0] == "on"
                with pytest.raises(psycopg.Error):
                    app.store.conn.execute("UPDATE sync_state SET native_index_revision=0 WHERE id=1")
                with pytest.raises(RuntimeError, match="read-only"):
                    app.ingest_documents([])
            finally:
                app.close()
            assert snapshot() == before
            assert admin.execute(sql.SQL("SELECT count(*) FROM {}.translation_cache").format(
                sql.Identifier(schema))).fetchone()[0] == 0
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
            admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(reader)))
