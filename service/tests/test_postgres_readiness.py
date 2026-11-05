"""Readiness must not wait behind source uploads or join their transaction."""
from __future__ import annotations

import threading
import os
import uuid
from types import SimpleNamespace
from unittest import mock

import pytest

from service.postgres import PostgresStore
from service.postgres_sync import PostgresSync


def test_probe_uses_bounded_independent_read_only_connection_and_closes():
    store = object.__new__(PostgresStore)
    store._migration_mode = False
    raw = mock.Mock()
    store._connect = mock.Mock(return_value=raw)
    store.lock = mock.Mock(side_effect=AssertionError("writer lock touched"))
    store.conn = mock.Mock()
    with mock.patch("service.postgres._verify_marker") as marker:
        store.probe_ready()
    store._connect.assert_called_once_with(connect_timeout=3)
    assert raw.execute.call_args_list == [
        mock.call("SET default_transaction_read_only=on"),
        mock.call("SET statement_timeout='3s'"),
    ]
    marker.assert_called_once_with(raw, require_ready=True)
    raw.close.assert_called_once_with()
    assert not store.conn.mock_calls
    assert not store.lock.mock_calls


def test_probe_closes_connection_when_marker_unavailable():
    store = object.__new__(PostgresStore)
    store._migration_mode = False
    raw = mock.Mock()
    store._connect = mock.Mock(return_value=raw)
    with mock.patch("service.postgres._verify_marker", side_effect=RuntimeError("unavailable")):
        with pytest.raises(RuntimeError, match="unavailable"):
            store.probe_ready()
    raw.close.assert_called_once_with()


def test_readiness_does_not_wait_for_upload_or_writer_locks():
    store = SimpleNamespace(lock=threading.RLock(), probe_ready=mock.Mock())
    syncer = PostgresSync(store)
    syncer.restored = True
    finished = threading.Event()
    result = []

    def probe():
        result.append(syncer.probe_ready())
        finished.set()

    # The probe thread must finish while both original locks remain held.
    with syncer.upload_lock, store.lock:
        thread = threading.Thread(target=probe)
        thread.start()
        completed = finished.wait(0.5)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert completed, "readiness waited for a writer"
    assert result == [True]
    store.probe_ready.assert_called_once_with()


def test_readiness_failure_is_closed_without_ack_or_writer_mutation():
    store = mock.Mock()
    store.probe_ready.side_effect = RuntimeError("private DSN must not escape")
    syncer = PostgresSync(store)
    syncer.restored = True
    before = dict(syncer.progress)
    assert syncer.probe_ready() is False
    assert syncer.progress == before
    assert not syncer.restore_failed
    store.reconnect.assert_not_called()
    store.ingest.assert_not_called()
    store.set_sync.assert_not_called()
    store.probe_ready.side_effect = None
    assert syncer.probe_ready() is True


def test_readiness_cannot_clear_a_concurrent_failed_write():
    store = mock.Mock()
    syncer = PostgresSync(store)
    syncer.restored = True
    store.probe_ready.side_effect = syncer._unavailable
    assert syncer.probe_ready() is False
    assert syncer.restore_failed


@pytest.mark.parametrize("restored,failed", [(False, False), (True, True)])
def test_restoring_probe_fails_fast_without_joining_restore(restored, failed):
    store = mock.Mock()
    syncer = PostgresSync(store)
    syncer.restored = restored
    syncer.restore_failed = failed
    syncer.restoring = True
    syncer.check_ready = mock.Mock(side_effect=AssertionError("restore lock touched"))
    result = []
    finished = threading.Event()

    def probe():
        result.append(syncer.probe_ready())
        finished.set()

    with syncer.upload_lock:
        thread = threading.Thread(target=probe)
        thread.start()
        completed = finished.wait(0.5)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert completed, "readiness waited for restoration"
    assert result == [False]
    syncer.check_ready.assert_not_called()
    store.probe_ready.assert_not_called()


@pytest.mark.parametrize("restored,failed", [(False, False), (True, True)])
def test_unvalidated_or_failed_writer_still_needs_full_check(restored, failed):
    store = mock.Mock()
    syncer = PostgresSync(store)
    syncer.restored = restored
    syncer.restore_failed = failed
    syncer.check_ready = mock.Mock(return_value=False)
    assert syncer.probe_ready() is False
    syncer.check_ready.assert_called_once_with()
    store.probe_ready.assert_not_called()


@pytest.fixture
def isolated_postgres(tmp_path):
    dsn = os.getenv("FUNES_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("FUNES_TEST_POSTGRES_DSN not configured")
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    schema = "funes_readiness_" + uuid.uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            options = conninfo_to_dict(dsn).get("options", "")
            scoped = make_conninfo(dsn, options=f"{options} -csearch_path={schema},public".strip())
            bootstrap = PostgresStore(str(tmp_path), scoped, initialize=True)
            try:
                bootstrap.finalize_migration(expected_count=0)
            finally:
                bootstrap.close()
            runtime = PostgresStore(str(tmp_path), scoped)
            try:
                yield runtime
            finally:
                runtime.close()
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_real_postgres_probe_finishes_during_open_writer_transaction(isolated_postgres):
    store = isolated_postgres
    syncer = PostgresSync(store)
    assert syncer.restore() == 0
    result = []
    finished = threading.Event()

    def probe():
        result.append(syncer.probe_ready())
        finished.set()

    with syncer.upload_lock, store.lock, store.conn:
        store.conn.execute("UPDATE sync_state SET last_error=? WHERE id=1", ("uncommitted-test",)).close()
        thread = threading.Thread(target=probe)
        thread.start()
        completed = finished.wait(2)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert completed, "real readiness waited for the writer transaction"
    assert result == [True]


def test_real_postgres_unready_marker_is_not_accepted(isolated_postgres):
    store = isolated_postgres
    syncer = PostgresSync(store)
    assert syncer.restore() == 0
    with store.lock, store.conn:
        store.conn.execute("UPDATE funes_schema_state SET migration_ready=FALSE WHERE id=1").close()
    assert syncer.probe_ready() is False
    # A read failure neither commits source data nor claims writer recovery.
    assert syncer.restored is True
    assert syncer.restore_failed is False
