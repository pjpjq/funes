"""Committed status diagnostics must not acquire writer locks or scan sources."""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from service import server
from service.postgres import PostgresNotReady, PostgresStore
from service.postgres_sync import PostgresSync
from service.tests.test_postgres_readiness import isolated_postgres


PROFILE = {"fingerprint": "voyage-test", "provider": "voyage", "model": "test", "dimensions": 4}
MEMORY = "owner/test-memory"


def state(**changes):
    return {
        "documents": 20, "native_checkpoint_profile": PROFILE["fingerprint"],
        "native_checkpoint_memory": MEMORY,
        "native_checkpoint_state_version": server.NATIVE_CHECKPOINT_STATE_VERSION,
        "native_index_revision": 7, "native_eligible_count": 12,
        "native_indexed_count": 9, "native_held_count": 2,
        "native_invalid_count": 0, **changes,
    }


def finish_while_locked(locks, operation):
    finished = threading.Event()
    result, errors = [], []

    def run():
        try:
            result.append(operation())
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    from contextlib import ExitStack
    with ExitStack() as stack:
        for lock in locks:
            stack.enter_context(lock)
        thread = threading.Thread(target=run)
        thread.start()
        completed = finished.wait(0.8)
    thread.join(timeout=5)
    assert completed, "status blocked behind upload/writer lock"
    assert not thread.is_alive()
    assert not errors
    return result[0]


def test_progress_does_not_wait_for_upload_lock():
    syncer = PostgresSync(mock.Mock())
    snapshot = finish_while_locked([syncer.upload_lock], lambda: syncer.progress)
    assert snapshot["phase"] == "idle"
    snapshot["phase"] = "caller-mutated"
    assert syncer.progress["phase"] == "idle"


def test_read_connection_sets_timeout_before_validating_marker():
    store = object.__new__(PostgresStore)
    store._migration_mode = False
    raw = mock.Mock()
    store._connect = mock.Mock(return_value=raw)

    def verify(connection, **kwargs):
        assert connection is raw
        assert raw.execute.call_args_list == [
            mock.call("SET default_transaction_read_only=on"),
            mock.call("SET statement_timeout='3s'"),
        ]
        raise PostgresNotReady("marker missing")

    with mock.patch("service.postgres._verify_marker", side_effect=verify):
        with pytest.raises(PostgresNotReady):
            store._read_connection(connect_timeout=3, statement_timeout=3)
    store._connect.assert_called_once_with(connect_timeout=3)
    raw.close.assert_called_once_with()


def test_sync_status_uses_single_committed_join_and_closes_reader():
    store = object.__new__(PostgresStore)
    store.lock = threading.RLock()
    store.conn = mock.Mock()
    reader = mock.Mock()
    reader.execute.return_value.fetchone.return_value = state()
    store._read_connection = mock.Mock(return_value=reader)
    snapshot = finish_while_locked([store.lock], store.sync_status)
    assert snapshot["documents"] == 20
    sql = reader.execute.call_args.args[0]
    assert "JOIN funes_schema_state" in sql
    assert "memories" not in sql.lower()
    assert "UPDATE" not in sql
    reader.execute.assert_called_once()
    reader.close.assert_called_once_with()
    assert not store.conn.mock_calls
    store._read_connection.assert_called_once_with(connect_timeout=3, statement_timeout=3)


@pytest.mark.parametrize("missing", [True, False])
def test_sync_status_closes_reader_on_missing_marker_or_query_error(missing):
    store = object.__new__(PostgresStore)
    reader = mock.Mock()
    store._read_connection = mock.Mock(return_value=reader)
    if missing:
        reader.execute.return_value.fetchone.return_value = None
    else:
        reader.execute.side_effect = RuntimeError("unavailable")
    with pytest.raises((PostgresNotReady, RuntimeError)):
        store.sync_status()
    reader.close.assert_called_once_with()


def test_snapshot_preserves_actual_counts_and_defers_expensive_failures():
    store = object.__new__(PostgresStore)
    store.sync_status = mock.Mock(return_value=state())
    value = store.status_snapshot(PROFILE, MEMORY)
    checkpoint = value["canonical_index"]
    assert (checkpoint["eligible"], checkpoint["indexed"], checkpoint["held"], checkpoint["pending"]) == (12, 9, 2, 1)
    assert checkpoint["checkpoint_current"] is True
    assert checkpoint["complete"] is False and checkpoint["cutover_ready"] is False
    assert checkpoint["failures"] is None
    assert checkpoint["failure_counts_status"] == "deferred"
    store.sync_status.assert_called_once_with()


@pytest.mark.parametrize("changes", [
    {"native_checkpoint_profile": "old-profile"},
    {"native_checkpoint_memory": "other/memory"},
    {"native_checkpoint_state_version": -1},
])
def test_other_checkpoint_cannot_be_mislabeled_as_current_or_complete(changes):
    store = object.__new__(PostgresStore)
    record = state(native_indexed_count=10, **changes)
    store.sync_status = mock.Mock(return_value=record)
    value = store.status_snapshot(PROFILE, MEMORY)["canonical_index"]
    assert value["pending"] == 0
    assert value["complete"] is False and value["cutover_ready"] is False
    assert value["checkpoint_current"] is False
    assert value["fingerprint"] == record["native_checkpoint_profile"]
    assert value["memory"] == record["native_checkpoint_memory"]
    assert "provider" not in value


def test_cutover_requires_matching_optimized_index_layout_and_fingerprint():
    store = object.__new__(PostgresStore)
    record = state(native_indexed_count=10)
    store.sync_status = lambda: dict(record)
    checkpoint = store.status_snapshot(PROFILE, MEMORY)["canonical_index"]
    record.update(native_optimize_status="optimized", native_optimize_fingerprint=PROFILE["fingerprint"],
                  native_optimize_memory=MEMORY, native_optimize_layout_version=1,
                  native_optimize_index_fingerprint=checkpoint["index_fingerprint"])
    assert store.status_snapshot(PROFILE, MEMORY)["canonical_index"]["cutover_ready"] is True
    assert store.status_snapshot(PROFILE, MEMORY, index_layout_version=2)["canonical_index"]["cutover_ready"] is False
    record["native_index_revision"] += 1
    assert store.status_snapshot(PROFILE, MEMORY)["canonical_index"]["cutover_ready"] is False


@pytest.mark.parametrize("method,path", [("GET", "/sync/status"), ("POST", "/sources/check")])
def test_service_read_only_routes_do_not_wait_for_upload_writer(method, path):
    store = SimpleNamespace(lock=threading.RLock(), probe_ready=mock.Mock(),
                            sync_status=lambda: {"documents": 12}, existing_identities=lambda ids: ids)
    syncer = PostgresSync(store)
    syncer.restored = True
    syncer.check_ready = mock.Mock(side_effect=AssertionError("writer validation called"))
    app = SimpleNamespace(store=store, syncer=syncer, restore_result=0)
    handler = object.__new__(server.make_handler(app))
    handler.path = path
    handler._authorized = lambda: True
    handler._json = lambda status, body: (status, body)
    handler._body = lambda: {"source_identities": ["test-id"]}
    code, value = finish_while_locked([syncer.upload_lock, store.lock], getattr(handler, "do_" + method))
    assert code == 200
    assert (value.get("documents") == 12 if method == "GET" else value["present"] == ["test-id"])
    syncer.check_ready.assert_not_called()


def test_real_status_observes_previous_commit_while_writer_has_new_markers(isolated_postgres):
    store = isolated_postgres
    syncer = PostgresSync(store)
    assert syncer.restore() == 0
    before = store.sync_status()
    result, error = [], []
    finished = threading.Event()

    def read():
        try:
            result.append(store.status_snapshot(PROFILE, MEMORY))
        except BaseException as exc:
            error.append(type(exc).__name__)
        finally:
            finished.set()

    with syncer.upload_lock, store.lock, store.conn:
        store.conn.execute("UPDATE sync_state SET last_error=? WHERE id=1", ("uncommitted-test",)).close()
        store.conn.execute("UPDATE funes_schema_state SET document_count=99 WHERE id=1").close()
        thread = threading.Thread(target=read)
        thread.start()
        completed = finished.wait(3)
    thread.join(timeout=5)
    assert completed and not thread.is_alive() and not error
    assert result[0]["documents"] == before["documents"]
    assert result[0]["sync"]["last_error"] == before["last_error"]
    assert store.sync_status()["documents"] == 99
