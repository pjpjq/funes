"""Tests for PostgreSQL status snapshot bridge integration and /sync/status routing."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

import space.server as bridge


PROFILE = {
    "fingerprint": "voyage-test",
    "provider": "voyage",
    "model": "test",
    "dimensions": 4,
    "schema_version": 1,
}
MEMORY = "owner/test-memory"


class SnapshotStoreDouble:
    def __init__(self, snapshot_data=None):
        self.lock = threading.RLock()
        self.snapshot_calls = []
        self._snapshot_data = snapshot_data or {
            "documents": 42,
            "sync": {"last_error": None, "schema_version": 1},
            "canonical_index": {
                "eligible": 42,
                "indexed": 42,
                "held": 0,
                "invalid": 0,
                "pending": 0,
                "complete": True,
                "cutover_ready": True,
                "failures": None,
                "failure_counts_status": "deferred",
            },
        }

    def status_snapshot(self, profile, memory, *, index_layout_version=1):
        self.snapshot_calls.append((profile, memory, index_layout_version))
        return dict(self._snapshot_data)

    def count(self):
        raise AssertionError("count() should not be called when status_snapshot is present")

    def native_index_checkpoint(self, profile, memory):
        raise AssertionError("native_index_checkpoint() should not be called")


def test_sync_status_uses_status_snapshot_and_bypasses_native_subprocess(monkeypatch):
    monkeypatch.setattr(bridge, "REMOTE", MEMORY)
    store = SnapshotStoreDouble()
    syncer = SimpleNamespace(
        backend="postgres",
        restoring=False,
        restore_failed=False,
        probe_ready=lambda: True,
    )
    app = SimpleNamespace(store=store, syncer=syncer, restore_result=0)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "ready"})
    monkeypatch.setattr(bridge, "run", mock.Mock(side_effect=AssertionError("run() must not be called")))

    code, payload = bridge.sync_status_payload()

    assert code == 200
    assert payload["ok"] is True
    assert payload["remote"] == MEMORY
    assert payload["status"] == ""
    assert payload["diagnostic"] == "not_run"
    assert payload["native_status"] == "deferred"
    assert payload["error"] == ""
    assert payload["native_warm"] == {"state": "ready"}

    sources = payload["source_store"]
    assert sources["configured"] is True
    assert sources["ready"] is True
    assert sources["documents"] == 42
    assert sources["canonical_index"]["complete"] is True
    assert sources["canonical_index"]["failures"] is None
    assert sources["canonical_index"]["failure_counts_status"] == "deferred"
    assert len(store.snapshot_calls) == 1
    assert store.snapshot_calls[0][1] == bridge.index_memory()


def test_sync_status_does_not_wait_for_upload_or_store_locks(monkeypatch):
    monkeypatch.setattr(bridge, "REMOTE", MEMORY)
    store = SnapshotStoreDouble()
    upload_lock = threading.Lock()
    syncer = SimpleNamespace(
        backend="postgres",
        restoring=False,
        restore_failed=False,
        probe_ready=lambda: True,
        upload_lock=upload_lock,
    )
    app = SimpleNamespace(store=store, syncer=syncer, restore_result=0)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "ready"})
    monkeypatch.setattr(bridge, "run", mock.Mock(side_effect=AssertionError("run() called")))

    finished = threading.Event()
    results = []
    errors = []

    def call_status():
        try:
            results.append(bridge.sync_status_payload())
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    with upload_lock, store.lock:
        thread = threading.Thread(target=call_status)
        thread.start()
        completed = finished.wait(1.0)
    thread.join(timeout=2.0)

    assert completed, "sync_status_payload blocked behind upload or store lock"
    assert not thread.is_alive()
    assert not errors
    code, payload = results[0]
    assert code == 200
    assert payload["ok"] is True
    assert payload["source_store"]["documents"] == 42


@pytest.mark.parametrize("warm_state_val,expected_error", [
    ({"state": "warming"}, "native_memory_warming"),
    ({"state": "error"}, "native_mcp_unavailable"),
    ({"state": "not_started"}, "native_mcp_unavailable"),
])
def test_sync_status_warm_state_gates(monkeypatch, warm_state_val, expected_error):
    monkeypatch.setattr(bridge, "REMOTE", MEMORY)
    store = SnapshotStoreDouble()
    syncer = SimpleNamespace(
        backend="postgres",
        restoring=False,
        restore_failed=False,
        probe_ready=lambda: True,
    )
    app = SimpleNamespace(store=store, syncer=syncer, restore_result=0)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "warm_state", lambda: warm_state_val)
    monkeypatch.setattr(bridge, "run", mock.Mock(side_effect=AssertionError("run() called")))

    code, payload = bridge.sync_status_payload()

    assert code == 503
    assert payload["ok"] is False
    assert payload["error"] == expected_error
    assert payload["diagnostic"] == "not_run"
    assert payload["native_status"] == "deferred"


def test_source_state_and_sync_status_failure_redaction(monkeypatch):
    monkeypatch.setattr(bridge, "REMOTE", MEMORY)
    store = SnapshotStoreDouble()
    store.status_snapshot = mock.Mock(
        side_effect=RuntimeError("postgres://secret_user:super_secret_pw@10.0.0.1:5432/funes")
    )
    syncer = SimpleNamespace(
        backend="postgres",
        restoring=False,
        restore_failed=False,
        probe_ready=lambda: True,
    )
    app = SimpleNamespace(store=store, syncer=syncer, restore_result=0)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "ready"})
    monkeypatch.setattr(bridge, "run", mock.Mock(side_effect=AssertionError("run() called")))

    code, payload = bridge.sync_status_payload()

    assert code == 503
    assert payload["ok"] is False
    assert payload["source_store"]["ready"] is False
    assert payload["source_store"]["error"] == "postgres_unavailable"

    rendered = json.dumps(payload)
    assert "postgres://" not in rendered
    assert "secret_user" not in rendered
    assert "super_secret_pw" not in rendered

