"""ANN maintenance must not wait for all source embeddings or alter checkpoints."""
import threading
from types import SimpleNamespace

import pytest

import space.server as bridge


class NoCheckpointStore:
    def __getattr__(self, name):
        raise AssertionError(f"maintenance must not access source/checkpoint: {name}")


@pytest.fixture
def setup(monkeypatch):
    app = SimpleNamespace(
        store=NoCheckpointStore(),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
        canonical_index_stop=threading.Event(),
    )
    bridge._initialize_canonical_reconcile_state(app)
    clock = [100.0]
    monkeypatch.setattr(bridge.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(bridge, "INDEX_MAINTENANCE_SETTINGS", {
        "enabled": True, "interval_seconds": 1800,
        "document_threshold": 2048, "retry_seconds": 300,
    })
    monkeypatch.setattr(bridge, "REMOTE", "owner/active")
    monkeypatch.setattr(bridge, "index_memory", lambda: "owner/active")
    monkeypatch.setattr(bridge, "index_embedding_profile", lambda: {"fingerprint": "active"})
    monkeypatch.setattr(bridge, "embedding_profile", lambda: {"fingerprint": "active"})
    calls, refreshes = [], []
    monkeypatch.setattr(bridge, "optimize_native_index", lambda memory, profile: calls.append((memory, profile)) or True)
    monkeypatch.setattr(bridge, "_request_canonical_refresh", lambda app, **kwargs: refreshes.append(kwargs))
    return app, clock, calls, refreshes


def cycle(app, indexed=1, **kwargs):
    return bridge._maybe_maintain_canonical_index(app, {
        "attempted": indexed, "indexed": indexed, "held": 0,
        "durable": True, **kwargs,
    })


def test_first_durable_cycle_repairs_existing_backlog_without_checkpoint(setup):
    app, clock, calls, refreshes = setup
    assert cycle(app, 37)
    state = bridge.index_maintenance_state(app)
    assert state["successes"] == 1
    assert state["pending_documents"] == 0
    assert state["last_error"] is None
    assert state["active"] is False
    assert len(calls) == 1
    assert refreshes == [{"force": True}]
    assert all(not key.startswith("_") for key in state)
    assert bridge.canonical_reconcile_state(app)["index_maintenance"] == state


def test_document_threshold_debounces_not_each_batch(setup):
    app, clock, calls, _ = setup
    assert cycle(app)
    assert not cycle(app, 2047)
    assert cycle(app, 1)
    assert len(calls) == 2
    assert bridge.index_maintenance_state(app)["pending_documents"] == 0


def test_time_threshold_requires_dirty_documents(setup):
    app, clock, calls, _ = setup
    assert cycle(app)
    clock[0] += 1800
    assert not cycle(app, 0)
    assert cycle(app, 1)
    assert len(calls) == 2


def test_time_below_threshold_preserves_dirty_documents(setup):
    app, clock, calls, _ = setup
    assert cycle(app)
    clock[0] += 1799
    assert not cycle(app, 3)
    assert bridge.index_maintenance_state(app)["pending_documents"] == 3
    clock[0] += 1
    assert cycle(app, 0)


def test_failed_attempt_cools_from_finish_and_keeps_all_dirty_rows(setup, monkeypatch):
    app, clock, calls, refreshes = setup
    def fail(*_args):
        clock[0] += 900
        return False
    monkeypatch.setattr(bridge, "optimize_native_index", fail)
    assert not cycle(app, 10)
    state = bridge.index_maintenance_state(app)
    assert state["pending_documents"] == 10
    assert state["failures"] == 1
    assert state["last_error"] == "native_optimize_failed"
    assert state["last_duration_ms"] == 900000
    assert not refreshes
    monkeypatch.setattr(bridge, "optimize_native_index", lambda *_: calls.append(True) or True)
    clock[0] += 299
    assert not cycle(app, 2)
    assert not calls
    assert bridge.index_maintenance_state(app)["pending_documents"] == 12
    clock[0] += 1
    assert cycle(app, 0)
    assert bridge.index_maintenance_state(app)["pending_documents"] == 0
    assert refreshes == [{"force": True}]


def test_exception_is_redacted_and_retryable(setup, monkeypatch):
    app, _, _, refreshes = setup
    def fail(*_args):
        raise RuntimeError("secret-raw-session")
    monkeypatch.setattr(bridge, "optimize_native_index", fail)
    assert not cycle(app, 5)
    state = bridge.index_maintenance_state(app)
    assert state["active"] is False
    assert state["pending_documents"] == 5
    assert state["last_error"] == "unexpected"
    assert "secret-raw-session" not in str(state)
    assert not refreshes


@pytest.mark.parametrize("result", [
    {"indexed": 0, "durable": True},
    {"indexed": 0, "held": 10, "durable": True},
    {"indexed": 10, "durable": False},
])
def test_idle_held_or_nondurable_does_not_start_maintenance(setup, result):
    app, _, calls, refreshes = setup
    assert not bridge._maybe_maintain_canonical_index(app, result)
    assert not calls and not refreshes


@pytest.mark.parametrize("guard", ["disabled", "restoring", "restore_failed", "stopped", "store", "syncer"])
def test_guarded_apps_do_not_run_native(setup, guard):
    app, _, calls, _ = setup
    if guard == "disabled":
        bridge.INDEX_MAINTENANCE_SETTINGS["enabled"] = False
    elif guard in {"restoring", "restore_failed"}:
        setattr(app.syncer, guard, True)
    elif guard == "stopped":
        app.canonical_index_stop.set()
    else:
        delattr(app, guard)
    assert not cycle(app)
    assert not calls


@pytest.mark.parametrize("facet", ["memory", "profile"])
def test_build_target_mismatch_does_not_refresh_active_worker(setup, monkeypatch, facet):
    app, _, calls, refreshes = setup
    if facet == "memory":
        monkeypatch.setattr(bridge, "index_memory", lambda: "owner/build")
    else:
        monkeypatch.setattr(bridge, "index_embedding_profile", lambda: {"fingerprint": "build"})
    assert cycle(app)
    assert len(calls) == 1
    assert not refreshes


def test_native_writes_serialized_but_raw_write_lock_is_free(setup, monkeypatch):
    app, _, _, _ = setup
    observed = []
    def optimize(*_args):
        def probe():
            raw = bridge.WRITE_LOCK.acquire(timeout=0.05)
            native = bridge.NATIVE_WRITE_LOCK.acquire(timeout=0.05)
            observed.append((raw, native))
            if raw:
                bridge.WRITE_LOCK.release()
            if native:
                bridge.NATIVE_WRITE_LOCK.release()
        thread = threading.Thread(target=probe)
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()
        return True
    monkeypatch.setattr(bridge, "optimize_native_index", optimize)
    assert cycle(app)
    assert observed == [(True, False)]


def test_background_invokes_maintenance_after_durable_reconcile(setup, monkeypatch):
    app, _, calls, _ = setup
    app.restore_done = threading.Event()
    app.restore_done.set()
    order = []
    def reconcile(_app):
        order.append("durable_reconcile")
        return {"attempted": 7, "indexed": 7, "held": 0, "durable": True}
    def optimize(*_args):
        order.append("maintenance")
        app.canonical_index_stop.set()
        return True
    monkeypatch.setattr(bridge, "reconcile_canonical_index", reconcile)
    monkeypatch.setattr(bridge, "optimize_native_index", optimize)
    bridge._canonical_reconcile_background(app)
    assert order == ["durable_reconcile", "maintenance"]
    state = bridge.canonical_reconcile_state(app)
    assert state["last_result"]["indexed"] == 7
    assert state["last_error"] is None
    assert state["index_maintenance"]["successes"] == 1


def test_restart_first_progress_runs_without_restoring_final_marker(setup):
    app, _, calls, _ = setup
    assert cycle(app)
    bridge._initialize_canonical_reconcile_state(app)
    assert cycle(app)
    assert len(calls) == 2


def test_settings_defaults_and_invalid_values():
    assert bridge._index_maintenance_settings({}) == {
        "enabled": True, "interval_seconds": 1800,
        "document_threshold": 2048, "retry_seconds": 300,
    }
    assert bridge._index_maintenance_settings({
        "FUNES_CANONICAL_MAINTENANCE_ENABLED": "false",
        "FUNES_CANONICAL_MAINTENANCE_ROWS": "bad",
    })["enabled"] is False
    assert bridge._index_maintenance_settings({"FUNES_CANONICAL_MAINTENANCE_ROWS": "bad"})["document_threshold"] == 2048
