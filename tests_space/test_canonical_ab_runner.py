"""A/B must use the existing canonical loop, locks, and durable pending rows."""
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import space.server as bridge
from service.server import Store
from space.canonical_ab import CanonicalABController, FIXED_PROFILES


class Syncer:
    restoring = False
    restore_failed = False
    restored = True

    def __init__(self):
        self.uploads = []

    def upload(self, documents):
        self.uploads.append(documents)
        return {"durable": True}


class BoundedStop:
    def __init__(self, cycles):
        self.cycles = cycles
        self.waits = []

    def is_set(self):
        return len(self.waits) >= self.cycles

    def wait(self, timeout):
        self.waits.append(timeout)
        return self.is_set()


@pytest.fixture
def app(tmp_path, monkeypatch):
    app = SimpleNamespace(
        store=Store(str(tmp_path / "source")), syncer=Syncer(),
        restore_done=threading.Event(), canonical_index_stop=threading.Event(),
        canonical_index_thread=SimpleNamespace(is_alive=lambda: True),
    )
    app.restore_done.set()
    bridge._initialize_canonical_reconcile_state(app)
    monkeypatch.setattr(bridge, "source_app", lambda: app)
    monkeypatch.setattr(bridge, "REMOTE", "owner/canonical")
    monkeypatch.setattr(bridge, "INDEX_REMOTE", "")
    monkeypatch.setattr(bridge, "request_warm", lambda **_kwargs: None)
    monkeypatch.setattr(bridge, "_maybe_maintain_canonical_index", lambda *_args: None)
    yield app
    app.canonical_index_stop.set() if hasattr(app.canonical_index_stop, "set") else None
    app.store.close()


@pytest.fixture
def http_server(app, monkeypatch):
    monkeypatch.setattr(bridge, "TOKEN", "test-app-token")
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    assert not thread.is_alive()


def request(server, method, action, headers=None, payload=None):
    conn = HTTPConnection(*server.server_address, timeout=2)
    try:
        conn.request(
            method, "/admin/canonical-ab/" + action,
            json.dumps({} if payload is None else payload).encode() if method == "POST" else None,
            headers or {},
        )
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


AUTH = {"Authorization": "Bearer test-app-token"}
PRIVATE_AUTH = {"Authorization": "Bearer test-hub-token", "X-Funes-Authorization": "Bearer test-app-token"}


@pytest.mark.parametrize("action,method", [("start", "POST"), ("abort", "POST"), ("status", "GET")])
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {**AUTH, "X-Funes-Authorization": "Bearer wrong"}])
def test_admin_routes_authenticate_before_touching_controller(http_server, app, action, method, headers):
    assert request(http_server, method, action, headers)[0] == 401
    assert app.canonical_ab.phase == "idle"


@pytest.mark.parametrize("headers", [AUTH, PRIVATE_AUTH, {"X-Funes-Token": "Bearer test-app-token"}])
def test_admin_routes_preserve_public_and_private_space_auth(http_server, app, headers):
    code, status = request(http_server, "GET", "status", headers)
    assert code == 200 and status["phase"] == "idle"
    code, status = request(http_server, "POST", "start", headers)
    assert code == 200 and status["phase"] == "running"
    assert status["deadline_seconds"] == 1200
    assert request(http_server, "POST", "start", headers)[0] == 409
    code, status = request(http_server, "POST", "abort", headers)
    assert code == 200 and status["phase"] == "aborted"
    assert app.canonical_index_thread.is_alive()


@pytest.mark.parametrize("guard,error", [
    ("restoring", "restore_in_progress"), ("restore_failed", "restore_failed"),
    ("unrestored", "restore_in_progress"), ("restore_wait", "restore_in_progress"),
    ("disabled", "canonical_index_disabled"), ("dead", "canonical_index_disabled"),
    ("stopped", "canonical_index_disabled"),
])
def test_start_requires_restored_running_reconciler(http_server, app, monkeypatch, guard, error):
    if guard in {"restoring", "restore_failed"}:
        setattr(app.syncer, guard, True)
    elif guard == "unrestored":
        app.syncer.restored = False
    elif guard == "restore_wait":
        app.restore_done.clear()
    elif guard == "disabled":
        monkeypatch.setattr(bridge, "index_memory", lambda: "")
    elif guard == "dead":
        app.canonical_index_thread = SimpleNamespace(is_alive=lambda: False)
    else:
        app.canonical_index_stop.set()
    code, body = request(http_server, "POST", "start", PRIVATE_AUTH)
    assert code == 503 and body["error"] == error
    assert app.canonical_ab.phase == "idle"


def test_admin_cannot_supply_runtime_override(http_server, app):
    code, body = request(http_server, "POST", "start", AUTH, {"concurrency": 32, "memory": "other"})
    assert code == 400
    assert body["error"] == "canonical_ab_accepts_no_parameters"
    assert app.canonical_ab.phase == "idle"


def populate(app, count, raw="safe canonical content"):
    app.store.ingest([
        {"source_identity": f"pending-{i:03d}", "source_version": "v1", "raw_text": raw,
         "source_agent": "codex", "source_type": "memory", "updated_at": "2026-10-04T00:00:00Z"}
        for i in range(count)
    ])


@pytest.mark.parametrize("fixed", FIXED_PROFILES)
def test_override_passes_exact_settings_without_changing_embedding_identity(app, monkeypatch, fixed):
    populate(app, 140)
    fingerprint = bridge.index_embedding_profile()["fingerprint"]
    monkeypatch.setattr(bridge, "CANONICAL_INDEX_BATCH", 2)
    monkeypatch.setattr(bridge, "CANONICAL_INDEX_REQUEST_ROWS", 3)
    monkeypatch.setattr(bridge, "CANONICAL_INDEX_MAX_CHARS", 12)
    candidate_calls = []
    original = app.store.canonical_index_candidates

    def candidates(*args):
        candidate_calls.append(args)
        return original(*args)

    app.store.canonical_index_candidates = candidates
    calls = []

    def run(*args, **kwargs):
        batch = [json.loads(line) for line in Path(args[1]).read_text().splitlines()]
        calls.append((args, kwargs, batch))
        return 0, f"ingested sources={len(batch)} chunks=1 unchanged=0 stale=0 held=0 commit=rev\n", ""

    monkeypatch.setattr(bridge, "run", run)
    result = bridge.reconcile_canonical_index(app, profile_override=dict(fixed))
    assert result == {"attempted": fixed["request_rows"], "indexed": fixed["request_rows"], "held": 0, "durable": True}
    assert candidate_calls == [(128, fingerprint, "owner/canonical")]
    args, kwargs, batch = calls[0]
    assert args[-2:] == ("--memory", "owner/canonical")
    profile = kwargs["profile"]
    assert profile == {**bridge.index_embedding_profile(), "concurrency": fixed["concurrency"]}
    assert bridge.native_environment(profile=profile)["FUNES_VOYAGE_CONCURRENCY"] == str(fixed["concurrency"])
    assert len(batch) == fixed["request_rows"]
    assert bridge.CANONICAL_INDEX_BATCH == 2 and bridge.CANONICAL_INDEX_REQUEST_ROWS == 3
    assert bridge.CANONICAL_INDEX_MAX_CHARS == 12
    assert bridge.index_embedding_profile()["fingerprint"] == fingerprint
    # Only the pending suffix can be selected after the same checkpoint advances.
    bridge.reconcile_canonical_index(app, profile_override=dict(fixed))
    assert set(doc["source_identity"] for doc in calls[0][2]).isdisjoint(
        doc["source_identity"] for doc in calls[1][2]
    )


def test_override_enforces_chars_and_rejects_nonfixed_profile(app, monkeypatch):
    populate(app, 20, raw="x" * 9000)
    batches = []

    def run(*args, **kwargs):
        batch = [json.loads(line) for line in Path(args[1]).read_text().splitlines()]
        batches.append(batch)
        return 0, f"ingested sources={len(batch)} chunks=1 unchanged=0 stale=0 held=0 commit=rev", ""

    monkeypatch.setattr(bridge, "run", run)
    with pytest.raises(ValueError, match="invalid_canonical_ab_profile"):
        bridge.reconcile_canonical_index(app, profile_override={**FIXED_PROFILES[0], "concurrency": 32})
    assert not batches
    result = bridge.reconcile_canonical_index(app, profile_override=dict(FIXED_PROFILES[0]))
    assert result["attempted"] == 10
    assert len(batches[0]) == 10


def test_loop_dispatches_exactly_twelve_cycles_then_uses_unmodified_defaults(app, monkeypatch):
    app.canonical_index_stop = BoundedStop(13)
    profiles = []
    app.canonical_ab.start()

    def reconcile(_app, **kwargs):
        profile = kwargs.get("profile_override")
        profiles.append(profile)
        return {"attempted": 1, "indexed": 1, "held": 0, "durable": True}

    monkeypatch.setattr(bridge, "reconcile_canonical_index", reconcile)
    bridge._canonical_reconcile_background(app)
    assert [profile["name"] for profile in profiles[:-1]] == [
        fixed["name"] for fixed in FIXED_PROFILES for _ in range(4)
    ]
    assert profiles[-1] is None
    assert app.canonical_ab.phase == "completed"
    assert len(app.canonical_ab.status()["records"]) == 12


def test_baseline_finishing_after_start_is_not_attributed_or_reuses_metrics(app, monkeypatch):
    app.canonical_index_stop = BoundedStop(2)
    calls = []

    def reconcile(_app, **kwargs):
        calls.append(kwargs)
        if not kwargs:
            app.canonical_ab.start()
            bridge._set_canonical_reconcile_state(app, last_metrics={"voyage_requests": 999})
        return {"attempted": 1, "indexed": 1, "held": 0, "durable": True}

    monkeypatch.setattr(bridge, "reconcile_canonical_index", reconcile)
    bridge._canonical_reconcile_background(app)
    assert calls[0] == {} and "profile_override" in calls[1]
    status = app.canonical_ab.status()
    assert len(status["records"]) == 1
    assert status["records"][0]["metrics"]["requests"] == 0


@pytest.mark.parametrize("failure", ["native_failure", "invalid_report", "exception"])
def test_failed_native_cycle_cannot_pass_because_retry_checkpoint_is_durable(app, monkeypatch, failure):
    populate(app, 1)
    app.canonical_index_stop = BoundedStop(1)
    app.canonical_ab.start()

    def run(*args, **kwargs):
        if failure == "exception":
            raise OSError("provider credential and private raw document")
        if failure == "invalid_report":
            return 0, "malformed report private raw document", ""
        return 1, "", "request failed private credential"

    monkeypatch.setattr(bridge, "run", run)
    bridge._canonical_reconcile_background(app)
    status = app.canonical_ab.status()
    assert status["phase"] == "error"
    assert status["error_reason"] == "no_progress"
    assert status["records"][0]["durable"] is True
    assert status["records"][0]["attempted"] == 1
    assert status["records"][0]["indexed"] == 0
    assert app.store.get("pending-000")["native_index_status"] == "retry"
    assert "private" not in json.dumps(status) and "credential" not in json.dumps(status)


def test_exception_aborts_run_and_next_cycle_uses_defaults(app, monkeypatch):
    app.canonical_index_stop = BoundedStop(2)
    app.canonical_ab.start()
    calls = []

    def reconcile(_app, **kwargs):
        calls.append(kwargs)
        if kwargs:
            raise RuntimeError("postgres://secret raw provider output")
        return {"attempted": 1, "indexed": 1, "held": 0, "durable": True}

    monkeypatch.setattr(bridge, "reconcile_canonical_index", reconcile)
    bridge._canonical_reconcile_background(app)
    assert "profile_override" in calls[0] and calls[1] == {}
    assert app.canonical_ab.phase == "error"
    assert "secret" not in json.dumps(app.canonical_ab.status())


@pytest.mark.parametrize("finish_mode", ["abort", "timeout"])
def test_abort_and_deadline_drain_writer_then_restore_global_configuration(app, monkeypatch, finish_mode):
    now = [100.0]
    app.canonical_ab = CanonicalABController(clock=lambda: now[0])
    app.canonical_index_stop = BoundedStop(2)
    app.canonical_ab.start()
    calls = []

    def reconcile(_app, **kwargs):
        calls.append(kwargs)
        if kwargs:
            if finish_mode == "abort":
                app.canonical_ab.abort()
            else:
                now[0] += 1200
                assert app.canonical_ab.status()["error_reason"] == "timeout"
            # A restart cannot replace the in-flight token before it drains.
            assert bridge.canonical_ab_payload("start")[0] == 409
            assert app.canonical_ab.status()["in_flight_cycles"] == 1
        return {"attempted": 1, "indexed": 1, "held": 0, "durable": True}

    monkeypatch.setattr(bridge, "reconcile_canonical_index", reconcile)
    bridge._canonical_reconcile_background(app)
    assert "profile_override" in calls[0] and calls[1] == {}
    assert app.canonical_ab.phase == "aborted"
    assert len(app.canonical_ab.status()["records"]) == 1
    assert app.canonical_ab.status()["in_flight_cycles"] == 0
    assert app.canonical_ab.begin_cycle() is None


def test_profile_writer_keeps_native_lock_without_holding_source_write_lock(app, monkeypatch):
    populate(app, 1)
    entered = threading.Event()
    released = threading.Event()
    result = []
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        entered.set()
        assert released.wait(2)
        return 0, "ingested sources=1 chunks=1 unchanged=0 stale=0 held=0 commit=rev", ""

    monkeypatch.setattr(bridge, "run", run)
    profile = dict(FIXED_PROFILES[0])
    bridge.NATIVE_WRITE_LOCK.acquire()
    thread = threading.Thread(target=lambda: result.append(bridge.reconcile_canonical_index(app, profile_override=profile)))
    try:
        thread.start()
        assert not entered.wait(0.05)
    finally:
        bridge.NATIVE_WRITE_LOCK.release()
    try:
        assert entered.wait(1)
        assert not bridge.NATIVE_WRITE_LOCK.acquire(timeout=0.05)
        assert bridge.WRITE_LOCK.acquire(timeout=0.05)
        bridge.WRITE_LOCK.release()
    finally:
        released.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(calls) == 1 and result[0]["indexed"] == 1


def test_live_metric_mapping_is_allowlisted_and_records_real_phases(app, monkeypatch):
    app.canonical_index_stop = BoundedStop(1)
    app.canonical_ab.start()
    metrics = bridge._native_metrics()
    bridge._record_native_metrics(metrics, '\n'.join([
        'funes_metric {"stage":"remote_open","duration_ms":10.5,"secret":"private"}',
        'funes_metric {"stage":"embedding","duration_ms":70.5}',
        'funes_metric {"stage":"voyage_request","duration_ms":12,"status_code":0,"attempt":1,"input_count":2,"backoff_ms":20}',
        'funes_metric {"stage":"voyage_request","duration_ms":15,"status_code":429,"attempt":2,"input_count":2,"backoff_ms":40}',
        'funes_metric {"stage":"voyage_request","duration_ms":20,"status_code":503,"attempt":3,"input_count":2}',
        'funes_metric {"stage":"voyage_request","duration_ms":23,"status_code":200,"attempt":4,"input_count":2,"token_usage":123}',
        'funes_metric {"stage":"untrusted_phase","duration_ms":999}',
    ]))
    metrics["raw_text"] = "provider secret private source"

    def reconcile(_app, **kwargs):
        bridge._set_canonical_reconcile_state(app, last_metrics=metrics)
        return {"attempted": 2, "indexed": 2, "held": 0, "durable": True}

    monkeypatch.setattr(bridge, "reconcile_canonical_index", reconcile)
    bridge._canonical_reconcile_background(app)
    record = app.canonical_ab.status()["records"][0]
    assert record["metrics"]["http_429"] == 1
    assert record["metrics"]["http_5xx"] == 1
    assert record["metrics"]["transport_errors"] == 1
    assert record["metrics"]["requests"] == 4
    assert record["metrics"]["retries"] == 3
    assert record["metrics"]["tokens"] == 123
    assert record["metrics"]["request_duration_ms"] == 70
    assert record["metrics"]["backoff_ms"] == 60
    assert record["phase_ms"] == {"remote_open": 10.5, "embedding": 70.5}
    assert app.canonical_ab.phase == "error"
    assert "private" not in json.dumps(record)


def test_restart_initialization_forgets_override_without_touching_store(app):
    app.canonical_ab.start()
    app.canonical_ab.begin_cycle()
    bridge._initialize_canonical_reconcile_state(app)
    assert app.canonical_ab.phase == "idle"
    assert app.canonical_ab.begin_cycle() is None
    assert app.syncer.uploads == []
