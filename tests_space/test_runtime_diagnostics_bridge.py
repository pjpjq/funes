from types import SimpleNamespace

import pytest

import space.server as bridge
from service import runtime_diagnostics as diag


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_runtime_resources_keep_existing_auth_boundary(monkeypatch, method):
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "sync_status_payload", lambda: (200, {"ok": True}))
    monkeypatch.setattr(bridge, "hub_cache_status", lambda: {})
    calls = []
    monkeypatch.setattr(diag, "status", lambda: calls.append(True) or {"snapshot": {"pid": 1}})
    handler = object.__new__(bridge.Handler)
    replies = []
    handler.path = "/sync/status"
    handler.headers = {}
    handler.body = lambda: {}
    handler.send_json = lambda code, body: replies.append((code, body))
    getattr(handler, "do_" + method)()
    assert replies[-1][0] == 401 and calls == []
    handler.headers = {"Authorization": "Bearer test-token"}
    getattr(handler, "do_" + method)()
    assert replies[-1] == (200, {"ok": True, "runtime_resources": {"snapshot": {"pid": 1}}, "hub_cache": {}})
    handler.path, handler.headers = "/health", {}
    handler.do_GET()
    assert replies[-1] == (200, {"ok": True, "service": "funes"})
    assert len(calls) == 1


def test_native_run_contract_survives_diagnostic_failure(monkeypatch):
    process = SimpleNamespace(pid=123, returncode=7, communicate=lambda **kwargs: ("output", "stderr"))
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(diag, "_memory", lambda: (_ for _ in ()).throw(OSError("PRIVATE_TOKEN")))
    assert bridge.run("optimize-index", "PRIVATE_MEMORY") == (7, "output", "stderr")


def test_native_exit_event_has_operation_but_no_raw_arguments(monkeypatch):
    process = SimpleNamespace(pid=123, returncode=-9, communicate=lambda **kwargs: ("PRIVATE_RAW", "PRIVATE_STDERR"))
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(diag, "_memory", lambda: {"available": False})
    emitted = []
    monkeypatch.setattr(diag, "_emit", lambda value: emitted.append(value))
    assert bridge.run("optimize-index", "PRIVATE_MEMORY")[0] == -9
    assert emitted[-1]["event"] == "native_exit" and emitted[-1]["exit_code"] == -9
    assert emitted[-1]["operation"] == "optimize-index" and "PRIVATE" not in str(emitted)


def test_ingest_reexec_logs_intent_without_changing_exec(monkeypatch):
    calls = []
    monkeypatch.setattr(bridge.time, "sleep", lambda value: calls.append(("sleep", value)))
    monkeypatch.setattr(diag, "event", lambda kind, **kwargs: calls.append(("event", kind)))
    monkeypatch.setattr(bridge.os, "execv", lambda executable, argv: calls.append(("exec", executable, argv)))
    monkeypatch.setattr(bridge.threading, "Thread", lambda target, **kwargs: SimpleNamespace(start=target))
    bridge.schedule_ingest_process_restart()
    assert calls[:2] == [("sleep", 0.25), ("event", "ingest_restart")]
    assert calls[2] == ("exec", bridge.sys.executable, [bridge.sys.executable, *bridge.sys.argv])


def test_diagnostic_status_errors_are_not_business_errors(monkeypatch):
    monkeypatch.setattr(diag.copy, "deepcopy", lambda value: (_ for _ in ()).throw(ValueError("PRIVATE_TOKEN")))
    assert diag.status() == {"running": False, "error": "diagnostic_unavailable"}
