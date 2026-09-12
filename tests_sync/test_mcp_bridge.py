import io
import json
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

import sync.mcp_bridge as bridge


class _Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_remote_search_waits_for_ready_and_retries_transient(monkeypatch):
    monkeypatch.setenv("FUNES_REMOTE_URL", "https://memory.example")
    monkeypatch.setenv("FUNES_API_TOKEN", "api-token")
    monkeypatch.setenv("FUNES_HF_TOKEN", "hub-token")
    monkeypatch.setenv("FUNES_REMOTE_TIMEOUT", "12")
    monkeypatch.setenv("FUNES_REMOTE_READY_TIMEOUT", "1")
    monkeypatch.setenv("FUNES_REMOTE_ATTEMPTS", "2")
    monkeypatch.setattr(bridge.time, "sleep", lambda _seconds: None)
    calls = []
    ready_states = iter(("warming", "ready"))

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, req.method, timeout))
        if req.full_url.endswith("/ready"):
            return _Response({"native_warm": {"state": next(ready_states)}})
        if len([call for call in calls if call[0].endswith("/search")]) == 1:
            raise OSError("connection reset")
        return _Response({"ok": True, "results": [{"raw_text": "原始中文"}]})

    monkeypatch.setattr(bridge, "open_no_redirect", fake_urlopen)
    result = bridge._remote_call("/search", {"query": "之前的决定"})

    assert result["ok"] is True
    assert [method for _, method, _ in calls] == ["GET", "GET", "POST", "POST"]
    assert all(timeout <= 50 for _, _, timeout in calls)


def test_remote_call_does_not_retry_auth_failure(monkeypatch):
    monkeypatch.setenv("FUNES_REMOTE_URL", "https://memory.example")
    monkeypatch.setenv("FUNES_API_TOKEN", "api-token")
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        raise HTTPError(req.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(bridge, "open_no_redirect", fake_urlopen)
    with pytest.raises(HTTPError):
        bridge._remote_call("/get", {"id": "record-1"})
    assert calls == ["https://memory.example/get"]


def test_remote_call_reads_url_from_config(monkeypatch):
    monkeypatch.delenv("FUNES_REMOTE_URL", raising=False)
    monkeypatch.setenv("FUNES_API_TOKEN", "api-token")
    monkeypatch.setattr(
        bridge.Config,
        "load",
        lambda: SimpleNamespace(remote_url="https://configured-memory.example"),
    )
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        return _Response({"ok": True})

    monkeypatch.setattr(bridge, "open_no_redirect", fake_urlopen)
    assert bridge._remote_call("/sync/status", {})["ok"] is True
    assert calls == ["https://configured-memory.example/sync/status"]


def test_mcp_get_uses_source_identity_and_recall_exposes_filters(monkeypatch):
    monkeypatch.setenv("FUNES_REMOTE_URL", "https://memory.example")
    monkeypatch.setenv("FUNES_API_TOKEN", "api-token")
    calls = []

    def fake_remote(path, payload):
        calls.append((path, payload))
        return {"ok": True}

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "get", "arguments": {"record_id": "memory-1"}},
        },
    ]
    stdin = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
    stdout = io.StringIO()
    monkeypatch.setattr(bridge, "_remote_call", fake_remote)
    monkeypatch.setattr(bridge.sys, "stdin", stdin)
    monkeypatch.setattr(bridge.sys, "stdout", stdout)

    bridge.serve(SimpleNamespace())

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    recall = responses[0]["result"]["tools"][0]
    assert {
        "source_agent",
        "source_type",
        "project",
        "repo",
        "device_id",
        "content_type",
        "since",
        "until",
    } <= set(recall["inputSchema"]["properties"])
    assert calls == [("/get", {"source_identity": "memory-1"})]
