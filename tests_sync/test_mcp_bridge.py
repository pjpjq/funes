import json
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

    monkeypatch.setattr(bridge.request, "urlopen", fake_urlopen)
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

    monkeypatch.setattr(bridge.request, "urlopen", fake_urlopen)
    with pytest.raises(HTTPError):
        bridge._remote_call("/get", {"id": "record-1"})
    assert calls == ["https://memory.example/get"]
