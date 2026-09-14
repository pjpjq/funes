import io
import json
import threading
import time
from http.client import RemoteDisconnected
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


class _HTTPResponse:
    def __init__(self, payload, status=200, headers=None, will_close=False):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self._body = io.BytesIO(body)
        self.status = status
        self.reason = "test response"
        self.headers = headers or {}
        self.will_close = will_close
        self.read_sizes = []

    def read(self, amount=-1):
        self.read_sizes.append(amount)
        return self._body.read(amount)


class _HTTPConnection:
    instances = []
    responses = []
    fail_first_request = False

    def __init__(self, host, port=None, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.requests = []
        self.tunnel = None
        self.closed = False
        type(self).instances.append(self)

    def set_tunnel(self, host, port=None, headers=None):
        self.tunnel = (host, port, headers)

    def request(self, method, target, body=None, headers=None):
        self.requests.append((method, target, body, headers or {}))
        if type(self).fail_first_request and len(type(self).instances) == 1:
            raise RemoteDisconnected("server closed the connection")

    def getresponse(self):
        return type(self).responses.pop(0)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def clear_connection_pool():
    bridge._close_connections()
    _HTTPConnection.instances = []
    _HTTPConnection.responses = []
    _HTTPConnection.fail_first_request = False
    yield
    bridge._close_connections()


def test_remote_search_waits_for_ready_and_retries_transient(monkeypatch):
    monkeypatch.setenv("FUNES_REMOTE_URL", "https://memory.example")
    monkeypatch.setenv("FUNES_API_TOKEN", "api-token")
    monkeypatch.setenv("FUNES_HF_TOKEN", "hub-token")
    monkeypatch.setenv("FUNES_REMOTE_TIMEOUT", "12")
    monkeypatch.setenv("FUNES_REMOTE_READY_TIMEOUT", "1")
    monkeypatch.setenv("FUNES_REMOTE_READY_POLLS", "2")
    monkeypatch.setenv("FUNES_REMOTE_ATTEMPTS", "2")
    monkeypatch.setattr(bridge.time, "sleep", lambda _seconds: None)
    calls = []
    ready_states = iter(("warming", "ready"))

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, req.method, timeout))
        if req.full_url.endswith("/ready/search"):
            return _Response(
                {"ok": True, "native_warm": {"state": next(ready_states)}}
            )
        if len([call for call in calls if call[1] == "POST"]) == 1:
            raise OSError("connection reset")
        return _Response({"ok": True, "results": [{"raw_text": "原始中文"}]})

    monkeypatch.setattr(bridge, "_open_remote", fake_urlopen)
    result = bridge._remote_call("/search", {"query": "之前的决定"})

    assert result["ok"] is True
    assert [method for _, method, _ in calls] == ["GET", "GET", "POST", "POST"]
    assert all(timeout <= 50 for _, _, timeout in calls)


def test_remote_search_polls_retryable_ready_503_until_ready(monkeypatch):
    monkeypatch.setenv("FUNES_REMOTE_URL", "https://memory.example")
    monkeypatch.setenv("FUNES_API_TOKEN", "api-token")
    monkeypatch.setenv("FUNES_REMOTE_READY_TIMEOUT", "1")
    monkeypatch.setenv("FUNES_REMOTE_READY_POLLS", "2")
    monkeypatch.setattr(bridge.time, "sleep", lambda _seconds: None)
    calls = []
    ready_attempt = 0

    def fake_urlopen(req, timeout):
        nonlocal ready_attempt
        calls.append((req.full_url, req.method, timeout))
        if req.full_url.endswith("/ready/search"):
            ready_attempt += 1
            if ready_attempt == 1:
                body = io.BytesIO(
                    json.dumps({"native_warm": {"state": "warming"}}).encode()
                )
                raise HTTPError(req.full_url, 503, "not ready", {}, body)
            return _Response({"ok": True, "native_warm": {"state": "ready"}})
        return _Response({"ok": True, "results": []})

    monkeypatch.setattr(bridge, "_open_remote", fake_urlopen)
    result = bridge._remote_call("/search", {"query": "previous decision"})

    assert result == {"ok": True, "results": []}
    assert [method for _, method, _ in calls] == ["GET", "GET", "POST"]


def test_remote_search_does_not_poll_while_source_store_restores(monkeypatch):
    monkeypatch.setenv("FUNES_REMOTE_URL", "https://memory.example")
    monkeypatch.setenv("FUNES_API_TOKEN", "api-token")
    calls = []

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, req.method, timeout))
        if req.full_url.endswith("/ready/search"):
            return _Response(
                {
                    "ok": True,
                    "native_warm": {"state": "ready"},
                    "source_store": {
                        "configured": True,
                        "ready": False,
                        "restoring": True,
                    },
                }
            )
        return _Response({"ok": True, "results": []})

    monkeypatch.setattr(bridge, "_open_remote", fake_urlopen)

    assert bridge._remote_call("/search", {"query": "previous"}) == {
        "ok": True,
        "results": [],
    }
    assert [url for url, _, _ in calls] == [
        "https://memory.example/ready/search",
        "https://memory.example/search",
    ]


def test_remote_search_default_budget_is_one_probe_and_one_attempt(monkeypatch):
    monkeypatch.setenv("FUNES_REMOTE_URL", "https://memory.example")
    monkeypatch.setenv("FUNES_API_TOKEN", "api-token")
    for name in (
        "FUNES_REMOTE_TIMEOUT",
        "FUNES_REMOTE_ATTEMPTS",
        "FUNES_REMOTE_ATTEMPT_TIMEOUT",
        "FUNES_REMOTE_READY_TIMEOUT",
        "FUNES_REMOTE_READY_POLLS",
    ):
        monkeypatch.delenv(name, raising=False)
    calls = []

    def unavailable(req, timeout):
        calls.append((req.full_url, req.method, timeout))
        raise OSError("unavailable")

    monkeypatch.setattr(bridge, "_open_remote", unavailable)
    monkeypatch.setattr(bridge.time, "sleep", lambda _seconds: None)

    assert bridge._remote_call("/search", {"query": "previous"}) is None
    assert [(url, method) for url, method, _ in calls] == [
        ("https://memory.example/ready/search", "GET"),
        ("https://memory.example/search", "POST"),
    ]
    assert all(0 < timeout <= 4 for _, _, timeout in calls)


def test_mcp_recall_returns_null_instead_of_json_rpc_error_when_unavailable(
    monkeypatch,
):
    monkeypatch.setenv("FUNES_REMOTE_URL", "https://memory.example")
    monkeypatch.setenv("FUNES_API_TOKEN", "api-token")
    for name in (
        "FUNES_REMOTE_TIMEOUT",
        "FUNES_REMOTE_ATTEMPTS",
        "FUNES_REMOTE_ATTEMPT_TIMEOUT",
        "FUNES_REMOTE_READY_TIMEOUT",
        "FUNES_REMOTE_READY_POLLS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        bridge, "_open_remote", lambda _req, timeout: (_ for _ in ()).throw(OSError())
    )
    monkeypatch.setattr(bridge.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        bridge.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "recall",
                        "arguments": {"query": "previous decision"},
                    },
                }
            )
            + "\n"
        ),
    )
    stdout = io.StringIO()
    monkeypatch.setattr(bridge.sys, "stdout", stdout)

    bridge.serve(SimpleNamespace())

    response = json.loads(stdout.getvalue())
    assert "error" not in response
    assert response["result"]["content"][0]["text"] == "null"


def test_remote_call_does_not_retry_auth_failure(monkeypatch):
    monkeypatch.setenv("FUNES_REMOTE_URL", "https://memory.example")
    monkeypatch.setenv("FUNES_API_TOKEN", "api-token")
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        raise HTTPError(req.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(bridge, "_open_remote", fake_urlopen)
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

    monkeypatch.setattr(bridge, "_open_remote", fake_urlopen)
    assert bridge._remote_call("/sync/status", {})["ok"] is True
    assert calls == ["https://configured-memory.example/sync/status"]


def test_ready_search_and_consecutive_calls_reuse_one_connection(monkeypatch):
    monkeypatch.setenv("FUNES_REMOTE_URL", "https://memory.example")
    monkeypatch.setenv("FUNES_API_TOKEN", "api-token")
    monkeypatch.setattr(bridge.request, "getproxies", lambda: {})
    monkeypatch.setattr(bridge.request, "proxy_bypass", lambda _host: False)
    monkeypatch.setattr(bridge.http.client, "HTTPSConnection", _HTTPConnection)
    _HTTPConnection.responses = [
        _HTTPResponse({"ok": True}),
        _HTTPResponse({"ok": True, "results": [1]}),
        _HTTPResponse({"ok": True}),
        _HTTPResponse({"ok": True, "results": [2]}),
    ]

    first = bridge._remote_call("/search", {"query": "first"})
    second = bridge._remote_call("/search", {"query": "second"})

    assert first["results"] == [1]
    assert second["results"] == [2]
    assert len(_HTTPConnection.instances) == 1
    assert [item[1] for item in _HTTPConnection.instances[0].requests] == [
        "/ready/search",
        "/search",
        "/ready/search",
        "/search",
    ]


def test_https_http_proxy_reuses_tunnel_without_auth_on_connect(monkeypatch):
    monkeypatch.setenv("FUNES_REMOTE_URL", "https://memory.example")
    monkeypatch.setenv("FUNES_API_TOKEN", "app-secret")
    monkeypatch.setenv("FUNES_HF_TOKEN", "hub-secret")
    monkeypatch.setattr(
        bridge.request,
        "getproxies",
        lambda: {"https": "http://127.0.0.1:6324"},
    )
    monkeypatch.setattr(bridge.request, "proxy_bypass", lambda _host: False)
    monkeypatch.setattr(bridge.http.client, "HTTPSConnection", _HTTPConnection)
    _HTTPConnection.responses = [
        _HTTPResponse({"ok": True}),
        _HTTPResponse({"ok": True, "results": []}),
        _HTTPResponse({"ok": True}),
    ]

    assert bridge._remote_call("/search", {"query": "safe"})["ok"] is True
    assert bridge._remote_call("/get", {"source_identity": "one"})["ok"] is True

    connection = _HTTPConnection.instances[0]
    assert len(_HTTPConnection.instances) == 1
    assert (connection.host, connection.port) == ("127.0.0.1", 6324)
    assert connection.tunnel == ("memory.example", 443, None)
    request_headers = {
        key.lower(): value for key, value in connection.requests[-1][3].items()
    }
    assert request_headers["authorization"] == "Bearer hub-secret"
    assert request_headers["x-funes-authorization"] == "Bearer app-secret"
    assert "app-secret" not in repr(list(bridge._CONNECTIONS))
    assert "hub-secret" not in repr(list(bridge._CONNECTIONS))


def test_remote_disconnect_discards_connection_and_existing_retry_rebuilds(monkeypatch):
    monkeypatch.setenv("FUNES_REMOTE_URL", "https://memory.example")
    monkeypatch.setenv("FUNES_API_TOKEN", "api-token")
    monkeypatch.setenv("FUNES_REMOTE_ATTEMPTS", "2")
    monkeypatch.setattr(bridge.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(bridge.request, "getproxies", lambda: {})
    monkeypatch.setattr(bridge.request, "proxy_bypass", lambda _host: False)
    monkeypatch.setattr(bridge.http.client, "HTTPSConnection", _HTTPConnection)
    _HTTPConnection.fail_first_request = True
    _HTTPConnection.responses = [_HTTPResponse({"ok": True})]

    assert bridge._remote_call("/get", {"source_identity": "one"}) == {"ok": True}
    assert len(_HTTPConnection.instances) == 2
    assert _HTTPConnection.instances[0].closed is True


def test_connection_access_is_serialized_across_threads(monkeypatch):
    class SerialConnection(_HTTPConnection):
        active = 0
        max_active = 0
        metric_lock = threading.Lock()

        def request(self, method, target, body=None, headers=None):
            with self.metric_lock:
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
            time.sleep(0.01)
            try:
                super().request(method, target, body, headers)
            finally:
                with self.metric_lock:
                    type(self).active -= 1

    SerialConnection.instances = []
    SerialConnection.responses = [
        _HTTPResponse({"worker": 1}),
        _HTTPResponse({"worker": 2}),
    ]
    monkeypatch.setattr(bridge.request, "getproxies", lambda: {})
    monkeypatch.setattr(bridge.request, "proxy_bypass", lambda _host: False)
    monkeypatch.setattr(bridge.http.client, "HTTPSConnection", SerialConnection)
    barrier = threading.Barrier(2)
    results = []
    failures = []

    def invoke(worker):
        try:
            barrier.wait()
            req = bridge.request.Request(
                f"https://memory.example/get?worker={worker}", method="GET"
            )
            with bridge._open_remote(req, timeout=1) as response:
                results.append(json.loads(response.read()))
        except Exception as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    threads = [threading.Thread(target=invoke, args=(worker,)) for worker in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert failures == []
    assert len(results) == 2
    assert len(SerialConnection.instances) == 1
    assert SerialConnection.max_active == 1


def test_persistent_error_body_is_bounded_and_connection_is_discarded(monkeypatch):
    monkeypatch.setattr(bridge.request, "getproxies", lambda: {})
    monkeypatch.setattr(bridge.request, "proxy_bypass", lambda _host: False)
    monkeypatch.setattr(bridge.http.client, "HTTPSConnection", _HTTPConnection)
    response = _HTTPResponse(b"x" * (bridge._ERROR_BODY_MAX + 100), status=503)
    _HTTPConnection.responses = [response]
    req = bridge.request.Request("https://memory.example/ready", method="GET")

    with pytest.raises(HTTPError):
        bridge._open_remote(req, timeout=1)

    assert response.read_sizes == [bridge._ERROR_BODY_MAX + 1]
    assert _HTTPConnection.instances[0].closed is True


@pytest.mark.parametrize(
    "proxy_url",
    ("https://proxy.example:443", "http://user:password@proxy.example:8080"),
)
def test_complex_proxy_falls_back_to_existing_no_redirect(monkeypatch, proxy_url):
    monkeypatch.setattr(
        bridge.request, "getproxies", lambda: {"https": proxy_url}
    )
    monkeypatch.setattr(bridge.request, "proxy_bypass", lambda _host: False)
    calls = []

    def fallback(req, timeout):
        calls.append((req.full_url, timeout))
        return _Response({"ok": True})

    monkeypatch.setattr(bridge, "open_no_redirect", fallback)
    req = bridge.request.Request("https://memory.example/get", method="GET")

    with bridge._open_remote(req, timeout=3) as response:
        assert json.loads(response.read())["ok"] is True
    assert calls == [("https://memory.example/get", 3)]
    assert _HTTPConnection.instances == []


@pytest.mark.parametrize(
    "url",
    ("ftp://memory.example", "https://user:secret@memory.example"),
)
def test_remote_url_rejects_non_http_and_userinfo(monkeypatch, url):
    monkeypatch.setenv("FUNES_REMOTE_URL", url)
    monkeypatch.setenv("FUNES_API_TOKEN", "api-token")

    with pytest.raises(ValueError):
        bridge._remote_call("/get", {})
    assert bridge._CONNECTIONS == {}


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
