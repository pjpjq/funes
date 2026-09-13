import gzip
import json
import threading
import time
from collections import deque
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import space.server as bridge
from service.server import persist_translation_documents
from service.server import Store as SourceStore


class _FakeStdout:
    def __init__(self):
        self.lines = deque()
        self.closed = False

    def fileno(self):
        raise OSError("in-memory fake")

    def readline(self):
        if self.lines:
            return self.lines.popleft() + "\n"
        return ""

    def push(self, value):
        self.lines.append(json.dumps(value))

    def close(self):
        self.closed = True


class _FakeStdin:
    def __init__(self, stdout, responder, messages):
        self.stdout = stdout
        self.responder = responder
        self.messages = messages
        self.closed = False

    def write(self, value):
        message = json.loads(value)
        self.messages.append(message)
        self.responder(message, self.stdout)

    def flush(self):
        return None

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self, responder):
        self.stdout = _FakeStdout()
        self.messages = []
        self.stdin = _FakeStdin(self.stdout, responder, self.messages)
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def _mcp_responder(message, stdout):
    if message.get("method") == "initialize":
        stdout.push({"jsonrpc": "2.0", "id": message["id"], "result": {"protocolVersion": bridge.MCP_PROTOCOL_VERSION}})
    elif message.get("method") == "tools/call":
        arguments = message.get("params", {}).get("arguments", {})
        name = message.get("params", {}).get("name")
        if name == "recall":
            text = "recall:" + arguments.get("query", "")
        else:
            text = "get:" + arguments.get("session_id", "")
        stdout.push({"jsonrpc": "2.0", "id": message["id"], "result": {"content": [{"type": "text", "text": text}]}})


def test_chinese_query_uses_ascii_retrieval_shadow(monkeypatch):
    monkeypatch.setattr(bridge, "LANGUAGE_MODE", "auto")
    query = bridge.query_text("CPA 第二轮为什么丢上下文？")
    assert "CPA" in query
    assert "second turn" in query
    assert "context loss" in query
    assert not any("\u4e00" <= char <= "\u9fff" for char in query)


def test_mixed_english_query_is_left_unchanged(monkeypatch):
    monkeypatch.setattr(bridge, "LANGUAGE_MODE", "auto")
    query = "Codex previous_response_id context loss"
    assert bridge.query_text(query) == query


def test_query_text_is_deterministic_and_never_calls_provider(monkeypatch):
    monkeypatch.setattr(bridge, "LANGUAGE_MODE", "translate")
    monkeypatch.setenv("TRANSLATION_API_KEY", "test-key")
    monkeypatch.setenv("TRANSLATION_MODEL", "test-model")
    monkeypatch.setenv("TRANSLATION_BASE_URL", "https://provider.example")
    monkeypatch.setattr(
        bridge.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("deterministic fallback must not call the provider")
        ),
    )
    assert bridge.query_text("中文 CPA 问题") == "CPA"


def test_native_mcp_worker_reuses_child_and_passes_hf_environment(monkeypatch, tmp_path):
    processes = []

    def fake_popen(args, **kwargs):
        assert args == ["fake-funes", "mcp", "owner/memory"]
        assert kwargs["env"]["FUNES_HOME"] == str(tmp_path)
        assert kwargs["env"]["HF_TOKEN"] == "not-a-real-token"
        process = _FakeProcess(_mcp_responder)
        processes.append(process)
        return process

    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    monkeypatch.setattr(bridge.subprocess, "Popen", fake_popen)
    worker = bridge.NativeMcpWorker("fake-funes", "owner/memory", tmp_path, timeout=1, handshake_timeout=1)
    try:
        assert worker.recall("ascii query", k=2) == "recall:ascii query"
        assert worker.get("session-1", from_=3, to=4) == "get:session-1"
    finally:
        worker.close()

    assert len(processes) == 1
    methods = [message["method"] for message in processes[0].messages]
    assert methods[:2] == ["initialize", "notifications/initialized"]
    assert methods[2:] == ["tools/call", "tools/call"]
    assert processes[0].messages[2]["params"]["name"] == "recall"
    assert processes[0].messages[3]["params"]["name"] == "get"


def test_native_mcp_worker_restarts_after_eof(monkeypatch, tmp_path):
    processes = []

    def first_responder(message, stdout):
        if message.get("method") == "initialize":
            _mcp_responder(message, stdout)
        elif message.get("method") == "tools/call":
            # Simulate a child dying after accepting a call.  The worker must
            # discard it and retry the read-only operation in a fresh child.
            stdout.closed = True

    def fake_popen(args, **kwargs):
        process = _FakeProcess(first_responder if not processes else _mcp_responder)
        processes.append(process)
        return process

    monkeypatch.setattr(bridge.subprocess, "Popen", fake_popen)
    worker = bridge.NativeMcpWorker("fake-funes", "owner/memory", tmp_path, timeout=1, handshake_timeout=1)
    try:
        assert worker.recall("retry me") == "recall:retry me"
    finally:
        worker.close()

    assert len(processes) == 2
    assert processes[0].terminated is True


def test_native_mcp_timeout_is_not_retried_with_a_second_full_budget(monkeypatch, tmp_path):
    worker = bridge.NativeMcpWorker(
        "fake-funes", "owner/memory", tmp_path, timeout=180, handshake_timeout=10
    )
    request_timeouts = []
    monkeypatch.setattr(worker, "_ensure_started_locked", lambda _deadline=None: None)

    def timeout_request(_method, _params, timeout):
        request_timeouts.append(timeout)
        raise bridge.NativeMcpTimeoutError("timed out")

    monkeypatch.setattr(worker, "_request_locked", timeout_request)
    monkeypatch.setattr(worker, "_stop_locked", lambda *_args: None)
    with pytest.raises(bridge.NativeMcpTimeoutError):
        worker.recall("slow query", timeout=12)
    assert len(request_timeouts) == 1
    assert 0 < request_timeouts[0] <= 12


def test_native_lock_wait_is_deducted_from_call_budget(monkeypatch):
    clock = [100.0]
    acquired = []
    released = []

    class Lock:
        def acquire(self, timeout):
            acquired.append(timeout)
            clock[0] += 0.04
            return True

        def release(self):
            released.append(True)

    remaining = []
    monkeypatch.setattr(bridge, "INDEX_LOCK", Lock())
    monkeypatch.setattr(bridge, "RECALL_LOCK_TIMEOUT", 2.0)
    monkeypatch.setattr(bridge.time, "monotonic", lambda: clock[0])
    bridge._locked_native_call(lambda timeout: remaining.append(timeout), 0.05)
    assert 0 < acquired[0] <= 0.05
    assert 0 < remaining[0] <= 0.011
    assert released == [True]


def test_timeout_cleanup_kills_without_synchronous_wait(monkeypatch, tmp_path):
    waits = []

    class Process:
        stdin = None
        stdout = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def kill(self):
            return None

        def wait(self, timeout=None):
            waits.append(timeout)
            return -9

    class Thread:
        def __init__(self, *, target, args, **_kwargs):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    worker = bridge.NativeMcpWorker("fake-funes", "owner/memory", tmp_path)
    worker._process = Process()
    monkeypatch.setattr(bridge.threading, "Thread", Thread)
    worker._stop_locked(deadline=100.0)
    assert waits == [None]


def test_materialize_native_session_uses_remaining_deadline(monkeypatch):
    calls = []
    app = SimpleNamespace(store=SimpleNamespace(get=lambda _identity: None))
    monkeypatch.setattr(bridge.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        bridge,
        "get",
        lambda identity, **kwargs: calls.append((identity, kwargs)) or "raw session",
    )
    results = bridge.materialize_native_results(
        "hit\n  → get session-1 --from 0 --to 0",
        app,
        1,
        deadline=112.0,
    )
    assert results[0]["raw_text"] == "raw session"
    assert calls == [("session-1", {"timeout": 12.0})]


def test_request_warm_reserves_state_before_start(monkeypatch):
    starts = []

    class FakeThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            starts.append(self.kwargs)

    monkeypatch.setattr(bridge.threading, "Thread", FakeThread)
    with bridge._WARM_STATE_LOCK:
        bridge._WARM_STATE.update(state="not_started", started_at=None, finished_at=None, refresh_pending=False)

    first = bridge.request_warm()
    second = bridge.request_warm(force=True)

    assert first["state"] == "warming"
    assert second["state"] == "warming"
    assert second["refresh_pending"] is True
    assert len(starts) == 1


def test_initial_warm_does_not_wait_on_recall_lock(monkeypatch):
    called = threading.Event()

    def fake_refresh():
        called.set()

    monkeypatch.setattr(bridge, "_refresh_native_worker", fake_refresh)
    with bridge._WARM_STATE_LOCK:
        bridge._WARM_STATE.update(state="not_started", started_at=None, finished_at=None, refresh_pending=False)
    bridge.INDEX_LOCK.acquire()
    thread = threading.Thread(target=bridge._warm_native_memory, kwargs={"replace": False})
    thread.start()
    try:
        assert called.wait(0.25), "initial warm must not hold INDEX_LOCK while loading native recall"
    finally:
        bridge.INDEX_LOCK.release()
        thread.join(timeout=1)
    assert not thread.is_alive()


def test_native_worker_does_not_spawn_during_initial_warm(monkeypatch):
    monkeypatch.setattr(bridge, "MCP_WORKER", None)
    monkeypatch.setattr(bridge, "_MCP_WORKER_CONFIG", None)
    with bridge._WARM_STATE_LOCK:
        bridge._WARM_STATE.update(state="warming", started_at="now", finished_at=None, refresh_pending=False)
    with pytest.raises(bridge.NativeMcpError, match="warming"):
        bridge.native_worker()


def test_recall_fails_fast_when_another_read_is_active(monkeypatch):
    monkeypatch.setattr(bridge, "RECALL_LOCK_TIMEOUT", 0.01)
    bridge.INDEX_LOCK.acquire()
    try:
        with pytest.raises(bridge.NativeMcpBusyError, match="busy"):
            bridge.recall("query")
    finally:
        bridge.INDEX_LOCK.release()


def test_search_and_get_use_native_worker_and_keep_raw_query(monkeypatch):
    calls = []

    class FakeWorker:
        def recall(self, query, **kwargs):
            calls.append(("recall", query, kwargs))
            return "english retrieval snippet\n  → get session-1 --from 2"

        def get(self, session_id, **kwargs):
            calls.append(("get", session_id, kwargs))
            return "verbatim native session"

    monkeypatch.setattr(bridge, "MCP_WORKER", FakeWorker())
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "LANGUAGE_MODE", "auto")
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection(*server.server_address)
        conn.request(
            "POST",
            "/search",
            json.dumps({"query": "第二轮为什么丢上下文？", "limit": 3}, ensure_ascii=False).encode(),
            {"Authorization": "Bearer test-token", "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        search = json.loads(response.read())
        conn.close()
        assert response.status == 200
        assert search["query"] == "第二轮为什么丢上下文？"
        assert search["results_text"] == "verbatim native session"
        assert "english retrieval snippet" not in search["results_text"]
        assert "第二轮" not in calls[0][1]

        conn = HTTPConnection(*server.server_address)
        conn.request(
            "POST",
            "/get",
            json.dumps({"session_id": "session-1", "from": 2}).encode(),
            {"Authorization": "Bearer test-token", "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        result = json.loads(response.read())
        conn.close()
        assert response.status == 200
        assert result["result"] == "verbatim native session"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert calls[1][0:2] == ("get", "session-1")
    assert 0 < calls[1][2]["timeout"] <= bridge.HTTP_NATIVE_TIMEOUT
    assert calls[2][0:2] == ("get", "session-1")
    assert calls[2][2]["from_"] == 2
    assert calls[2][2]["to"] is None
    assert 0 < calls[2][2]["timeout"] <= bridge.HTTP_NATIVE_TIMEOUT


def _request(server, payload, token="test-token"):
    conn = HTTPConnection(*server.server_address)
    conn.request(
        "POST",
        "/ingest",
        json.dumps(payload, ensure_ascii=False).encode(),
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    response = conn.getresponse()
    body = json.loads(response.read())
    conn.close()
    return response.status, body


def test_http_ingest_requires_sidecar_and_never_runs_native(monkeypatch, tmp_path):
    calls = []
    warm_calls = []
    monkeypatch.setattr(bridge, "HOME", tmp_path)
    (tmp_path / "sources").mkdir()
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "request_warm", lambda **kwargs: warm_calls.append(kwargs))

    def fake_run(*args, **kwargs):
        calls.append(args)
        return 0, "native output", ""

    monkeypatch.setattr(bridge, "run", fake_run)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(server, {"documents": [{"source_identity": "s1", "raw_text": "原始中文"}]})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert status == 503
    assert body["durable"] is False
    assert body["error"] == "FUNES_STORAGE_REPO is not configured"
    assert calls == []
    assert warm_calls == []


def test_ingest_does_not_wait_for_recall_lock(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "HOME", tmp_path)
    (tmp_path / "sources").mkdir()
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("request must not run native")),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    bridge.INDEX_LOCK.acquire()
    result = {}
    done = threading.Event()

    def submit():
        try:
            result["value"] = _request(server, {"raw_text": "写入但不阻塞读取"})
        finally:
            done.set()

    request_thread = threading.Thread(target=submit)
    request_thread.start()
    try:
        assert done.wait(0.25), "ingest must use WRITE_LOCK, not the recall lock"
    finally:
        bridge.INDEX_LOCK.release()
        request_thread.join(timeout=2)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        app.store.close()
    assert result["value"][0] == 200


def test_remote_without_source_sidecar_is_not_accepted(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "HOME", tmp_path)
    (tmp_path / "sources").mkdir()
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "TOKEN", "test-token")

    def fake_run(*args, **kwargs):
        if args and args[0] == "push":
            return 2, "", "blocked"
        return 0, "", ""

    monkeypatch.setattr(bridge, "run", fake_run)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(server, {"raw_text": "待上传"})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert status == 503
    assert body["durable"] is False
    assert body["error"] == "FUNES_STORAGE_REPO is not configured"


def test_sync_status_alias_returns_ready_payload(monkeypatch):
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")

    def fake_run(*args, **kwargs):
        assert args[:2] == ("status", "owner/memory")
        return 0, "chunks: 12\n", ""

    monkeypatch.setattr(bridge, "run", fake_run)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection(*server.server_address)
        conn.request(
            "POST",
            "/sync/status",
            b"{}",
            {"Authorization": "Bearer test-token", "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read())
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert response.status == 200
    assert body["ok"] is True
    assert body["remote"] == "owner/memory"
    assert "chunks: 12" in body["status"]


def test_sync_checkpoint_acknowledges_existing_durable_push(monkeypatch):
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    calls = []
    monkeypatch.setattr(bridge, "run", lambda *args, **kwargs: calls.append(args))
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection(*server.server_address)
        conn.request(
            "POST",
            "/sync",
            b"{}",
            {"Authorization": "Bearer test-token", "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read())
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert response.status == 200
    assert body == {"ok": True, "durable": True, "remote": "owner/memory"}
    assert calls == []


def test_ingest_preserves_source_metadata_and_timestamp(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "HOME", tmp_path)
    (tmp_path / "sources").mkdir()
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "TOKEN", "test-token")

    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("request must not run native")),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    document = {
        "source_identity": "stable-memory",
        "source_agent": "codex",
        "source_type": "agents_md",
        "device_id": "device-safe-hash",
        "source_path": "~/code/project/AGENTS.md",
        "content_hash": "content-hash",
        "timestamp": "2026-09-01T01:02:03Z",
        "updated_at": "2026-09-01T02:03:04Z",
        "raw_text": "原始中文",
    }
    try:
        status, body = _request(server, {"documents": [document]})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        stored = app.store.get("stable-memory")
        app.store.close()
    assert status == 200
    assert body["durable"] is True
    assert stored["timestamp"] == document["timestamp"]
    assert stored["source_agent"] == "codex"
    assert stored["source_type"] == "agents_md"
    assert stored["device_id"] == "device-safe-hash"
    assert stored["source_path"] == "~/code/project/AGENTS.md"
    assert stored["updated_at"] == "2026-09-01T02:03:04Z"


class _SourceTranslator:
    model = "test-model"
    max_per_ingest = 16

    def pending_document(self, raw):
        return raw, "translation-hash", "translation-version", "pending_provider"

    def normalize_many(self, raws):
        return [
            (
                "english retrieval needle",
                "translation-hash",
                "translation-version",
                "ok",
            )
            for _raw in raws
        ]

    def rewrite_query(self, query):
        return query


class _SourceSyncer:
    restoring = False
    restore_failed = False

    def __init__(self, durable=True):
        self.durable = durable
        self.uploads = []
        self.controls = []

    def upload(self, docs=None):
        self.uploads.append(docs)
        return {"uploaded": self.durable, "durable": self.durable}

    def upload_reindex_control(self, control):
        self.controls.append(dict(control))
        return {"uploaded": self.durable, "durable": self.durable}


def _source_app(tmp_path, durable=True):
    store = SourceStore(str(tmp_path / "source-store"))
    return SimpleNamespace(
        store=store,
        translator=_SourceTranslator(),
        syncer=_SourceSyncer(durable),
        restore_result=0,
    )


def _post(server, path, payload, token="test-token"):
    conn = HTTPConnection(*server.server_address)
    conn.request(
        "POST",
        path,
        json.dumps(payload, ensure_ascii=False).encode(),
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    response = conn.getresponse()
    body = json.loads(response.read())
    conn.close()
    return response.status, body


def test_sources_check_requires_auth_and_returns_no_raw_payload(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    secret = "raw-secret-must-not-leak"
    app.store.ingest([{"source_identity": "present", "raw_text": secret}])
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    payload = {"source_identities": ["missing", "present", "missing", "present"]}
    try:
        status, _body = _post(server, "/sources/check", payload, token="bad")
        assert status == 401
        status, body = _post(server, "/sources/check", payload)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == 200
    assert body == {"ok": True, "present": ["present"], "missing": ["missing"]}
    assert secret not in json.dumps(body)


def test_sources_check_validates_bounds_and_source_restore_availability(
    monkeypatch, tmp_path
):
    app = _source_app(tmp_path)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(server, "/sources/check", {"source_identities": [1]})
        assert status == 400
        assert body["error"] == "source_identities must contain only non-empty strings"
        status, body = _post(
            server,
            "/sources/check",
            {"source_identities": [f"id-{index}" for index in range(5001)]},
        )
        assert status == 400
        assert "at most 5000" in body["error"]
        app.syncer.restore_failed = True
        status, body = _post(
            server, "/sources/check", {"source_identities": ["present"]}
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == 503
    assert body == {"ok": False, "error": "restore_failed"}


def _post_bytes(server, path, body, headers=None):
    conn = HTTPConnection(*server.server_address)
    request_headers = {"Authorization": "Bearer test-token", "Content-Type": "application/json"}
    request_headers.update(headers or {})
    conn.request("POST", path, body, request_headers)
    response = conn.getresponse()
    payload = json.loads(response.read())
    conn.close()
    return response.status, payload


def _async_post(server, payload, token="test-token"):
    conn = HTTPConnection(*server.server_address)
    conn.request(
        "POST",
        "/ingest",
        json.dumps(payload, ensure_ascii=False).encode(),
        {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": "respond-async",
        },
    )
    response = conn.getresponse()
    body = json.loads(response.read())
    headers = {name.lower(): value for name, value in response.getheaders()}
    conn.close()
    return response.status, body, headers


def _operation_get(server, status_url, token="test-token"):
    conn = HTTPConnection(*server.server_address)
    conn.request("GET", status_url, headers={"Authorization": f"Bearer {token}"})
    response = conn.getresponse()
    body = json.loads(response.read())
    headers = {name.lower(): value for name, value in response.getheaders()}
    conn.close()
    return response.status, body, headers


def _clear_ingest_operations():
    with bridge.INGEST_OPERATION_LOCK:
        bridge.INGEST_OPERATIONS.clear()
        bridge.INGEST_ACTIVE_OPERATION = None
        if hasattr(bridge, "INGEST_RESTART_SCHEDULED"):
            bridge.INGEST_RESTART_SCHEDULED = False


def test_async_ingest_deduplicates_bounds_concurrency_and_requires_status_auth(monkeypatch):
    _clear_ingest_operations()
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setenv("FUNES_INGEST_RETRY_AFTER", "7")
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls = []
    secret = "raw-secret-must-not-be-returned"

    def fake_ingest(docs):
        calls.append(docs)
        started.set()
        assert release.wait(2)
        finished.set()
        return (
            200,
            {
                "durable": True,
                "accepted": len(docs),
                "created": len(docs),
                "items": [{"raw_text": secret}],
            },
            [{"raw_text": secret}],
        )

    monkeypatch.setattr(bridge, "ingest_source_documents", fake_ingest)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    document = {"raw_text": secret, "source_identity": "async-one"}
    try:
        status, first, headers = _async_post(server, {"documents": [document]})
        assert status == 202
        assert headers["location"] == first["status_url"]
        assert headers["retry-after"] == "7"
        assert started.wait(1)
        assert first["durable"] is False
        assert secret not in json.dumps(first)

        reordered = {"source_identity": "async-one", "raw_text": secret}
        duplicate_status, duplicate, _headers = _async_post(
            server, {"documents": [reordered]}
        )
        assert duplicate_status == 202
        assert duplicate["operation_id"] == first["operation_id"]
        assert len(calls) == 1

        busy_status, busy, busy_headers = _async_post(
            server,
            {"documents": [{"source_identity": "async-two", "raw_text": "different"}]},
        )
        assert busy_status == 429
        assert busy == {
            "ok": False,
            "durable": False,
            "error": "ingest_busy",
            "retry_after": 7,
        }
        assert busy_headers["retry-after"] == "7"

        unauthorized, _body, _headers = _operation_get(
            server, first["status_url"], token="wrong-token"
        )
        assert unauthorized == 401
        running_status, running, running_headers = _operation_get(
            server, first["status_url"]
        )
        assert running_status == 202
        assert running["durable"] is False
        assert running_headers["retry-after"] == "7"

        release.set()
        assert finished.wait(1)
        deadline = time.monotonic() + 1
        while True:
            done_status, done, _headers = _operation_get(server, first["status_url"])
            if done_status != 202 or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert done_status == 200
        assert done["durable"] is True
        assert done["accepted"] == 1
        assert secret not in json.dumps(done)

        cached_status, cached, _headers = _async_post(
            server, {"documents": [reordered]}
        )
        assert cached_status == 200
        assert cached == done
        assert len(calls) == 1
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        _clear_ingest_operations()


def test_failed_async_ingest_can_be_retried_without_exposing_failure_details(monkeypatch):
    _clear_ingest_operations()
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    starts = [threading.Event(), threading.Event()]
    releases = [threading.Event(), threading.Event()]
    calls = []
    secret = "provider-secret-and-raw-document"

    def fake_ingest(docs):
        attempt = len(calls)
        calls.append(docs)
        starts[attempt].set()
        assert releases[attempt].wait(2)
        if attempt == 0:
            return 503, {"durable": False, "error": secret, "items": docs}, []
        return 200, {"durable": True, "accepted": len(docs), "items": docs}, docs

    monkeypatch.setattr(bridge, "ingest_source_documents", fake_ingest)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    payload = {"documents": [{"source_identity": "retry", "raw_text": secret}]}
    try:
        status, first, _headers = _async_post(server, payload)
        assert status == 202
        assert starts[0].wait(1)
        releases[0].set()

        deadline = time.monotonic() + 1
        while True:
            failed_status, failed, _headers = _operation_get(server, first["status_url"])
            if failed_status != 202 or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert failed_status == 503
        assert failed["status"] == "failed"
        assert failed["durable"] is False
        assert failed["error"] == "ingest_failed"
        assert secret not in json.dumps(failed)

        retry_status, retry, _headers = _async_post(server, payload)
        assert retry_status == 202
        assert retry["operation_id"] == first["operation_id"]
        assert starts[1].wait(1)
        releases[1].set()

        deadline = time.monotonic() + 1
        while True:
            done_status, done, _headers = _operation_get(server, retry["status_url"])
            if done_status != 202 or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert done_status == 200
        assert done["durable"] is True
        assert done["accepted"] == 1
        assert len(calls) == 2
    finally:
        for event in releases:
            event.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        _clear_ingest_operations()


def test_completed_ingest_operation_cleanup_is_ttl_and_count_bounded(monkeypatch):
    _clear_ingest_operations()
    monkeypatch.setenv("FUNES_INGEST_OPERATION_TTL", "100")
    monkeypatch.setenv("FUNES_INGEST_OPERATION_MAX_COMPLETED", "2")
    active_id = "d" * 64
    with bridge.INGEST_OPERATION_LOCK:
        for index, operation_id in enumerate(("a" * 64, "b" * 64, "c" * 64)):
            bridge.INGEST_OPERATIONS[operation_id] = {
                "operation_id": operation_id,
                "state": "succeeded",
                "accepted": 1,
                "completed_at": float(index),
            }
        bridge.INGEST_OPERATIONS[active_id] = {
            "operation_id": active_id,
            "state": "running",
            "documents": [{"raw_text": "bounded-active"}],
        }
        bridge.INGEST_ACTIVE_OPERATION = active_id
        bridge._cleanup_ingest_operations_locked(10.0)
        assert set(bridge.INGEST_OPERATIONS) == {"b" * 64, "c" * 64, active_id}

        monkeypatch.setenv("FUNES_INGEST_OPERATION_TTL", "1")
        bridge._cleanup_ingest_operations_locked(20.0)
        assert set(bridge.INGEST_OPERATIONS) == {active_id}
    _clear_ingest_operations()


def test_first_async_response_survives_concurrent_completed_operation_cleanup(monkeypatch):
    _clear_ingest_operations()
    monkeypatch.setenv("FUNES_INGEST_OPERATION_MAX_COMPLETED", "1")
    docs_a = [{"source_identity": "race-a", "raw_text": "first"}]
    docs_b = [{"source_identity": "race-b", "raw_text": "second"}]
    operation_a = bridge._ingest_operation_id(docs_a)
    nested = []

    monkeypatch.setattr(
        bridge,
        "ingest_source_documents",
        lambda docs: (200, {"durable": True, "accepted": len(docs)}, docs),
    )

    class ImmediateThread:
        def __init__(self, *, target, args, **_kwargs):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)
            if self.args[0] == operation_a:
                nested.append(bridge.start_ingest_operation(docs_b))

    monkeypatch.setattr(bridge.threading, "Thread", ImmediateThread)
    status_a, result_a = bridge.start_ingest_operation(docs_a)

    assert status_a == 200
    assert result_a["operation_id"] == operation_a
    assert result_a["durable"] is True
    assert nested[0][0] == 200
    assert set(bridge.INGEST_OPERATIONS) == {bridge._ingest_operation_id(docs_b)}
    _clear_ingest_operations()


def test_timed_out_async_ingest_fail_stops_once_without_releasing_active_slot(monkeypatch):
    _clear_ingest_operations()
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setenv("FUNES_INGEST_OPERATION_TIMEOUT", "1")
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls = []
    restarts = []

    def fake_ingest(docs):
        calls.append(docs)
        started.set()
        assert release.wait(2)
        finished.set()
        return 200, {"durable": True, "accepted": len(docs)}, docs

    monkeypatch.setattr(bridge, "ingest_source_documents", fake_ingest)
    monkeypatch.setattr(
        bridge,
        "schedule_ingest_process_restart",
        lambda: restarts.append("scheduled"),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    payload = {"documents": [{"source_identity": "hung", "raw_text": "blocked"}]}
    try:
        status, first, _headers = _async_post(server, payload)
        assert status == 202
        assert started.wait(1)
        with bridge.INGEST_OPERATION_LOCK:
            bridge.INGEST_OPERATIONS[first["operation_id"]]["started_at"] -= 2.0

        timed_out_status, timed_out, _headers = _operation_get(
            server, first["status_url"]
        )
        assert timed_out_status == 503
        assert timed_out["durable"] is False
        assert timed_out["error"] == "ingest_operation_timeout"
        assert restarts == ["scheduled"]

        repeated_status, repeated, _headers = _operation_get(
            server, first["status_url"]
        )
        assert repeated_status == 503
        assert repeated["error"] == "ingest_operation_timeout"
        assert restarts == ["scheduled"]

        blocked_status, blocked, _headers = _async_post(
            server,
            {"documents": [{"source_identity": "other", "raw_text": "must not start"}]},
        )
        assert blocked_status == 503
        assert blocked["error"] == "ingest_operation_timeout"
        assert blocked["status"] == "restart_pending"
        assert "operation_id" not in blocked
        assert len(calls) == 1
        with bridge.INGEST_OPERATION_LOCK:
            assert bridge.INGEST_ACTIVE_OPERATION == first["operation_id"]
            assert bridge.INGEST_OPERATIONS[first["operation_id"]]["state"] == "running"
    finally:
        release.set()
        assert finished.wait(1)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        _clear_ingest_operations()


def test_ingest_fail_stop_replaces_pid_one_and_has_hard_exit_fallback(monkeypatch):
    calls = []

    class ImmediateThread:
        def __init__(self, *, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(bridge.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(bridge.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        bridge.os,
        "execv",
        lambda executable, argv: calls.append((executable, argv))
        or (_ for _ in ()).throw(OSError("exec failed")),
    )
    monkeypatch.setattr(bridge.os, "_exit", lambda code: calls.append(("exit", code)))

    bridge.schedule_ingest_process_restart()

    assert calls[0][0] == bridge.sys.executable
    assert calls[0][1][0] == bridge.sys.executable
    assert calls[1] == ("exit", 75)


def test_http_gzip_ingest_rejects_invalid_and_oversized_payloads(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        valid = gzip.compress(json.dumps({"documents": [{"source_identity": "gzip-valid", "raw_text": "compressed"}]}).encode())
        status, result = _post_bytes(server, "/ingest", valid, {"Content-Encoding": "gzip"})
        assert status == 200
        assert result["created"] == 1
        assert app.store.get("gzip-valid") is not None

        status, result = _post_bytes(server, "/ingest", b"not-a-gzip-stream", {"Content-Encoding": "gzip"})
        assert status == 400
        assert result["error"] == "invalid gzip request body"
        status, result = _post_bytes(server, "/ingest", valid[:-1], {"Content-Encoding": "gzip"})
        assert status == 400
        assert result["error"] == "invalid gzip request body"
        corrupt_deflate = bytes.fromhex("1f8b0800000000000003ffff0000000000000000")
        status, result = _post_bytes(server, "/ingest", corrupt_deflate, {"Content-Encoding": "gzip"})
        assert status == 400
        assert result["error"] == "invalid gzip request body"
        status, result = _post_bytes(server, "/ingest", valid, {"Content-Encoding": "br"})
        assert status == 400
        assert result["error"] == "unsupported Content-Encoding"

        monkeypatch.setenv("FUNES_MAX_BODY_BYTES", "128")
        oversized = gzip.compress(json.dumps({"documents": [{"source_identity": "gzip-oversized", "raw_text": "x" * 1024}]}).encode())
        status, result = _post_bytes(server, "/ingest", oversized, {"Content-Encoding": "gzip"})
        assert status == 400
        assert result["error"] == "request too large"
        assert app.store.get("gzip-oversized") is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()


def test_reindex_queues_durable_control_without_native_or_provider_work(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    app.reindex_lock = threading.Lock()
    app.translation_lock = threading.Lock()
    app.reindex_wake = threading.Event()
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("HTTP reindex must not run native work")
        ),
    )
    app.translator.normalize_many = lambda _raws: (_ for _ in ()).throw(
        AssertionError("HTTP reindex must not call the provider")
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(server, "/reindex", {"scope": "all"})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == 202
    assert body == {"queued": True, "durable": True, "scope": "all", "generation": 1}
    assert app.syncer.controls[0]["_funes_record"] == "reindex_control"
    assert "retrieval_text" not in body
    assert "raw_text" not in body


def test_native_session_and_low_value_records_skip_provider_translation(tmp_path):
    class Translator:
        model = "test-model"

        def normalize_many(self, _raws):
            raise AssertionError("session/tool records must not call the provider")

    store = SourceStore(str(tmp_path / "skip-store"))
    app = SimpleNamespace(store=store, translator=Translator())
    try:
        prepared = bridge.prepare_source_documents(
            app,
            [
                {"source_identity": "session", "source_type": "session", "raw_text": "中文 session"},
                {"source_identity": "tool", "content_type": "tool_result", "raw_text": "中文 tool"},
            ],
        )
    finally:
        store.close()
    assert [item["translation_status"] for item in prepared] == [
        "skipped_native_session",
        "skipped_low_value",
    ]


def test_source_sidecar_updates_in_place_and_returns_only_raw_text(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native ingest must not run")),
    )
    monkeypatch.setattr(
        bridge,
        "recall",
        lambda *_args, **_kwargs: "ranked\n  → get memory-section",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = {
        "source_identity": "memory-section",
        "source_version": "v1",
        "source_agent": "codex",
        "source_type": "memory",
        "project": "demo",
        "raw_text": "旧的中文正文",
    }
    try:
        status, first = _post(server, "/ingest", {"documents": [base]})
        status2, second = _post(
            server,
            "/ingest",
            {"documents": [{**base, "source_version": "v2", "raw_text": "新的中文正文"}]},
        )
        search_status, found = _post(
            server,
            "/search",
            {"query": "english retrieval needle", "source_type": "memory", "limit": 3},
        )
        get_status, item = _post(server, "/get", {"id": "memory-section"})
        count = app.store.count()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == status2 == search_status == get_status == 200
    assert first["created"] == 1
    assert second["updated"] == 1
    assert len(app.syncer.uploads) == 2
    assert app.syncer.uploads[0][0]["translation_status"] == "pending_provider"
    assert app.syncer.uploads[-1][0]["raw_text"] == "新的中文正文"
    assert found["results"][0]["raw_text"] == "新的中文正文"
    assert "retrieval_text" not in found["results"][0]
    assert item["result"]["raw_text"] == "新的中文正文"
    assert count == 1


def test_source_sidecar_does_not_ack_before_encrypted_durability(monkeypatch, tmp_path):
    app = _source_app(tmp_path, durable=False)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(
            server,
            "/ingest",
            {"source_identity": "pending", "raw_text": "must remain queued"},
        )
        item = app.store.get("pending")
        candidates = app.store.canonical_index_candidates(10)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == 503
    assert body["durable"] is False
    assert body["error"] == "durability_pending"
    assert item["native_index_status"] == "waiting_durability"
    assert candidates == []


def test_ingest_never_runs_provider_and_persists_all_raw_first(tmp_path):
    class LimitedTranslator:
        model = "test-model"
        max_per_ingest = 2

        def __init__(self):
            self.calls = []

        def pending_document(self, raw):
            return raw, "pending-hash", "translation-version", "pending_provider"

        def normalize_many(self, raws):
            self.calls.extend(raws)
            return [
                (f"{raw} english-shadow", "translated-hash", "translation-version", "ok")
                for raw in raws
            ]

    translator = LimitedTranslator()
    app = SimpleNamespace(
        store=SourceStore(str(tmp_path / "limited-store")),
        translator=translator,
        syncer=_SourceSyncer(),
    )
    docs = [
        {"source_identity": f"doc-{index}", "raw_text": f"中文正文 {index}"}
        for index in range(5)
    ]
    try:
        result = bridge.persist_source_ingest(app, docs)
        stored = [app.store.get(f"doc-{index}") for index in range(5)]
    finally:
        app.store.close()
    assert result["durable"] is True
    assert result["translation"]["attempted"] == 0
    assert translator.calls == []
    assert len(app.syncer.uploads[0]) == 5
    assert {item["translation_status"] for item in app.syncer.uploads[0]} == {"pending_provider"}
    assert {item["translation_status"] for item in stored} == {"pending_provider"}


def test_pending_translation_survives_restore_and_reconciles_in_place(tmp_path):
    source = SourceStore(str(tmp_path / "before-restart"))
    source.ingest(
        [{
            "source_identity": "restart-doc",
            "source_version": "v1",
            "raw_text": "重启后恢复",
            "retrieval_text": "重启后恢复",
            "translation_hash": "pending-hash",
            "translation_version": "translation-version",
            "translation_status": "pending_provider",
        }]
    )
    snapshot = tmp_path / "restart.jsonl.gz"
    source.snapshot(snapshot)
    source.close()

    restored = SourceStore(str(tmp_path / "after-restart"))
    restored.restore(snapshot)

    class SuccessfulTranslator(_SourceTranslator):
        batch_size = 1
        concurrency = 1

    app = bridge.SourceApp.__new__(bridge.SourceApp)
    app.store = restored
    app.translator = SuccessfulTranslator()
    app.syncer = _SourceSyncer()
    app.restore_done = threading.Event()
    app.restore_done.set()
    app.reconcile_stop = threading.Event()
    app.reconcile_interval = 0.01
    app.translation_lock = threading.Lock()
    app.restore_thread = None
    app.reconcile_thread = threading.Thread(target=app._reconcile_background, daemon=True)
    app.reconcile_thread.start()
    try:
        deadline = time.monotonic() + 1
        item = restored.get("restart-doc")
        while item["translation_status"] != "ok" and time.monotonic() < deadline:
            time.sleep(0.01)
            item = restored.get("restart-doc")
    finally:
        app.close()
    assert item["raw_text"] == "重启后恢复"
    assert item["retrieval_text"] == "english retrieval needle"
    assert item["translation_status"] == "ok"


def test_provider_failure_stays_pending_and_later_retry_succeeds(tmp_path):
    class RetryTranslator(_SourceTranslator):
        max_per_ingest = 0

        def __init__(self):
            self.fail = True

        def normalize_many(self, raws):
            if self.fail:
                raise RuntimeError("provider unavailable")
            return super().normalize_many(raws)

    translator = RetryTranslator()
    app = SimpleNamespace(
        store=SourceStore(str(tmp_path / "retry-store")),
        translator=translator,
        syncer=_SourceSyncer(),
    )
    try:
        bridge.persist_source_ingest(
            app,
            [{"source_identity": "retry-doc", "raw_text": "失败后重试"}],
        )
        pending = app.store.pending_translations(1)
        first = persist_translation_documents(app, pending)
        translator.fail = False
        second = persist_translation_documents(
            app,
            app.store.pending_translations(1),
        )
        item = app.store.get("retry-doc")
    finally:
        app.store.close()
    assert first["updated"] == 0
    assert second["updated"] == 1
    assert item["raw_text"] == "失败后重试"
    assert item["translation_status"] == "ok"


def _canonical_source(store, identity, *, raw="原始中文", retrieval="english retrieval", **extra):
    document = {
        "source_identity": identity,
        "source_version": "raw-v1",
        "source_agent": "codex",
        "source_type": "memory",
        "project": "demo",
        "raw_text": raw,
        "retrieval_text": retrieval,
        "translation_hash": "translation-hash",
        "translation_version": "translation-version",
        "translation_status": "ok",
        "retrieval_updated_at": "2026-09-13T01:02:03Z",
        "updated_at": "2026-09-13T01:00:00Z",
        **extra,
    }
    store.ingest([document])
    return store.get(identity)


def test_canonical_source_version_has_unambiguous_field_boundaries():
    base = {
        "translation_hash": "translation-hash",
        "translation_version": "translation-version",
        "retrieval_text": "english retrieval",
    }
    left = {**base, "source_version": "ab", "content_hash": "c"}
    right = {**base, "source_version": "a", "content_hash": "bc"}
    assert bridge.canonical_source_version(left) != bridge.canonical_source_version(right)
    assert bridge.canonical_source_version(
        {**base, "native_generation": 1}
    ) != bridge.canonical_source_version({**base, "native_generation": 2})


def test_pending_translation_is_not_sent_to_native(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    app.store.ingest(
        [{
            "source_identity": "pending-doc",
            "source_version": "v1",
            "source_type": "memory",
            "raw_text": "等待翻译",
            "retrieval_text": "等待翻译",
            "translation_status": "pending_provider",
        }]
    )
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("pending row must not index")),
    )
    try:
        result = bridge.reconcile_canonical_index(app)
    finally:
        app.store.close()
    assert result == {"attempted": 0, "indexed": 0, "held": 0, "durable": True}


def test_canonical_scan_waits_for_raw_durability_lock(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    item = _canonical_source(app.store, "durability-race")
    scanned = threading.Event()
    original_candidates = app.store.canonical_index_candidates

    def candidates(limit):
        scanned.set()
        return original_candidates(limit)

    app.store.canonical_index_candidates = candidates
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("undurable row must not index")),
    )
    result = {}
    bridge.WRITE_LOCK.acquire()
    thread = threading.Thread(
        target=lambda: result.setdefault("value", bridge.reconcile_canonical_index(app))
    )
    thread.start()
    try:
        assert not scanned.wait(0.05)
        app.store.update_native_index(
            [{
                "source_identity": item["source_identity"],
                "source_version": item["source_version"],
                "content_hash": item["content_hash"],
                "native_index_version": None,
                "native_index_status": "waiting_durability",
                "native_indexed_at": None,
                "native_index_error": "durability_pending",
            }]
        )
    finally:
        bridge.WRITE_LOCK.release()
        thread.join(timeout=1)
        app.store.close()
    assert scanned.is_set()
    assert result["value"]["attempted"] == 0


def test_canonical_jsonl_has_no_raw_and_success_is_durable(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    _canonical_source(
        app.store,
        "memory-section",
        raw="PRIVATE RAW SOURCE",
        metadata={"raw_text": "PRIVATE RAW SOURCE", "nested": {"text": "PRIVATE RAW SOURCE"}},
    )
    captured = {}
    warm = []
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "request_warm", lambda **kwargs: warm.append(kwargs))

    def fake_run(*args, **kwargs):
        assert args[0] == "ingest-docs"
        assert args[2:] == ("--memory", "owner/memory")
        assert kwargs["timeout"] == bridge.CANONICAL_INDEX_TIMEOUT
        text = Path(args[1]).read_text(encoding="utf-8")
        captured["text"] = text
        captured["document"] = json.loads(text)
        return 0, "ingested sources=1 chunks=1 unchanged=0 stale=0 held=0 commit=abc123\n", ""

    monkeypatch.setattr(bridge, "run", fake_run)
    try:
        result = bridge.reconcile_canonical_index(app)
        stored = app.store.get("memory-section")
    finally:
        app.store.close()
    document = captured["document"]
    assert "PRIVATE RAW SOURCE" not in captured["text"]
    assert "raw_text" not in document
    assert "text" not in document
    assert set(("source_identity", "source_version", "retrieval_text", "content_hash", "updated_at", "metadata")) <= set(document)
    assert document["updated_at"] == "2026-09-13T01:02:03Z"
    assert document["metadata"]["source_version"] == "raw-v1"
    assert document["session_id"] == bridge.canonical_reference("memory-section")
    assert result == {"attempted": 1, "indexed": 1, "held": 0, "durable": True}
    assert stored["native_index_status"] == "indexed"
    assert stored["native_index_version"] == document["source_version"]
    assert stored["native_indexed_at"]
    assert warm == [{"force": True}]


def test_canonical_held_batch_is_bisected_and_source_revision_retries(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    _canonical_source(app.store, "clean-doc")
    _canonical_source(app.store, "dirty-doc")
    calls = []
    warm = []
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "request_warm", lambda **kwargs: warm.append(kwargs))

    def held_run(*args, **_kwargs):
        documents = [json.loads(line) for line in Path(args[1]).read_text().splitlines()]
        identities = [item["source_identity"] for item in documents]
        calls.append(identities)
        held = int("dirty-doc" in identities)
        sources = len(identities) - held
        commit = " commit=commit1" if sources else ""
        return 0, f"ingested sources={sources} chunks={sources} unchanged=0 stale=0 held={held}{commit}\n", ""

    monkeypatch.setattr(bridge, "run", held_run)
    try:
        first = bridge.reconcile_canonical_index(app)
        clean = app.store.get("clean-doc")
        dirty = app.store.get("dirty-doc")
        second = bridge.reconcile_canonical_index(app)
        previous_source = {
            key: value
            for key, value in dirty.items()
            if key not in {
                "id",
                "content_hash",
                "native_index_version",
                "native_index_status",
                "native_indexed_at",
                "native_index_error",
            }
        }
        app.store.ingest(
            [{
                **previous_source,
                "source_version": "raw-v2",
                "raw_text": "修订后原文",
                "retrieval_text": "revised clean retrieval",
                "translation_hash": "translation-hash-v2",
                "retrieval_updated_at": "2026-09-13T02:00:00Z",
            }]
        )
        monkeypatch.setattr(
            bridge,
            "run",
            lambda *_args, **_kwargs: (0, "ingested sources=1 chunks=1 unchanged=0 stale=0 held=0 commit=commit2\n", ""),
        )
        third = bridge.reconcile_canonical_index(app)
        revised = app.store.get("dirty-doc")
    finally:
        app.store.close()
    assert calls == [["clean-doc", "dirty-doc"], ["clean-doc"], ["dirty-doc"]]
    assert first == {"attempted": 2, "indexed": 1, "held": 1, "durable": True}
    assert clean["native_index_status"] == "indexed"
    assert dirty["native_index_status"] == "held_secret"
    assert second["attempted"] == 0
    assert third["indexed"] == 1
    assert revised["native_index_status"] == "indexed"
    assert len(warm) == 2


def test_canonical_failure_persists_retry_and_next_pass_succeeds(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    _canonical_source(app.store, "retry-doc")
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "request_warm", lambda **_kwargs: None)
    outcomes = iter(
        [
            (1, "", "must not persist"),
            (0, "ingested sources=1 chunks=1 unchanged=0 stale=0 held=0 commit=commit1\n", ""),
        ]
    )
    monkeypatch.setattr(bridge, "run", lambda *_args, **_kwargs: next(outcomes))
    try:
        first = bridge.reconcile_canonical_index(app)
        failed = app.store.get("retry-doc")
        second = bridge.reconcile_canonical_index(app)
        recovered = app.store.get("retry-doc")
    finally:
        app.store.close()
    assert first["indexed"] == 0
    assert failed["native_index_status"] == "retry"
    assert failed["native_index_error"] == "native_exit"
    assert second["indexed"] == 1
    assert recovered["native_index_status"] == "indexed"
    assert recovered["native_index_error"] is None


def test_native_stale_report_is_retryable_not_indexed(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    _canonical_source(app.store, "stale-doc")
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (
            0,
            "ingested sources=0 chunks=0 unchanged=0 stale=1 held=0\n",
            "",
        ),
    )
    try:
        result = bridge.reconcile_canonical_index(app)
        item = app.store.get("stale-doc")
    finally:
        app.store.close()
    assert result["indexed"] == 0
    assert item["native_index_status"] == "retry"
    assert item["native_index_error"] == "native_stale"


def test_restart_restores_pending_canonical_and_indexes_it(monkeypatch, tmp_path):
    before = SourceStore(str(tmp_path / "before"))
    _canonical_source(before, "restart-canonical")
    snapshot = tmp_path / "restart.jsonl.gz"
    before.snapshot(snapshot)
    before.close()
    restored = SourceStore(str(tmp_path / "after"))
    restored.restore(snapshot)
    app = SimpleNamespace(store=restored, syncer=_SourceSyncer())
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "request_warm", lambda **_kwargs: None)
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (0, "ingested sources=1 chunks=1 unchanged=0 stale=0 held=0 commit=commit1\n", ""),
    )
    try:
        result = bridge.reconcile_canonical_index(app)
        item = restored.get("restart-canonical")
    finally:
        restored.close()
    assert result["indexed"] == 1
    assert item["native_index_status"] == "indexed"


def test_native_rank_maps_canonical_to_raw_and_forwards_facets(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    provider_calls = []

    def rewrite_query(query):
        provider_calls.append(query)
        return "PROVIDER REWRITE"

    app.translator.rewrite_query = rewrite_query
    _canonical_source(
        app.store,
        "canonical id",
        raw="原始 sidecar 正文",
        retrieval="CANONICAL ENGLISH SHADOW",
        source_missing=False,
    )
    calls = []

    class FakeWorker:
        def recall(self, query, **kwargs):
            calls.append((query, kwargs))
            return "CANONICAL ENGLISH SHADOW\n  → get canonical id --from 0 --to 0"

        def get(self, *_args, **_kwargs):
            raise AssertionError("canonical hit must resolve from sidecar")

    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "MCP_WORKER", FakeWorker())
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "query_text",
        lambda _query: (_ for _ in ()).throw(AssertionError("provider rewrite must be reused")),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(
            server,
            "/search",
            {
                "query": "english retrieval",
                "limit": 3,
                "facets": {
                    "source_agent": "codex",
                    "source_type": "memory",
                    "project": "demo",
                    "repo": "owner/repo",
                    "device_id": "device-1",
                    "content_type": "summary",
                    "source_missing": False,
                },
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == 200
    assert body["results"][0]["raw_text"] == "原始 sidecar 正文"
    assert "retrieval_text" not in body["results"][0]
    assert "CANONICAL ENGLISH SHADOW" not in body["results_text"]
    assert provider_calls == ["english retrieval"]
    assert calls[0][0] == "PROVIDER REWRITE"
    assert calls[0][1]["source_agent"] == "codex"
    assert calls[0][1]["source_type"] == "memory"
    assert calls[0][1]["source_missing"] is False


def test_sidecar_search_rrf_queries_raw_and_rewrite_with_filters(monkeypatch):
    calls = []

    class Translator:
        def rewrite_query(self, query):
            calls.append(("rewrite", query))
            return "provider query"

    class Store:
        def search(self, query, limit, filters):
            calls.append(("search", query, limit, filters))
            if query == "raw query":
                return [
                    {"source_identity": "raw-only", "raw_text": "raw BM25", "retrieval_text": "shadow"},
                    {"source_identity": "both", "raw_text": "both raw", "retrieval_text": "shadow"},
                ]
            return [
                {"source_identity": "provider-only", "raw_text": "provider raw", "retrieval_text": "shadow"},
                {"source_identity": "both", "raw_text": "both raw", "retrieval_text": "shadow"},
            ]

    app = SimpleNamespace(
        translator=Translator(),
        store=Store(),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    filters = {"project": "demo", "role": "user"}
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    rewritten, results = bridge.search_source_documents("raw query", 3, filters)
    assert rewritten == "provider query"
    assert [item["source_identity"] for item in results] == [
        "both",
        "raw-only",
        "provider-only",
    ]
    assert all("retrieval_text" not in item for item in results)
    assert calls == [
        ("rewrite", "raw query"),
        ("search", "raw query", 9, filters),
        ("search", "provider query", 9, filters),
    ]


def test_sidecar_search_partitions_native_sessions_and_filters_harness(monkeypatch):
    class Store:
        def search(self, query, _limit, filters):
            assert filters == {}
            if query == "raw query":
                return [
                    {"source_identity": "memory", "source_type": "memory", "raw_text": "memory raw", "retrieval_text": "shadow"},
                    {"source_identity": "codex", "source_type": "session", "source_agent": "codex", "raw_text": "codex raw", "retrieval_text": "shadow"},
                    {"source_identity": "claude", "source_type": "claude_session", "source_agent": "claude", "raw_text": "claude raw", "retrieval_text": "shadow"},
                    {"source_identity": "claude-code", "source_type": "session", "source_agent": "claude_code", "raw_text": "claude code raw", "retrieval_text": "shadow"},
                ]
            return [
                {"source_identity": "pi", "source_type": "pi_session", "source_agent": "pi", "raw_text": "pi raw", "retrieval_text": "shadow"}
            ]

    app = SimpleNamespace(
        translator=SimpleNamespace(rewrite_query=lambda _query: "provider query"),
        store=Store(),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)

    rewritten, primary, fallback = bridge.search_source_rankings("raw query", 5, {})
    assert rewritten == "provider query"
    assert [[item["source_identity"] for item in ranking] for ranking in primary] == [["memory"]]
    assert [[item["source_identity"] for item in ranking] for ranking in fallback] == [
        ["codex", "claude", "claude-code"],
        ["pi"],
    ]
    assert all(
        "retrieval_text" not in item
        for ranking in [*primary, *fallback]
        for item in ranking
    )

    _, _, claude_fallback = bridge.search_source_rankings(
        "raw query", 5, {}, harness="claude_code"
    )
    assert [item["source_identity"] for item in claude_fallback[0]] == [
        "claude",
        "claude-code",
    ]
    _, _, unknown_fallback = bridge.search_source_rankings(
        "raw query", 5, {}, harness="unknown"
    )
    assert unknown_fallback == []


@pytest.mark.parametrize(
    "filters",
    (
        {"role": "user"},
        {"since": "2026-09-01"},
        {"until": "2026-09-02"},
        {"source_type": "session"},
    ),
)
def test_explicit_sidecar_filters_keep_sessions_primary(monkeypatch, filters):
    session = {
        "source_identity": "filtered-session",
        "source_type": "session",
        "source_agent": "codex",
        "raw_text": "filtered raw",
        "retrieval_text": "shadow",
    }
    app = SimpleNamespace(
        translator=SimpleNamespace(rewrite_query=lambda query: query),
        store=SimpleNamespace(search=lambda _query, _limit, filters: [session]),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    _, primary, fallback = bridge.search_source_rankings("query", 3, filters)
    assert primary == [[{key: value for key, value in session.items() if key != "retrieval_text"}]]
    assert fallback == []


def test_http_rrf_promotes_dual_hit_and_keeps_sidecar_raw(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    for identity, raw in (
        ("both", "dual raw"),
        ("native-only", "native raw"),
    ):
        _canonical_source(app.store, identity, raw=raw, retrieval="ENGLISH SHADOW")
    source_rankings = [
        [
            {"source_identity": "sidecar-only", "raw_text": "raw BM25"},
            {"source_identity": "both", "raw_text": "triple raw"},
        ],
        [
            {"source_identity": "provider-only", "raw_text": "provider raw"},
            {"source_identity": "both", "raw_text": "triple raw"},
        ],
    ]
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "search_source_rankings",
        lambda query, limit, filters, harness: (
            "provider query",
            source_rankings,
            [[{"source_identity": "session-fallback", "raw_text": "fallback raw"}]],
        ),
    )
    monkeypatch.setattr(
        bridge,
        "recall",
        lambda *_args, **_kwargs: "\n".join(
            (
                "native\n  → get native-only --from 0 --to 0",
                "dual\n  → get both --from 0 --to 0",
            )
        ),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(server, "/search", {"query": "raw query", "limit": 3})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == 200
    assert [item["source_identity"] for item in body["results"]] == [
        "both",
        "sidecar-only",
        "provider-only",
    ]
    assert body["results_text"] == "triple raw\n\nraw BM25\n\nprovider raw"
    assert "ENGLISH SHADOW" not in json.dumps(body["results"], ensure_ascii=False)
    assert "fallback raw" not in body["results_text"]


@pytest.mark.parametrize(
    ("native_error", "degraded"),
    (
        (bridge.NativeMcpError("timed out"), "native_mcp_unavailable"),
        (bridge.NativeMcpBusyError("busy"), "native_mcp_busy"),
    ),
)
def test_http_native_failure_returns_raw_sidecar_results(
    monkeypatch, tmp_path, native_error, degraded
):
    app = _source_app(tmp_path)
    source_rankings = [[{
        "source_identity": "sidecar-session",
        "source_type": "session",
        "raw_text": "原始 sidecar session 结果",
    }]]
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "search_source_rankings",
        lambda query, limit, filters, harness: ("provider query", [], source_rankings),
    )
    monkeypatch.setattr(
        bridge,
        "recall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(native_error),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(server, "/search", {"query": "raw query", "limit": 3})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == 200
    assert body["retrieval_degraded"] == degraded
    assert body["results_text"] == "原始 sidecar session 结果"
    assert body["results"] == source_rankings[0]


def test_http_native_empty_uses_session_sidecar_fallback(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    fallback = [[{
        "source_identity": "empty-native-session",
        "source_type": "session",
        "raw_text": "native 空结果时的原始 session",
    }]]
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "search_source_rankings",
        lambda query, limit, filters, harness: ("provider query", [], fallback),
    )
    monkeypatch.setattr(bridge, "recall", lambda *_args, **_kwargs: "")
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(server, "/search", {"query": "raw query", "limit": 3})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == 200
    assert body["results"] == fallback[0]
    assert body["results_text"] == "native 空结果时的原始 session"
    assert "retrieval_degraded" not in body


@pytest.mark.parametrize(
    ("native_error", "expected_status", "expected_error"),
    (
        (bridge.NativeMcpError("failed"), 503, "native_mcp_unavailable"),
        (bridge.NativeMcpBusyError("busy"), 429, "native_mcp_busy"),
    ),
)
def test_http_native_failure_without_sidecar_stays_retryable(
    monkeypatch, tmp_path, native_error, expected_status, expected_error
):
    app = _source_app(tmp_path)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "search_source_rankings",
        lambda query, limit, filters, harness: ("provider query", [], []),
    )
    monkeypatch.setattr(
        bridge,
        "recall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(native_error),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(server, "/search", {"query": "private query", "limit": 3})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == expected_status
    assert body["error"] == expected_error
    assert "private query" not in json.dumps(body)


def test_http_caps_cjk_native_candidates_after_query_tuning(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    calls = []
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "LANGUAGE_MODE", "auto")
    monkeypatch.setattr(bridge, "HTTP_MAX_CANDIDATES", 12)
    monkeypatch.setattr(
        bridge,
        "search_source_rankings",
        lambda query, limit, filters, harness: (
            "CPA previous_response_id context loss",
            [],
            [],
        ),
    )

    def fake_recall(query, **kwargs):
        calls.append((query, kwargs))
        return ""

    monkeypatch.setattr(bridge, "recall", fake_recall)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _body = _post(
            server,
            "/search",
            {"query": "CPA previous_response_id 为什么丢上下文？", "limit": 10},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == 200
    assert calls[0][0] == "CPA previous_response_id context loss"
    assert calls[0][1]["candidates"] == 12
    assert 0 < calls[0][1]["timeout"] <= bridge.HTTP_NATIVE_TIMEOUT


def test_missing_canonical_reference_never_falls_back_to_native_get(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    reference = bridge.canonical_reference("missing-canonical")
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must fail closed")),
    )
    assert bridge.materialize_native_results(f"shadow\n  → get {reference}", app, 3) == []
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(server, "/get", {"id": reference})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == 404
    assert body["error"] == "not_found"
