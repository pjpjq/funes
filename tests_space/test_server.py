import gzip
import hashlib
import json
import os
import socket
import subprocess
import sys
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


def test_selective_bm25_trigger_accepts_repeated_groups_two_two_one():
    raw_query = "CPA 第二轮为什么丢上下文？"
    results = [
        {
            "source_identity": "a-1",
            "session_id": "session-a",
            "raw_text": "Northflank 调用 previous_response_id 后继续执行。",
        },
        {
            "source_identity": "a-2",
            "session_id": "session-a",
            "raw_text": "同一 session 的普通记录。",
        },
        {
            "source_identity": "b-1",
            "session_id": "session-b",
            "raw_text": "previous_response_id 与下一轮上下文有关。",
        },
        {
            "source_identity": "b-2",
            "session_id": "session-b",
            "raw_text": "另一个重复 session 的普通记录。",
        },
        {
            "source_identity": "c-1",
            "session_id": "session-c",
            "raw_text": "单独的 session。",
        },
    ]

    assert bridge.should_augment_with_local_bm25(raw_query, results) is True


@pytest.mark.parametrize(
    ("query", "results"),
    (
        (
            "why was context lost?",
            [
                {"session_id": "same", "raw_text": "retry_state"},
                {"session_id": "same", "raw_text": "retry_state"},
                {"session_id": "same", "raw_text": "retry_state"},
            ],
        ),
        (
            "please investigate the context loss 的",
            [
                {"session_id": "same", "raw_text": "retry_state"},
                {"session_id": "same", "raw_text": "retry_state"},
                {"session_id": "same", "raw_text": "retry_state"},
            ],
        ),
        (
            "为什么丢上下文？",
            [
                {"session_id": "one", "raw_text": "retry_state"},
                {"session_id": "two", "raw_text": "retry_state"},
                {"session_id": "three", "raw_text": "retry_state"},
            ],
        ),
        (
            "为什么丢上下文？",
            [
                {"session_id": "same", "raw_text": "retry_state"},
                {"session_id": "same", "raw_text": "no identifier here"},
                {"session_id": "same", "raw_text": "another note"},
            ],
        ),
        (
            "retry_state 为什么丢上下文？",
            [
                {"session_id": "same", "raw_text": "retry_state"},
                {"session_id": "same", "raw_text": "retry_state"},
                {"session_id": "same", "raw_text": "retry_state"},
            ],
        ),
    ),
)
def test_selective_bm25_trigger_keeps_ordinary_requests_one_pass(query, results):
    assert bridge.should_augment_with_local_bm25(query, results) is False


def test_selective_bm25_trigger_rejects_secret_and_unstructured_terms():
    unsafe = "ghp_" + "x" * 36
    rows = [
        {
            "session_id": "same",
            "raw_text": (
                f"Northflank access_token {unsafe}"
            ),
        }
        for _ in range(3)
    ]

    assert bridge.should_augment_with_local_bm25("这个问题为什么失败？", rows) is False


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


def test_native_mcp_worker_can_return_structured_recall_result(monkeypatch, tmp_path):
    def responder(message, stdout):
        if message.get("method") == "initialize":
            _mcp_responder(message, stdout)
            return
        if message.get("method") == "tools/call":
            stdout.push(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "content": [{"type": "text", "text": "agent rendering"}],
                        "structuredContent": {
                            "hits": [{"raw_text": "raw", "session_id": "s1"}]
                        },
                    },
                }
            )

    monkeypatch.setattr(
        bridge.subprocess,
        "Popen",
        lambda *_args, **_kwargs: _FakeProcess(responder),
    )
    worker = bridge.NativeMcpWorker(
        "fake-funes", "owner/memory", tmp_path, timeout=1, handshake_timeout=1
    )
    try:
        result = worker.recall_result("query", k=1)
    finally:
        worker.close()

    assert result["structuredContent"]["hits"] == [
        {"raw_text": "raw", "session_id": "s1"}
    ]


def test_native_mcp_worker_rejects_tool_error_text(monkeypatch, tmp_path):
    worker = bridge.NativeMcpWorker(
        "fake-funes", "owner/memory", tmp_path, timeout=1, handshake_timeout=1
    )
    monkeypatch.setattr(
        worker,
        "call_tool",
        lambda name, _arguments, **_kwargs: f"{name} error: provider detail",
    )

    with pytest.raises(bridge.NativeMcpError, match="native recall failed"):
        worker.recall("probe")
    with pytest.raises(bridge.NativeMcpError, match="native get failed"):
        worker.get("session-1")


def test_refresh_native_worker_rejects_error_probe_and_keeps_old_worker(
    monkeypatch, tmp_path
):
    workers = []

    class FakeWorker:
        def __init__(self, *_args, **_kwargs):
            self.calls = []
            self.closed = False
            workers.append(self)

        def recall(self, query, **kwargs):
            self.calls.append((query, kwargs))
            return "recall error: provider unavailable"

        def close(self):
            self.closed = True

    old_worker = FakeWorker()
    old_config = ("old",)
    monkeypatch.setattr(bridge, "HOME", tmp_path)
    monkeypatch.setattr(bridge, "NativeMcpWorker", FakeWorker)
    monkeypatch.setattr(bridge, "MCP_WORKER", old_worker)
    monkeypatch.setattr(bridge, "_MCP_WORKER_CONFIG", old_config)
    monkeypatch.setattr(
        bridge,
        "_WARM_STATE",
        {
            "state": "warming",
            "started_at": "now",
            "finished_at": None,
            "refresh_pending": False,
        },
    )

    with pytest.raises(bridge.NativeMcpError, match="native recall failed"):
        bridge._refresh_native_worker()

    candidate = workers[-1]
    assert candidate.calls == [
        ("memory", {"k": 1, "candidates": 1, "half_life": 0, "neighbors": 0})
    ]
    assert candidate.closed is True
    assert old_worker.closed is False
    assert bridge.MCP_WORKER is old_worker
    assert bridge._MCP_WORKER_CONFIG == old_config

    bridge._warm_native_memory()
    assert bridge.warm_state()["state"] == "error"
    assert workers[-1].closed is True
    assert bridge.MCP_WORKER is old_worker


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


def test_search_and_get_use_native_worker_and_keep_voyage_raw_query(monkeypatch):
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
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
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
        assert calls[0][1] == "第二轮为什么丢上下文？"
        assert 0 < calls[0][2]["timeout"] <= bridge.VOYAGE_NATIVE_TIMEOUT

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


def test_get_ready_is_constant_time_and_never_touches_storage(monkeypatch):
    class Store:
        def __getattribute__(self, name):
            raise AssertionError(f"GET /ready must not access store.{name}")

    app = SimpleNamespace(
        store=Store(),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
        restore_result=12,
    )
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setenv("FUNES_STORAGE_REPO", "owner/source")
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(
        bridge,
        "source_state",
        lambda: (_ for _ in ()).throw(
            AssertionError("GET /ready must not build full source status")
        ),
    )
    monkeypatch.setattr(
        bridge,
        "source_app",
        lambda: (_ for _ in ()).throw(
            AssertionError("GET /ready must not initialize the source app")
        ),
    )
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "ready"})
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("GET /ready must not run native status")
        ),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection(*server.server_address)
        conn.request("GET", "/ready", headers={"Authorization": "Bearer test-token"})
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
    assert body["source_store"] == {
        "configured": True,
        "ready": True,
        "restoring": False,
        "restored": 12,
    }
    assert body["native_warm"]["state"] == "ready"
    assert body["status"] == ""


def test_ready_retries_failed_warm_without_sidecar_then_recovers(monkeypatch):
    refreshes = []
    queued_warms = []
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.delenv("FUNES_STORAGE_REPO", raising=False)
    monkeypatch.setattr(bridge, "SOURCE_APP", None)
    monkeypatch.setattr(
        bridge,
        "_WARM_STATE",
        {
            "state": "not_started",
            "started_at": None,
            "finished_at": None,
            "refresh_pending": False,
        },
    )
    monkeypatch.setattr(
        bridge,
        "source_state",
        lambda: (_ for _ in ()).throw(
            AssertionError("GET /ready must not build full source status")
        ),
    )
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("GET /ready must stay lightweight")
        ),
    )

    def first_refresh_fails():
        refreshes.append("failed")
        raise bridge.NativeMcpError("provider unavailable")

    monkeypatch.setattr(bridge, "_refresh_native_worker", first_refresh_fails)
    bridge._warm_native_memory()
    assert bridge.warm_state()["state"] == "error"

    def refreshed_worker_is_ready():
        refreshes.append("ready")

    monkeypatch.setattr(bridge, "_refresh_native_worker", refreshed_worker_is_ready)

    class QueuedThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            queued_warms.append(self)

    # Do not patch the global Thread while a ThreadingHTTPServer is alive.
    # The queued worker makes the retry/warming transition deterministic.
    with monkeypatch.context() as context:
        context.setattr(bridge.threading, "Thread", QueuedThread)
        code, payload = bridge.ready_payload()
        assert code == 503
        assert payload["native_warm"]["state"] == "warming"
        assert len(queued_warms) == 1
        repeat_code, repeat = bridge.ready_payload()
        assert repeat_code == 503
        assert repeat["native_warm"]["state"] == "warming"
        assert len(queued_warms) == 1
        queued_warms[0].kwargs["target"](**queued_warms[0].kwargs["kwargs"])

    assert refreshes == ["failed", "ready"]
    assert bridge.warm_state()["state"] == "ready"
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection(*server.server_address)
        conn.request("GET", "/ready", headers={"Authorization": "Bearer test-token"})
        response = conn.getresponse()
        body = json.loads(response.read())
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status == 200
    assert body["ok"] is True


def test_get_ready_returns_immediate_503_during_source_restore(monkeypatch):
    class Store:
        def __getattribute__(self, name):
            raise AssertionError(f"GET /ready must not access store.{name}")

    app = SimpleNamespace(
        store=Store(),
        syncer=SimpleNamespace(restoring=True, restore_failed=False),
        restore_result=0,
    )
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setenv("FUNES_STORAGE_REPO", "owner/source")
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(
        bridge,
        "source_state",
        lambda: (_ for _ in ()).throw(
            AssertionError("GET /ready must not build full source status")
        ),
    )
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "ready"})
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("GET /ready must not run native status")
        ),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        conn = HTTPConnection(*server.server_address)
        conn.request("GET", "/ready", headers={"Authorization": "Bearer test-token"})
        response = conn.getresponse()
        body = json.loads(response.read())
        conn.close()
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert elapsed < 0.25
    assert response.status == 503
    assert body["error"] == "restore_in_progress"
    assert body["source_store"]["restoring"] is True


@pytest.mark.parametrize(
    ("restoring", "restore_failed", "source_error"),
    (
        (True, False, "restore_in_progress"),
        (False, True, "restore_failed"),
    ),
)
def test_search_ready_stays_200_while_source_is_unavailable(
    monkeypatch, restoring, restore_failed, source_error
):
    class Store:
        def __getattribute__(self, name):
            raise AssertionError(f"readiness must not access store.{name}")

    app = SimpleNamespace(
        store=Store(),
        syncer=SimpleNamespace(restoring=restoring, restore_failed=restore_failed),
        restore_result=0,
    )
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setenv("FUNES_STORAGE_REPO", "owner/source")
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "ready"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        responses = {}
        for path in ("/ready/search", "/ready/ingest", "/ready"):
            conn = HTTPConnection(*server.server_address)
            conn.request("GET", path, headers={"Authorization": "Bearer test-token"})
            response = conn.getresponse()
            responses[path] = response.status, json.loads(response.read())
            conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert responses["/ready/search"][0] == 200
    assert responses["/ready/search"][1]["ok"] is True
    assert responses["/ready/ingest"][0] == 503
    assert responses["/ready/ingest"][1]["error"] == source_error
    assert responses["/ready"][0] == 503
    assert responses["/ready"][1]["error"] == source_error


def test_search_ready_stays_200_during_active_worker_replacement(monkeypatch):
    process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.delenv("FUNES_STORAGE_REPO", raising=False)
    monkeypatch.setattr(bridge, "SOURCE_APP", None)
    monkeypatch.setattr(bridge, "MCP_WORKER", SimpleNamespace(process=process))
    monkeypatch.setattr(bridge, "_MCP_WORKER_CONFIG", bridge._native_worker_config())
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "warming"})

    search_code, search = bridge.search_ready_payload()
    ingest_code, ingest = bridge.ready_payload()

    assert search_code == 200
    assert search["ok"] is True
    assert search["error"] == ""
    assert search["native_warm"]["state"] == "warming"
    assert ingest_code == 503
    assert ingest["error"] == "native_warm_warming"

    monkeypatch.setattr(bridge, "_MCP_WORKER_CONFIG", ("stale",))
    assert bridge.search_ready_payload()[0] == 503
    monkeypatch.setattr(bridge, "_MCP_WORKER_CONFIG", bridge._native_worker_config())
    process.poll = lambda: 1
    assert bridge.search_ready_payload()[0] == 503


def test_search_ready_stays_503_during_initial_warm_without_active_worker(monkeypatch):
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.delenv("FUNES_STORAGE_REPO", raising=False)
    monkeypatch.setattr(bridge, "SOURCE_APP", None)
    monkeypatch.setattr(bridge, "MCP_WORKER", None)
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "warming"})

    code, payload = bridge.search_ready_payload()

    assert code == 503
    assert payload["ok"] is False
    assert payload["error"] == "native_warm_warming"


def test_search_ready_requires_source_for_legacy_local_provider(monkeypatch):
    app = SimpleNamespace(
        syncer=SimpleNamespace(restoring=True, restore_failed=False),
        restore_result=0,
    )
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setenv("FUNES_STORAGE_REPO", "owner/source")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "ready"})

    code, payload = bridge.search_ready_payload()

    assert code == 503
    assert payload["ok"] is False
    assert payload["error"] == "restore_in_progress"


@pytest.mark.parametrize(
    ("path", "payload", "extra_headers"),
    (
        ("/sync", {}, {}),
        ("/reindex", {"scope": "all"}, {}),
        (
            "/ingest",
            {"documents": [{"raw_text": "must not be retained"}]},
            {"Prefer": "respond-async"},
        ),
    ),
)
@pytest.mark.parametrize(
    ("restoring", "restore_failed", "source_error"),
    (
        (True, False, "restore_in_progress"),
        (False, True, "restore_failed"),
    ),
)
def test_writes_fail_fast_while_source_is_restoring(
    monkeypatch,
    path,
    payload,
    extra_headers,
    restoring,
    restore_failed,
    source_error,
):
    class Store:
        def __getattribute__(self, name):
            raise AssertionError(f"restore-gated write must not access store.{name}")

    app = SimpleNamespace(
        store=Store(),
        syncer=SimpleNamespace(restoring=restoring, restore_failed=restore_failed),
        restore_result=0,
    )
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        conn = HTTPConnection(*server.server_address)
        headers = {
            "Authorization": "Bearer test-token",
            "Content-Type": "application/json",
            **extra_headers,
        }
        conn.request("POST", path, json.dumps(payload).encode(), headers)
        response = conn.getresponse()
        body = json.loads(response.read())
        conn.close()
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert elapsed < 0.25
    assert response.status == 503
    assert body["error"] == source_error
    assert body["durable"] is False


def test_sync_status_native_timeout_keeps_sanitized_reconciler_state(monkeypatch):
    monkeypatch.setattr(bridge, "REMOTE", "owner/private")
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            bridge.subprocess.TimeoutExpired(
                ["/private/path", "exception-only-private"],
                30,
                output="PRIVATE RAW SESSION",
                stderr="SECRET PROVIDER PAYLOAD",
            )
        ),
    )
    monkeypatch.setattr(
        bridge,
        "source_state",
        lambda: {
            "configured": True,
            "ready": True,
            "documents": 17,
            "canonical_reconciler": {
                "thread_alive": True,
                "phase": "native_ingest",
            },
        },
    )
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "ready"})

    code, payload = bridge.sync_status_payload()

    assert code == 503
    assert payload["ok"] is False
    assert payload["status"] == ""
    assert payload["error"] == "native_status_timeout"
    assert payload["source_store"]["canonical_reconciler"] == {
        "thread_alive": True,
        "phase": "native_ingest",
    }
    rendered = json.dumps(payload)
    assert "exception-only-private" not in rendered
    assert "/private/path" not in rendered
    assert "PRIVATE RAW SESSION" not in rendered
    assert "SECRET PROVIDER PAYLOAD" not in rendered


def test_get_sync_status_still_runs_full_native_status(monkeypatch):
    calls = []
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(
        bridge,
        "source_state",
        lambda: {"configured": True, "ready": True, "documents": 0},
    )
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "ready"})

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return 0, "chunks: 12\n", ""

    monkeypatch.setattr(bridge, "run", fake_run)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection(*server.server_address)
        conn.request(
            "GET", "/sync/status", headers={"Authorization": "Bearer test-token"}
        )
        response = conn.getresponse()
        body = json.loads(response.read())
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status == 200
    assert calls == [(("status", "owner/memory"), {"timeout": 30})]
    assert body["status"] == "chunks: 12\n"


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
    status, body, _headers = _post_with_headers(server, path, payload, token)
    return status, body


def _post_with_headers(server, path, payload, token="test-token"):
    conn = HTTPConnection(*server.server_address)
    conn.request(
        "POST",
        path,
        json.dumps(payload, ensure_ascii=False).encode(),
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    response = conn.getresponse()
    body = json.loads(response.read())
    headers = dict(response.getheaders())
    conn.close()
    return response.status, body, headers


def test_server_script_entry_can_filter_warm_memory(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    env = os.environ.copy()
    for name in ("FUNES_STORAGE_REPO", "FUNES_INDEX_MEMORY"):
        env.pop(name, None)
    env.update(
        {
            "FUNES_API_TOKEN": "test-token",
            "FUNES_BIN": "/usr/bin/false",
            "FUNES_HOME": str(tmp_path),
            "FUNES_MEMORY": "owner/canonical-memory",
            "PYTHONPATH": str(Path(bridge.__file__).parents[1]),
            "PORT": str(port),
        }
    )
    process = subprocess.Popen(
        [sys.executable, str(Path(bridge.__file__))],
        cwd=Path(bridge.__file__).parents[1],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    server = SimpleNamespace(server_address=("127.0.0.1", port))
    deadline = time.monotonic() + 60
    try:
        while True:
            if process.poll() is not None:
                pytest.fail(f"server exited early: {(process.stderr.read() or '')[-1000:]}")
            try:
                status, body = _post(
                    server, "/warm", {"memory": "owner/legacy-memory"}
                )
                break
            except OSError:
                if time.monotonic() >= deadline:
                    pytest.fail("server did not accept warm requests within 60 seconds")
                time.sleep(0.05)
        assert status == 200
        assert body["skipped"] is True
        assert body["reason"] == "memory_mismatch"
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def test_warm_skips_mismatched_memory_without_breaking_existing_callers(monkeypatch):
    calls = []
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "REMOTE", "owner/canonical-memory")
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "ready"})
    monkeypatch.setattr(
        bridge,
        "request_warm",
        lambda **kwargs: calls.append(kwargs) or {"state": "warming"},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, mismatch = _post(
            server, "/warm", {"memory": "owner/legacy-memory"}
        )
        uri_status, uri_matching = _post(
            server,
            "/warm",
            {"memory": "hf://datasets/owner/canonical-memory"},
        )
        matching_status, matching = _post(
            server, "/warm", {"memory": "owner/canonical-memory"}
        )
        monkeypatch.setattr(
            bridge, "REMOTE", "hf://datasets/owner/canonical-memory"
        )
        shorthand_status, shorthand_matching = _post(
            server, "/warm", {"memory": "owner/canonical-memory"}
        )
        legacy_status, legacy = _post(server, "/warm", {})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 200
    assert mismatch == {
        "ok": True,
        "skipped": True,
        "reason": "memory_mismatch",
        "native_warm": {"state": "ready"},
    }
    assert matching_status == 202
    assert matching["native_warm"]["state"] == "warming"
    assert uri_status == 202
    assert uri_matching["native_warm"]["state"] == "warming"
    assert shorthand_status == 202
    assert shorthand_matching["native_warm"]["state"] == "warming"
    assert legacy_status == 202
    assert legacy["native_warm"]["state"] == "warming"
    assert calls == [
        {"force": True},
        {"force": True},
        {"force": True},
        {"force": True},
    ]


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
            {"query": "新的中文正文", "source_type": "memory", "limit": 3},
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
    assert bridge.canonical_source_version(
        {**base, "embedding_generation": 1}
    ) != bridge.canonical_source_version({**base, "embedding_generation": 2})


def test_canonical_source_version_generation_zero_preserves_legacy_hash():
    item = {
        "source_version": "raw-v1",
        "content_hash": "hash",
        "translation_hash": "translation-hash",
        "translation_version": "translation-version",
        "retrieval_text": "english retrieval",
        "source_missing": False,
        "native_generation": 3,
        "embedding_generation": 0,
    }
    assert bridge.canonical_source_version(
        item, {"fingerprint": "fixed-profile"}
    ) == "a16edaffcadde01adc63c9e51e346fc10dc3bbe77c6276582e268f6370421ea5"
    assert bridge.canonical_source_version(
        {**item, "embedding_generation": 7}, {"fingerprint": "fixed-profile"}
    ) == (
        "~funes-eg-v1:00000000000000000007:"
        "a16edaffcadde01adc63c9e51e346fc10dc3bbe77c6276582e268f6370421ea5"
    )


def test_canonical_source_version_keeps_epoch_prefix_across_shadow_changes():
    base = {
        "source_version": "raw-v1",
        "content_hash": "hash",
        "embedding_generation": 7,
    }
    left = bridge.canonical_source_version(
        {**base, "retrieval_text": "first"}, {"fingerprint": "fixed-profile"}
    )
    right = bridge.canonical_source_version(
        {**base, "retrieval_text": "second"}, {"fingerprint": "fixed-profile"}
    )
    assert left != right
    assert left.startswith("~funes-eg-v1:00000000000000000007:")
    assert right.startswith("~funes-eg-v1:00000000000000000007:")


def test_canonical_document_carries_embedding_generation():
    item = {
        "source_identity": "memory-section",
        "source_version": "raw-v1",
        "raw_text": "raw",
        "retrieval_text": "retrieval",
        "content_hash": "hash",
        "updated_at": "2026-09-15T00:00:00Z",
        "embedding_generation": 7,
    }
    assert bridge.canonical_document(item)["embedding_generation"] == 7


def test_opaque_source_id_matches_native_domain_separated_contract():
    assert bridge._opaque_source_id("identity-with-secret-marker") == (
        "sha256:1cfefd9638aac282a2a112dc1cfa62a18376b837315a9b67f6687f697168af9c"
    )


def test_embedding_profile_matches_native_contract(monkeypatch):
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("FUNES_EMBEDDING_MODEL", "voyage-4-lite")
    monkeypatch.setenv("FUNES_EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("FUNES_EMBEDDING_SCHEMA_VERSION", "2")
    profile = bridge.embedding_profile()
    contract = (
        "provider=voyage\n"
        "model=voyage-4-lite\n"
        "dimensions=1024\n"
        "schema_version=2\n"
        "document_input=document\n"
        "query_input=query\n"
        "normalization=l2\n"
        "metric=l2"
    )
    assert profile == {
        "provider": "voyage",
        "model": "voyage-4-lite",
        "dimensions": 1024,
        "schema_version": 2,
        "fingerprint": hashlib.sha256(contract.encode()).hexdigest(),
    }


def test_native_environment_uses_safe_production_defaults(monkeypatch, tmp_path):
    for name in (
        "FUNES_EMBEDDING_PROVIDER",
        "FUNES_EMBEDDING_MODEL",
        "FUNES_EMBEDDING_DIMENSIONS",
        "FUNES_EMBEDDING_SCHEMA_VERSION",
        "FUNES_RERANK_PROVIDER",
        "FUNES_NATIVE_FALLBACK",
        "FUNES_RETRIEVAL_LANGUAGE_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    env = bridge.native_environment(tmp_path)
    assert env["FUNES_HOME"] == str(tmp_path)
    assert env["FUNES_EMBEDDING_PROVIDER"] == "voyage"
    assert env["FUNES_EMBEDDING_MODEL"] == "voyage-4-lite"
    assert env["FUNES_EMBEDDING_DIMENSIONS"] == "1024"
    assert env["FUNES_EMBEDDING_SCHEMA_VERSION"] == "2"
    assert env["FUNES_RERANK_PROVIDER"] == "none"
    assert env["FUNES_NATIVE_FALLBACK"] == "false"
    assert env["FUNES_MCP_PIN_MEMORY"] == "true"
    assert env["FUNES_RETRIEVAL_LANGUAGE_MODE"] == "raw"


def test_blue_green_build_profile_isolated_from_active_query_profile(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(bridge, "REMOTE", "owner/bge-active")
    monkeypatch.setattr(bridge, "INDEX_REMOTE", "owner/voyage-build")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("FUNES_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("FUNES_EMBEDDING_DIMENSIONS", "384")
    monkeypatch.setenv("FUNES_EMBEDDING_SCHEMA_VERSION", "1")
    monkeypatch.setenv("FUNES_INDEX_EMBEDDING_PROVIDER", "voyage")
    for name in (
        "FUNES_INDEX_EMBEDDING_MODEL",
        "FUNES_INDEX_EMBEDDING_DIMENSIONS",
        "FUNES_INDEX_EMBEDDING_SCHEMA_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)

    active = bridge.embedding_profile()
    build = bridge.index_embedding_profile()
    active_env = bridge.native_environment(tmp_path, active)
    build_env = bridge.native_environment(tmp_path, build)

    assert bridge.index_memory() == "owner/voyage-build"
    assert (active["provider"], active["model"], active["dimensions"]) == (
        "local",
        "BAAI/bge-small-en-v1.5",
        384,
    )
    assert (build["provider"], build["model"], build["dimensions"]) == (
        "voyage",
        "voyage-4-lite",
        1024,
    )
    assert active["fingerprint"] != build["fingerprint"]
    assert active_env["FUNES_EMBEDDING_PROVIDER"] == "local"
    assert build_env["FUNES_EMBEDDING_PROVIDER"] == "voyage"
    assert build_env["FUNES_EMBEDDING_DIMENSIONS"] == "1024"


def test_run_uses_explicit_build_profile_without_mutating_active_profile(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("FUNES_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("FUNES_EMBEDDING_DIMENSIONS", "384")
    monkeypatch.setenv("FUNES_EMBEDDING_SCHEMA_VERSION", "1")
    build = bridge._embedding_profile("voyage", "voyage-4-lite", 1024, 2)
    captured = {}

    class FakeProcess:
        pid = 123
        returncode = 0

        @staticmethod
        def communicate(**kwargs):
            captured["timeout"] = kwargs["timeout"]
            return "ok", ""

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        captured["start_new_session"] = kwargs["start_new_session"]
        return FakeProcess()

    monkeypatch.setattr(bridge, "HOME", tmp_path)
    monkeypatch.setattr(bridge.subprocess, "Popen", fake_popen)
    assert bridge.run("status", "owner/build", profile=build) == (0, "ok", "")
    assert captured["argv"] == [bridge.FUNES_BIN, "status", "owner/build"]
    assert captured["env"]["FUNES_EMBEDDING_PROVIDER"] == "voyage"
    assert captured["timeout"] == 180
    assert captured["start_new_session"] is (bridge.os.name == "posix")
    assert bridge.embedding_profile()["provider"] == "local"


def test_run_timeout_reaps_process_group_and_discards_sensitive_output(
    monkeypatch, tmp_path
):
    captured = {"calls": 0}

    class TimedOutProcess:
        pid = 456
        returncode = None

        def communicate(self, **kwargs):
            captured["calls"] += 1
            if captured["calls"] == 1:
                raise bridge.subprocess.TimeoutExpired(
                    ["PRIVATE PATH", "owner/private"],
                    kwargs["timeout"],
                    output="PRIVATE RAW SESSION",
                    stderr="SECRET PROVIDER PAYLOAD",
                )
            assert kwargs == {}
            self.returncode = -9
            return "PRIVATE RAW SESSION", "SECRET PROVIDER PAYLOAD"

        def kill(self):
            captured["killed"] = True

    monkeypatch.setattr(bridge, "HOME", tmp_path)
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *_args, **_kwargs: TimedOutProcess())
    monkeypatch.setattr(
        bridge.os,
        "killpg",
        lambda pid, sig: captured.update(killpg=(pid, sig)),
    )

    with pytest.raises(bridge.subprocess.TimeoutExpired) as raised:
        bridge.run("status", "owner/private", timeout=0.01)

    assert captured["calls"] == 2
    if bridge.os.name == "posix":
        assert captured["killpg"] == (456, bridge.signal.SIGKILL)
    else:
        assert captured["killed"] is True
    rendered = str(raised.value)
    assert "owner/private" not in rendered
    assert "PRIVATE RAW SESSION" not in rendered
    assert "SECRET PROVIDER PAYLOAD" not in rendered


@pytest.mark.skipif(bridge.os.name != "posix", reason="POSIX process groups required")
def test_run_timeout_reaps_real_descendant_and_cleans_private_tmpdir(
    monkeypatch, tmp_path
):
    pid_path = tmp_path / "descendant.pid"
    marker = "PRIVATE RAW SESSION MARKER"
    code = (
        "import os,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        f"open({str(pid_path)!r},'w').write(str(child.pid));"
        f"open(os.path.join(os.environ['TMPDIR'],'raw.txt'),'w').write({marker!r});"
        "time.sleep(60)"
    )
    monkeypatch.setattr(bridge, "FUNES_BIN", bridge.sys.executable)
    monkeypatch.setattr(bridge.tempfile, "tempdir", str(tmp_path))

    with pytest.raises(bridge.subprocess.TimeoutExpired) as raised:
        bridge.run("-c", code, timeout=0.25)

    descendant = int(pid_path.read_text())
    state = ""
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        probe = bridge.subprocess.run(
            ["/bin/ps", "-o", "stat=", "-p", str(descendant)],
            capture_output=True,
            text=True,
        )
        state = probe.stdout.strip()
        if not state or state.startswith("Z"):
            break
        time.sleep(0.025)
    assert not state or state.startswith("Z")
    assert list(tmp_path.glob("funes-native-*")) == []
    rendered = str(raised.value)
    assert marker not in rendered
    assert str(pid_path) not in rendered


def test_blue_green_reconcile_writes_only_build_target_and_does_not_warm_active(
    monkeypatch, tmp_path
):
    row = {
        "source_identity": "durable-row",
        "source_version": "raw-v1",
        "content_hash": "content-hash",
        "raw_text": "原始中文",
        "retrieval_text": "legacy shadow",
        "updated_at": "2026-09-13T01:02:03Z",
        "native_generation": 0,
    }

    class FakeStore:
        def __init__(self):
            self.candidate_args = None
            self.updates = []
            self.marker = None

        def canonical_index_candidates(self, limit, profile, memory):
            self.candidate_args = (limit, profile, memory)
            return [dict(row)]

        def get(self, identity):
            return dict(row) if identity == row["source_identity"] else None

        def native_optimize_checkpoint(self):
            return {
                "status": None,
                "fingerprint": None,
                "memory": None,
                "revision": 0,
            }

        def update_native_index(self, updates):
            self.updates.extend(updates)

        def set_native_optimize_checkpoint(self, marker):
            self.marker = marker

        def native_index_state_record(self, updates):
            return {
                "_funes_record": "native_index_state",
                "state_version": 1,
                "profile": updates[0]["native_index_profile"],
                "memory": updates[0]["native_index_memory"],
                "revision": 1,
                "eligible": 1,
                "indexed": 1,
                "held": 0,
                "invalid": 0,
                "index_fingerprint": "projected",
            }

    store = FakeStore()
    syncer = _SourceSyncer()
    app = SimpleNamespace(store=store, syncer=syncer)
    calls = []
    monkeypatch.setattr(bridge, "REMOTE", "owner/bge-active")
    monkeypatch.setattr(bridge, "INDEX_REMOTE", "owner/voyage-build")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("FUNES_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("FUNES_EMBEDDING_DIMENSIONS", "384")
    monkeypatch.setenv("FUNES_EMBEDDING_SCHEMA_VERSION", "1")
    monkeypatch.setenv("FUNES_INDEX_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("FUNES_INDEX_EMBEDDING_MODEL", "voyage-4-lite")
    monkeypatch.setenv("FUNES_INDEX_EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("FUNES_INDEX_EMBEDDING_SCHEMA_VERSION", "2")
    monkeypatch.setattr(
        bridge,
        "request_warm",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("a staging commit must not warm the active BGE memory")
        ),
    )

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return 0, "ingested sources=1 chunks=1 unchanged=0 stale=0 held=0 commit=voyage\n", ""

    monkeypatch.setattr(bridge, "run", fake_run)
    result = bridge.reconcile_canonical_index(app)

    build = bridge.index_embedding_profile()
    assert result == {"attempted": 1, "indexed": 1, "held": 0, "durable": True}
    assert store.candidate_args == (
        bridge.CANONICAL_INDEX_BATCH,
        build["fingerprint"],
        "owner/voyage-build",
    )
    assert calls[0][0][2:] == ("--memory", "owner/voyage-build")
    assert calls[0][1]["profile"] == build
    assert store.updates[0]["native_index_profile"] == build["fingerprint"]
    assert store.updates[0]["native_index_memory"] == "owner/voyage-build"
    assert store.marker["memory"] == "owner/voyage-build"


def test_pending_translation_raw_is_sent_to_native(monkeypatch, tmp_path):
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
    captured = {}

    def fake_run(*args, **_kwargs):
        captured.update(json.loads(Path(args[1]).read_text()))
        return 0, "ingested sources=1 chunks=1 unchanged=0 stale=0 held=0 commit=raw\n", ""

    monkeypatch.setattr(bridge, "run", fake_run)
    monkeypatch.setattr(bridge, "request_warm", lambda **_kwargs: None)
    try:
        result = bridge.reconcile_canonical_index(app)
    finally:
        app.store.close()
    assert result == {"attempted": 1, "indexed": 1, "held": 0, "durable": True}
    assert captured["raw_text"] == "等待翻译"


def test_canonical_scan_waits_for_raw_durability_lock(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    item = _canonical_source(app.store, "durability-race")
    scanned = threading.Event()
    original_candidates = app.store.canonical_index_candidates

    def candidates(limit, profile_fingerprint="", memory=""):
        scanned.set()
        return original_candidates(limit, profile_fingerprint, memory)

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


def test_canonical_jsonl_sends_raw_and_profile_checkpoint_is_durable(monkeypatch, tmp_path):
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
        optimize_checkpoint = app.store.native_optimize_checkpoint()
    finally:
        app.store.close()
    document = captured["document"]
    assert document["raw_text"] == "PRIVATE RAW SOURCE"
    assert "text" not in document
    assert set(("source_identity", "source_version", "raw_text", "retrieval_text", "content_hash", "updated_at", "metadata")) <= set(document)
    assert document["updated_at"] == "2026-09-13T01:02:03Z"
    assert document["metadata"]["source_version"] == "raw-v1"
    assert document["session_id"] == bridge.canonical_reference("memory-section")
    assert result == {"attempted": 1, "indexed": 1, "held": 0, "durable": True}
    assert stored["native_index_status"] == "indexed"
    assert stored["native_index_version"] == document["source_version"]
    assert stored["native_index_profile"] == bridge.embedding_profile()["fingerprint"]
    assert stored["native_index_memory"] == "owner/memory"
    assert stored["native_indexed_at"]
    assert optimize_checkpoint["status"] == "pending"
    assert optimize_checkpoint["revision"] == 1
    durable_records = app.syncer.uploads[-1]
    assert any(
        item.get("_funes_record") == "native_optimize_checkpoint"
        for item in durable_records
    )
    assert durable_records[-1]["_funes_record"] == "native_index_state"
    assert "raw_text" not in durable_records[-1]
    assert warm == [{"force": True}]


def test_canonical_commit_refresh_uses_app_scoped_cooldown(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    for identity in ("first", "second", "third"):
        _canonical_source(app.store, identity)
    clock = [100.0]
    warm = []
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "INDEX_REMOTE", "")
    monkeypatch.setattr(bridge, "CANONICAL_INDEX_BATCH", 1)
    monkeypatch.setattr(bridge, "CANONICAL_REFRESH_COOLDOWN", 300.0)
    monkeypatch.setattr(bridge.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(bridge, "request_warm", lambda **kwargs: warm.append(kwargs))
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (
            0,
            "ingested sources=1 chunks=1 unchanged=0 stale=0 held=0 commit=revision\n",
            "",
        ),
    )
    try:
        assert bridge.reconcile_canonical_index(app)["indexed"] == 1
        clock[0] = 101.0
        assert bridge.reconcile_canonical_index(app)["indexed"] == 1
        assert warm == [{"force": True}]

        clock[0] = 401.0
        assert bridge.reconcile_canonical_index(app)["indexed"] == 1
        assert warm == [{"force": True}, {"force": True}]
    finally:
        app.store.close()


def test_canonical_final_backlog_flushes_dirty_refresh_once(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    for identity in ("first", "second"):
        _canonical_source(app.store, identity)
    warm = []
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "INDEX_REMOTE", "")
    monkeypatch.setattr(bridge, "CANONICAL_INDEX_BATCH", 1)
    monkeypatch.setattr(bridge, "CANONICAL_REFRESH_COOLDOWN", 300.0)
    monkeypatch.setattr(bridge.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(bridge, "request_warm", lambda **kwargs: warm.append(kwargs))
    # A failed optimize must not strand the final committed revision behind the
    # cooldown; the trailing dirty flush is independent of optimize success.
    monkeypatch.setattr(bridge, "optimize_native_index", lambda *_args: False)
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *_args, **_kwargs: (
            0,
            "ingested sources=1 chunks=1 unchanged=0 stale=0 held=0 commit=revision\n",
            "",
        ),
    )
    try:
        assert bridge.reconcile_canonical_index(app)["indexed"] == 1
        assert bridge.reconcile_canonical_index(app)["indexed"] == 1
        assert warm == [{"force": True}]

        assert bridge.reconcile_canonical_index(app)["attempted"] == 0
        assert warm == [{"force": True}, {"force": True}]
        assert bridge.reconcile_canonical_index(app)["attempted"] == 0
        assert warm == [{"force": True}, {"force": True}]
    finally:
        app.store.close()


def test_profile_change_rebuilds_durable_session_without_client_reupload(monkeypatch, tmp_path):
    before = SourceStore(str(tmp_path / "before-profile-change"))
    session = _canonical_source(
        before,
        "durable-session",
        source_type="session",
        translation_status="skipped_native_session",
        native_index_status="indexed",
        native_index_profile="old-profile",
    )
    snapshot = tmp_path / "profile-change.jsonl.gz"
    before.snapshot(snapshot)
    before.close()
    restored = SourceStore(str(tmp_path / "after-profile-change"))
    restored.restore(snapshot)
    app = SimpleNamespace(store=restored, syncer=_SourceSyncer())
    captured = []
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "request_warm", lambda **_kwargs: None)

    def fake_run(*args, **_kwargs):
        captured.extend(json.loads(line) for line in Path(args[1]).read_text().splitlines())
        return 0, "ingested sources=1 chunks=1 unchanged=0 stale=0 held=0 commit=new-profile\n", ""

    monkeypatch.setattr(bridge, "run", fake_run)
    try:
        result = bridge.reconcile_canonical_index(app)
        rebuilt = restored.get("durable-session")
    finally:
        restored.close()
    assert session["raw_text"] == captured[0]["raw_text"]
    assert captured[0]["source_identity"] == "durable-session"
    assert result["indexed"] == 1
    assert rebuilt["native_index_profile"] == bridge.embedding_profile()["fingerprint"]


def test_canonical_phase_reports_write_lock_wait_without_row_data(monkeypatch):
    class Store:
        @staticmethod
        def canonical_index_candidates(*_args):
            return []

    app = SimpleNamespace(
        store=Store(),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    bridge._initialize_canonical_reconcile_state(app)
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "INDEX_REMOTE", "")
    monkeypatch.setattr(bridge, "optimize_canonical_index", lambda *_args: False)

    bridge.WRITE_LOCK.acquire()
    thread = threading.Thread(target=bridge.reconcile_canonical_index, args=(app,))
    thread.start()
    try:
        deadline = time.monotonic() + 1
        while (
            bridge.canonical_reconcile_state(app)["phase"] != "waiting_write_lock"
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        status = bridge.canonical_reconcile_state(app)
        assert status["phase"] == "waiting_write_lock"
        assert "owner/memory" not in json.dumps(status)
    finally:
        bridge.WRITE_LOCK.release()
        thread.join(timeout=1)
    assert not thread.is_alive()


def test_source_state_reports_reconciler_while_restore_is_running(monkeypatch):
    app = SimpleNamespace(
        syncer=SimpleNamespace(restoring=True, restore_failed=False),
        store=SimpleNamespace(count=lambda: 17),
        restore_result=0,
    )
    bridge._initialize_canonical_reconcile_state(app)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)

    status = bridge.source_state()

    assert status["ready"] is False
    assert status["restoring"] is True
    assert status["documents"] == 17
    assert status["canonical_reconciler"]["phase"] == "waiting_restore"


def test_canonical_background_marks_non_durable_result_as_failure(monkeypatch):
    app = SimpleNamespace(
        restore_done=threading.Event(),
        canonical_index_stop=threading.Event(),
    )
    app.restore_done.set()
    bridge._initialize_canonical_reconcile_state(app)

    def not_durable(_app):
        app.canonical_index_stop.set()
        return {"attempted": 2, "indexed": 0, "held": 0, "durable": False}

    monkeypatch.setattr(bridge, "reconcile_canonical_index", not_durable)
    bridge._canonical_reconcile_background(app)

    status = bridge.canonical_reconcile_state(app)
    assert status["last_error"] == "status_not_durable"
    assert status["consecutive_failures"] == 1
    assert status["last_result"]["durable"] is False


def test_canonical_background_reports_only_allowlisted_error(monkeypatch):
    app = SimpleNamespace(
        restore_done=threading.Event(),
        canonical_index_stop=threading.Event(),
    )
    app.restore_done.set()
    bridge._initialize_canonical_reconcile_state(app)

    def explode(_app):
        app.canonical_index_stop.set()
        raise RuntimeError("PRIVATE RAW SESSION provider payload")

    monkeypatch.setattr(bridge, "reconcile_canonical_index", explode)
    bridge._canonical_reconcile_background(app)

    status = bridge.canonical_reconcile_state(app)
    assert status["active"] is False
    assert status["phase"] == "sleeping"
    assert status["last_error"] == "unexpected"
    assert status["consecutive_failures"] == 1
    assert status["last_result"] is None
    assert status["last_duration_ms"] >= 0
    assert status["batch_size"] == bridge.CANONICAL_INDEX_BATCH
    assert status["timeout_seconds"] == bridge.CANONICAL_INDEX_TIMEOUT
    assert "PRIVATE RAW SESSION" not in json.dumps(status)
    assert "provider payload" not in json.dumps(status)


def test_canonical_catchup_optimizes_once_and_restores_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "INDEX_REMOTE", "")
    store = SourceStore(str(tmp_path / "optimized-source"))
    item = _canonical_source(store, "optimized-row", raw="durable raw")
    profile = bridge.index_embedding_profile()
    memory = bridge.index_memory()
    document = bridge.canonical_document(item, profile)
    store.update_native_index(
        [
            bridge._native_update(
                item,
                "indexed",
                document["source_version"],
                profile=profile,
                memory=memory,
            )
        ]
    )
    # Simulate an optimized marker written before source_agent_idx became a
    # required native index. It must trigger exactly one structural upgrade.
    legacy_marker = bridge.native_optimize_marker(
        profile,
        "optimized",
        {},
        store.native_index_checkpoint(profile, memory)["index_fingerprint"],
        memory=memory,
    )
    legacy_marker.pop("index_layout_version")
    assert store.set_native_optimize_checkpoint(legacy_marker)
    app = SimpleNamespace(store=store, syncer=_SourceSyncer(), restore_result=0)
    optimize_calls = []
    warm_calls = []
    monkeypatch.setattr(
        bridge,
        "optimize_native_index",
        lambda target, target_profile: optimize_calls.append(
            (target, target_profile)
        )
        or True,
    )
    monkeypatch.setattr(bridge, "request_warm", lambda **kwargs: warm_calls.append(kwargs))
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    checkpoint_scans = []
    native_index_checkpoint = store.native_index_checkpoint

    def counted_checkpoint(target_profile, target):
        checkpoint_scans.append((target_profile["fingerprint"], target))
        return native_index_checkpoint(target_profile, target)

    store.native_index_checkpoint = counted_checkpoint
    try:
        first = bridge.reconcile_canonical_index(app)
        assert len(checkpoint_scans) == 1
        store.native_index_checkpoint = lambda _profile, _memory: (_ for _ in ()).throw(
            AssertionError("steady state must not scan/hash indexed rows")
        )
        second = bridge.reconcile_canonical_index(app)
        store.native_index_checkpoint = native_index_checkpoint
        checkpoint = store.native_optimize_checkpoint()
        status = bridge.source_state()
        snapshot = tmp_path / "optimized.jsonl.gz"
        store.snapshot(snapshot)
    finally:
        store.close()

    restored = SourceStore(str(tmp_path / "restored-optimized-source"))
    restored.restore(snapshot)
    restored_app = SimpleNamespace(store=restored, syncer=_SourceSyncer())
    restored.native_index_checkpoint = lambda _profile, _memory: (_ for _ in ()).throw(
        AssertionError("restored steady state must not scan/hash indexed rows")
    )
    try:
        third = bridge.reconcile_canonical_index(restored_app)
        restored_checkpoint = restored.native_optimize_checkpoint()
    finally:
        restored.close()
    assert first["attempted"] == second["attempted"] == third["attempted"] == 0
    assert optimize_calls == [(memory, profile)]
    assert warm_calls == [{"force": True}]
    assert {key: checkpoint[key] for key in profile} == profile
    assert checkpoint["status"] == "optimized"
    assert checkpoint["memory"] == memory
    assert checkpoint["index_layout_version"] == bridge.CANONICAL_INDEX_LAYOUT_VERSION
    assert restored_checkpoint == checkpoint
    assert {key: status["canonical_index"][key] for key in profile} == profile
    assert status["canonical_index"]["optimize"]["status"] == "optimized"
    assert status["canonical_index"]["cutover_ready"] is True


def test_optimize_native_index_uses_remote_only_command(monkeypatch):
    calls = []
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or (0, "", ""),
    )
    assert bridge.optimize_native_index() is True
    profile = bridge.index_embedding_profile()
    assert calls == [
        (
            ("optimize-index", "owner/memory"),
            {
                "timeout": bridge.CANONICAL_OPTIMIZE_TIMEOUT,
                "profile": profile,
            },
        )
    ]


def test_invalid_canonical_row_isolated_without_blocking_clean_rows(
    monkeypatch, tmp_path
):
    app = _source_app(tmp_path)
    _canonical_source(app.store, "clean-doc")
    _canonical_source(app.store, "poison-doc")
    calls = []
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "INDEX_REMOTE", "")
    monkeypatch.setattr(bridge, "request_warm", lambda **_kwargs: None)

    def invalid_run(*args, **_kwargs):
        identities = [
            json.loads(line)["source_identity"]
            for line in Path(args[1]).read_text().splitlines()
        ]
        calls.append(identities)
        if "poison-doc" in identities:
            return 1, "", "Error: invalid canonical JSONL record at line 1"
        return 0, "ingested sources=1 chunks=1 unchanged=0 stale=0 held=0 commit=clean\n", ""

    monkeypatch.setattr(bridge, "run", invalid_run)
    try:
        result = bridge.reconcile_canonical_index(app)
        clean = app.store.get("clean-doc")
        poison = app.store.get("poison-doc")
        profile = bridge.index_embedding_profile()
        pending = app.store.canonical_index_candidates(
            10, str(profile["fingerprint"]), bridge.index_memory()
        )
    finally:
        app.store.close()

    assert calls == [
        ["clean-doc", "poison-doc"],
        ["clean-doc"],
        ["poison-doc"],
    ]
    assert result == {"attempted": 2, "indexed": 1, "held": 1, "durable": True}
    assert clean["native_index_status"] == "indexed"
    assert poison["native_index_status"] == "held_invalid"
    assert poison["native_index_error"] == "invalid_source"
    assert pending == []


def test_canonical_held_batch_uses_opaque_fast_path_and_source_revision_retries(monkeypatch, tmp_path):
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
        held_source_ids = (
            [bridge._opaque_source_id("dirty-doc")] if held else []
        )
        return 0, (
            f"ingested sources={sources} chunks={sources} unchanged=0 "
            f"stale=0 held={held}{commit} "
            f"held_source_ids={json.dumps(held_source_ids)}\n"
        ), ""

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
    assert calls == [["clean-doc", "dirty-doc"]]
    assert first == {"attempted": 2, "indexed": 1, "held": 1, "durable": True}
    assert clean["native_index_status"] == "indexed"
    assert dirty["native_index_status"] == "held_secret"
    assert second["attempted"] == 0
    assert third["indexed"] == 1
    assert revised["native_index_status"] == "indexed"
    assert len(warm) == 1


@pytest.mark.parametrize(
    "reported_ids",
    [
        None,
        "[not-json]",
        json.dumps(["sha256:" + "f" * 64]),
        "[]",
    ],
    ids=["legacy", "malformed", "unknown", "count-mismatch"],
)
def test_canonical_held_fast_path_rejects_untrusted_identity_reports(
    monkeypatch, tmp_path, reported_ids
):
    app = _source_app(tmp_path)
    _canonical_source(app.store, "clean-doc")
    _canonical_source(app.store, "dirty-doc")
    calls = []
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "request_warm", lambda **_kwargs: None)

    def held_run(*args, **_kwargs):
        identities = [
            json.loads(line)["source_identity"]
            for line in Path(args[1]).read_text().splitlines()
        ]
        calls.append(identities)
        if len(identities) == 2:
            extension = (
                "" if reported_ids is None else f" held_source_ids={reported_ids}"
            )
            return 0, (
                "ingested sources=1 chunks=1 unchanged=0 stale=0 held=1 "
                f"commit=initial{extension}\n"
            ), ""
        if identities == ["dirty-doc"]:
            held_source_ids = json.dumps(
                [bridge._opaque_source_id("dirty-doc")]
            )
            return 0, (
                "ingested sources=0 chunks=0 unchanged=0 stale=0 held=1 "
                f"held_source_ids={held_source_ids}\n"
            ), ""
        return 0, (
            "ingested sources=0 chunks=0 unchanged=1 stale=0 held=0 "
            "held_source_ids=[]\n"
        ), ""

    monkeypatch.setattr(bridge, "run", held_run)
    try:
        result = bridge.reconcile_canonical_index(app)
        clean = app.store.get("clean-doc")
        dirty = app.store.get("dirty-doc")
    finally:
        app.store.close()

    assert calls == [
        ["clean-doc", "dirty-doc"],
        ["clean-doc"],
        ["dirty-doc"],
    ]
    assert result == {"attempted": 2, "indexed": 1, "held": 1, "durable": True}
    assert clean["native_index_status"] == "indexed"
    assert dirty["native_index_status"] == "held_secret"


def test_canonical_mixed_stale_and_held_report_still_bisects(
    monkeypatch, tmp_path
):
    app = _source_app(tmp_path)
    _canonical_source(app.store, "stale-doc")
    _canonical_source(app.store, "dirty-doc")
    calls = []
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "request_warm", lambda **_kwargs: None)

    def mixed_run(*args, **_kwargs):
        identities = [
            json.loads(line)["source_identity"]
            for line in Path(args[1]).read_text().splitlines()
        ]
        calls.append(identities)
        if len(identities) == 2:
            held_source_ids = json.dumps(
                [bridge._opaque_source_id("dirty-doc")]
            )
            return 0, (
                "ingested sources=0 chunks=0 unchanged=0 stale=1 held=1 "
                f"held_source_ids={held_source_ids}\n"
            ), ""
        if identities == ["dirty-doc"]:
            held_source_ids = json.dumps(
                [bridge._opaque_source_id("dirty-doc")]
            )
            return 0, (
                "ingested sources=0 chunks=0 unchanged=0 stale=0 held=1 "
                f"held_source_ids={held_source_ids}\n"
            ), ""
        return 0, (
            "ingested sources=0 chunks=0 unchanged=0 stale=1 held=0\n"
        ), ""

    monkeypatch.setattr(bridge, "run", mixed_run)
    try:
        result = bridge.reconcile_canonical_index(app)
        stale = app.store.get("stale-doc")
        dirty = app.store.get("dirty-doc")
    finally:
        app.store.close()

    assert calls == [
        ["stale-doc", "dirty-doc"],
        ["stale-doc"],
        ["dirty-doc"],
    ]
    assert result == {"attempted": 2, "indexed": 0, "held": 1, "durable": True}
    assert stale["native_index_status"] == "retry"
    assert stale["native_index_error"] == "native_stale"
    assert dirty["native_index_status"] == "held_secret"


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


def test_voyage_native_rank_maps_canonical_to_raw_with_indexed_filter(
    monkeypatch, tmp_path
):
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
    app.store.search = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("successful Voyage recall must not run sidecar FTS")
    )
    calls = []

    class FakeWorker:
        def recall(self, query, **kwargs):
            calls.append((query, kwargs))
            reference = bridge.canonical_reference("canonical id")
            return f"CANONICAL ENGLISH SHADOW\n  → get {reference} --from 0 --to 0"

        def get(self, *_args, **_kwargs):
            raise AssertionError("canonical hit must resolve from sidecar")

    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "MCP_WORKER", FakeWorker())
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "query_text",
        lambda _query: (_ for _ in ()).throw(AssertionError("raw mode must keep the raw query")),
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
                "facets": {"source_agent": "codex"},
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == 200
    # This is the wire representation returned by native recall, not a raw
    # sidecar identity that happens to resolve in the test fixture.
    assert bridge.canonical_reference("canonical id") == "funes-doc:Y2Fub25pY2FsIGlk"
    assert body["results"][0]["raw_text"] == "原始 sidecar 正文"
    assert "retrieval_text" not in body["results"][0]
    assert "CANONICAL ENGLISH SHADOW" not in body["results_text"]
    assert provider_calls == []
    assert calls[0][0] == "english retrieval"
    assert body["retrieval_query"] == "english retrieval"
    assert calls[0][1]["source_agent"] == "codex"


@pytest.mark.parametrize(
    ("restoring", "restore_failed", "degraded"),
    (
        (True, False, "source_restore_in_progress"),
        (False, True, "source_restore_failed"),
    ),
)
def test_voyage_native_search_uses_structured_hits_while_source_is_unavailable(
    monkeypatch, restoring, restore_failed, degraded
):
    class Store:
        def __getattribute__(self, name):
            raise AssertionError(f"degraded native search must not access store.{name}")

    calls = []

    class FakeWorker:
        process = SimpleNamespace(poll=lambda: None)

        def recall_result(self, query, **kwargs):
            calls.append((query, kwargs))
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "agent text must never be parsed or returned",
                    }
                ],
                "structuredContent": {
                    "hits": [
                        {
                            "raw_text": "原始 native 正文",
                            "session_id": "session-1",
                            "seq": 7,
                            "timestamp": "2026-09-14T00:00:00Z",
                            "block_type": "response",
                            "harness": "codex",
                            "score": 0.91,
                            "neighbors": [
                                {
                                    "raw_text": "相邻原文",
                                    "seq": 6,
                                    "role": "user",
                                    "block_type": "prompt",
                                    "retrieval_text": "NEIGHBOR SHADOW",
                                }
                            ],
                            "retrieval_text": "DERIVED SECRET SHADOW",
                            "secret": "must-not-leak",
                        }
                    ]
                },
            }

    app = SimpleNamespace(
        store=Store(),
        syncer=SimpleNamespace(
            restoring=restoring,
            restore_failed=restore_failed,
        ),
        restore_result=0,
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "MCP_WORKER", FakeWorker())
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setattr(bridge, "_MCP_WORKER_CONFIG", bridge._native_worker_config())
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "warming"})
    monkeypatch.setattr(
        bridge,
        "search_source_bm25_rankings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("restore degradation must disable all sidecar BM25")
        ),
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(
            server,
            "/search",
            {"query": "cold restore query", "limit": 3, "source_agent": "codex"},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 200
    assert calls and calls[0][0] == "cold restore query"
    assert body["retrieval_degraded"] == degraded
    assert body["retrieval_backend"] == "voyage_lance_bm25_rrf"
    assert body["results_text"] == "原始 native 正文"
    assert body["results"] == [
        {
            "raw_text": "原始 native 正文",
            "session_id": "session-1",
            "seq": 7,
            "timestamp": "2026-09-14T00:00:00Z",
            "block_type": "response",
            "harness": "codex",
            "score": 0.91,
            "neighbors": [
                {
                    "raw_text": "相邻原文",
                    "seq": 6,
                    "role": "user",
                    "block_type": "prompt",
                }
            ],
            "retrieval_backend": "native_funes",
        }
    ]
    assert "agent text" not in json.dumps(body, ensure_ascii=False)
    assert "DERIVED SECRET SHADOW" not in json.dumps(body, ensure_ascii=False)
    assert "NEIGHBOR SHADOW" not in json.dumps(body, ensure_ascii=False)
    assert "must-not-leak" not in json.dumps(body, ensure_ascii=False)


@pytest.mark.parametrize(
    "payload",
    (
        {"query": "query", "role": "user"},
        {"query": "query", "since": "2026-09-01"},
        {"query": "query", "until": "2026-09-14"},
    ),
)
def test_source_only_search_filters_fail_fast_during_restore(monkeypatch, payload):
    class Store:
        def __getattribute__(self, name):
            raise AssertionError(f"restore gate must not access store.{name}")

    class FakeWorker:
        def recall_result(self, *_args, **_kwargs):
            raise AssertionError("source-only filters must not call native")

    app = SimpleNamespace(
        store=Store(),
        syncer=SimpleNamespace(restoring=True, restore_failed=False),
        restore_result=0,
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "MCP_WORKER", FakeWorker())
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")

    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        status, body = _post(server, "/search", payload)
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert elapsed < 0.25
    assert status == 503
    assert body["error"] == "restore_in_progress"


def test_native_unavailable_during_restore_fails_without_sidecar_fallback(monkeypatch):
    class Store:
        def __getattribute__(self, name):
            raise AssertionError(f"native failure must not access store.{name}")

    class FakeWorker:
        def recall_result(self, *_args, **_kwargs):
            raise bridge.NativeMcpError("native unavailable")

    app = SimpleNamespace(
        store=Store(),
        syncer=SimpleNamespace(restoring=True, restore_failed=False),
        restore_result=0,
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "MCP_WORKER", FakeWorker())
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "warm_state", lambda: {"state": "ready"})
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setattr(
        bridge,
        "search_source_bm25_rankings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native failure during restore must not fall back")
        ),
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        status, body = _post(server, "/search", {"query": "query"})
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert elapsed < 0.25
    assert status == 503
    assert body["error"] == "native_mcp_unavailable"


def test_sidecar_search_rrf_queries_raw_and_rewrite_with_filters(monkeypatch):
    calls = []

    class Translator:
        def rewrite_query(self, query):
            calls.append(("rewrite", query))
            return "provider query"

    class Store:
        def search(self, query, limit, filters, *, allow_broad_scan):
            calls.append(("search", query, limit, filters, allow_broad_scan))
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
        ("search", "raw query", 9, filters, False),
        ("search", "provider query", 9, filters, False),
    ]


def test_sidecar_search_partitions_native_sessions_and_filters_harness(monkeypatch):
    class Store:
        def search(self, query, _limit, filters, *, allow_broad_scan):
            assert filters == {}
            assert allow_broad_scan is False
            if query == "raw query":
                return [
                    {"source_identity": "memory", "source_type": "memory", "source_agent": "codex", "raw_text": "memory raw", "retrieval_text": "shadow"},
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
    assert [[item["source_identity"] for item in ranking] for ranking in primary] == [
        ["memory", "codex", "claude", "claude-code"],
        ["pi"],
    ]
    assert [[item["source_identity"] for item in ranking] for ranking in fallback] == [
        ["codex", "claude", "claude-code"],
        ["pi"],
    ]
    assert all(
        "retrieval_text" not in item
        for ranking in [*primary, *fallback]
        for item in ranking
    )

    _, claude_primary, claude_fallback = bridge.search_source_rankings(
        "raw query", 5, {}, harness="claude_code"
    )
    assert [item["source_identity"] for item in claude_primary[0]] == [
        "claude",
        "claude-code",
    ]
    assert [item["source_identity"] for item in claude_fallback[0]] == [
        "claude",
        "claude-code",
    ]
    _, _, unknown_fallback = bridge.search_source_rankings(
        "raw query", 5, {}, harness="unknown"
    )
    assert unknown_fallback == []


def test_voyage_bm25_fallback_is_one_raw_fts_and_strips_retrieval_text(monkeypatch):
    calls = []
    hit = {
        "source_identity": "raw-hit",
        "source_type": "memory",
        "raw_text": "raw source of truth",
        "retrieval_text": "derived shadow",
    }

    class Store:
        def search(self, query, limit, filters, *, allow_broad_scan):
            calls.append((query, limit, filters, allow_broad_scan))
            return [hit]

    app = SimpleNamespace(
        translator=SimpleNamespace(
            rewrite_query=lambda _query: (_ for _ in ()).throw(
                AssertionError("fast Voyage fallback must not call the provider")
            )
        ),
        store=Store(),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)

    primary, fallback = bridge.search_source_bm25_rankings(
        "previous_response_id", 3, {"repo": "owner/repo"}
    )

    assert calls == [("previous_response_id", 9, {"repo": "owner/repo"}, False)]
    assert primary == [[{key: value for key, value in hit.items() if key != "retrieval_text"}]]
    assert fallback == []


def test_deadline_runner_is_single_flight_when_dependency_ignores_timeout():
    release = threading.Event()
    finished = threading.Event()
    calls = []
    thread_name = "test-funes-single-flight"

    def blocked():
        calls.append(True)
        try:
            release.wait(40)
        finally:
            finished.set()

    try:
        for _ in range(6):
            with pytest.raises((RuntimeError, TimeoutError)):
                bridge._run_before_deadline(
                    blocked,
                    time.monotonic() + 0.02,
                    thread_name=thread_name,
                )
        assert calls == [True]
        assert sum(
            thread.name == thread_name for thread in threading.enumerate()
        ) == 1
    finally:
        release.set()
        finished.wait(1)


def test_deadline_runner_rejects_foreign_invocation_without_sharing_result():
    release = threading.Event()
    started = threading.Event()
    first_done = threading.Event()
    first_result = []
    calls = []
    thread_name = "test-funes-single-flight-foreign-result"

    def first():
        calls.append("A")
        started.set()
        release.wait(1)
        return "sentinel A"

    def call_first():
        try:
            first_result.append(
                bridge._run_before_deadline(
                    first, time.monotonic() + 1, thread_name=thread_name
                )
            )
        finally:
            first_done.set()

    caller = threading.Thread(target=call_first, daemon=True)
    caller.start()
    try:
        assert started.wait(1)
        with pytest.raises(bridge._DeadlineRunBusyError):
            bridge._run_before_deadline(
                lambda: calls.append("B") or "sentinel B",
                time.monotonic() + 1,
                thread_name=thread_name,
            )
        assert calls == ["A"]
        assert sum(thread.name == thread_name for thread in threading.enumerate()) == 1
    finally:
        release.set()
        assert first_done.wait(1)
        caller.join(timeout=1)

    assert first_result == ["sentinel A"]


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
        store=SimpleNamespace(
            search=lambda _query, _limit, filters, **_kwargs: [session]
        ),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    _, primary, fallback = bridge.search_source_rankings("query", 3, filters)
    assert primary == [[{key: value for key, value in session.items() if key != "retrieval_text"}]]
    assert fallback == []


@pytest.mark.parametrize(
    "filters",
    (
        {"role": "user"},
        {"since": "2026-09-01"},
        {"until": "2026-09-02"},
        {"source_type": "memory"},
        {"project": "funes"},
        {"repo": "pjpjq/funes"},
        {"device_id": "dev-test"},
        {"content_type": "memory"},
        {"source_missing": False},
    ),
)
def test_http_voyage_unindexed_filters_remain_sidecar_authoritative(
    monkeypatch, filters
):
    calls = []
    hit = {
        "source_identity": "filtered",
        "source_type": "memory",
        "raw_text": "filtered raw",
        "retrieval_text": "derived shadow",
    }

    class Store:
        def search(self, query, limit, *, filters, allow_broad_scan):
            calls.append((query, limit, filters, allow_broad_scan))
            return [hit]

    app = SimpleNamespace(
        translator=SimpleNamespace(rewrite_query=lambda query: query),
        store=Store(),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "recall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unindexed filters must not enter native recall")
        ),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(
            server,
            "/search",
            {"query": "raw query", "limit": 3, **filters},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 200
    assert calls == [("raw query", 9, filters, False)]
    assert body["retrieval_backend"] == "bm25"
    assert body["results"] == [
        {key: value for key, value in hit.items() if key != "retrieval_text"}
    ]
    assert body["results_text"] == "filtered raw"
    assert "retrieval_degraded" not in body


def test_http_voyage_role_filter_never_waits_for_query_translator(monkeypatch):
    release_translator = threading.Event()
    translator_finished = threading.Event()
    translator_calls = []
    hit = {"source_identity": "role-hit", "raw_text": "role filtered raw"}

    def blocked_rewrite(query):
        translator_calls.append(query)
        try:
            release_translator.wait(40)
            return query
        finally:
            translator_finished.set()

    app = SimpleNamespace(
        translator=SimpleNamespace(rewrite_query=blocked_rewrite),
        store=SimpleNamespace(search=lambda *_args, **_kwargs: [hit]),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    watchdog = threading.Timer(5.2, release_translator.set)
    watchdog.daemon = True
    watchdog.start()
    started = time.monotonic()
    try:
        status, body = _post(
            server,
            "/search",
            {"query": "raw query", "limit": 3, "role": "user"},
        )
        elapsed = time.monotonic() - started
    finally:
        release_translator.set()
        watchdog.cancel()
        translator_finished.wait(1)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert bridge.VOYAGE_HTTP_TIMEOUT <= 4.5
    assert elapsed < 5
    assert translator_calls == []
    assert status == 200
    assert body["results"] == [hit]


def test_voyage_role_filter_applies_harness_to_sidecar_sessions(monkeypatch):
    hits = [
        {
            "source_identity": "codex-session",
            "source_type": "session",
            "source_agent": "codex",
            "role": "user",
            "raw_text": "codex raw",
        },
        {
            "source_identity": "claude-session",
            "source_type": "session",
            "source_agent": "claude",
            "role": "user",
            "raw_text": "claude raw",
        },
    ]
    app = SimpleNamespace(
        store=SimpleNamespace(search=lambda *_args, **_kwargs: hits),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)

    primary, fallback = bridge.search_source_bm25_rankings(
        "raw query", 3, {"role": "user"}, harness="codex"
    )

    assert [item["source_identity"] for item in primary[0]] == ["codex-session"]
    assert fallback == []


def test_http_voyage_uses_native_rrf_order_and_keeps_sidecar_raw(monkeypatch, tmp_path):
    app = _source_app(tmp_path)
    for identity, raw in (
        ("both", "dual raw"),
        ("native-only", "native raw"),
    ):
        _canonical_source(app.store, identity, raw=raw, retrieval="ENGLISH SHADOW")
    native_queries = []
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "search_source_rankings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("successful Voyage recall must not run sidecar FTS")
        ),
    )
    def native_recall(query, **_kwargs):
        native_queries.append(query)
        return "\n".join(
            (
                "native\n  → get native-only --from 0 --to 0",
                "dual\n  → get both --from 0 --to 0",
            )
        )

    monkeypatch.setattr(bridge, "recall", native_recall)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(server, "/search", {"query": "raw query", "limit": 5})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == 200
    assert native_queries == ["raw query"]
    assert body["retrieval_query"] == "raw query"
    assert body["retrieval_backend"] == "voyage_lance_bm25_rrf"
    assert body["embedding_profile"]["provider"] == "voyage"
    assert [item["source_identity"] for item in body["results"]] == [
        "native-only",
        "both",
    ]
    assert body["results_text"] == "native raw\n\ndual raw"
    assert "ENGLISH SHADOW" not in json.dumps(body["results"], ensure_ascii=False)


def test_http_selective_bm25_fuses_full_raw_union_with_filters(monkeypatch):
    raw_query = "CPA 第二轮为什么丢上下文？"
    first_results = [
        {
            "source_identity": "dominant-1",
            "session_id": "session-a",
            "raw_text": "Northflank previous_response_id 第一条原文",
        },
        {
            "source_identity": "dominant-2",
            "session_id": "session-a",
            "raw_text": "previous_response_id 第二条原文",
        },
        {
            "source_identity": "dominant-3",
            "session_id": "session-a",
            "raw_text": "previous_response_id 第三条原文",
        },
    ]
    recovered = {
        "source_identity": "recovered-history",
        "session_id": "session-c",
        "raw_text": "历史原文：CPA Northflank previous_response_id",
    }
    recall_calls = []
    bm25_calls = []

    class Store:
        def search(self, query, limit, *, filters, allow_broad_scan):
            assert query == raw_query
            assert limit == 9
            assert filters == {"source_agent": "codex"}
            assert allow_broad_scan is False
            return [
                {**item, "source_type": "session", "source_agent": "codex", "retrieval_text": "DERIVED"}
                for item in [*first_results, recovered]
            ]

    app = SimpleNamespace(
        store=Store(),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    real_bm25 = bridge.search_source_bm25_rankings

    def tracked_bm25(query, limit, filters, harness):
        bm25_calls.append((query, limit, filters, harness))
        return real_bm25(query, limit, filters, harness)

    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setattr(
        bridge,
        "recall",
        lambda query, **_kwargs: recall_calls.append(query) or "first pass",
    )
    monkeypatch.setattr(
        bridge,
        "materialize_native_results",
        lambda *_args, **_kwargs: first_results,
    )
    monkeypatch.setattr(bridge, "search_source_bm25_rankings", tracked_bm25)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(
            server,
            "/search",
            {
                "query": raw_query,
                "limit": 3,
                "source_agent": "codex",
                "harness": "codex",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 200
    assert recall_calls == [raw_query]
    assert bm25_calls == [
        (raw_query, 3, {"source_agent": "codex"}, "codex")
    ]
    assert body["query"] == raw_query
    assert body["retrieval_query"] == raw_query
    recovered_result = next(
        item
        for item in body["results"]
        if item["source_identity"] == recovered["source_identity"]
    )
    assert recovered_result["raw_text"] == recovered["raw_text"]
    assert body["results_text"].split("\n\n") == [
        item["raw_text"] for item in body["results"]
    ]
    assert "DERIVED" not in json.dumps(body, ensure_ascii=False)
    assert not any(
        word in key.lower()
        for key in body
        for word in ("feedback", "identifier", "expanded")
    )
    assert sum(
        item.get("session_id") == "session-a" for item in body["results"][:3]
    ) <= 2


def test_http_selective_bm25_non_trigger_never_opens_sqlite(monkeypatch):
    first_results = [
        {
            "source_identity": f"dominant-{index}",
            "session_id": "session-a",
            "raw_text": f"retry_state raw {index}",
        }
        for index in range(3)
    ]
    recall_calls = []
    app = SimpleNamespace(
        store=SimpleNamespace(),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setattr(
        bridge,
        "recall",
        lambda query, **_kwargs: recall_calls.append(query) or "first pass",
    )
    monkeypatch.setattr(
        bridge,
        "materialize_native_results",
        lambda *_args, **_kwargs: first_results,
    )
    monkeypatch.setattr(
        bridge,
        "search_source_bm25_rankings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ordinary requests must not open SQLite")
        ),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(
            server, "/search", {"query": "why was context lost?", "limit": 3}
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 200
    assert recall_calls == ["why was context lost?"]
    assert body["results"] == first_results


def test_http_selective_bm25_timeout_returns_first_pass_within_shared_deadline(
    monkeypatch,
):
    raw_query = "为什么第二轮丢上下文？"
    first_results = [
        {
            "source_identity": f"dominant-{index}",
            "session_id": "session-a",
            "raw_text": f"retry_state 原始记录 {index}",
        }
        for index in range(3)
    ]
    release = threading.Event()
    finished = threading.Event()
    recall_calls = []
    bm25_calls = []

    def native_recall(query, **_kwargs):
        recall_calls.append(query)
        return "first pass"

    def blocked_bm25(query, limit, filters, harness):
        bm25_calls.append((query, limit, filters, harness))
        try:
            release.wait(40)
            return [], []
        finally:
            finished.set()

    app = SimpleNamespace(
        store=SimpleNamespace(),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setattr(bridge, "VOYAGE_HTTP_TIMEOUT", 0.12)
    monkeypatch.setattr(bridge, "VOYAGE_NATIVE_TIMEOUT", 0.08)
    monkeypatch.setattr(bridge, "recall", native_recall)
    monkeypatch.setattr(
        bridge,
        "materialize_native_results",
        lambda output, *_args, **_kwargs: first_results if output == "first pass" else [],
    )
    monkeypatch.setattr(bridge, "search_source_bm25_rankings", blocked_bm25)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        status, body = _post(server, "/search", {"query": raw_query, "limit": 3})
        elapsed = time.monotonic() - started
    finally:
        release.set()
        finished.wait(1)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert elapsed < 1
    assert status == 200
    assert recall_calls == [raw_query]
    assert bm25_calls == [(raw_query, 3, {}, None)]
    assert body["results"] == first_results
    assert body["retrieval_query"] == raw_query
    assert "retrieval_degraded" not in body


def test_http_selective_bm25_failure_keeps_first_pass_and_hides_error(monkeypatch):
    raw_query = "为什么第二轮丢上下文？"
    first_results = [
        {
            "source_identity": f"dominant-{index}",
            "session_id": "session-a",
            "raw_text": f"retry_state 原始记录 {index}",
        }
        for index in range(3)
    ]
    recall_calls = []

    app = SimpleNamespace(
        store=SimpleNamespace(),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setattr(
        bridge,
        "recall",
        lambda query, **_kwargs: recall_calls.append(query) or "first pass",
    )
    monkeypatch.setattr(
        bridge,
        "materialize_native_results",
        lambda *_args, **_kwargs: first_results,
    )
    monkeypatch.setattr(
        bridge,
        "search_source_bm25_rankings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("sidecar-secret-detail")
        ),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(server, "/search", {"query": raw_query, "limit": 3})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    encoded = json.dumps(body, ensure_ascii=False)
    assert status == 200
    assert recall_calls == [raw_query]
    assert body["results"] == first_results
    assert "sidecar-secret-detail" not in encoded


def test_http_native_bm25_keeps_exact_identifier_without_sidecar_fts(monkeypatch):
    exact = {
        "source_identity": "exact-session",
        "source_type": "session",
        "source_agent": "codex",
        "raw_text": "exact identifier",
        "retrieval_text": "shadow",
    }
    native_queries = []
    app = SimpleNamespace(
        translator=SimpleNamespace(
            rewrite_query=lambda _query: (_ for _ in ()).throw(
                AssertionError("successful Voyage recall must not rewrite the query")
            )
        ),
        store=SimpleNamespace(
            search=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("native BM25 owns exact identifiers")
            )
        ),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "recall",
        lambda query, **_kwargs: native_queries.append(query) or "native exact hit",
    )
    monkeypatch.setattr(
        bridge,
        "materialize_native_results",
        lambda *_args, **_kwargs: [
            {key: value for key, value in exact.items() if key != "retrieval_text"}
        ],
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(server, "/search", {"query": "identifier", "limit": 1})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 200
    assert native_queries == ["identifier"]
    assert body["results"] == [
        {
            "source_identity": "exact-session",
            "source_type": "session",
            "source_agent": "codex",
            "raw_text": "exact identifier",
        }
    ]


@pytest.mark.parametrize(
    ("requested_limit", "expected_identities"),
    ((2, ["first", "second"]), (0, ["first"]), (-1, ["first"])),
)
def test_http_local_exact_sidecar_skips_slow_native(
    monkeypatch, requested_limit, expected_identities
):
    sidecar = [[
        {"source_identity": "first", "raw_text": "previous_response_id exact raw"},
        {"source_identity": "second", "raw_text": "second raw"},
    ]]
    app = SimpleNamespace(syncer=SimpleNamespace(restoring=False, restore_failed=False))
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "LANGUAGE_MODE", "auto")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(
        bridge,
        "search_source_rankings",
        lambda *_args, **_kwargs: ("rewritten query", sidecar, []),
    )
    monkeypatch.setattr(
        bridge,
        "recall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an exact sidecar page must not wait for native")
        ),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(
            server,
            "/search",
            {
                "query": "Codex 调用 CPA 时 previous_response_id 和 chatcmpl-* 的问题",
                "limit": requested_limit,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 200
    assert [item["source_identity"] for item in body["results"]] == expected_identities


@pytest.mark.parametrize(
    ("query", "weak_raw"),
    (
        ("之前 API 为什么失败？", "capital allocation notes"),
        ("之前 error: 为什么失败？", "error details"),
        ("之前 src/ 为什么失败？", "src code"),
        ("之前 foo* 为什么失败？", "foo result"),
        (
            "mostly English API request with many filler words 为什么失败？",
            "capital request notes",
        ),
    ),
)
def test_http_weak_cjk_sidecar_uses_short_native_budget(
    monkeypatch, query, weak_raw
):
    sidecar = [[
        {"source_identity": "weak-first", "raw_text": weak_raw},
        {"source_identity": "weak-second", "raw_text": "另一个弱匹配"},
    ]]
    timeouts = []
    app = SimpleNamespace(syncer=SimpleNamespace(restoring=False, restore_failed=False))
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "LANGUAGE_MODE", "auto")
    monkeypatch.setattr(bridge, "CJK_NATIVE_TIMEOUT", 5.0)
    monkeypatch.setattr(
        bridge,
        "search_source_rankings",
        lambda *_args, **_kwargs: ("rewritten query", sidecar, []),
    )

    def native_recall(*_args, **kwargs):
        timeouts.append(kwargs["timeout"])
        return "native"

    monkeypatch.setattr(bridge, "recall", native_recall)
    monkeypatch.setattr(
        bridge,
        "materialize_native_results",
        lambda *_args, **_kwargs: [
            {"source_identity": "semantic", "raw_text": "强语义结果"}
        ],
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(
            server,
            "/search",
            {"query": query, "limit": 2},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 200
    assert len(timeouts) == 1
    assert 0 < timeouts[0] <= 5.0
    assert any(item["source_identity"] == "semantic" for item in body["results"])


@pytest.mark.parametrize(
    ("native_error", "degraded"),
    (
        (bridge.NativeMcpError("timed out"), "voyage_unavailable"),
        (bridge.NativeMcpBusyError("busy"), "voyage_unavailable"),
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
        "search_source_bm25_rankings",
        lambda query, limit, filters, harness: (source_rankings, source_rankings),
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
    assert body["retrieval_backend"] == "bm25"
    assert body["embedding_profile"]["provider"] == "voyage"
    assert body["results_text"] == "原始 sidecar session 结果"
    assert body["results"] == source_rankings[0]


def test_voyage_blocked_native_returns_sidecar_bm25_within_http_budget(monkeypatch):
    release_native = threading.Event()
    native_finished = threading.Event()
    search_calls = []
    legacy_calls = []

    class Store:
        def get(self, _identity):
            return None

        def search(self, query, limit, filters, *, allow_broad_scan):
            search_calls.append((query, limit, filters, allow_broad_scan))
            return [{"source_identity": "bm25-result", "raw_text": "raw BM25 result"}]

    app = SimpleNamespace(
        translator=SimpleNamespace(
            rewrite_query=lambda _query: (_ for _ in ()).throw(
                AssertionError("degraded Voyage fallback must stay raw and local")
            )
        ),
        store=Store(),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("FUNES_NATIVE_FALLBACK", "false")

    def blocked_native(*_args, **_kwargs):
        try:
            release_native.wait(40)
            return ""
        finally:
            native_finished.set()

    def slow_legacy_native(*_args, **_kwargs):
        legacy_calls.append(True)
        time.sleep(40)
        return 0, "", ""

    monkeypatch.setattr(bridge, "recall", blocked_native)
    monkeypatch.setattr(bridge, "run", slow_legacy_native)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    watchdog = threading.Timer(5.2, release_native.set)
    watchdog.daemon = True
    watchdog.start()
    started = time.monotonic()
    try:
        status, body = _post(server, "/search", {"query": "raw query", "limit": 3})
        elapsed = time.monotonic() - started
    finally:
        release_native.set()
        watchdog.cancel()
        native_finished.wait(1)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert elapsed < 5
    assert status == 200
    assert body["retrieval_degraded"] == "voyage_unavailable"
    assert body["retrieval_backend"] == "bm25"
    assert body["results"] == [
        {"source_identity": "bm25-result", "raw_text": "raw BM25 result"}
    ]
    assert search_calls == [("raw query", 9, {}, False)]
    assert legacy_calls == []


def test_concurrent_voyage_native_request_returns_busy_without_foreign_result(
    monkeypatch,
):
    release_native = threading.Event()
    native_started = threading.Event()
    first_complete = threading.Event()
    native_queries = []
    first_response = []

    class Store:
        def search(self, *_args, **_kwargs):
            # The busy request may use bounded raw BM25 before returning 429.
            return []

    app = SimpleNamespace(
        store=Store(),
        syncer=SimpleNamespace(restoring=False, restore_failed=False),
    )
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")

    def blocked_native(query, **_kwargs):
        native_queries.append(query)
        native_started.set()
        release_native.wait(2)
        return ""

    monkeypatch.setattr(bridge, "recall", blocked_native)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    def first_request():
        try:
            first_response.append(_post(server, "/search", {"query": "first"}))
        finally:
            first_complete.set()

    request_thread = threading.Thread(target=first_request, daemon=True)
    request_thread.start()
    try:
        assert native_started.wait(1)
        status, body, headers = _post_with_headers(
            server, "/search", {"query": "second"}
        )
        assert status == 429
        assert headers["Retry-After"] == "3"
        assert body["error"] == "native_mcp_busy"
        assert native_queries == ["first"]
        assert sum(
            thread.name == "funes-http-native-recall"
            for thread in threading.enumerate()
        ) == 1
    finally:
        release_native.set()
        assert first_complete.wait(1)
        request_thread.join(timeout=1)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    assert first_response[0][0] == 200
    assert first_response[0][1]["query"] == "first"


@pytest.mark.parametrize("has_sidecar", (False, True))
def test_voyage_failure_skips_slow_legacy_fallback(monkeypatch, has_sidecar):
    source_rankings = (
        [[{
            "source_identity": "bm25-result",
            "raw_text": "raw BM25 result",
        }]]
        if has_sidecar
        else []
    )
    app = SimpleNamespace(syncer=SimpleNamespace(restoring=False, restore_failed=False))
    legacy_calls = []
    native_timeouts = []
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("FUNES_NATIVE_FALLBACK", "false")
    monkeypatch.setattr(
        bridge,
        "search_source_bm25_rankings",
        lambda *_args, **_kwargs: (source_rankings, []),
    )

    def unavailable_voyage(*_args, **kwargs):
        native_timeouts.append(kwargs["timeout"])
        raise bridge.NativeMcpError("provider unavailable")

    def slow_legacy_native(*_args, **_kwargs):
        legacy_calls.append(True)
        time.sleep(40)
        return 0, "", ""

    monkeypatch.setattr(bridge, "recall", unavailable_voyage)
    monkeypatch.setattr(bridge, "run", slow_legacy_native)
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        status, body = _post(server, "/search", {"query": "raw query", "limit": 3})
    finally:
        elapsed = time.monotonic() - started
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert elapsed < 5
    assert native_timeouts and 0 < native_timeouts[0] <= bridge.VOYAGE_NATIVE_TIMEOUT
    assert native_timeouts[0] < 5
    assert legacy_calls == []
    if has_sidecar:
        assert status == 200
        assert body["retrieval_degraded"] == "voyage_unavailable"
        assert body["retrieval_backend"] == "bm25"
        assert body["embedding_profile"]["provider"] == "voyage"
        assert body["results"] == source_rankings[0]
    else:
        assert status == 503
        assert body["error"] == "native_mcp_unavailable"
        assert body["retrieval_degraded"] == "voyage_unavailable"
        assert body["retrieval_backend"] == "unavailable"
        assert body["embedding_profile"]["provider"] == "voyage"


def test_http_voyage_native_empty_is_success_without_sidecar_fallback(
    monkeypatch, tmp_path
):
    app = _source_app(tmp_path)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(
        bridge,
        "search_source_rankings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an empty native ranking is still a successful query")
        ),
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
    assert body["results"] == []
    assert body["results_text"] == ""
    assert body["retrieval_backend"] == "voyage_lance_bm25_rrf"
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
    assert calls[0][0] == "CPA previous_response_id 为什么丢上下文？"
    assert calls[0][1]["candidates"] == 12
    assert 0 < calls[0][1]["timeout"] <= bridge.VOYAGE_NATIVE_TIMEOUT


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
