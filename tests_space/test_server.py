import json
import threading
from collections import deque
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import space.server as bridge


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


def test_search_and_get_use_native_worker_and_keep_raw_query(monkeypatch):
    calls = []

    class FakeWorker:
        def recall(self, query, **kwargs):
            calls.append(("recall", query, kwargs))
            return "verbatim native recall"

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
        assert search["results_text"] == "verbatim native recall"
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
    assert calls[1] == ("get", "session-1", {"from_": 2, "to": None})


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


def test_native_bridge_ack_requires_push(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bridge, "HOME", tmp_path)
    (tmp_path / "sources").mkdir()
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "TOKEN", "test-token")

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
    assert status == 200
    assert body["durable"] is True
    assert body["accepted"] == 1
    assert any(call[0] == "push" for call in calls)
    assert any(call[0] == "index" for call in calls)


def test_native_bridge_keeps_queue_when_push_fails(monkeypatch, tmp_path):
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
    assert body["accepted"] == 0
