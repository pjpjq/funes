import json
import threading
from collections import deque
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import space.server as bridge
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


def test_translation_provider_accepts_base_url_with_or_without_v1(monkeypatch):
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"context loss CPA"}}]}'

    def fake_urlopen(request, timeout):
        seen.append(request.full_url)
        return Response()

    monkeypatch.setattr(bridge, "LANGUAGE_MODE", "translate")
    monkeypatch.setenv("TRANSLATION_API_KEY", "test-key")
    monkeypatch.setenv("TRANSLATION_MODEL", "test-model")
    monkeypatch.setattr(bridge.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("TRANSLATION_BASE_URL", "https://provider.example/v1")
    assert bridge.query_text("中文 CPA 问题").startswith("context loss CPA")
    monkeypatch.setenv("TRANSLATION_BASE_URL", "https://provider.example")
    assert bridge.query_text("中文 CPA 问题").startswith("context loss CPA")
    assert seen == [
        "https://provider.example/v1/chat/completions",
        "https://provider.example/v1/chat/completions",
    ]


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


def test_request_warm_reserves_state_before_start(monkeypatch):
    starts = []

    class FakeThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            starts.append(self.kwargs)

    monkeypatch.setattr(bridge.threading, "Thread", FakeThread)
    with bridge._WARM_STATE_LOCK:
        bridge._WARM_STATE.update(state="not_started", started_at=None, finished_at=None)

    first = bridge.request_warm()
    second = bridge.request_warm()

    assert first["state"] == "warming"
    assert second["state"] == "warming"
    assert len(starts) == 1


def test_initial_warm_does_not_wait_on_recall_lock(monkeypatch):
    called = threading.Event()

    def fake_refresh():
        called.set()

    monkeypatch.setattr(bridge, "_refresh_native_worker", fake_refresh)
    with bridge._WARM_STATE_LOCK:
        bridge._WARM_STATE.update(state="not_started", started_at=None, finished_at=None)
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
        bridge._WARM_STATE.update(state="warming", started_at="now", finished_at=None)
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
    assert status == 200
    assert body["durable"] is True
    assert body["accepted"] == 1
    assert any(call[0] == "push" for call in calls)
    assert any(call[0] == "index" for call in calls)
    assert warm_calls == []


def test_ingest_does_not_wait_for_recall_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "HOME", tmp_path)
    (tmp_path / "sources").mkdir()
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setattr(bridge, "run", lambda *args, **kwargs: (0, "", ""))
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
    assert result["value"][0] == 200


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
    captured = {}
    monkeypatch.setattr(bridge, "HOME", tmp_path)
    (tmp_path / "sources").mkdir()
    monkeypatch.setattr(bridge, "REMOTE", "owner/memory")
    monkeypatch.setattr(bridge, "TOKEN", "test-token")

    def fake_run(*args, **kwargs):
        if args and args[0] == "index":
            path = next(Path(args[1]).glob("*.jsonl"))
            captured["records"] = [json.loads(line) for line in path.read_text().splitlines()]
        return 0, "", ""

    monkeypatch.setattr(bridge, "run", fake_run)
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
    assert status == 200
    assert body["durable"] is True
    session = captured["records"][0]
    assert session["timestamp"] == document["timestamp"]
    metadata = session["payload"]["metadata"]
    assert metadata["source_agent"] == "codex"
    assert metadata["source_type"] == "agents_md"
    assert metadata["device_id"] == "device-safe-hash"
    assert metadata["source_path"] == "~/code/project/AGENTS.md"
    assert metadata["content_hash"] == "content-hash"
    assert metadata["updated_at"] == "2026-09-01T02:03:04Z"


class _SourceTranslator:
    model = "test-model"

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

    def rewrite(self, query):
        return query


class _SourceSyncer:
    restoring = False
    restore_failed = False

    def __init__(self, durable=True):
        self.durable = durable
        self.uploads = []

    def upload(self, docs=None):
        self.uploads.append(docs)
        return {"uploaded": self.durable, "durable": self.durable}


def _source_app(tmp_path, durable=True):
    store = SourceStore(str(tmp_path / "source-store"))
    return SimpleNamespace(
        store=store,
        translator=_SourceTranslator(),
        syncer=_SourceSyncer(durable),
        restore_result=0,
    )


def _post(server, path, payload):
    conn = HTTPConnection(*server.server_address)
    conn.request(
        "POST",
        path,
        json.dumps(payload, ensure_ascii=False).encode(),
        {"Authorization": "Bearer test-token", "Content-Type": "application/json"},
    )
    response = conn.getresponse()
    body = json.loads(response.read())
    conn.close()
    return response.status, body


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
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        app.store.close()
    assert status == 503
    assert body["durable"] is False
    assert body["error"] == "durability_pending"
