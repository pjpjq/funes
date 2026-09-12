import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import space.server as bridge


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
