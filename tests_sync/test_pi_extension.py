from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

NODE = shutil.which("node")
EXTENSION = Path(__file__).parents[1] / "integrations" / "pi" / "funes-remote.ts"


def run_harness(
    tmp_path: Path,
    source: str,
    body: str,
    extra_env: dict[str, str] | None = None,
    timeout: int = 15,
) -> subprocess.CompletedProcess[str]:
    config = tmp_path / "config.toml"
    config.write_text(source, encoding="utf-8")
    harness = tmp_path / "harness.mjs"
    harness.write_text(
        f"import extension, {{ configuredRemoteUrl }} from {json.dumps(EXTENSION.as_uri())};\n"
        + body,
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.pop("FUNES_REMOTE_URL", None)
    env.update({
        "HOME": str(tmp_path),
        "FUNES_CONFIG": str(config),
        "FUNES_API_TOKEN": "synthetic-api-token",
        "FUNES_HF_TOKEN": "synthetic-hf-token",
    })
    env.update(extra_env or {})
    return subprocess.run(
        [NODE, "--experimental-strip-types", str(harness)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('[remote]\nurl = "https://remote.example"\n', "https://remote.example"),
        ('[sync]\nremote_url = "https://sync.example"\n', "https://sync.example"),
        ('remote_url = "https://top.example"\n', "https://top.example"),
    ],
)
def test_pi_reads_current_and_legacy_remote_config(tmp_path, source, expected):
    result = run_harness(tmp_path, source, "console.log(configuredRemoteUrl());\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.skipif(
    NODE is None or sys.platform != "linux", reason="Linux secret-file fallback is required"
)
def test_pi_reads_tokens_from_private_env_file(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(dict(self.headers.items()))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true,"results":[]}')

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env_file = tmp_path / "funes.env"
        env_file.write_text(
            "FUNES_API_TOKEN=from-private-file\nFUNES_HF_TOKEN=hf-from-private-file\n",
            encoding="utf-8",
        )
        env_file.chmod(0o600)
        body = (
            "const tools = {};\n"
            "const pi = {registerTool(tool) { tools[tool.name] = tool; }, on() {}};\n"
            "extension(pi);\n"
            "await tools.funes_recall.execute('test', {query: 'history'});\n"
        )
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{server.server_port}"\n',
            body,
            {
                "FUNES_API_TOKEN": "",
                "FUNES_HF_TOKEN": "",
                "HF_TOKEN": "",
                "FUNES_ENV_FILE": str(env_file),
            },
        )
        assert result.returncode == 0, result.stderr
        assert len(requests) == 1
        assert requests[0]["Authorization"] == "Bearer hf-from-private-file"
        assert requests[0]["X-Funes-Authorization"] == "Bearer from-private-file"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(
    NODE is None or sys.platform != "linux", reason="Linux secret-file fallback is required"
)
def test_pi_rejects_public_env_file(tmp_path):
    env_file = tmp_path / "funes.env"
    env_file.write_text("FUNES_API_TOKEN=public-token\n", encoding="utf-8")
    env_file.chmod(0o644)
    body = (
        "const tools = {};\n"
        "const pi = {registerTool(tool) { tools[tool.name] = tool; }, on() {}};\n"
        "extension(pi);\n"
        "try { await tools.funes_recall.execute('test', {query: 'history'}); }\n"
        "catch (error) { console.log(JSON.stringify({text: error.message, isError: true})); }\n"
    )
    result = run_harness(
        tmp_path,
        '[remote]\nurl = "http://127.0.0.1:1"\n',
        body,
        {
            "FUNES_API_TOKEN": "",
            "FUNES_HF_TOKEN": "",
            "HF_TOKEN": "",
            "FUNES_ENV_FILE": str(env_file),
        },
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "text": "funes_recall error: Funes remote API token is not configured",
        "isError": True,
    }


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_pi_does_not_follow_or_retry_redirect(tmp_path):
    sink_requests = []
    source_requests = []

    class SinkHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            sink_requests.append(dict(self.headers.items()))
            self.send_response(204)
            self.end_headers()

        def log_message(self, format, *args):
            return

    sink = ThreadingHTTPServer(("127.0.0.1", 0), SinkHandler)

    class SourceHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            source_requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true,"native_warm":{"state":"ready"}}')

        def do_POST(self):
            source_requests.append(self.path)
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{sink.server_port}/capture")
            self.end_headers()

        def log_message(self, format, *args):
            return

    source = ThreadingHTTPServer(("127.0.0.1", 0), SourceHandler)
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (source, sink)
    ]
    for thread in threads:
        thread.start()
    try:
        body = (
            "const tools = {};\n"
            "const pi = {registerTool(tool) { tools[tool.name] = tool; }, on() {}};\n"
            "extension(pi);\n"
            "try { await tools.funes_recall.execute('test', {query: 'history'}); }\n"
            "catch (error) { console.log(JSON.stringify({text: error.message, isError: true})); }\n"
        )
        started = time.monotonic()
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{source.server_port}"\n',
            body,
        )
        elapsed = time.monotonic() - started
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout)
        assert parsed["isError"] is True
        assert parsed["text"].startswith("funes_recall error:")
        assert source_requests == ["/search"]
        assert sink_requests == []
        assert elapsed < 3
    finally:
        source.shutdown()
        sink.shutdown()
        source.server_close()
        sink.server_close()


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_pi_before_agent_start_allows_slow_search_within_automatic_budget(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.path)
            time.sleep(6.1)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"ok":true,"results":[{"raw_text":"remembered decision"}]}'
            )

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            "let beforeAgentStart;\n"
            "const pi = {registerTool() {}, on(name, handler) { "
            "if (name === 'before_agent_start') beforeAgentStart = handler; }};\n"
            "extension(pi);\n"
            "const result = await beforeAgentStart({"
            "prompt: 'What did we decide previously about the remote memory?', "
            "systemPrompt: 'base'});\n"
            "console.log(JSON.stringify(result ?? null));\n"
        )
        started = time.monotonic()
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{server.server_port}"\n',
            body,
            {"FUNES_REMOTE_AUTO_RECALL_TIMEOUT_MS": "8000"},
        )
        elapsed = time.monotonic() - started

        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {
            "systemPrompt": "base\n\n## Funes unified memory\n1. remembered decision"
        }
        assert requests == ["/search"]
        assert 5.8 < elapsed < 8.0
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
@pytest.mark.parametrize("failure", ("hang", "503"))
def test_pi_before_agent_start_fails_open_with_automatic_budget(tmp_path, failure):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if failure == "hang":
                time.sleep(10)
            self.send_response(503 if failure == "503" else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                self.wfile.write(b'{"ok":false,"native_warm":{"state":"warming"}}')
            except BrokenPipeError:
                pass

        def do_POST(self):
            requests.append(self.path)
            if failure == "hang":
                time.sleep(10)
            self.send_response(503)
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            "let beforeAgentStart;\n"
            "const pi = {registerTool() {}, on(name, handler) { "
            "if (name === 'before_agent_start') beforeAgentStart = handler; }};\n"
            "extension(pi);\n"
            "const result = await beforeAgentStart({"
            "prompt: 'What did we decide previously about the remote memory?', "
            "systemPrompt: 'base'});\n"
            "console.log(JSON.stringify(result ?? null));\n"
        )
        started = time.monotonic()
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{server.server_port}"\n',
            body,
            {
                "FUNES_REMOTE_TIMEOUT": "180",
                "FUNES_REMOTE_ATTEMPTS": "5",
                "FUNES_REMOTE_READY_POLLS": "8",
            },
        )
        elapsed = time.monotonic() - started

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "null"
        assert requests == ["/search"]
        assert elapsed < 5
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_pi_funes_recall_uses_short_503_backoff_and_recovers(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, time.monotonic()))
            if len(requests) == 1:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"ok":true,"results":[{"raw_text":"recovered decision"}]}'
            )

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            "const tools = {};\n"
            "const pi = {registerTool(tool) { tools[tool.name] = tool; }, on() {}};\n"
            "extension(pi);\n"
            "const result = await tools.funes_recall.execute('test', {query: 'decision'});\n"
            "console.log(result.content[0].text);\n"
        )
        started = time.monotonic()
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{server.server_port}"\n',
            body,
        )
        elapsed = time.monotonic() - started
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout)
        assert parsed["ok"] is True
        assert parsed["results"][0]["raw_text"] == "recovered decision"
        assert len(requests) == 2
        delay = requests[1][1] - requests[0][1]
        assert 2.5 <= delay < 4.5, f"expected ~3s backoff, got {delay}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_pi_funes_recall_ignores_long_remote_env_and_defaults_to_two_attempts(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, time.monotonic()))
            self.send_response(503)
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            "const tools = {};\n"
            "const pi = {registerTool(tool) { tools[tool.name] = tool; }, on() {}};\n"
            "extension(pi);\n"
            "try { await tools.funes_recall.execute('test', {query: 'decision'}); }\n"
            "catch (error) { console.log(JSON.stringify({text: error.message, isError: true})); }\n"
        )
        started = time.monotonic()
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{server.server_port}"\n',
            body,
            {
                "FUNES_REMOTE_TIMEOUT": "180",
                "FUNES_REMOTE_ATTEMPTS": "5",
                "FUNES_REMOTE_ATTEMPT_TIMEOUT": "50",
            },
        )
        elapsed = time.monotonic() - started
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout)
        assert parsed["isError"] is True
        assert "HTTP 503" in parsed["text"]
        assert len(requests) == 2
        assert elapsed < 6.0
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_pi_funes_recall_reads_dedicated_recall_attempts_env(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, time.monotonic()))
            self.send_response(503)
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            "const tools = {};\n"
            "const pi = {registerTool(tool) { tools[tool.name] = tool; }, on() {}};\n"
            "extension(pi);\n"
            "try { await tools.funes_recall.execute('test', {query: 'decision'}); }\n"
            "catch (error) { console.log(JSON.stringify({text: error.message, isError: true})); }\n"
        )
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{server.server_port}"\n',
            body,
            {
                "FUNES_REMOTE_RECALL_ATTEMPTS": "3",
            },
        )
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout)
        assert parsed["isError"] is True
        assert "HTTP 503" in parsed["text"]
        assert len(requests) == 3
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_pi_funes_get_keeps_standard_backoff_and_remote_budget(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, time.monotonic()))
            if len(requests) == 1:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true,"record":"found"}')

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            "const tools = {};\n"
            "const pi = {registerTool(tool) { tools[tool.name] = tool; }, on() {}};\n"
            "extension(pi);\n"
            "const result = await tools.funes_get.execute('test', {record_id: 'rec-1'});\n"
            "console.log(result.content[0].text);\n"
        )
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{server.server_port}"\n',
            body,
        )
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout)
        assert parsed["ok"] is True
        assert len(requests) == 2
        assert requests[0][0] == "/get"
        delay = requests[1][1] - requests[0][1]
        assert delay >= 1.8, f"expected standard backoff >= 1.8s, got {delay}"
    finally:
        server.shutdown()
        server.server_close()

@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_pi_funes_recall_allows_slow_search_matching_large_index_latency(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.path)
            time.sleep(6.1)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"ok":true,"results":[{"raw_text":"large index raw text"}]}'
            )

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            "const tools = {};\n"
            "const pi = {registerTool(tool) { tools[tool.name] = tool; }, on() {}};\n"
            "extension(pi);\n"
            "const result = await tools.funes_recall.execute('test', {query: 'bm25'});\n"
            "console.log(result.content[0].text);\n"
        )
        started = time.monotonic()
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{server.server_port}"\n',
            body,
            {"FUNES_REMOTE_AUTO_RECALL_TIMEOUT_MS": "8000"},
        )
        elapsed = time.monotonic() - started
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout)
        assert parsed["ok"] is True
        assert parsed["results"][0]["raw_text"] == "large index raw text"
        assert len(requests) == 1
        assert 5.8 < elapsed < 8.0
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_pi_before_agent_start_reads_auto_recall_timeout_from_config(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.path)
            time.sleep(6.1)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"ok":true,"results":[{"raw_text":"remembered decision"}]}'
            )

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            "let beforeAgentStart;\n"
            "const pi = {registerTool() {}, on(name, handler) { "
            "if (name === 'before_agent_start') beforeAgentStart = handler; }};\n"
            "extension(pi);\n"
            "const result = await beforeAgentStart({"
            "prompt: 'What did we decide previously about the remote memory?', "
            "systemPrompt: 'base'});\n"
            "console.log(JSON.stringify(result ?? null));\n"
        )
        started = time.monotonic()
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{server.server_port}"\nauto_recall_timeout_ms = 8000\n',
            body,
        )
        elapsed = time.monotonic() - started

        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {
            "systemPrompt": "base\n\n## Funes unified memory\n1. remembered decision"
        }
        assert requests == ["/search"]
        assert 5.8 < elapsed < 8.0
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_pi_before_agent_start_can_disable_auto_recall(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.path)
            self.send_response(500)
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            "let beforeAgentStart;\n"
            "const pi = {registerTool() {}, on(name, handler) { "
            "if (name === 'before_agent_start') beforeAgentStart = handler; }};\n"
            "extension(pi);\n"
            "const result = await beforeAgentStart({"
            "prompt: 'What did we decide previously about the remote memory?', "
            "systemPrompt: 'base'});\n"
            "console.log(JSON.stringify(result ?? null));\n"
        )
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{server.server_port}"\n',
            body,
            {"FUNES_REMOTE_AUTO_RECALL_TIMEOUT_MS": "0"},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "null"
        assert requests == []
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_pi_funes_recall_exposes_and_forwards_filters(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true,"results":[]}')

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            "const tools = {};\n"
            "const pi = {registerTool(tool) { tools[tool.name] = tool; }, on() {}};\n"
            "extension(pi);\n"
            "const result = await tools.funes_recall.execute('test', {"
            "query: 'context', limit: 3, source_agent: 'pi', source_type: 'session', "
            "project: 'funes', repo: 'owner/funes', device_id: 'macminim2', role: 'user', "
            "content_type: 'user_message', source_missing: false, since: '2026-01-01', until: '2026-09-24', "
            "facets: {source_agent: 'pi'} });\n"
            "console.log(JSON.stringify({result, schema: tools.funes_recall.parameters}));\n"
        )
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{server.server_port}"\n',
            body,
        )
        assert result.returncode == 0, result.stderr
        output = json.loads(result.stdout)
        assert output["result"].get("isError") is not True
        assert json.loads(output["result"]["content"][0]["text"])["ok"] is True
        assert output["schema"]["properties"]["source_agent"]["type"] == "string"
        assert output["schema"]["properties"]["source_missing"]["type"] == "boolean"
        assert requests == [{
            "query": "context",
            "limit": 3,
            "source_agent": "pi",
            "source_type": "session",
            "project": "funes",
            "repo": "owner/funes",
            "device_id": "macminim2",
            "role": "user",
            "content_type": "user_message",
            "source_missing": False,
            "since": "2026-01-01",
            "until": "2026-09-24",
            "facets": {"source_agent": "pi"},
        }]
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_pi_funes_recall_marks_429_as_tool_error(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.path)
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            "const tools = {};\n"
            "const pi = {registerTool(tool) { tools[tool.name] = tool; }, on() {}};\n"
            "extension(pi);\n"
            "try { await tools.funes_recall.execute('test', {query: 'rate'}); }\n"
            "catch (error) { console.log(JSON.stringify({text: error.message, isError: true})); }\n"
        )
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{server.server_port}"\n',
            body,
            {"FUNES_REMOTE_RECALL_ATTEMPTS": "1"},
        )
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout)
        assert parsed["isError"] is True
        assert "HTTP 429" in parsed["text"]
        assert requests == ["/search"]
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_pi_funes_recall_marks_timeout_as_tool_error(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            time.sleep(0.3)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            "const tools = {};\n"
            "const pi = {registerTool(tool) { tools[tool.name] = tool; }, on() {}};\n"
            "extension(pi);\n"
            "try { await tools.funes_recall.execute('test', {query: 'timeout'}); }\n"
            "catch (error) { console.log(JSON.stringify({text: error.message, isError: true})); }\n"
        )
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{server.server_port}"\n',
            body,
            {
                "FUNES_REMOTE_RECALL_ATTEMPTS": "1",
                "FUNES_REMOTE_RECALL_TIMEOUT_MS": "200",
                "FUNES_REMOTE_RECALL_ATTEMPT_TIMEOUT_MS": "100",
            },
        )
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout)
        assert parsed["isError"] is True
        assert "timed out" in parsed["text"]
    finally:
        server.shutdown()
        server.server_close()
