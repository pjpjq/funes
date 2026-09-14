from __future__ import annotations

import json
import os
import shutil
import subprocess
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
        timeout=10,
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
            "const result = await tools.funes_recall.execute('test', {query: 'history'});\n"
            "console.log(result.content[0].text);\n"
        )
        started = time.monotonic()
        result = run_harness(
            tmp_path,
            f'[remote]\nurl = "http://127.0.0.1:{source.server_port}"\n',
            body,
        )
        elapsed = time.monotonic() - started
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "null"
        assert source_requests == ["/ready/search", "/search"]
        assert sink_requests == []
        assert elapsed < 3
    finally:
        source.shutdown()
        sink.shutdown()
        source.server_close()
        sink.server_close()


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
        assert requests == ["/ready/search", "/search"]
        assert elapsed < 5
    finally:
        server.shutdown()
        server.server_close()
