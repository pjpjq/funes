from __future__ import annotations

import importlib.util
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

MODULE_PATH = Path(__file__).parents[1] / "deploy" / "funes-sync" / "warm-space.py"
SPEC = importlib.util.spec_from_file_location("funes_warm_space", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
WARM_SPACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WARM_SPACE)


class WarmSpaceTest(unittest.TestCase):
    def test_recently_warmed_uses_success_stamp_age(self) -> None:
        with TemporaryDirectory() as directory:
            stamp = Path(directory) / "warm.stamp"
            stamp.touch()
            self.assertTrue(WARM_SPACE.recently_warmed(stamp, 300))
            os.utime(stamp, (time.time() - 301, time.time() - 301))
            self.assertFalse(WARM_SPACE.recently_warmed(stamp, 300))

    def test_stamp_is_scoped_to_remote_and_state_directory(self) -> None:
        with TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"FUNES_SYNC_STATE_DIR": directory}, clear=False
        ):
            first = WARM_SPACE.stamp_path("https://one.example")
            second = WARM_SPACE.stamp_path("https://two.example")
            self.assertEqual(first.parent, Path(directory))
            self.assertNotEqual(first, second)

    def test_invalid_interval_falls_back_to_five_minutes(self) -> None:
        with mock.patch.dict(
            os.environ, {"FUNES_NATIVE_WARM_MIN_INTERVAL": "invalid"}, clear=False
        ):
            self.assertEqual(WARM_SPACE.warm_interval(), 300.0)

    def test_warm_request_names_pushed_memory_and_scopes_stamp(self) -> None:
        bodies = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                bodies.append(self.rfile.read(length))
                self.send_response(202)
                self.end_headers()

            def log_message(self, format, *args) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with TemporaryDirectory() as directory, mock.patch.dict(
                os.environ,
                {
                    "FUNES_REMOTE_URL": f"http://127.0.0.1:{server.server_port}",
                    "FUNES_API_TOKEN": "synthetic-api-token",
                    "HF_TOKEN": "synthetic-hf-token",
                    "FUNES_NATIVE_MEMORY": "owner/legacy-memory",
                    "FUNES_SYNC_STATE_DIR": directory,
                    "FUNES_NATIVE_WARM_MIN_INTERVAL": "0",
                },
                clear=False,
            ):
                WARM_SPACE.main()
                self.assertEqual(
                    bodies, [b'{"memory":"owner/legacy-memory"}']
                )
                memory_stamp = WARM_SPACE.stamp_path(
                    os.environ["FUNES_REMOTE_URL"], "owner/legacy-memory"
                )
                other_stamp = WARM_SPACE.stamp_path(
                    os.environ["FUNES_REMOTE_URL"], "owner/canonical-memory"
                )
                uri_stamp = WARM_SPACE.stamp_path(
                    os.environ["FUNES_REMOTE_URL"],
                    "hf://datasets/owner/legacy-memory",
                )
                self.assertTrue(memory_stamp.exists())
                self.assertNotEqual(memory_stamp, other_stamp)
                self.assertEqual(memory_stamp, uri_stamp)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_zshrc_env_rejects_unknown_variable(self) -> None:
        with mock.patch.object(WARM_SPACE.subprocess, "run") as run:
            self.assertEqual(WARM_SPACE.zshrc_env("UNTRUSTED"), "")
            run.assert_not_called()

    def test_cross_origin_redirect_does_not_forward_tokens(self) -> None:
        sink_headers = []

        class SinkHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                sink_headers.append(dict(self.headers.items()))
                self.send_response(204)
                self.end_headers()

            def log_message(self, format, *args) -> None:
                return

        sink = ThreadingHTTPServer(("127.0.0.1", 0), SinkHandler)

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.send_response(302)
                self.send_header(
                    "Location", f"http://127.0.0.1:{sink.server_port}/capture"
                )
                self.end_headers()

            def log_message(self, format, *args) -> None:
                return

        source = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in (source, sink)
        ]
        for thread in threads:
            thread.start()
        try:
            with TemporaryDirectory() as directory, mock.patch.dict(
                os.environ,
                {
                    "FUNES_REMOTE_URL": f"http://127.0.0.1:{source.server_port}",
                    "FUNES_API_TOKEN": "synthetic-api-token",
                    "HF_TOKEN": "synthetic-hf-token",
                    "FUNES_SYNC_STATE_DIR": directory,
                    "FUNES_NATIVE_WARM_MIN_INTERVAL": "0",
                },
                clear=False,
            ):
                WARM_SPACE.main()
                self.assertEqual(sink_headers, [])
                self.assertFalse(
                    WARM_SPACE.stamp_path(os.environ["FUNES_REMOTE_URL"]).exists()
                )
        finally:
            source.shutdown()
            sink.shutdown()
            source.server_close()
            sink.server_close()


if __name__ == "__main__":
    unittest.main()
