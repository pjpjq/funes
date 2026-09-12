import json
import os
import tempfile
import threading
import unittest
from unittest import mock
from http.client import HTTPConnection
from pathlib import Path
from http.server import ThreadingHTTPServer

from service.server import App, Store, Translator, make_handler


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = {k: os.environ.get(k) for k in ("FUNES_DATA_DIR", "FUNES_AUTH_TOKEN", "FUNES_API_TOKEN", "FUNES_STORAGE_REPO", "HF_TOKEN", "FUNES_REQUIRE_DURABLE_ACK", "FUNES_ALLOW_EMPTY_REMOTE", "TRANSLATION_BASE_URL", "TRANSLATION_API_KEY", "TRANSLATION_MODEL")}
        os.environ["FUNES_DATA_DIR"] = self.tmp.name
        os.environ["FUNES_AUTH_TOKEN"] = "test-token"
        for k in ("FUNES_STORAGE_REPO", "HF_TOKEN", "FUNES_REQUIRE_DURABLE_ACK", "FUNES_ALLOW_EMPTY_REMOTE", "TRANSLATION_BASE_URL", "TRANSLATION_API_KEY", "TRANSLATION_MODEL"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_dedupe_and_update(self):
        store = Store(self.tmp.name)
        first = {"source_path": "a.md", "source_version": "1", "raw_text": "hello world", "project": "p"}
        self.assertEqual(store.ingest([first])["created"], 1)
        self.assertEqual(store.ingest([first])["deduped"], 1)
        changed = dict(first, source_version="2", raw_text="hello revised")
        self.assertEqual(store.ingest([changed])["updated"], 1)
        self.assertEqual(store.count(), 1)
        self.assertEqual(store.get("a.md")["raw_text"], "hello revised")
        store.close()

    def test_multiple_chunks_same_source_path(self):
        store = Store(self.tmp.name)
        result = store.ingest([
            {"source_path": "a.md", "chunk_id": "0", "raw_text": "first chunk"},
            {"source_path": "a.md", "chunk_id": "1", "raw_text": "second chunk"},
        ])
        self.assertEqual(result["created"], 2)
        self.assertEqual(store.count(), 2)
        self.assertEqual({r["source_identity"] for r in store.sources()}, {"a.md#chunk:0", "a.md#chunk:1"})
        store.close()

    def _server(self):
        app = App()
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), app.close()))
        return server

    def _request(self, server, method, path, payload=None, token="test-token"):
        conn = HTTPConnection(*server.server_address)
        headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
        conn.request(method, path, json.dumps(payload or {}).encode(), headers)
        response = conn.getresponse()
        data = json.loads(response.read())
        conn.close()
        return response.status, data

    def test_auth_search_get(self):
        server = self._server()
        status, _ = self._request(server, "POST", "/ingest", {"raw_text": "alpha beta", "source_path": "x"}, token="bad")
        self.assertEqual(status, 401)
        status, result = self._request(server, "POST", "/ingest", {"raw_text": "alpha beta", "source_path": "x"})
        self.assertEqual(status, 200)
        ident = result["items"][0]["id"]
        status, found = self._request(server, "POST", "/search", {"query": "alpha"})
        self.assertEqual(status, 200)
        self.assertEqual(found["results"][0]["raw_text"], "alpha beta")
        status, item = self._request(server, "POST", "/get", {"id": ident})
        self.assertEqual(status, 200)
        self.assertEqual(item["source_path"], "x")

    def test_chinese_fallback_and_normalization_cache(self):
        store = Store(self.tmp.name)
        tr = Translator(store)
        self.assertEqual(tr.rewrite("  中文   查询 "), "中文 查询")
        # Configure an unreachable endpoint: retries must fall back without raising.
        os.environ.update(TRANSLATION_BASE_URL="http://127.0.0.1:1", TRANSLATION_API_KEY="x", TRANSLATION_MODEL="m")
        tr = Translator(store)
        self.assertEqual(tr.rewrite("中文 查询"), "中文 查询")
        store.translation_put(tr._cache_key("中文 查询"), "cached words")
        self.assertEqual(tr.rewrite("中文 查询"), "cached words")
        store.ingest([{"source_path": "zh", "raw_text": "中文检索内容"}])
        self.assertEqual(store.search("中文")[0]["raw_text"], "中文检索内容")
        store.close()

    def test_records_and_api_token_alias(self):
        os.environ.pop("FUNES_AUTH_TOKEN")
        os.environ["FUNES_API_TOKEN"] = "api-token"
        server = self._server()
        status, result = self._request(server, "POST", "/ingest", {"records": [{"source_path": "r", "raw_text": "record"}],}, token="api-token")
        self.assertEqual(status, 200)
        self.assertEqual(result["created"], 1)

    def test_snapshot_roundtrip(self):
        source = Store(self.tmp.name)
        source.ingest([{"source_path": "s", "raw_text": "durable text", "role": "user", "metadata": {"project": "demo"}}])
        snap = Path(self.tmp.name) / "snapshot.jsonl"
        source.snapshot(snap)
        second_dir = tempfile.TemporaryDirectory()
        target = Store(second_dir.name)
        self.assertEqual(target.restore(snap), 1)
        self.assertEqual(target.restore(snap), 0)
        target.reindex()
        self.assertEqual(target.search("durable")[0]["raw_text"], "durable text")
        source.close(); target.close(); second_dir.cleanup()

    def test_ingest_does_not_ack_before_durable_snapshot(self):
        os.environ["FUNES_REQUIRE_DURABLE_ACK"] = "true"
        server = self._server()
        status, result = self._request(server, "POST", "/ingest", {"source_path": "pending", "raw_text": "must persist"})
        self.assertEqual(status, 503)
        self.assertEqual(result["error"], "durability_pending")
        self.assertFalse(result["sync"]["durable"])

    def test_secret_gate_blocks_remote_snapshot(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        store.ingest([{"source_path": "secret", "raw_text": "bearer token"}])
        os.environ.update(FUNES_STORAGE_REPO="owner/private", HF_TOKEN="hf-test")
        syncer = SnapshotSync(store)
        with mock.patch.object(syncer, "_secret_gate", return_value=(False, "secret_detected")):
            result = syncer.upload()
        self.assertFalse(result["durable"])
        self.assertEqual(result["reason"], "secret_detected")
        store.close()

    def test_restore_failure_is_fail_closed(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        os.environ.update(FUNES_STORAGE_REPO="owner/private", HF_TOKEN="hf-test")
        syncer = SnapshotSync(store)
        with mock.patch.dict("sys.modules", {"huggingface_hub": mock.Mock(hf_hub_download=mock.Mock(side_effect=RuntimeError("offline")))}):
            self.assertEqual(syncer.restore(), -1)
        self.assertTrue(syncer.restore_failed)
        result = syncer.upload()
        self.assertFalse(result["durable"])
        self.assertEqual(result["reason"], "restore_failed")
        store.close()


if __name__ == "__main__":
    unittest.main()
