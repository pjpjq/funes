import json
import os
import tempfile
import threading
import unittest
import urllib.error
from unittest import mock
from http.client import HTTPConnection
from pathlib import Path
from http.server import ThreadingHTTPServer

from service.server import App, Store, Translator, make_handler


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = {k: os.environ.get(k) for k in ("FUNES_DATA_DIR", "FUNES_AUTH_TOKEN", "FUNES_API_TOKEN", "FUNES_STORAGE_KEY", "FUNES_STORAGE_REPO", "FUNES_SNAPSHOT_FILE", "HF_TOKEN", "FUNES_REQUIRE_DURABLE_ACK", "FUNES_ALLOW_EMPTY_REMOTE", "TRANSLATION_BASE_URL", "TRANSLATION_API_KEY", "TRANSLATION_MODEL", "TRANSLATION_MAX_PER_INGEST", "TRANSLATION_RECONCILE_INTERVAL", "RETURN_RETRIEVAL_TEXT")}
        os.environ["FUNES_DATA_DIR"] = self.tmp.name
        os.environ["FUNES_AUTH_TOKEN"] = "test-token"
        for k in ("FUNES_API_TOKEN", "FUNES_STORAGE_KEY", "FUNES_STORAGE_REPO", "FUNES_SNAPSHOT_FILE", "HF_TOKEN", "FUNES_REQUIRE_DURABLE_ACK", "FUNES_ALLOW_EMPTY_REMOTE", "TRANSLATION_BASE_URL", "TRANSLATION_API_KEY", "TRANSLATION_MODEL", "TRANSLATION_MAX_PER_INGEST", "TRANSLATION_RECONCILE_INTERVAL", "RETURN_RETRIEVAL_TEXT"):
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

    def test_translation_defaults_to_background_only(self):
        store = Store(self.tmp.name)
        try:
            self.assertEqual(Translator(store).max_per_ingest, 0)
        finally:
            store.close()

    def test_same_revision_updates_only_derived_fields_in_place(self):
        store = Store(self.tmp.name)
        raw = "原始正文"
        original = {
            "source_identity": "derived-only",
            "source_version": "v1",
            "raw_text": raw,
            "retrieval_text": raw,
            "translation_hash": "same-hash",
            "translation_version": "v1",
            "translation_status": "pending_provider",
        }
        self.assertEqual(store.ingest([original])["created"], 1)
        reconciled = dict(
            original,
            retrieval_text=raw + " english retrieval",
            translation_status="ok",
        )
        result = store.ingest([reconciled])
        item = store.get("derived-only")
        self.assertEqual(result["updated"], 1)
        self.assertEqual(result["items"][0]["status"], "derived_updated")
        self.assertEqual(store.count(), 1)
        self.assertEqual(item["raw_text"], raw)
        self.assertEqual(item["retrieval_text"], raw + " english retrieval")
        self.assertEqual(item["translation_status"], "ok")
        # An older/raw pending delta can restore later without regressing the
        # successfully reconciled shadow.
        self.assertEqual(store.ingest([original])["deduped"], 1)
        self.assertEqual(store.get("derived-only")["translation_status"], "ok")
        store.close()

    def test_same_revision_native_status_update_preserves_raw_and_retrieval(self):
        store = Store(self.tmp.name)
        raw = "原始正文"
        store.ingest(
            [{
                "source_identity": "native-state",
                "source_version": "v1",
                "raw_text": raw,
                "retrieval_text": "english retrieval shadow",
                "translation_status": "ok",
                "retrieval_updated_at": "2026-09-13T01:00:00Z",
            }]
        )
        result = store.ingest(
            [{
                "source_identity": "native-state",
                "source_version": "v1",
                "raw_text": raw,
                "native_index_version": "canonical-v1",
                "native_index_status": "indexed",
                "native_indexed_at": "2026-09-13T02:00:00Z",
            }]
        )
        item = store.get("native-state")
        self.assertEqual(result["items"][0]["status"], "derived_updated")
        self.assertEqual(item["raw_text"], raw)
        self.assertEqual(item["retrieval_text"], "english retrieval shadow")
        self.assertEqual(item["translation_status"], "ok")
        self.assertEqual(item["native_index_status"], "indexed")
        changed = store.ingest(
            [{
                "source_identity": "native-state",
                "source_version": "v1",
                "raw_text": raw,
                "source_missing": True,
            }]
        )
        item = store.get("native-state")
        self.assertEqual(changed["items"][0]["status"], "derived_updated")
        self.assertTrue(item["source_missing"])
        self.assertIsNone(item["native_index_status"])
        store.close()

    def test_restore_does_not_regress_terminal_native_state(self):
        store = Store(self.tmp.name)
        base = {
            "source_identity": "monotonic-native",
            "source_version": "raw-v1",
            "raw_text": "原始正文",
            "retrieval_text": "english retrieval shadow",
            "translation_hash": "translation-hash",
            "translation_version": "translation-version",
            "translation_status": "ok",
            "retrieval_updated_at": "2026-09-13T01:00:00Z",
            "native_index_version": None,
            "native_index_status": None,
            "native_indexed_at": None,
            "native_index_error": None,
        }
        store.ingest([base])
        indexed = {
            **base,
            "native_index_version": "canonical-v1",
            "native_index_status": "indexed",
            "native_indexed_at": "2026-09-13T02:00:00Z",
        }
        store.ingest([indexed])
        store.restore_documents([base])
        item = store.get("monotonic-native")
        self.assertEqual(item["native_index_status"], "indexed")
        self.assertEqual(item["native_index_version"], "canonical-v1")
        held = {
            **indexed,
            "native_index_status": "held_secret",
            "native_indexed_at": None,
        }
        store.ingest([held])
        store.restore_documents(
            [{
                **base,
                "native_index_status": "retry",
                "native_index_error": "native_exit",
            }]
        )
        item = store.get("monotonic-native")
        self.assertEqual(item["native_index_status"], "held_secret")
        self.assertIsNone(item["native_index_error"])
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

    def test_numeric_source_identity_precedes_sqlite_id(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {"source_identity": "first", "raw_text": "row one"},
                {"source_identity": "1", "raw_text": "numeric identity"},
            ]
        )
        self.assertEqual(store.get("1")["raw_text"], "numeric identity")
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

    def test_translation_cache_key_is_hashed_and_permanent_error_opens_circuit(self):
        store = Store(self.tmp.name)
        os.environ.update(
            TRANSLATION_BASE_URL="https://provider.example/v1",
            TRANSLATION_API_KEY="test-key",
            TRANSLATION_MODEL="test-model",
        )
        tr = Translator(store)
        cache_key = tr._cache_key("中文 查询")
        self.assertEqual(len(cache_key), 64)
        self.assertNotIn("中文", cache_key)
        failure = urllib.error.HTTPError(
            "https://provider.example/v1/chat/completions", 402, "payment", {}, None
        )
        with mock.patch("service.server.urllib.request.urlopen", side_effect=failure) as request:
            self.assertEqual(tr.rewrite("中文 查询"), "中文 查询")
            self.assertEqual(tr.rewrite("另一个中文查询"), "另一个中文查询")
        self.assertEqual(request.call_count, 1)
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
        source.translation_put("cache-key", "cached retrieval text")
        snap = Path(self.tmp.name) / "snapshot.jsonl.gz"
        source.snapshot(snap)
        second_dir = tempfile.TemporaryDirectory()
        target = Store(second_dir.name)
        self.assertEqual(target.restore(snap), 1)
        self.assertEqual(target.restore(snap), 0)
        target.reindex()
        self.assertEqual(target.search("durable")[0]["raw_text"], "durable text")
        self.assertEqual(target.translation_get("cache-key"), "cached retrieval text")
        source.close(); target.close(); second_dir.cleanup()

    def test_snapshot_roundtrip_preserves_retrieval_and_native_state(self):
        source = Store(self.tmp.name)
        source.ingest(
            [{
                "source_identity": "stateful",
                "source_version": "raw-v1",
                "raw_text": "原文",
                "retrieval_text": "retrieval shadow",
                "translation_status": "ok",
                "retrieval_updated_at": "2026-09-13T01:00:00Z",
                "native_index_version": "canonical-v1",
                "native_index_status": "indexed",
                "native_indexed_at": "2026-09-13T02:00:00Z",
                "native_index_error": None,
            }]
        )
        snapshot = Path(self.tmp.name) / "state.jsonl.gz"
        source.snapshot(snapshot)
        second_dir = tempfile.TemporaryDirectory()
        target = Store(second_dir.name)
        self.assertEqual(target.restore(snapshot), 1)
        item = target.get("stateful")
        self.assertEqual(item["retrieval_updated_at"], "2026-09-13T01:00:00Z")
        self.assertEqual(item["native_index_version"], "canonical-v1")
        self.assertEqual(item["native_index_status"], "indexed")
        self.assertEqual(item["native_indexed_at"], "2026-09-13T02:00:00Z")
        source.close(); target.close(); second_dir.cleanup()

    def test_http_never_returns_retrieval_text_even_when_legacy_flag_is_set(self):
        os.environ["RETURN_RETRIEVAL_TEXT"] = "true"
        server = self._server()
        status, _ = self._request(
            server,
            "POST",
            "/ingest",
            {"source_identity": "raw-only", "raw_text": "原文", "retrieval_text": "english shadow"},
        )
        self.assertEqual(status, 200)
        status, item = self._request(server, "POST", "/get", {"id": "raw-only"})
        self.assertEqual(status, 200)
        self.assertEqual(item["raw_text"], "原文")
        self.assertNotIn("retrieval_text", item)

    def test_ingest_does_not_ack_before_durable_snapshot(self):
        os.environ["FUNES_REQUIRE_DURABLE_ACK"] = "true"
        server = self._server()
        status, result = self._request(server, "POST", "/ingest", {"source_path": "pending", "raw_text": "must persist"})
        self.assertEqual(status, 503)
        self.assertEqual(result["error"], "durability_pending")
        self.assertFalse(result["sync"]["durable"])

    def test_configured_remote_without_hf_token_is_not_durable(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "remote", "raw_text": "must reach hub"}])
        os.environ["FUNES_STORAGE_REPO"] = "owner/private"
        result = SnapshotSync(store).upload(store.get_many(["remote"]))
        self.assertFalse(result["durable"])
        self.assertEqual(result["reason"], "HF storage not configured")
        store.close()

    def test_encrypted_source_snapshot_roundtrip_hides_plaintext(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        marker = "sensitive bearer material must remain encrypted"
        store.ingest([{"source_path": "secret", "raw_text": marker}])
        os.environ["FUNES_STORAGE_KEY"] = "test-storage-key"
        syncer = SnapshotSync(store)
        plain = Path(self.tmp.name) / "source.jsonl.gz"
        encrypted = Path(self.tmp.name) / "source.jsonl.gz.enc"
        decrypted = Path(self.tmp.name) / "restored.jsonl.gz"
        store.snapshot(plain)
        syncer._encrypt_file(plain, encrypted)
        self.assertNotIn(marker.encode(), encrypted.read_bytes())
        syncer._decrypt_file(encrypted, decrypted)
        second_dir = tempfile.TemporaryDirectory()
        restored = Store(second_dir.name)
        self.assertEqual(restored.restore(decrypted), 1)
        self.assertEqual(restored.get("secret")["raw_text"], marker)
        restored.close(); second_dir.cleanup()
        store.close()

    def test_remote_source_upload_fails_closed_without_encryption_key(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        store.ingest([{"source_path": "secret", "raw_text": "private source"}])
        os.environ.update(FUNES_STORAGE_REPO="owner/private", HF_TOKEN="hf-test")
        os.environ.pop("FUNES_AUTH_TOKEN", None)
        os.environ.pop("FUNES_API_TOKEN", None)
        os.environ.pop("FUNES_STORAGE_KEY", None)
        result = SnapshotSync(store).upload()
        self.assertFalse(result["durable"])
        self.assertEqual(result["reason"], "RuntimeError")
        store.close()

    def test_restore_file_decrypts_inside_temporary_directory(self):
        from service.server import SnapshotSync
        source = Store(self.tmp.name)
        source.ingest([{"source_identity": "restore-one", "raw_text": "encrypted restore"}])
        os.environ["FUNES_STORAGE_KEY"] = "test-storage-key"
        writer = SnapshotSync(source)
        plaintext = Path(self.tmp.name) / "funes-delta-test.jsonl.gz"
        encrypted = Path(self.tmp.name) / "funes-delta-test.jsonl.gz.enc"
        writer._write_jsonl_gzip(source.get_many(["restore-one"]), plaintext)
        writer._encrypt_file(plaintext, encrypted)
        target_dir = tempfile.TemporaryDirectory()
        target = Store(target_dir.name)
        reader = SnapshotSync(target)
        with mock.patch(
            "huggingface_hub.hf_hub_download", return_value=str(encrypted)
        ):
            self.assertEqual(reader._restore_file(encrypted.name), 1)
        self.assertEqual(target.get("restore-one")["raw_text"], "encrypted restore")
        source.close(); target.close(); target_dir.cleanup()

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
