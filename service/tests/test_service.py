import gzip
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

from service.server import QUERY_PROMPT_VERSION, QUERY_RETRIEVAL_PROMPT, RETRIEVAL_PROMPT, App, Store, Translator, make_handler, persist_translation_documents, prepare_ingest_documents


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = {k: os.environ.get(k) for k in ("FUNES_DATA_DIR", "FUNES_AUTH_TOKEN", "FUNES_API_TOKEN", "FUNES_STORAGE_KEY", "FUNES_STORAGE_REPO", "FUNES_SNAPSHOT_FILE", "HF_TOKEN", "FUNES_REQUIRE_DURABLE_ACK", "FUNES_ALLOW_EMPTY_REMOTE", "FUNES_MAX_BODY_BYTES", "TRANSLATION_BASE_URL", "TRANSLATION_API_KEY", "TRANSLATION_MODEL", "TRANSLATION_MAX_PER_INGEST", "TRANSLATION_QUERY_MAX_TOKENS", "TRANSLATION_RECONCILE_INTERVAL", "RETURN_RETRIEVAL_TEXT")}
        os.environ["FUNES_DATA_DIR"] = self.tmp.name
        os.environ["FUNES_AUTH_TOKEN"] = "test-token"
        for k in ("FUNES_API_TOKEN", "FUNES_STORAGE_KEY", "FUNES_STORAGE_REPO", "FUNES_SNAPSHOT_FILE", "HF_TOKEN", "FUNES_REQUIRE_DURABLE_ACK", "FUNES_ALLOW_EMPTY_REMOTE", "FUNES_MAX_BODY_BYTES", "TRANSLATION_BASE_URL", "TRANSLATION_API_KEY", "TRANSLATION_MODEL", "TRANSLATION_MAX_PER_INGEST", "TRANSLATION_QUERY_MAX_TOKENS", "TRANSLATION_RECONCILE_INTERVAL", "RETURN_RETRIEVAL_TEXT"):
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

    def test_prepare_ingest_reads_reindex_generation_once_per_batch(self):
        app = App()
        try:
            with mock.patch.object(
                app.store,
                "latest_reindex_generation",
                wraps=app.store.latest_reindex_generation,
            ) as latest:
                prepared = prepare_ingest_documents(
                    app,
                    [
                        {"source_identity": "batch-one", "source_type": "session", "raw_text": "one"},
                        {"source_identity": "batch-two", "source_type": "session", "raw_text": "two"},
                    ],
                )
            self.assertEqual(latest.call_count, 1)
            self.assertEqual(
                {item["retrieval_generation"] for item in prepared},
                {app.store.latest_reindex_generation()},
            )
        finally:
            app.close()

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

    def test_existing_identities_is_ordered_chunked_and_selects_no_raw_payload(self):
        store = Store(self.tmp.name)
        store.ingest(
            [{"source_identity": "present", "raw_text": "raw-secret-must-not-load"}]
        )
        statements = []
        store.conn.set_trace_callback(statements.append)
        try:
            identities = ["missing-first", "present", "present"] + [
                f"missing-{index}" for index in range(500)
            ]
            self.assertEqual(store.existing_identities(identities), ["present"])
        finally:
            store.conn.set_trace_callback(None)
            store.close()
        selects = [statement for statement in statements if statement.startswith("SELECT")]
        self.assertEqual(len(selects), 2)
        self.assertTrue(all(statement.startswith("SELECT source_identity FROM") for statement in selects))
        self.assertTrue(all("raw_text" not in statement for statement in selects))

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

    def _request_bytes(self, server, body, headers=None):
        conn = HTTPConnection(*server.server_address)
        request_headers = {"Authorization": "Bearer test-token", "Content-Type": "application/json"}
        request_headers.update(headers or {})
        conn.request("POST", "/ingest", body, request_headers)
        response = conn.getresponse()
        data = json.loads(response.read())
        conn.close()
        return response.status, data

    def test_gzip_ingest_rejects_invalid_and_oversized_payloads(self):
        server = self._server()
        valid = gzip.compress(json.dumps({"documents": [{"source_identity": "gzip-valid", "raw_text": "compressed"}]}).encode())
        status, result = self._request_bytes(server, valid, {"Content-Encoding": "gzip"})
        self.assertEqual(status, 200)
        self.assertEqual(result["created"], 1)

        status, result = self._request_bytes(server, b"not-a-gzip-stream", {"Content-Encoding": "gzip"})
        self.assertEqual(status, 400)
        self.assertEqual(result["error"], "invalid gzip request body")
        status, result = self._request_bytes(server, valid[:-1], {"Content-Encoding": "gzip"})
        self.assertEqual(status, 400)
        self.assertEqual(result["error"], "invalid gzip request body")
        corrupt_deflate = bytes.fromhex("1f8b0800000000000003ffff0000000000000000")
        status, result = self._request_bytes(server, corrupt_deflate, {"Content-Encoding": "gzip"})
        self.assertEqual(status, 400)
        self.assertEqual(result["error"], "invalid gzip request body")
        status, result = self._request_bytes(server, valid, {"Content-Encoding": "br"})
        self.assertEqual(status, 400)
        self.assertEqual(result["error"], "unsupported Content-Encoding")
        status, _ = self._request(server, "POST", "/get", {"id": "gzip-invalid"})
        self.assertEqual(status, 404)

        os.environ["FUNES_MAX_BODY_BYTES"] = "128"
        oversized = gzip.compress(json.dumps({"documents": [{"source_identity": "gzip-oversized", "raw_text": "x" * 1024}]}).encode())
        status, result = self._request_bytes(server, oversized, {"Content-Encoding": "gzip"})
        self.assertEqual(status, 400)
        self.assertEqual(result["error"], "request too large")
        status, _ = self._request(server, "POST", "/get", {"id": "gzip-oversized"})
        self.assertEqual(status, 404)

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

    def test_sources_check_requires_auth_and_returns_only_present_and_missing(self):
        server = self._server()
        secret = "raw-secret-must-not-leak"
        status, _ = self._request(
            server,
            "POST",
            "/ingest",
            {"source_identity": "present", "raw_text": secret},
        )
        self.assertEqual(status, 200)
        payload = {"source_identities": ["missing", "present", "missing", "present"]}
        status, body = self._request(
            server, "POST", "/sources/check", payload, token="bad"
        )
        self.assertEqual(status, 401)
        status, body = self._request(server, "POST", "/sources/check", payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {"ok": True, "present": ["present"], "missing": ["missing"]},
        )
        self.assertNotIn(secret, json.dumps(body))

    def test_sources_check_validates_bounds_and_restore_readiness(self):
        server = self._server()
        status, body = self._request(
            server, "POST", "/sources/check", {"source_identities": "invalid"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "source_identities must be a list")
        status, body = self._request(
            server,
            "POST",
            "/sources/check",
            {"source_identities": [f"id-{index}" for index in range(5001)]},
        )
        self.assertEqual(status, 400)
        self.assertIn("at most 5000", body["error"])

        app = App()
        app.syncer.restoring = True
        unavailable = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        thread = threading.Thread(target=unavailable.serve_forever, daemon=True)
        thread.start()
        try:
            status, body = self._request(
                unavailable,
                "POST",
                "/sources/check",
                {"source_identities": ["id"]},
            )
        finally:
            app.syncer.restoring = False
            unavailable.shutdown()
            unavailable.server_close()
            thread.join(timeout=2)
            app.close()
        self.assertEqual(status, 503)
        self.assertEqual(body, {"ok": False, "error": "restore_in_progress"})

    def test_compatibility_search_fuses_expanded_raw_and_rewrite_rankings(self):
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
                        {"source_identity": "raw-only", "raw_text": "raw BM25", "retrieval_text": "raw shadow"},
                        {"source_identity": "both", "raw_text": "both raw", "retrieval_text": "both shadow"},
                    ]
                return [
                    {"source_identity": "provider-only", "raw_text": "provider raw", "retrieval_text": "provider shadow"},
                    {"source_identity": "both", "raw_text": "both raw", "retrieval_text": "both shadow"},
                ]

        app = type("CompatibilityApp", (), {})()
        app.translator = Translator()
        app.store = Store()
        app.syncer = type("Syncer", (), {"restoring": False})()
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, body = self._request(
                server,
                "POST",
                "/search",
                {"query": "raw query", "limit": 3, "project": "demo"},
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(status, 200)
        self.assertEqual(
            [item["source_identity"] for item in body["results"]],
            ["both", "raw-only", "provider-only"],
        )
        self.assertTrue(all("retrieval_text" not in item for item in body["results"]))
        self.assertEqual(
            calls,
            [
                ("rewrite", "raw query"),
                ("search", "raw query", 9, {"project": "demo"}),
                ("search", "provider query", 9, {"project": "demo"}),
            ],
        )

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

    def test_chinese_query_uses_technical_fts_before_character_scan(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": "target",
                    "source_agent": "pi",
                    "role": "user",
                    "raw_text": "MacBook Pro通过Tailscale连接办公室Mac mini时高延迟。",
                },
                {
                    "source_identity": "wrong-agent",
                    "source_agent": "codex",
                    "role": "user",
                    "raw_text": "Pi Tailscale 延迟",
                },
                {
                    "source_identity": "wrong-role",
                    "source_agent": "pi",
                    "role": "assistant",
                    "raw_text": "Pi Tailscale 延迟",
                },
            ]
        )
        statements = []
        store.conn.set_trace_callback(statements.append)

        results = store.search(
            "之前 Pi 里讨论过的 Tailscale 延迟",
            filters={"source_agent": "pi", "role": "user"},
        )

        store.conn.set_trace_callback(None)
        self.assertEqual([item["source_identity"] for item in results], ["target"])
        traced = "\n".join(statements).upper()
        self.assertIn("LIKE '%TAILSCALE%'", traced)
        self.assertNotIn("LIKE '%之%'", traced)

        statements.clear()
        store.conn.set_trace_callback(statements.append)
        malformed_results = store.search(
            '之前 Tailscale "',
            filters={"source_agent": "pi", "role": "user"},
        )
        store.conn.set_trace_callback(None)
        self.assertEqual(
            [item["source_identity"] for item in malformed_results], ["target"]
        )
        malformed_trace = "\n".join(statements).upper()
        self.assertIn("LIKE '%TAILSCALE%'", malformed_trace)
        self.assertNotIn("LIKE '%之前 TAILSCALE", malformed_trace)
        store.close()

    def test_chinese_technical_fts_skips_like_when_one_hit_covers_all_terms(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": "fast-target",
                    "source_agent": "pi",
                    "project": "fast",
                    "raw_text": "Pi Tailscale latency analysis",
                },
                {
                    "source_identity": "wrong-project",
                    "source_agent": "pi",
                    "project": "other",
                    "raw_text": "Pi Tailscale unrelated",
                },
            ]
        )
        statements = []
        store.conn.set_trace_callback(statements.append)

        results = store.search(
            "之前 Pi Tailscale 的延迟",
            filters={"project": "fast"},
        )

        store.conn.set_trace_callback(None)
        self.assertEqual([item["source_identity"] for item in results], ["fast-target"])
        self.assertFalse(any(" LIKE " in statement.upper() for statement in statements))
        store.close()

    def test_chinese_technical_like_recovers_strong_mixed_token(self):
        store = Store(self.tmp.name)
        documents = [
            {
                "source_identity": "mixed-target",
                "source_agent": "pi",
                "role": "user",
                "project": "mixed",
                "raw_text": "讨论通过Tailscale连接办公室网络",
                "updated_at": "2020-01-01T00:00:00Z",
            }
        ]
        documents.extend(
            {
                "source_identity": f"weak-distractor-{index}",
                "source_agent": "pi",
                "role": "user",
                "project": "mixed",
                "raw_text": "Mac unrelated notes",
                "updated_at": f"2026-01-01T00:00:0{index}Z",
            }
            for index in range(6)
        )
        store.ingest(documents)

        results = store.search(
            "讨论 Mac Tailscale",
            limit=1,
            filters={"source_agent": "pi", "role": "user"},
        )

        self.assertEqual(
            [item["source_identity"] for item in results], ["mixed-target"]
        )
        store.close()

    def test_chinese_ngrams_rerank_older_technical_candidate(self):
        store = Store(self.tmp.name)
        documents = [
            {
                "source_identity": "context-loss-target",
                "source_agent": "codex",
                "role": "user",
                "raw_text": "CPA 部署后第二轮 previous_response_id 导致上下文丢失。",
                "updated_at": "2020-01-01T00:00:00Z",
            }
        ]
        documents.extend(
            {
                "source_identity": f"generic-cpa-{index}",
                "source_agent": "codex",
                "role": "user",
                "raw_text": "CPA unrelated deployment note",
                "updated_at": f"2026-01-01T00:00:{index:02d}Z",
            }
            for index in range(20)
        )
        store.ingest(documents)

        results = store.search(
            "CPA 第二轮为什么丢上下文？",
            limit=1,
            filters={"source_agent": "codex", "role": "user"},
        )

        self.assertEqual(
            [item["source_identity"] for item in results],
            ["context-loss-target"],
        )
        store.close()

    def test_single_cjk_character_technical_query_respects_limit(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": f"cpa-note-{index}",
                    "source_agent": "codex",
                    "raw_text": "CPA note",
                }
                for index in range(40)
            ]
        )

        results = store.search(
            "查 CPA",
            limit=1,
            filters={"source_agent": "codex"},
        )

        self.assertEqual(len(results), 1)
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

    def test_query_rewrite_rejects_hallucinations_and_does_not_cache_them(self):
        store = Store(self.tmp.name)
        os.environ.update(
            TRANSLATION_BASE_URL="https://provider.example/v1",
            TRANSLATION_API_KEY="test-key",
            TRANSLATION_MODEL="test-model",
        )
        tr = Translator(store)

        class Response:
            def __init__(self, content):
                self.content = content

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": self.content}}]}).encode()

        invalid = (
            ("Funes MCP 怎么配置？", "Funes MCP use 1.0 with 1024 tokens and temperature 0.9"),
            ("Funes MCP 怎么检索？", "Funes MCP " + "detailed answer " * 40),
            ("请检索最近 30 天的 CPA 记录", "recent CPA records"),
            (
                "CPA previous_response_id 为什么丢上下文？",
                "CPA context loss",
            ),
        )
        try:
            for raw, hallucination in invalid:
                with self.subTest(raw=raw), mock.patch(
                    "service.server.urllib.request.urlopen",
                    return_value=Response(hallucination),
                ) as request:
                    self.assertEqual(tr.rewrite_query(raw), raw)
                    self.assertIsNone(
                        store.translation_get(tr._cache_key(raw, QUERY_PROMPT_VERSION))
                    )
                    self.assertEqual(request.call_count, 1)
        finally:
            store.close()

    def test_query_rewrite_uses_query_prompt_token_limit_and_versioned_cache(self):
        store = Store(self.tmp.name)
        os.environ.update(
            TRANSLATION_BASE_URL="https://provider.example/v1",
            TRANSLATION_API_KEY="test-key",
            TRANSLATION_MODEL="test-model",
            TRANSLATION_QUERY_MAX_TOKENS="96",
        )
        tr = Translator(store)
        raw_query = "Funes MCP CPA previous_response_id 请问之前为什么会丢上下文，应该如何检索？"

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"Funes MCP CPA previous_response_id retrieval configuration"}}]}'

        try:
            with mock.patch(
                "service.server.urllib.request.urlopen", return_value=Response()
            ) as request:
                self.assertEqual(
                    tr.rewrite_query(raw_query),
                    "Funes MCP CPA previous_response_id retrieval configuration",
                )
                self.assertEqual(
                    tr.rewrite_query(raw_query),
                    "Funes MCP CPA previous_response_id retrieval configuration",
                )
            self.assertEqual(request.call_count, 1)
            payload = json.loads(request.call_args.args[0].data)
            self.assertEqual(payload["max_tokens"], 96)
            self.assertEqual(payload["messages"][0]["content"], QUERY_RETRIEVAL_PROMPT)
            self.assertNotEqual(
                tr._cache_key(raw_query),
                tr._cache_key(raw_query, QUERY_PROMPT_VERSION),
            )
        finally:
            store.close()

    def test_document_normalization_keeps_original_prompt_and_english_only_shadow(self):
        store = Store(self.tmp.name)
        os.environ.update(
            TRANSLATION_BASE_URL="https://provider.example/v1",
            TRANSLATION_API_KEY="test-key",
            TRANSLATION_MODEL="test-model",
        )
        tr = Translator(store)

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"English CPA retrieval text"}}]}'

        try:
            with mock.patch(
                "service.server.urllib.request.urlopen", return_value=Response()
            ) as request:
                normalized = tr.normalize_document("中文 CPA 正文")
            self.assertEqual(normalized[0], "English CPA retrieval text")
            self.assertEqual(normalized[3], "ok")
            payload = json.loads(request.call_args.args[0].data)
            self.assertEqual(payload["messages"][0]["content"], RETRIEVAL_PROMPT)
            self.assertNotIn("max_tokens", payload)
        finally:
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

    def test_reindex_generations_reset_only_derived_state(self):
        store = Store(self.tmp.name)
        original = {
            "source_identity": "provider-row",
            "source_version": "raw-v1",
            "raw_text": "原始中文",
            "retrieval_text": "english shadow",
            "translation_hash": "old-hash",
            "translation_version": "old-version",
            "translation_status": "ok",
            "native_index_version": "native-v1",
            "native_index_status": "indexed",
            "metadata": {"keep": "unchanged"},
        }
        store.ingest(
            [
                original,
                {
                    "source_identity": "english-row",
                    "raw_text": "plain english",
                    "retrieval_text": "plain english",
                    "translation_status": "skipped_non_cjk",
                    "native_index_version": "native-en",
                    "native_index_status": "indexed",
                },
                {
                    "source_identity": "old-raw-mode-row",
                    "raw_text": "需要按当前模式重新判断",
                    "retrieval_text": "旧 raw mode shadow",
                    "translation_status": "skipped_raw_mode",
                    "native_index_status": "indexed",
                },
                {
                    "source_identity": "waiting-row",
                    "raw_text": "等待持久化",
                    "retrieval_text": "old waiting shadow",
                    "translation_status": "ok",
                    "native_index_status": "waiting_durability",
                    "native_index_error": "durability_pending",
                },
            ]
        )
        store.record_reindex_control(
            {"generation": 2, "scope": "retrieval_text", "created_at": "2026-09-13T00:00:02Z"}
        )
        store.drain_reindex_controls(1)

        provider = store.get("provider-row")
        self.assertEqual(provider["raw_text"], original["raw_text"])
        self.assertEqual(provider["source_version"], original["source_version"])
        self.assertEqual(provider["metadata"], original["metadata"])
        self.assertEqual(provider["retrieval_text"], original["raw_text"])
        self.assertEqual(provider["translation_status"], "pending_provider")
        self.assertIsNone(provider["native_index_status"])
        self.assertEqual(provider["retrieval_generation"], 2)
        self.assertEqual(provider["native_generation"], 2)
        english = store.get("english-row")
        self.assertEqual(english["translation_status"], "pending_provider")
        self.assertIsNone(english["native_index_status"])
        self.assertEqual(
            store.get("old-raw-mode-row")["translation_status"], "pending_provider"
        )
        waiting = store.get("waiting-row")
        self.assertEqual(waiting["translation_status"], "pending_provider")
        self.assertEqual(waiting["native_index_status"], "waiting_durability")
        self.assertEqual(
            {item["source_identity"] for item in store.pending_translations(10)},
            {"provider-row", "english-row", "old-raw-mode-row"},
        )

        # A pre-control derived delta must not overwrite generation 2.
        store.ingest([original])
        self.assertEqual(store.get("provider-row")["translation_status"], "pending_provider")

        # An older all-control arriving after generation 2 still clears the
        # independent native generation of canonical-eligible English rows.
        store.record_reindex_control(
            {"generation": 1, "scope": "all", "created_at": "2026-09-13T00:00:01Z"}
        )
        store.drain_reindex_controls(1)
        self.assertIsNone(store.get("english-row")["native_index_status"])
        self.assertEqual(store.get("waiting-row")["native_index_status"], "waiting_durability")
        store.close()

    def test_reindex_row_cursor_is_bounded_and_survives_restart(self):
        directory = tempfile.TemporaryDirectory()
        store = Store(directory.name)
        store.ingest(
            [
                {
                    "source_identity": f"row-{index}",
                    "raw_text": f"raw {index}",
                    "retrieval_text": f"old shadow {index}",
                    "translation_status": "skipped_non_cjk",
                    "native_index_status": "indexed",
                }
                for index in range(5)
            ]
        )
        control = {
            "generation": 1,
            "scope": "retrieval_text",
            "created_at": "2026-09-13T00:00:01Z",
        }
        store.record_reindex_control(control)
        first = store.apply_pending_reindex_controls(2)
        state = store.conn.execute(
            "SELECT row_cursor, applied_at FROM reindex_controls WHERE generation=1"
        ).fetchone()
        self.assertEqual(first["scanned"], 2)
        self.assertEqual(first["applied"], 0)
        self.assertEqual(state["row_cursor"], 2)
        self.assertIsNone(state["applied_at"])
        self.assertEqual(store.get("row-1")["translation_status"], "pending_provider")
        self.assertEqual(store.get("row-2")["translation_status"], "skipped_non_cjk")
        store.close()

        restored = Store(directory.name)
        restored.record_reindex_control(control)
        resumed = restored.conn.execute(
            "SELECT row_cursor FROM reindex_controls WHERE generation=1"
        ).fetchone()
        self.assertEqual(resumed["row_cursor"], 2)
        second = restored.apply_pending_reindex_controls(2)
        final = restored.apply_pending_reindex_controls(2)
        state = restored.conn.execute(
            "SELECT row_cursor, applied_at FROM reindex_controls WHERE generation=1"
        ).fetchone()
        self.assertEqual(second["scanned"], 2)
        self.assertEqual(second["applied"], 0)
        self.assertEqual(final["scanned"], 1)
        self.assertEqual(final["applied"], 1)
        self.assertEqual(state["row_cursor"], 5)
        self.assertIsNotNone(state["applied_at"])
        self.assertEqual(
            {restored.get(f"row-{index}")["translation_status"] for index in range(5)},
            {"pending_provider"},
        )
        restored.close()
        directory.cleanup()

    def test_reindex_rechecks_raw_with_current_translation_configuration(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": "old-skipped",
                    "raw_text": "原始内容",
                    "retrieval_text": "old shadow",
                    "translation_hash": "old-model-hash",
                    "translation_status": "skipped_non_cjk",
                }
            ]
        )
        store.record_reindex_control(
            {"generation": 1, "scope": "retrieval_text", "created_at": "2026-09-13T00:00:01Z"}
        )
        store.drain_reindex_controls(10)
        self.assertEqual(store.get("old-skipped")["translation_status"], "pending_provider")

        class Syncer:
            def upload(self, _documents):
                return {"durable": True}

        with mock.patch.dict(
            os.environ,
            {"FUNES_RETRIEVAL_LANGUAGE_MODE": "raw", "TRANSLATION_MODEL": "current-model"},
        ):
            app = mock.Mock()
            app.store = store
            app.translator = Translator(store)
            app.syncer = Syncer()
            app.translation_lock = threading.Lock()
            result = persist_translation_documents(app, store.pending_translations(10))
        item = store.get("old-skipped")
        self.assertTrue(result["durable"])
        self.assertEqual(item["translation_status"], "skipped_raw_mode")
        self.assertNotEqual(item["translation_hash"], "old-model-hash")
        store.close()

    def test_full_hub_restore_rewinds_partial_cursor_before_drain(self):
        from service.server import SnapshotSync

        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": f"hub-row-{index}",
                    "source_version": "v1",
                    "raw_text": f"old raw {index}",
                    "retrieval_text": f"old shadow {index}",
                    "translation_status": "ok",
                }
                for index in range(4)
            ]
        )
        control = {
            "generation": 1,
            "scope": "retrieval_text",
            "created_at": "2026-09-13T00:00:01Z",
        }
        store.record_reindex_control(control)
        first = store.apply_pending_reindex_controls(2)
        self.assertEqual(first["scanned"], 2)
        self.assertEqual(first["applied"], 0)

        syncer = SnapshotSync(store)
        syncer.repo = "owner/private"
        syncer.token = "test-token"
        syncer.restore_batch = 2

        def restore_cursor_prefix_revision(_filename):
            result = store.ingest(
                [
                    {
                        "source_identity": "hub-row-0",
                        "source_version": "v2",
                        "raw_text": "new raw revision before cursor",
                        "retrieval_text": "stale generation-zero shadow",
                        "translation_status": "skipped_non_cjk",
                        "retrieval_generation": 0,
                        "native_generation": 0,
                        "updated_at": "9999-01-01T00:00:00Z",
                    }
                ]
            )
            return result["updated"]

        with mock.patch.object(syncer, "_repo_files", return_value=["delta.enc"]), mock.patch.object(
            syncer, "_restore_file", side_effect=restore_cursor_prefix_revision
        ):
            self.assertEqual(syncer.restore(), 1)

        restored = store.get("hub-row-0")
        state = store.conn.execute(
            "SELECT row_cursor, applied_at FROM reindex_controls WHERE generation=1"
        ).fetchone()
        self.assertEqual(restored["raw_text"], "new raw revision before cursor")
        self.assertEqual(restored["retrieval_text"], "new raw revision before cursor")
        self.assertEqual(restored["translation_status"], "pending_provider")
        self.assertEqual(restored["retrieval_generation"], 1)
        self.assertEqual(state["row_cursor"], 4)
        self.assertIsNotNone(state["applied_at"])
        store.close()

    def test_restore_compacts_history_and_skips_satisfied_updates(self):
        store = Store(self.tmp.name)
        store.ingest(
            [
                {
                    "source_identity": f"compact-row-{index}",
                    "raw_text": f"raw {index}",
                    "retrieval_text": f"old shadow {index}",
                    "translation_status": "ok",
                    "native_index_status": "indexed",
                }
                for index in range(20)
            ]
        )
        for generation, scope in enumerate(
            ("all", "retrieval_text", "retrieval_text", "all", "retrieval_text"),
            start=1,
        ):
            store.record_reindex_control(
                {
                    "generation": generation,
                    "scope": scope,
                    "created_at": f"2026-09-13T00:00:0{generation}Z",
                }
            )

        compacted = store.compact_reindex_controls(replay=True)
        first = store.drain_reindex_controls(7)
        controls = store.conn.execute(
            "SELECT generation, scope FROM reindex_controls ORDER BY generation"
        ).fetchall()
        self.assertEqual(compacted, {"kept": 2, "deleted": 3})
        self.assertEqual(
            [(row["generation"], row["scope"]) for row in controls],
            [(4, "all"), (5, "retrieval_text")],
        )
        self.assertEqual(first["scanned"], 40)
        self.assertEqual(first["updated"], 20)

        store.compact_reindex_controls(replay=True)
        replay = store.drain_reindex_controls(7)
        self.assertEqual(replay["scanned"], 40)
        self.assertEqual(replay["updated"], 0)
        store.close()

    def test_reindex_http_returns_202_only_after_durable_queue(self):
        class Syncer:
            restoring = False
            restore_failed = False

            def __init__(self, durable):
                self.durable = durable
                self.controls = []

            def upload_reindex_control(self, control):
                self.controls.append(dict(control))
                return {"durable": self.durable, "reason": "not_durable"}

        def request(durable):
            directory = tempfile.TemporaryDirectory()
            app = mock.Mock()
            app.store = Store(directory.name)
            app.syncer = Syncer(durable)
            app.reindex_lock = threading.Lock()
            app.translation_lock = threading.Lock()
            app.reindex_wake = threading.Event()
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                status, body = self._request(
                    server, "POST", "/reindex", {"scope": "retrieval_text"}
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                app.store.close()
                directory.cleanup()
            return status, body, app.syncer.controls

        status, body, controls = request(True)
        self.assertEqual(status, 202)
        self.assertEqual(body, {"queued": True, "durable": True, "scope": "retrieval_text", "generation": 1})
        self.assertEqual(controls[0]["_funes_record"], "reindex_control")
        self.assertNotIn("raw_text", json.dumps(body))
        status, body, _ = request(False)
        self.assertEqual(status, 503)
        self.assertFalse(body["queued"])

    def test_reindex_control_is_encrypted_before_local_durable_ack(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        syncer = SnapshotSync(store)
        control = store.next_reindex_control("all")
        result = syncer.upload_reindex_control(control)
        encrypted = next((Path(self.tmp.name) / "reindex-queue").glob("*.enc"))
        self.assertTrue(result["durable"])
        self.assertTrue(encrypted.read_bytes().startswith(b"FUNES-SOURCE-V1\0"))
        self.assertNotIn(b"reindex_control", encrypted.read_bytes())
        store.close()

    def test_configured_remote_without_hf_token_is_not_durable(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "remote", "raw_text": "must reach hub"}])
        os.environ["FUNES_STORAGE_REPO"] = "owner/private"
        result = SnapshotSync(store).upload(store.get_many(["remote"]))
        self.assertFalse(result["durable"])
        self.assertEqual(result["reason"], "HF storage not configured")
        store.close()

    def test_existing_content_addressed_delta_is_durable_without_reupload(self):
        from service.server import SnapshotSync
        store = Store(self.tmp.name)
        store.ingest([{"source_identity": "remote", "raw_text": "already durable"}])
        os.environ.update(
            FUNES_STORAGE_REPO="owner/private",
            FUNES_STORAGE_KEY="test-storage-key",
            HF_TOKEN="hf-test",
        )
        api = mock.Mock()
        api.file_exists.return_value = True
        module = mock.Mock(HfApi=mock.Mock(return_value=api))
        syncer = SnapshotSync(store)
        with mock.patch.dict("sys.modules", {"huggingface_hub": module}):
            result = syncer.upload(store.get_many(["remote"]))
        self.assertTrue(result["durable"])
        self.assertTrue(result["already_uploaded"])
        api.upload_file.assert_not_called()
        self.assertFalse(list(Path(self.tmp.name).glob("funes-delta-*.jsonl.gz")))
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
