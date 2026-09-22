"""Tests for PostgreSQL B-baseline native retrieval routing and hydration."""
from __future__ import annotations

import json
import os
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from unittest import mock

import pytest

import space.server as bridge
from space.server import canonical_reference


def _post(server, path, payload, token="test-token"):
    conn = HTTPConnection(*server.server_address)
    conn.request(
        "POST",
        path,
        json.dumps(payload, ensure_ascii=False).encode(),
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    response = conn.getresponse()
    body = json.loads(response.read())
    conn.close()
    return response.status, body


class MemoryStoreDouble:
    """In-memory store double implementing get and get_many with fts_ready=False."""

    def __init__(self, records: dict[str, dict] | None = None):
        self.records = dict(records or {})
        self.get_calls = []
        self.get_many_calls = []
        self.fail_get = False
        self.fail_get_many = False

    def fts_ready(self) -> bool:
        return False

    def get(self, identity: str) -> dict | None:
        self.get_calls.append(identity)
        if self.fail_get:
            raise RuntimeError("postgresql://user:secret_pass@pg-host:5432/db connection broken")
        item = self.records.get(identity)
        return dict(item) if item else None

    def get_many(self, identities: list[str]) -> list[dict]:
        self.get_many_calls.append(list(identities))
        if self.fail_get_many:
            raise RuntimeError("postgresql://user:secret_pass@pg-host:5432/db connection broken")
        results = []
        for identity in identities:
            item = self.records.get(identity)
            if item:
                results.append(dict(item))
        return results


def test_materialize_native_results_batch_hydration_order_and_fallback():
    """Verify batch hydration preserves native result rank order and falls back to get."""
    doc1_id = "doc/chatcmpl-9xyz8765/previous_response_id"
    doc2_id = "doc/Northflank/CPA/v1.2.3"
    doc3_id = "doc/path/Users/pwd/code/funes/https/api"

    ref1 = canonical_reference(doc1_id)
    ref2 = canonical_reference(doc2_id)
    ref3 = canonical_reference(doc3_id)

    raw1 = "Exact identifier: previous_response_id in chatcmpl-9xyz8765"
    raw2 = "Exact identifier: Northflank/CPA release v1.2.3"
    raw3 = "Exact identifier: path /Users/pwd/code/funes and URL https://example.com/v1"

    records = {
        doc1_id: {
            "source_identity": doc1_id,
            "raw_text": raw1,
            "retrieval_text": "SHOULD_BE_STRIPPED_1",
            "source_type": "doc",
        },
        doc2_id: {
            "source_identity": doc2_id,
            "raw_text": raw2,
            "retrieval_text": "SHOULD_BE_STRIPPED_2",
            "source_type": "doc",
        },
        doc3_id: {
            "source_identity": doc3_id,
            "raw_text": raw3,
            "retrieval_text": "SHOULD_BE_STRIPPED_3",
            "source_type": "doc",
        },
    }

    class ScrambledStore(MemoryStoreDouble):
        def get_many(self, identities: list[str]) -> list[dict]:
            self.get_many_calls.append(list(identities))
            return [dict(records[doc2_id]), dict(records[doc1_id])]

    store = ScrambledStore(records)
    app = mock.Mock()
    app.store = store

    native_output = f"search results:\n  → get {ref1} --from 1\n  → get {ref2} --from 2\n  → get {ref3} --from 3\n"

    results = bridge.materialize_native_results(native_output, app, limit=3)

    assert len(results) == 3
    assert results[0]["source_identity"] == doc1_id
    assert results[0]["raw_text"] == raw1
    assert "retrieval_text" not in results[0]
    assert results[0]["retrieval_backend"] == "native_funes"

    assert results[1]["source_identity"] == doc2_id
    assert results[1]["raw_text"] == raw2
    assert "retrieval_text" not in results[1]
    assert results[1]["retrieval_backend"] == "native_funes"

    assert results[2]["source_identity"] == doc3_id
    assert results[2]["raw_text"] == raw3
    assert "retrieval_text" not in results[2]
    assert results[2]["retrieval_backend"] == "native_funes"

    assert doc3_id in store.get_calls


def test_materialize_native_results_get_many_failure_falls_back_to_get():
    """Verify get_many exception gracefully degrades to single get lookups."""
    doc_id = "doc/test-fallback"
    ref = canonical_reference(doc_id)
    store = MemoryStoreDouble({
        doc_id: {
            "source_identity": doc_id,
            "raw_text": "fallback text",
            "retrieval_text": "strip-me",
        }
    })
    store.fail_get_many = True
    app = mock.Mock()
    app.store = store

    native_output = f"  → get {ref}\n"
    results = bridge.materialize_native_results(native_output, app, limit=1)

    assert len(results) == 1
    assert results[0]["source_identity"] == doc_id
    assert results[0]["raw_text"] == "fallback text"
    assert "retrieval_text" not in results[0]
    assert doc_id in store.get_calls


def test_voyage_pg_b_baseline_routes_directly_with_technical_identifiers(monkeypatch):
    """Voyage + PG B-baseline (fts_ready=False) routes to native without false 503."""
    doc_id = "doc/chatcmpl-9xyz8765/previous_response_id"
    ref = canonical_reference(doc_id)
    raw_content = (
        "Previous response ID: previous_response_id, "
        "Chat completion: chatcmpl-9xyz8765, "
        "Service: Northflank/CPA v1.2.3, "
        "File: /Users/pwd/code/funes, URL: https://api.northflank.com/v1"
    )

    store = MemoryStoreDouble({
        doc_id: {
            "source_identity": doc_id,
            "raw_text": raw_content,
            "retrieval_text": "STRIP_SECRET",
        }
    })
    syncer = mock.Mock()
    syncer.restoring = False
    syncer.restore_failed = False
    app = mock.Mock()
    app.store = store
    app.syncer = syncer

    recall_calls = []

    def fake_recall(query, **kwargs):
        recall_calls.append((query, kwargs))
        return f"result:\n  → get {ref} --from 1\n"

    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("FUNES_POSTGRES_DSN", "postgresql://user:pass@localhost:5432/funes")
    monkeypatch.setattr(bridge, "recall", fake_recall)

    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        query = "previous_response_id chatcmpl-9xyz8765 Northflank/CPA /Users/pwd/code/funes https://api.northflank.com/v1 v1.2.3"
        status, body = _post(
            server,
            "/search",
            {"query": query, "limit": 3},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 200
    assert body["ok"] is True
    assert body["query"] == query
    assert body["retrieval_backend"] == "voyage_lance_bm25_rrf"
    assert len(body["results"]) == 1
    item = body["results"][0]
    assert item["source_identity"] == doc_id
    assert item["raw_text"] == raw_content
    assert "retrieval_text" not in item
    assert recall_calls[0][0] == query


def test_supported_canonical_filters_passed_to_native_tuning_without_false_503(monkeypatch):
    """Supported canonical filters are passed to native recall tuning when PG FTS is disabled."""
    doc_id = "doc/filtered"
    ref = canonical_reference(doc_id)
    store = MemoryStoreDouble({
        doc_id: {
            "source_identity": doc_id,
            "raw_text": "Filtered content",
            "source_type": "doc",
            "project": "funes",
            "repo": "funes-repo",
            "device_id": "dev-001",
            "content_type": "text/plain",
            "source_missing": False,
        }
    })
    syncer = mock.Mock(restoring=False, restore_failed=False)
    app = mock.Mock(store=store, syncer=syncer)

    recall_calls = []

    def fake_recall(query, **kwargs):
        recall_calls.append((query, kwargs))
        return f"result:\n  → get {ref}\n"

    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("FUNES_POSTGRES_DSN", "postgresql://user:pass@localhost:5432/funes")
    monkeypatch.setattr(bridge, "recall", fake_recall)

    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(
            server,
            "/search",
            {
                "query": "test query",
                "limit": 5,
                "source_agent": "codex",
                "source_type": "doc",
                "project": "funes",
                "repo": "funes-repo",
                "device_id": "dev-001",
                "content_type": "text/plain",
                "source_missing": False,
                "harness": "codex",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 200
    assert body["ok"] is True
    assert len(recall_calls) == 1
    query, kwargs = recall_calls[0]
    assert query == "test query"
    assert kwargs["source_agent"] == "codex"
    assert kwargs["source_type"] == "doc"
    assert kwargs["project"] == "funes"
    assert kwargs["repo"] == "funes-repo"
    assert kwargs["device_id"] == "dev-001"
    assert kwargs["content_type"] == "text/plain"
    assert kwargs["source_missing"] is False
    assert kwargs["harness"] == "codex"


@pytest.mark.parametrize("unsupported_filter", [
    {"role": "user"},
    {"since": "2026-09-01T00:00:00Z"},
    {"until": "2026-09-22T00:00:00Z"},
])
def test_unsupported_filters_fail_fast_503_when_pg_fts_disabled(monkeypatch, unsupported_filter):
    """Unsupported sidecar filters (role, since, until) return 503 source_fts_unavailable."""
    store = MemoryStoreDouble()
    syncer = mock.Mock(restoring=False, restore_failed=False)
    app = mock.Mock(store=store, syncer=syncer)

    recall_called = False

    def fake_recall(*_args, **_kwargs):
        nonlocal recall_called
        recall_called = True
        return ""

    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("FUNES_POSTGRES_DSN", "postgresql://user:pass@localhost:5432/funes")
    monkeypatch.setattr(bridge, "recall", fake_recall)

    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = {"query": "test query", "limit": 3, **unsupported_filter}
        status, body = _post(server, "/search", payload)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 503
    assert body["ok"] is False
    assert body["error"] == "source_fts_unavailable"
    assert not recall_called


def test_postgres_disconnect_during_search_sanitized_503(monkeypatch):
    """PG disconnect during search returns 503 postgres_unavailable without leaking secrets."""
    doc_id = "doc/disconnect-test"
    ref = canonical_reference(doc_id)

    store = MemoryStoreDouble({
        doc_id: {"source_identity": doc_id, "raw_text": "sensitive content"}
    })
    store.fail_get_many = True
    store.fail_get = True
    syncer = mock.Mock(restoring=False, restore_failed=False)
    app = mock.Mock(store=store, syncer=syncer)

    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    monkeypatch.setattr(bridge, "TOKEN", "test-token")
    monkeypatch.setenv("FUNES_EMBEDDING_PROVIDER", "voyage")
    secret_dsn = "postgresql://dbuser:super_secret_password_123@db.internal:5432/funes"
    monkeypatch.setenv("FUNES_POSTGRES_DSN", secret_dsn)
    monkeypatch.setattr(bridge, "recall", lambda *_args, **_kwargs: f"  → get {ref}\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(server, "/search", {"query": "test disconnect"})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 503
    assert body["ok"] is False
    assert body["error"] == "postgres_unavailable"
    body_str = json.dumps(body)
    assert "super_secret_password_123" not in body_str
    assert "dbuser" not in body_str
    assert "db.internal" not in body_str


def test_real_native_identifiers_offline(tmp_path, monkeypatch):
    """Opt-in real Lance recall; cached BGE, no production config or paid API.

    This verifies identifier retrieval and source hydration, not Voyage semantic
    quality. Other tests independently exercise the real PostgreSQL contract.
    """
    import hashlib
    from pathlib import Path
    import shutil
    import signal
    import subprocess
    import time
    from types import SimpleNamespace

    binary_value = os.getenv("FUNES_TEST_NATIVE_BINARY")
    cache_value = os.getenv("FUNES_TEST_HF_HOME")
    if not binary_value or not cache_value:
        pytest.skip("set FUNES_TEST_NATIVE_BINARY and a populated FUNES_TEST_HF_HOME")
    binary = Path(binary_value).resolve()
    cache = Path(cache_value).resolve()
    assert binary.is_file(), "native test binary must already exist; do not build"
    snapshots = cache / "hub/models--BAAI--bge-small-en-v1.5/snapshots"
    assert any(
        (p / "model.safetensors").is_file() and (p / "tokenizer.json").is_file()
        for p in snapshots.glob("*")
    ), "populate the test model cache before running; downloads are disabled"
    scanner_value = os.getenv("FUNES_TEST_TRUFFLEHOG") or shutil.which("trufflehog")
    assert scanner_value and Path(scanner_value).is_file(), "use the real offline secret scanner"
    monkeypatch.setenv("RETURN_RETRIEVAL_TEXT", "false")

    cases = [
        ("previous", "previous_response_id", "之前 CPA 第二轮上下文丢失涉及 previous_response_id。"),
        ("completion", "chatcmpl-*", "返回 ID 类型为 chatcmpl-*，实例是 chatcmpl-review0922。"),
        ("url", "https://api.example.invalid/v1/responses?stream=false",
         "请求地址是 https://api.example.invalid/v1/responses?stream=false。"),
        ("path", "/Users/example/worktree/funes/service/postgres.py",
         "修改文件 /Users/example/worktree/funes/service/postgres.py，保留原文。"),
        ("model", "voyage-4-lite", "默认 embedding 模型版本为 voyage-4-lite。"),
        ("error", "ERR_PG_08006", "连接中断时错误码为 ERR_PG_08006，保留待传队列。"),
    ]
    distractors = [
        "The unrelated identifier is conversation_id.",
        "A different completion type is completion-other.",
        "An unrelated endpoint is https://other.invalid/v2/messages.",
        "The unrelated source file is /var/tmp/different.py.",
        "A different model is bge-small-en-v1.5.",
        "A separate warning is WARN_STORE_BUSY.",
    ]
    records = {}
    documents = []
    for ordinal, (name, raw) in enumerate(
        [(name, raw) for name, _query, raw in cases]
        + [(f"distractor-{i}", raw) for i, raw in enumerate(distractors)]
    ):
        identity = f"offline-identifier/{name}"
        doc = {
            "source_identity": identity,
            "source_version": "offline-v1",
            "content_hash": hashlib.sha256(raw.encode()).hexdigest(),
            "updated_at": "2026-09-22T00:00:00Z",
            "raw_text": raw,
            "retrieval_text": raw,
            "session_id": canonical_reference(identity),
            "source_agent": ("codex", "pi", "claude_code")[ordinal % 3],
            "source_type": "memory",
            "content_type": "memory",
        }
        documents.append(doc)
        records[identity] = dict(doc, retrieval_text="derived text must not be returned")
    jsonl = tmp_path / "identifiers.jsonl"
    jsonl.write_text("".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents))
    home = tmp_path / "home"
    home.mkdir()
    # An allow-list is deliberate: do not inherit HF/Voyage tokens, a remote
    # memory, a PG DSN, or the operator's real FUNES_HOME into this subprocess.
    env = {
        "PATH": os.defpath,
        "HOME": str(home),
        "TMPDIR": str(tmp_path),
        "HF_HOME": str(cache),
        "HF_HUB_OFFLINE": "1",
        "FUNES_HOME": str(tmp_path / "funes"),
        "FUNES_EMBEDDING_PROVIDER": "local",
        "FUNES_RERANK_PROVIDER": "none",
        "FUNES_TRUFFLEHOG": str(Path(scanner_value).resolve()),
        "RAYON_NUM_THREADS": "2",
        "TOKIO_WORKER_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
        "OMP_NUM_THREADS": "2",
    }
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env[key] = "http://127.0.0.1:9"
    started = time.monotonic()
    deadline = started + 120

    def run(*args):
        remaining = deadline - time.monotonic()
        assert remaining > 0, "native acceptance exceeded the 120-second budget"
        process = subprocess.Popen(
            [str(binary), *args], env=env, cwd=tmp_path,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=min(40, remaining))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            pytest.fail(f"native {args[0]} timed out; isolated process group terminated")
        assert process.returncode == 0, f"native {args[0]} failed: {stderr}"
        return stdout

    version = run("--version").strip()
    ingestion = run("ingest-docs", str(jsonl))
    assert "sources=12" in ingestion and "held=0" in ingestion, ingestion
    store = MemoryStoreDouble(records)
    app = SimpleNamespace(store=store)
    results = []
    queries = [(name, query) for name, query, _raw in cases]
    queries.append(("completion", "chatcmpl-review0922"))
    for name, query in queries:
        output = run("recall", query, "-k", "1", "--half-life", "0", "--neighbors", "0")
        expected = f"offline-identifier/{name}"
        ids = bridge.native_result_ids(output)
        assert ids and bridge.canonical_reference_identity(ids[0]) == expected, output
        hydrated = bridge.materialize_native_results(output, app, limit=1)
        assert len(hydrated) == 1
        assert hydrated[0]["raw_text"] == records[expected]["raw_text"]
        assert query in hydrated[0]["raw_text"]
        assert "retrieval_text" not in hydrated[0]
        results.append({"query": query, "expected_source": expected, "rank": 1, "raw_exact": True})
    assert len(store.get_many_calls) == len(queries)
    assert list((tmp_path / "funes").rglob("*.lance")), "test must create an actual Lance dataset"
    report_path = os.getenv("FUNES_TEST_NATIVE_REPORT")
    if report_path:
        with binary.open("rb") as handle:
            binary_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        Path(report_path).write_text(json.dumps({
            "binary_version": version, "binary_sha256": binary_sha,
            "model": "BAAI/bge-small-en-v1.5", "dimensions": 384,
            "paid_api_calls": 0, "production_requests": 0,
            "native_queries_mocked": False, "source_hydration_store": "in_memory_double",
            "voyage_semantic_quality_tested": False,
            "documents": len(documents), "queries": results,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }, ensure_ascii=False, indent=2) + "\n")
