#!/usr/bin/env python3
"""Benchmark the Chinese retrieval fixture with Voyage embeddings and reranking.

With no mode flag this command performs an offline self-check: it does not read
credentials, open sockets, or write result/cache files. ``--live-voyage`` runs
the paid Voyage matrix, while ``--http-latency`` measures 50 warm requests to a
configured Funes HTTP service. Credentials are read only from their documented
environment variables and are never serialized or printed.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_chinese_retrieval import Memory, Query, _memories, _queries


EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"
RERANK_URL = "https://api.voyageai.com/v1/rerank"
CACHE_VERSION = 1
DEFAULT_CACHE = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "funes"
    / "voyage-retrieval-embeddings.json"
)
SHARED_VOYAGE_4_SPACE = frozenset({"voyage-4", "voyage-4-lite"})


class BenchmarkError(RuntimeError):
    """A sanitized benchmark failure that contains no request or response body."""


@dataclass(frozen=True)
class BenchmarkArm:
    name: str
    document_model: str
    query_model: str
    rerank_model: str | None = None


MATRIX = (
    BenchmarkArm("voyage-4-lite", "voyage-4-lite", "voyage-4-lite"),
    BenchmarkArm("voyage-4", "voyage-4", "voyage-4"),
    BenchmarkArm(
        "voyage-4-doc__voyage-4-lite-query",
        "voyage-4",
        "voyage-4-lite",
    ),
    BenchmarkArm("voyage-code-4", "voyage-code-4", "voyage-code-4"),
    BenchmarkArm(
        "voyage-4-lite__rerank-3-lite",
        "voyage-4-lite",
        "voyage-4-lite",
        "rerank-3-lite",
    ),
)

HTTP_LATENCY_CASES = (
    ("chinese", "部署失败时如何自动恢复？"),
    ("english", "How does a failed deployment return to the last healthy release?"),
    ("mixed", "Funes 的 previous_response_id context 为什么在 second turn 丢失？"),
    ("semantic_paraphrase", "服务发布出错后，用什么机制恢复到上一个可用版本？"),
    ("exact_identifier", "查找 source_agent=codex 和 device_id 的跨代理检索记录"),
    ("code_error", "HTTP 429 FREE_MODEL_FAILED and EmptyProviderResponseError fallback"),
)
DEFAULT_HTTP_RETRIEVAL_BACKEND = "voyage_lance_bm25_rrf"
DEFAULT_HTTP_EMBEDDING_PROFILE = {
    "provider": "voyage",
    "model": "voyage-4-lite",
    "dimensions": 1024,
    "schema_version": 2,
}
EMBEDDING_PROFILE_FIELDS = (
    "provider",
    "model",
    "dimensions",
    "schema_version",
    "fingerprint",
)
KNOWN_RETRIEVAL_DEGRADATIONS = frozenset(
    {"voyage_unavailable", "native_mcp_busy", "native_mcp_unavailable"}
)
MAX_HTTP_ERROR_BODY_BYTES = 16 * 1024


@dataclass(frozen=True)
class ClientStats:
    embedding_api_calls: int
    rerank_api_calls: int
    embedding_cache_hits: int
    embedding_cache_misses: int
    request_seconds: float
    embedding_request_seconds: tuple[float, ...]
    rerank_request_seconds: tuple[float, ...]


@dataclass(frozen=True)
class HttpRecallObservation:
    status: int
    retrieval_backend: str | None
    embedding_profile: dict[str, object] | None
    degraded_reason: str | None
    validation_failures: tuple[str, ...]

    @property
    def verified_voyage(self) -> bool:
        return not self.validation_failures


@dataclass(frozen=True)
class _HttpTransportConfig:
    target_host: str
    target_port: int
    connect_host: str
    connect_port: int
    request_target: str
    target_scheme: str
    transport: str
    proxy_used: bool
    tunnel: bool


_RETRYABLE_HTTP_DISCONNECTS = (
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)


class _PersistentFunesHttpClient:
    """One active HTTP/1.1 connection, with one safe retry after disconnect."""

    def __init__(self, *, endpoint: str, token: str, timeout: float) -> None:
        self._config = _http_transport_config(endpoint)
        self._token = token
        self._timeout = timeout
        self._connection: http.client.HTTPConnection | None = None
        self.connection_attempts = 0
        self.connections_opened = 0
        self.request_attempts = 0
        self.disconnect_retries = 0

    @property
    def transport(self) -> str:
        return self._config.transport

    @property
    def proxy_used(self) -> bool:
        return self._config.proxy_used

    def _new_connection(self) -> http.client.HTTPConnection:
        config = self._config
        if config.target_scheme == "https":
            connection: http.client.HTTPConnection = http.client.HTTPSConnection(
                config.connect_host,
                config.connect_port,
                timeout=self._timeout,
            )
        else:
            connection = http.client.HTTPConnection(
                config.connect_host,
                config.connect_port,
                timeout=self._timeout,
            )
        if config.tunnel:
            # Target Authorization belongs only to the request inside the tunnel.
            connection.set_tunnel(config.target_host, config.target_port, headers={})
        return connection

    def _ensure_connected(self) -> http.client.HTTPConnection:
        if self._connection is None:
            self._connection = self._new_connection()
        if self._connection.sock is None:
            self.connection_attempts += 1
            self._connection.connect()
            self.connections_opened += 1
        return self._connection

    def _discard_connection(self) -> None:
        if self._connection is not None:
            connection = self._connection
            self._connection = None
            try:
                connection.close()
            except (OSError, http.client.HTTPException):
                pass

    def close(self) -> None:
        self._discard_connection()

    def post_recall(self, query: str) -> tuple[int, object]:
        body = json.dumps(
            {"query": query, "limit": 5},
            ensure_ascii=False,
        ).encode("utf-8")
        response_body: bytes
        status: int
        for retry in range(2):
            try:
                connection = self._ensure_connected()
                self.request_attempts += 1
                connection.request(
                    "POST",
                    self._config.request_target,
                    body=body,
                    headers={
                        "Authorization": "Bearer " + self._token,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Connection": "keep-alive",
                    },
                )
                response = connection.getresponse()
                status = int(response.status)
                if status < 200 or status >= 300:
                    response_reusable = False
                    try:
                        error_bytes_read = len(
                            response.read(MAX_HTTP_ERROR_BODY_BYTES + 1)
                        )
                        isclosed = getattr(response, "isclosed", None)
                        response_reusable = (
                            error_bytes_read <= MAX_HTTP_ERROR_BODY_BYTES
                            and callable(isclosed)
                            and bool(isclosed())
                        )
                    except (
                        TimeoutError,
                        OSError,
                        http.client.HTTPException,
                        TypeError,
                        ValueError,
                    ):
                        pass
                    finally:
                        try:
                            response.close()
                        except (OSError, http.client.HTTPException, ValueError):
                            response_reusable = False
                    if not response_reusable:
                        self._discard_connection()
                    raise BenchmarkError(f"Funes recall HTTP {status}")
                try:
                    # Exhausting the response is required before this connection can
                    # safely carry the next benchmark request.
                    response_body = response.read()
                finally:
                    response.close()
                break
            except _RETRYABLE_HTTP_DISCONNECTS as exc:
                self._discard_connection()
                if retry == 0:
                    # POST /recall is retrieval-only, so one replay on a fresh
                    # connection is safe. Never retry timeouts or arbitrary errors.
                    self.disconnect_retries += 1
                    continue
                raise BenchmarkError(
                    f"Funes recall transport failure ({type(exc).__name__})"
                ) from None
            except (TimeoutError, OSError, http.client.HTTPException) as exc:
                self._discard_connection()
                raise BenchmarkError(
                    f"Funes recall transport failure ({type(exc).__name__})"
                ) from None
        else:  # pragma: no cover - the bounded loop either breaks or raises
            raise AssertionError("unreachable HTTP retry state")

        try:
            return status, json.loads(response_body)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise BenchmarkError("Funes recall returned malformed JSON") from None


class EmbeddingCache:
    """Persistent embeddings keyed by endpoint, model, role, and text hash."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self._entries: dict[str, dict[str, object]] = {}
        self._dirty = False
        if self.path.exists():
            self._load()

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def _key_from_hash(
        cls,
        *,
        endpoint: str,
        model: str,
        input_type: str,
        text_hash: str,
    ) -> str:
        identity = json.dumps(
            [endpoint, model, input_type, text_hash],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @classmethod
    def _key(cls, *, endpoint: str, model: str, input_type: str, text: str) -> str:
        return cls._key_from_hash(
            endpoint=endpoint,
            model=model,
            input_type=input_type,
            text_hash=cls._text_hash(text),
        )

    @staticmethod
    def _valid_vector(value: object) -> bool:
        return (
            isinstance(value, list)
            and bool(value)
            and all(
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                and math.isfinite(float(number))
                for number in value
            )
        )

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            entries = payload["entries"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise BenchmarkError(f"invalid embedding cache: {self.path}") from exc
        if payload.get("version") != CACHE_VERSION or not isinstance(entries, dict):
            raise BenchmarkError(f"unsupported embedding cache: {self.path}")
        for key, row in entries.items():
            if not isinstance(key, str) or not isinstance(row, dict):
                raise BenchmarkError(f"invalid embedding cache entry: {self.path}")
            endpoint = row.get("endpoint")
            model = row.get("model")
            input_type = row.get("input_type")
            text_hash = row.get("text_sha256")
            vector = row.get("embedding")
            if not all(isinstance(value, str) for value in (endpoint, model, input_type, text_hash)):
                raise BenchmarkError(f"invalid embedding cache entry: {self.path}")
            expected_key = self._key_from_hash(
                endpoint=str(endpoint),
                model=str(model),
                input_type=str(input_type),
                text_hash=str(text_hash),
            )
            if key != expected_key or not self._valid_vector(vector):
                raise BenchmarkError(f"invalid embedding cache entry: {self.path}")
        self._entries = entries

    def get(
        self,
        *,
        endpoint: str,
        model: str,
        input_type: str,
        text: str,
    ) -> list[float] | None:
        key = self._key(
            endpoint=endpoint,
            model=model,
            input_type=input_type,
            text=text,
        )
        row = self._entries.get(key)
        if row is None:
            return None
        if row.get("text_sha256") != self._text_hash(text):
            raise BenchmarkError("embedding cache hash mismatch")
        return [float(value) for value in row["embedding"]]  # type: ignore[index]

    def put(
        self,
        *,
        endpoint: str,
        model: str,
        input_type: str,
        text: str,
        embedding: list[float],
    ) -> None:
        if not self._valid_vector(embedding):
            raise BenchmarkError("Voyage returned an invalid embedding vector")
        text_hash = self._text_hash(text)
        key = self._key_from_hash(
            endpoint=endpoint,
            model=model,
            input_type=input_type,
            text_hash=text_hash,
        )
        self._entries[key] = {
            "endpoint": endpoint,
            "model": model,
            "input_type": input_type,
            "text_sha256": text_hash,
            "embedding": [float(value) for value in embedding],
        }
        self._dirty = True

    @property
    def entries(self) -> int:
        return len(self._entries)

    def save(self) -> None:
        if not self._dirty:
            return
        payload = {"version": CACHE_VERSION, "entries": self._entries}
        _atomic_write_json(self.path, payload)
        self._dirty = False


class VoyageClient:
    """Small native Voyage REST client with bounded retries and embedding cache."""

    def __init__(
        self,
        *,
        api_key: str,
        cache: EmbeddingCache,
        timeout: float,
        attempts: int,
    ) -> None:
        if not api_key:
            raise BenchmarkError("VOYAGE_API_KEY is empty")
        self._api_key = api_key
        self.cache = cache
        self.timeout = timeout
        self.attempts = attempts
        self._embedding_api_calls = 0
        self._rerank_api_calls = 0
        self._embedding_cache_hits = 0
        self._embedding_cache_misses = 0
        self._request_seconds = 0.0
        self._embedding_request_seconds: list[float] = []
        self._rerank_request_seconds: list[float] = []

    @property
    def stats(self) -> ClientStats:
        return ClientStats(
            embedding_api_calls=self._embedding_api_calls,
            rerank_api_calls=self._rerank_api_calls,
            embedding_cache_hits=self._embedding_cache_hits,
            embedding_cache_misses=self._embedding_cache_misses,
            request_seconds=self._request_seconds,
            embedding_request_seconds=tuple(self._embedding_request_seconds),
            rerank_request_seconds=tuple(self._rerank_request_seconds),
        )

    def _post_json(self, url: str, body: dict[str, object], operation: str) -> dict[str, object]:
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        last_error = f"Voyage {operation} request failed"
        for attempt in range(self.attempts):
            request = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            started = time.monotonic()
            try:
                if operation == "embeddings":
                    self._embedding_api_calls += 1
                else:
                    self._rerank_api_calls += 1
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    decoded = json.loads(response.read())
                if not isinstance(decoded, dict):
                    raise ValueError("JSON object required")
                return decoded
            except urllib.error.HTTPError as exc:
                last_error = f"Voyage {operation} HTTP {exc.code}"
                retryable = exc.code in (408, 425, 429) or exc.code >= 500
                if not retryable:
                    raise BenchmarkError(last_error) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"Voyage {operation} transport failure ({type(exc).__name__})"
            except (TypeError, ValueError, json.JSONDecodeError):
                last_error = f"Voyage {operation} returned malformed JSON"
            finally:
                elapsed = time.monotonic() - started
                self._request_seconds += elapsed
                if operation == "embeddings":
                    self._embedding_request_seconds.append(elapsed)
                else:
                    self._rerank_request_seconds.append(elapsed)
            if attempt + 1 < self.attempts:
                time.sleep(0.5 * (2**attempt))
        raise BenchmarkError(f"{last_error} after {self.attempts} attempts")

    @staticmethod
    def _parse_embeddings(payload: dict[str, object], expected: int) -> list[list[float]]:
        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != expected:
            raise BenchmarkError("Voyage embeddings response has the wrong row count")
        ordered: list[list[float] | None] = [None] * expected
        for row in rows:
            if not isinstance(row, dict):
                raise BenchmarkError("Voyage embeddings response contains an invalid row")
            index = row.get("index")
            vector = row.get("embedding")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= expected
                or ordered[index] is not None
                or not EmbeddingCache._valid_vector(vector)
            ):
                raise BenchmarkError("Voyage embeddings response contains an invalid row")
            ordered[index] = [float(value) for value in vector]  # type: ignore[union-attr]
        if any(vector is None for vector in ordered):
            raise BenchmarkError("Voyage embeddings response is missing a row")
        dimensions = {len(vector) for vector in ordered if vector is not None}
        if len(dimensions) != 1:
            raise BenchmarkError("Voyage embeddings response has inconsistent dimensions")
        return [vector for vector in ordered if vector is not None]

    def embed(
        self,
        texts: list[str],
        *,
        model: str,
        input_type: str,
        batch_size: int = 128,
    ) -> list[list[float]]:
        if input_type not in ("document", "query"):
            raise BenchmarkError("embedding input_type must be document or query")
        if not texts or any(not isinstance(text, str) or not text for text in texts):
            raise BenchmarkError("embedding input must contain non-empty strings")
        if batch_size < 1:
            raise BenchmarkError("embedding batch_size must be positive")

        resolved: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, list[int]] = {}
        for index, text in enumerate(texts):
            cached = self.cache.get(
                endpoint=EMBEDDINGS_URL,
                model=model,
                input_type=input_type,
                text=text,
            )
            if cached is not None:
                self._embedding_cache_hits += 1
                resolved[index] = cached
            else:
                self._embedding_cache_misses += 1
                missing.setdefault(text, []).append(index)

        unique_missing = list(missing)
        for offset in range(0, len(unique_missing), batch_size):
            batch = unique_missing[offset : offset + batch_size]
            payload = self._post_json(
                EMBEDDINGS_URL,
                {
                    "input": batch,
                    "model": model,
                    "input_type": input_type,
                },
                "embeddings",
            )
            if payload.get("model") != model:
                raise BenchmarkError("Voyage embeddings response model does not match request")
            vectors = self._parse_embeddings(payload, len(batch))
            for text, vector in zip(batch, vectors):
                self.cache.put(
                    endpoint=EMBEDDINGS_URL,
                    model=model,
                    input_type=input_type,
                    text=text,
                    embedding=vector,
                )
                for index in missing[text]:
                    resolved[index] = vector
            self.cache.save()
        if any(vector is None for vector in resolved):
            raise BenchmarkError("embedding resolution failed")
        return [vector for vector in resolved if vector is not None]

    def rerank(self, query: str, documents: list[str], *, model: str) -> list[int]:
        if not query or not documents or any(not document for document in documents):
            raise BenchmarkError("rerank requires a non-empty query and documents")
        payload = self._post_json(
            RERANK_URL,
            {
                "query": query,
                "documents": documents,
                "model": model,
                "top_k": len(documents),
                "return_documents": False,
            },
            "rerank",
        )
        if payload.get("model") != model:
            raise BenchmarkError("Voyage rerank response model does not match request")
        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != len(documents):
            raise BenchmarkError("Voyage rerank response has the wrong row count")
        ranked: list[int] = []
        for row in rows:
            if not isinstance(row, dict):
                raise BenchmarkError("Voyage rerank response contains an invalid row")
            index = row.get("index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= len(documents)
                or index in ranked
            ):
                raise BenchmarkError("Voyage rerank response contains an invalid index")
            ranked.append(index)
        return ranked


def _models_share_space(document_model: str, query_model: str) -> bool:
    if document_model == query_model:
        return True
    return {document_model, query_model}.issubset(SHARED_VOYAGE_4_SPACE)


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise BenchmarkError("document and query embeddings have incompatible dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _embedding_ranking(
    memories: list[Memory],
    document_embeddings: list[list[float]],
    query_embedding: list[float],
) -> list[int]:
    if len(memories) != len(document_embeddings):
        raise BenchmarkError("memory and document embedding counts differ")
    return sorted(
        range(len(memories)),
        key=lambda index: (
            -_cosine(query_embedding, document_embeddings[index]),
            memories[index].ident,
        ),
    )


def _quality_metrics(ranks: list[int | None]) -> dict[str, float]:
    if not ranks:
        raise BenchmarkError("quality metrics require at least one rank")
    metrics = {
        f"recall@{k}": round(
            sum(rank is not None and rank <= k for rank in ranks) / len(ranks),
            4,
        )
        for k in (1, 3, 5)
    }
    metrics["mrr"] = round(
        sum(1.0 / rank for rank in ranks if rank is not None) / len(ranks),
        4,
    )
    return metrics


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _latency_metrics(milliseconds: list[float]) -> dict[str, float]:
    if not milliseconds:
        raise BenchmarkError("latency metrics require at least one sample")
    return {
        "p50": round(_percentile(milliseconds, 0.50), 3),
        "p95": round(_percentile(milliseconds, 0.95), 3),
        "max": round(max(milliseconds), 3),
    }


def _stats_delta(before: ClientStats, after: ClientStats) -> dict[str, object]:
    return {
        "embedding_api_calls": after.embedding_api_calls - before.embedding_api_calls,
        "rerank_api_calls": after.rerank_api_calls - before.rerank_api_calls,
        "embedding_cache_hits": after.embedding_cache_hits - before.embedding_cache_hits,
        "embedding_cache_misses": after.embedding_cache_misses - before.embedding_cache_misses,
        "request_seconds": round(after.request_seconds - before.request_seconds, 3),
    }


def _api_network_latency_delta(
    before: ClientStats,
    after: ClientStats,
) -> dict[str, dict[str, object]]:
    embedding_seconds = list(
        after.embedding_request_seconds[len(before.embedding_request_seconds) :]
    )
    rerank_seconds = list(after.rerank_request_seconds[len(before.rerank_request_seconds) :])

    def summary(seconds: list[float]) -> dict[str, object]:
        milliseconds = [value * 1000 for value in seconds]
        return {
            "requests": len(milliseconds),
            "latency_ms": _latency_metrics(milliseconds) if milliseconds else None,
        }

    return {
        "embedding": summary(embedding_seconds),
        "rerank": summary(rerank_seconds),
        "all": summary([*embedding_seconds, *rerank_seconds]),
    }


def _arm_cache_path(cache_path: Path, arm: BenchmarkArm) -> Path:
    digest = hashlib.sha256(arm.name.encode("utf-8")).hexdigest()[:12]
    suffix = cache_path.suffix or ".json"
    stem = cache_path.stem if cache_path.suffix else cache_path.name
    return cache_path.with_name(f"{stem}.{digest}{suffix}")


def _warm_arm_embeddings(
    *,
    arm: BenchmarkArm,
    memories: list[Memory],
    queries: list[Query],
    client: VoyageClient,
    batch_size: int,
) -> dict[str, object]:
    if not _models_share_space(arm.document_model, arm.query_model):
        raise BenchmarkError(
            f"models do not share an embedding space: {arm.document_model} and {arm.query_model}"
        )
    stats_before = client.stats
    document_embeddings = client.embed(
        [memory.raw for memory in memories],
        model=arm.document_model,
        input_type="document",
        batch_size=batch_size,
    )
    query_embeddings = client.embed(
        [query.text for query in queries],
        model=arm.query_model,
        input_type="query",
        batch_size=batch_size,
    )
    dimensions = {len(vector) for vector in [*document_embeddings, *query_embeddings]}
    if len(dimensions) != 1:
        raise BenchmarkError(f"warmup returned incompatible dimensions in arm {arm.name}")
    stats_after = client.stats
    return {
        "protocol": "populate_all_document_and_query_embeddings_before_measurement",
        "document_inputs": len(memories),
        "query_inputs": len(queries),
        "embedding_dimension": dimensions.pop(),
        "api_and_cache": _stats_delta(stats_before, stats_after),
        "api_network_latency_ms": _api_network_latency_delta(stats_before, stats_after),
    }


def run_arm(
    *,
    arm: BenchmarkArm,
    memories: list[Memory],
    queries: list[Query],
    client: VoyageClient,
    batch_size: int,
    candidate_k: int,
) -> dict[str, object]:
    if not _models_share_space(arm.document_model, arm.query_model):
        raise BenchmarkError(
            f"models do not share an embedding space: {arm.document_model} and {arm.query_model}"
        )
    stats_before = client.stats
    document_started = time.monotonic()
    document_embeddings = client.embed(
        [memory.raw for memory in memories],
        model=arm.document_model,
        input_type="document",
        batch_size=batch_size,
    )
    document_embedding_ms = (time.monotonic() - document_started) * 1000
    dimension = len(document_embeddings[0])

    ranks: list[int | None] = []
    latencies: list[float] = []
    rows: list[dict[str, object]] = []
    for query in queries:
        started = time.monotonic()
        query_embedding = client.embed(
            [query.text],
            model=arm.query_model,
            input_type="query",
            batch_size=batch_size,
        )[0]
        if len(query_embedding) != dimension:
            raise BenchmarkError(
                f"shared-space models returned different dimensions in arm {arm.name}"
            )
        candidate_indexes = _embedding_ranking(memories, document_embeddings, query_embedding)
        if arm.rerank_model is not None:
            candidate_indexes = candidate_indexes[:candidate_k]
            reranked_offsets = client.rerank(
                query.text,
                [memories[index].raw for index in candidate_indexes],
                model=arm.rerank_model,
            )
            candidate_indexes = [candidate_indexes[index] for index in reranked_offsets]
        elapsed_ms = (time.monotonic() - started) * 1000
        ranked_ids = [memories[index].ident for index in candidate_indexes]
        rank = ranked_ids.index(query.expected) + 1 if query.expected in ranked_ids else None
        ranks.append(rank)
        latencies.append(elapsed_ms)
        rows.append(
            {
                "expected": query.expected,
                "rank": rank,
                "latency_ms": round(elapsed_ms, 3),
            }
        )

    stats_after = client.stats
    api_and_cache = _stats_delta(stats_before, stats_after)
    if api_and_cache["embedding_api_calls"] != 0:
        raise BenchmarkError(f"arm {arm.name} measurement started with a cold embedding cache")
    end_to_end_latency = _latency_metrics(latencies)
    return {
        "name": arm.name,
        "document_model": arm.document_model,
        "query_model": arm.query_model,
        "rerank_model": arm.rerank_model,
        "actual_backend": "Voyage native REST API",
        "actual_profile": {
            "document_model": arm.document_model,
            "query_model": arm.query_model,
            "rerank_model": arm.rerank_model,
            "embedding_dimension": dimension,
        },
        "shared_embedding_space": True,
        "embedding_dimension": dimension,
        "rerank_candidates": candidate_k if arm.rerank_model else None,
        "metrics": _quality_metrics(ranks),
        "measured_queries": len(queries),
        "end_to_end_latency_ms": end_to_end_latency,
        "query_latency_ms": end_to_end_latency,
        "warm_document_cache_read_ms": round(document_embedding_ms, 3),
        "api_network_latency_ms": _api_network_latency_delta(stats_before, stats_after),
        "api_and_cache": api_and_cache,
        "rows": rows,
    }


def run_voyage_benchmark(
    *,
    api_key: str,
    cache_path: Path,
    timeout: float,
    attempts: int,
    batch_size: int,
    candidate_k: int,
) -> dict[str, object]:
    memories = _memories()
    queries = _queries(memories)
    _validate_fixture(memories, queries)
    arms = []
    cache_entries_by_arm: dict[str, int] = {}
    for index, arm in enumerate(MATRIX, start=1):
        print(f"Voyage matrix {index}/{len(MATRIX)}: {arm.name}", file=sys.stderr, flush=True)
        cache = EmbeddingCache(_arm_cache_path(cache_path, arm))
        client = VoyageClient(
            api_key=api_key,
            cache=cache,
            timeout=timeout,
            attempts=attempts,
        )
        warmup = _warm_arm_embeddings(
            arm=arm,
            memories=memories,
            queries=queries,
            client=client,
            batch_size=batch_size,
        )
        arm_result = run_arm(
            arm=arm,
            memories=memories,
            queries=queries,
            client=client,
            batch_size=batch_size,
            candidate_k=candidate_k,
        )
        arm_result["cache_namespace"] = arm.name
        arm_result["warmup"] = warmup
        arms.append(arm_result)
        cache_entries_by_arm[arm.name] = cache.entries
    return {
        "status": "completed",
        "mode": "live_voyage",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": "voyage_chinese_retrieval_matrix_v1",
        "provider": {
            "protocol": "Voyage native REST API",
            "embeddings_endpoint": EMBEDDINGS_URL,
            "rerank_endpoint": RERANK_URL,
            "credential_source": "environment variable VOYAGE_API_KEY",
            "credential_value_recorded": False,
        },
        "dataset": {
            "fixture": "scripts/benchmark_chinese_retrieval.py",
            "memories": len(memories),
            "queries": len(queries),
            "distinct_document_texts": len({memory.raw for memory in memories}),
        },
        "method": {
            "documents_input_type": "document",
            "queries_input_type": "query",
            "similarity": "cosine with fixture id ascending as the stable tie-break",
            "cache_protocol": "isolated_per_arm_then_fully_warm",
            "end_to_end_latency": "20 fixture queries after all document and query embeddings for that arm are cached; includes cache lookup, local ranking, and configured reranking",
            "api_network_latency": "actual Voyage REST attempts only; cache hits are never represented as API latency",
            "percentiles": "linear interpolation",
        },
        "embedding_cache": {
            "persistent": True,
            "isolated_per_arm": True,
            "stores_raw_text": False,
            "entries_by_namespace": cache_entries_by_arm,
        },
        "arms": arms,
    }


def _validate_remote_url(value: str) -> str:
    if (
        not value
        or "?" in value
        or "#" in value
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise BenchmarkError("FUNES_REMOTE_URL must be an http(s) base URL without credentials")
    try:
        parsed = urllib.parse.urlsplit(value)
        hostname = parsed.hostname
        parsed.port
    except ValueError:
        raise BenchmarkError(
            "FUNES_REMOTE_URL must be an http(s) base URL without credentials"
        ) from None
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise BenchmarkError("FUNES_REMOTE_URL must be an http(s) base URL without credentials")
    return value.rstrip("/") + "/recall"


def _http_transport_config(endpoint: str) -> _HttpTransportConfig:
    parsed = urllib.parse.urlsplit(endpoint)
    target_host = parsed.hostname
    if target_host is None:  # _validate_remote_url has already checked this.
        raise AssertionError("validated HTTP endpoint has no hostname")
    target_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    proxy_value = urllib.request.getproxies().get(parsed.scheme)
    proxy_used = bool(proxy_value) and not urllib.request.proxy_bypass(parsed.netloc)
    request_target = parsed.path or "/"

    if not proxy_used:
        return _HttpTransportConfig(
            target_host=target_host,
            target_port=target_port,
            connect_host=target_host,
            connect_port=target_port,
            request_target=request_target,
            target_scheme=parsed.scheme,
            transport=f"direct_{parsed.scheme}_keep_alive",
            proxy_used=False,
            tunnel=False,
        )

    if not isinstance(proxy_value, str):
        raise BenchmarkError("HTTP proxy configuration is unsupported")
    try:
        proxy = urllib.parse.urlsplit(proxy_value)
        proxy_host = proxy.hostname
        proxy_port = proxy.port
    except ValueError:
        raise BenchmarkError("HTTP proxy configuration is unsupported") from None
    if (
        proxy.scheme != "http"
        or proxy_host is None
        or proxy.username is not None
        or proxy.password is not None
        or proxy.query
        or proxy.fragment
        or proxy.path not in ("", "/")
    ):
        raise BenchmarkError(
            "HTTP proxy must be an unauthenticated http:// host for this benchmark"
        )
    proxy_port = proxy_port or 80
    if parsed.scheme == "https":
        transport = "https_over_http_connect_keep_alive"
        tunnel = True
        request_target = parsed.path or "/"
    else:
        transport = "http_via_http_proxy_keep_alive"
        tunnel = False
        request_target = endpoint
    return _HttpTransportConfig(
        target_host=target_host,
        target_port=target_port,
        connect_host=proxy_host,
        connect_port=proxy_port,
        request_target=request_target,
        target_scheme=parsed.scheme,
        transport=transport,
        proxy_used=True,
        tunnel=tunnel,
    )


def _recorded_embedding_profile(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    recorded: dict[str, object] = {}
    for field in EMBEDDING_PROFILE_FIELDS:
        item = value.get(field)
        if field == "dimensions":
            if isinstance(item, int) and not isinstance(item, bool):
                recorded[field] = item
        elif field == "schema_version":
            if isinstance(item, (int, str)) and not isinstance(item, bool):
                recorded[field] = item
        elif isinstance(item, str) and item:
            recorded[field] = item
    return recorded


def _profile_matches(
    actual: dict[str, object] | None,
    expected: dict[str, object],
) -> bool:
    if actual is None:
        return False
    for field in ("provider", "model", "dimensions"):
        if actual.get(field) != expected.get(field):
            return False
    if str(actual.get("schema_version", "")) != str(expected.get("schema_version", "")):
        return False
    fingerprint = actual.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        return False
    expected_fingerprint = expected.get("fingerprint")
    return expected_fingerprint in (None, "") or fingerprint == expected_fingerprint


def _degraded_reason(value: object) -> str | None:
    if not value:
        return None
    if isinstance(value, str) and value in KNOWN_RETRIEVAL_DEGRADATIONS:
        return value
    return "other_nonempty"


def _funes_recall(
    client: _PersistentFunesHttpClient,
    query: str,
    *,
    expected_backend: str,
    expected_profile: dict[str, object],
) -> HttpRecallObservation:
    status, payload = client.post_recall(query)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise BenchmarkError("Funes recall response has no results list")
    backend_value = payload.get("retrieval_backend")
    retrieval_backend = backend_value if isinstance(backend_value, str) else None
    embedding_profile = _recorded_embedding_profile(payload.get("embedding_profile"))
    degraded_reason = _degraded_reason(payload.get("retrieval_degraded"))
    failures = []
    if status < 200 or status >= 300:
        failures.append("http_status")
    if degraded_reason is not None:
        failures.append("degraded")
    if retrieval_backend != expected_backend:
        failures.append("backend_mismatch")
    if not _profile_matches(embedding_profile, expected_profile):
        failures.append("profile_mismatch")
    return HttpRecallObservation(
        status=status,
        retrieval_backend=retrieval_backend,
        embedding_profile=embedding_profile,
        degraded_reason=degraded_reason,
        validation_failures=tuple(failures),
    )


def _http_observation_counts(
    observations: list[HttpRecallObservation],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    backend_counts: dict[str | None, int] = {}
    profile_counts: dict[str, tuple[dict[str, object] | None, int]] = {}
    for observation in observations:
        backend_counts[observation.retrieval_backend] = (
            backend_counts.get(observation.retrieval_backend, 0) + 1
        )
        profile_key = json.dumps(
            observation.embedding_profile,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        profile, count = profile_counts.get(
            profile_key,
            (observation.embedding_profile, 0),
        )
        profile_counts[profile_key] = (profile, count + 1)
    backends = [
        {"retrieval_backend": backend, "observations": count}
        for backend, count in sorted(
            backend_counts.items(),
            key=lambda item: (item[0] is None, item[0] or ""),
        )
    ]
    profiles = [
        {"embedding_profile": profile, "observations": count}
        for _key, (profile, count) in sorted(profile_counts.items())
    ]
    return backends, profiles


def run_http_latency(
    *,
    remote_url: str,
    token: str,
    requests: int = 50,
    timeout: float,
    expected_backend: str = DEFAULT_HTTP_RETRIEVAL_BACKEND,
    expected_profile: dict[str, object] | None = None,
) -> dict[str, object]:
    if not token:
        raise BenchmarkError("FUNES_API_TOKEN is empty")
    if requests < 1:
        raise BenchmarkError("latency requests must be positive")
    expected_profile = dict(
        DEFAULT_HTTP_EMBEDDING_PROFILE if expected_profile is None else expected_profile
    )
    schema_version = expected_profile.get("schema_version")
    fingerprint = expected_profile.get("fingerprint")
    if (
        not expected_backend
        or expected_profile.get("provider") != "voyage"
        or not isinstance(expected_profile.get("model"), str)
        or not expected_profile.get("model")
        or not isinstance(expected_profile.get("dimensions"), int)
        or isinstance(expected_profile.get("dimensions"), bool)
        or int(expected_profile["dimensions"]) < 1
        or not (
            isinstance(schema_version, str)
            and bool(schema_version)
            or isinstance(schema_version, int)
            and not isinstance(schema_version, bool)
            and schema_version > 0
        )
        or fingerprint is not None
        and (not isinstance(fingerprint, str) or not fingerprint)
    ):
        raise BenchmarkError("expected HTTP Voyage backend/profile is invalid")
    endpoint = _validate_remote_url(remote_url)
    client = _PersistentFunesHttpClient(endpoint=endpoint, token=token, timeout=timeout)
    samples: dict[str, list[float]] = {category: [] for category, _ in HTTP_LATENCY_CASES}
    statuses: dict[str, int] = {}
    observations: list[HttpRecallObservation] = []
    try:
        warmup = _funes_recall(
            client,
            HTTP_LATENCY_CASES[0][1],
            expected_backend=expected_backend,
            expected_profile=expected_profile,
        )
        for index in range(requests):
            category, query = HTTP_LATENCY_CASES[index % len(HTTP_LATENCY_CASES)]
            started = time.monotonic()
            observation = _funes_recall(
                client,
                query,
                expected_backend=expected_backend,
                expected_profile=expected_profile,
            )
            elapsed_ms = (time.monotonic() - started) * 1000
            samples[category].append(elapsed_ms)
            observations.append(observation)
            statuses[str(observation.status)] = statuses.get(str(observation.status), 0) + 1
    finally:
        client.close()

    all_samples = [sample for category_samples in samples.values() for sample in category_samples]
    failure_counts: dict[str, int] = {}
    degraded_counts: dict[str, int] = {}
    for observation in observations:
        for failure in observation.validation_failures:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
        if observation.degraded_reason is not None:
            degraded_counts[observation.degraded_reason] = (
                degraded_counts.get(observation.degraded_reason, 0) + 1
            )
    verified_requests = sum(observation.verified_voyage for observation in observations)
    all_observations = [warmup, *observations]
    observed_backends, observed_profiles = _http_observation_counts(all_observations)
    validation_failed = not warmup.verified_voyage or verified_requests != requests
    end_to_end_latency = _latency_metrics(all_samples)
    return {
        "status": "failed_validation" if validation_failed else "completed",
        "mode": "warm_funes_http_latency",
        "requests": requests,
        "warmup_requests": 1,
        "end_to_end_latency_ms": end_to_end_latency,
        "latency_ms": end_to_end_latency,
        "latency_scope": (
            "client-observed POST /recall round trip after one unmeasured warmup; "
            "uses the warmup connection when available and includes reconnect/retry time"
        ),
        "transport": client.transport,
        "proxy_used": client.proxy_used,
        "connection_reuse": {
            "mode": "single_persistent_connection_with_bounded_reconnect",
            "scope": "warmup_and_measured_requests",
            "logical_requests": requests + 1,
            "request_attempts": client.request_attempts,
            "transport_connection_attempts": client.connection_attempts,
            "transport_connections_opened": client.connections_opened,
            "transport_reconnections": max(0, client.connections_opened - 1),
            "disconnect_retries": client.disconnect_retries,
            "max_disconnect_retries_per_request": 1,
            "single_transport_connection_for_warmup_and_measurements": (
                client.connections_opened == 1
            ),
            "response_bodies_fully_read": True,
            "redirects_followed": False,
        },
        "by_category": {
            category: {
                "requests": len(category_samples),
                **_latency_metrics(category_samples),
            }
            for category, category_samples in samples.items()
            if category_samples
        },
        "http_status_counts": statuses,
        "voyage_validation": {
            "expected_retrieval_backend": expected_backend,
            "expected_embedding_profile": expected_profile,
            "warmup_verified_voyage": warmup.verified_voyage,
            "warmup_validation_failures": list(warmup.validation_failures),
            "warmup_degraded_reason": warmup.degraded_reason,
            "verified_voyage_requests": verified_requests,
            "non_voyage_requests": requests - verified_requests,
            "failure_counts": failure_counts,
            "degraded_counts": degraded_counts,
            "observed_retrieval_backends": observed_backends,
            "observed_embedding_profiles": observed_profiles,
        },
        "response_content_recorded": False,
        "credential_value_recorded": False,
    }


def _validate_fixture(memories: list[Memory], queries: list[Query]) -> None:
    if len(memories) != 60 or len(queries) != 20:
        raise BenchmarkError("fixture must contain exactly 60 memories and 20 queries")
    identifiers = {memory.ident for memory in memories}
    if len(identifiers) != len(memories):
        raise BenchmarkError("fixture memory identifiers must be unique")
    if any(query.expected not in identifiers for query in queries):
        raise BenchmarkError("fixture query references an unknown memory")


def _self_check() -> dict[str, object]:
    memories = _memories()
    queries = _queries(memories)
    _validate_fixture(memories, queries)
    if any(not _models_share_space(arm.document_model, arm.query_model) for arm in MATRIX):
        raise BenchmarkError("matrix contains incompatible embedding models")
    if _quality_metrics([1, 3, 5, None]) != {
        "recall@1": 0.25,
        "recall@3": 0.5,
        "recall@5": 0.75,
        "mrr": 0.3833,
    }:
        raise BenchmarkError("quality metric invariant failed")
    if {category for category, _query in HTTP_LATENCY_CASES} != {
        "chinese",
        "english",
        "mixed",
        "semantic_paraphrase",
        "exact_identifier",
        "code_error",
    }:
        raise BenchmarkError("HTTP latency coverage invariant failed")
    return {
        "status": "ok",
        "mode": "offline_self_check",
        "network_requests": 0,
        "paid_api_calls": 0,
        "memories": len(memories),
        "queries": len(queries),
        "distinct_document_texts": len({memory.raw for memory in memories}),
        "matrix": [arm.name for arm in MATRIX],
        "http_latency_categories": [category for category, _query in HTTP_LATENCY_CASES],
        "default_http_latency_requests": 50,
    }


def _environment_value(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise BenchmarkError(f"required environment variable is missing: {name}")
    return value


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Default: offline self-check (no network and no paid API).\n"
            "Live Voyage reads only VOYAGE_API_KEY.\n"
            "HTTP latency reads only FUNES_REMOTE_URL and FUNES_API_TOKEN."
        ),
    )
    modes = out.add_mutually_exclusive_group()
    modes.add_argument(
        "--self-check",
        dest="mode",
        action="store_const",
        const="self-check",
        help="validate fixture/matrix/metrics offline (default)",
    )
    modes.add_argument(
        "--live-voyage",
        dest="mode",
        action="store_const",
        const="live-voyage",
        help="run the paid five-arm Voyage embedding/rerank matrix",
    )
    modes.add_argument(
        "--http-latency",
        dest="mode",
        action="store_const",
        const="http-latency",
        help="warm Funes once, then measure HTTP recall latency (50 requests by default)",
    )
    out.set_defaults(mode="self-check")
    out.add_argument("--output", type=Path, help="optionally write the sanitized JSON result here")
    out.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help=f"embedding cache (default: {DEFAULT_CACHE})")
    out.add_argument("--batch-size", type=int, default=128, help="maximum texts per Voyage embedding request (default: 128)")
    out.add_argument("--candidate-k", type=int, default=20, help="embedding candidates sent to rerank-3-lite (default: 20)")
    out.add_argument("--request-timeout", type=float, default=180.0, help="per-request timeout in seconds (default: 180)")
    out.add_argument("--attempts", type=int, default=3, help="maximum Voyage attempts per request (default: 3)")
    out.add_argument("--latency-requests", type=int, default=50, help="measured warm Funes requests (default: 50)")
    out.add_argument(
        "--expected-http-backend",
        default=DEFAULT_HTTP_RETRIEVAL_BACKEND,
        help=f"required /recall retrieval_backend (default: {DEFAULT_HTTP_RETRIEVAL_BACKEND})",
    )
    out.add_argument(
        "--expected-http-model",
        default=DEFAULT_HTTP_EMBEDDING_PROFILE["model"],
        help="required Voyage embedding profile model (default: voyage-4-lite)",
    )
    out.add_argument(
        "--expected-http-dimensions",
        type=int,
        default=DEFAULT_HTTP_EMBEDDING_PROFILE["dimensions"],
        help="required Voyage embedding profile dimensions (default: 1024)",
    )
    out.add_argument(
        "--expected-http-schema-version",
        type=int,
        default=DEFAULT_HTTP_EMBEDDING_PROFILE["schema_version"],
        help="required Voyage embedding profile schema version (default: 2)",
    )
    out.add_argument(
        "--expected-http-profile-fingerprint",
        help="optionally require one exact embedding profile fingerprint",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if (
        args.batch_size < 1
        or args.candidate_k < 5
        or args.candidate_k > 60
        or args.request_timeout <= 0
        or args.attempts < 1
        or args.latency_requests < 1
        or args.expected_http_dimensions < 1
        or args.expected_http_schema_version < 1
    ):
        raise SystemExit(
            "batch-size/attempts/latency-requests must be positive, request-timeout must be positive, and candidate-k must be 5..60"
        )
    try:
        if args.mode == "self-check":
            result = _self_check()
        elif args.mode == "live-voyage":
            result = run_voyage_benchmark(
                api_key=_environment_value("VOYAGE_API_KEY"),
                cache_path=args.cache,
                timeout=args.request_timeout,
                attempts=args.attempts,
                batch_size=args.batch_size,
                candidate_k=args.candidate_k,
            )
        else:
            result = run_http_latency(
                remote_url=_environment_value("FUNES_REMOTE_URL"),
                token=_environment_value("FUNES_API_TOKEN"),
                requests=args.latency_requests,
                timeout=args.request_timeout,
                expected_backend=args.expected_http_backend,
                expected_profile={
                    "provider": "voyage",
                    "model": args.expected_http_model,
                    "dimensions": args.expected_http_dimensions,
                    "schema_version": args.expected_http_schema_version,
                    "fingerprint": args.expected_http_profile_fingerprint,
                },
            )
    except BenchmarkError as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1
    if args.output is not None:
        _atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("status") == "failed_validation" else 0


if __name__ == "__main__":
    raise SystemExit(main())
