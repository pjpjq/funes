#!/usr/bin/env python3
"""Small authenticated HTTP bridge for the Funes CLI.

The durable source of truth is the configured HF Hub dataset (FUNES_MEMORY).  The
container's /data/.funes directory is only a warm cache and may be recreated.
"""
import hashlib
import atexit
import base64
import gzip
import io
import json
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from service.server import App as SourceApp
from service.server import expanded_candidate_limit
from service.server import ingest_documents as persist_source_ingest
from service.server import NATIVE_SESSION_TYPES
from service.server import prepare_ingest_documents as prepare_source_ingest_documents
from service.server import queue_reindex as queue_source_reindex
from service.server import stable_rrf
from service.server import utc_now
from service.server import validate_source_identity_batch


FUNES_BIN = os.getenv("FUNES_BIN", "/usr/local/bin/funes")
REMOTE = os.getenv("FUNES_MEMORY", "")
INDEX_REMOTE = os.getenv("FUNES_INDEX_MEMORY", "")
TOKEN = os.getenv("FUNES_API_TOKEN", "")
HOME = Path(os.getenv("FUNES_HOME", "/data/.funes"))
PORT = int(os.getenv("PORT", "7860"))
TRANSLATION_THRESHOLD = float(os.getenv("TRANSLATE_CHINESE_THRESHOLD", "0.15"))
INGEST_INDEX_TIMEOUT = int(os.getenv("FUNES_INGEST_INDEX_TIMEOUT", "900"))
INGEST_PUSH_TIMEOUT = int(os.getenv("FUNES_INGEST_PUSH_TIMEOUT", "1800"))
CANONICAL_INDEX_BATCH = max(1, int(os.getenv("FUNES_CANONICAL_INDEX_BATCH", "32")))
CANONICAL_INDEX_INTERVAL = max(0.01, float(os.getenv("FUNES_CANONICAL_INDEX_INTERVAL", "30")))
CANONICAL_INDEX_TIMEOUT = max(1, int(os.getenv("FUNES_CANONICAL_INDEX_TIMEOUT", "900")))
CANONICAL_OPTIMIZE_TIMEOUT = max(1, int(os.getenv("FUNES_CANONICAL_OPTIMIZE_TIMEOUT", "900")))
MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_TIMEOUT = float(os.getenv("FUNES_MCP_TIMEOUT", "180"))
MCP_HANDSHAKE_TIMEOUT = float(os.getenv("FUNES_MCP_HANDSHAKE_TIMEOUT", "10"))
try:
    RECALL_LOCK_TIMEOUT = min(55.0, max(0.1, float(os.getenv("FUNES_RECALL_LOCK_TIMEOUT", "2"))))
except ValueError:
    RECALL_LOCK_TIMEOUT = 2.0
try:
    HTTP_MAX_CANDIDATES = max(1, int(os.getenv("FUNES_HTTP_MAX_CANDIDATES", "12")))
except ValueError:
    HTTP_MAX_CANDIDATES = 12
try:
    HTTP_NATIVE_TIMEOUT = min(50.0, max(0.1, float(os.getenv("FUNES_HTTP_NATIVE_TIMEOUT", "12"))))
except ValueError:
    HTTP_NATIVE_TIMEOUT = 12.0
try:
    CJK_NATIVE_TIMEOUT = min(
        HTTP_NATIVE_TIMEOUT,
        max(0.1, float(os.getenv("FUNES_CJK_NATIVE_TIMEOUT", "5"))),
    )
except ValueError:
    CJK_NATIVE_TIMEOUT = min(HTTP_NATIVE_TIMEOUT, 5.0)
try:
    # Leave enough of the Space ingress window for a raw BM25 fallback.  This
    # is a hard cap: an operator may lower it, but cannot accidentally restore
    # an unbounded provider wait by raising an environment value.
    VOYAGE_NATIVE_TIMEOUT = min(
        3.0,
        HTTP_NATIVE_TIMEOUT,
        max(0.1, float(os.getenv("FUNES_VOYAGE_NATIVE_TIMEOUT", "3"))),
    )
except ValueError:
    VOYAGE_NATIVE_TIMEOUT = min(HTTP_NATIVE_TIMEOUT, 3.0)
try:
    VOYAGE_HTTP_TIMEOUT = min(
        4.5,
        max(0.2, float(os.getenv("FUNES_VOYAGE_HTTP_TIMEOUT", "4.5"))),
    )
except ValueError:
    VOYAGE_HTTP_TIMEOUT = 4.5
PROMPT_VERSION = "funes-retrieval-v1"
LANGUAGE_MODE = os.getenv("FUNES_RETRIEVAL_LANGUAGE_MODE", "raw").lower()
INDEX_LOCK = threading.Lock()
# Writes use the native memory lock inside the `funes` process.  Keep their
# Python-level serialization separate from reads so a background ingest/push
# cannot make an HTTP recall wait for the full upload duration.
WRITE_LOCK = threading.Lock()
SOURCE_APP = None
SOURCE_APP_LOCK = threading.Lock()
INGEST_OPERATION_LOCK = threading.Lock()
INGEST_OPERATIONS: dict[str, dict[str, object]] = {}
INGEST_ACTIVE_OPERATION: str | None = None
INGEST_RESTART_SCHEDULED = False


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _embedding_profile(
    provider: str,
    model: str,
    dimensions: int,
    schema_version: int,
) -> dict[str, object]:
    """Build the exact native embedding contract and its stable fingerprint."""
    contract = (
        f"provider={provider}\n"
        f"model={model}\n"
        f"dimensions={dimensions}\n"
        f"schema_version={schema_version}\n"
        "document_input=document\n"
        "query_input=query\n"
        "normalization=l2\n"
        "metric=l2"
    )
    return {
        "provider": provider,
        "model": model,
        "dimensions": dimensions,
        "schema_version": schema_version,
        "fingerprint": hashlib.sha256(contract.encode("utf-8")).hexdigest(),
    }


def embedding_profile() -> dict[str, object]:
    """Return the active query embedding contract."""
    provider = os.getenv("FUNES_EMBEDDING_PROVIDER", "voyage").strip().lower() or "voyage"
    model = os.getenv("FUNES_EMBEDDING_MODEL", "voyage-4-lite").strip() or "voyage-4-lite"
    dimensions = _positive_int_env("FUNES_EMBEDDING_DIMENSIONS", 1024)
    schema_version = _positive_int_env("FUNES_EMBEDDING_SCHEMA_VERSION", 2)
    return _embedding_profile(provider, model, dimensions, schema_version)


def index_memory() -> str:
    """Return the blue/green build target, defaulting to the active memory."""
    return INDEX_REMOTE.strip() or REMOTE


def index_embedding_profile() -> dict[str, object]:
    """Return the build profile without changing the active query profile."""
    active = embedding_profile()
    provider = (
        os.getenv("FUNES_INDEX_EMBEDDING_PROVIDER", "").strip().lower()
        or str(active["provider"])
    )
    provider_changed = provider != active["provider"]
    defaults = {
        "voyage": ("voyage-4-lite", 1024, 2),
        "local": ("BAAI/bge-small-en-v1.5", 384, 1),
    }
    default_model, default_dimensions, default_schema = defaults.get(
        provider,
        (str(active["model"]), int(active["dimensions"]), int(active["schema_version"])),
    )
    model = (
        os.getenv("FUNES_INDEX_EMBEDDING_MODEL", "").strip()
        or (default_model if provider_changed else str(active["model"]))
    )
    dimensions = _positive_int_env(
        "FUNES_INDEX_EMBEDDING_DIMENSIONS",
        default_dimensions if provider_changed else int(active["dimensions"]),
    )
    schema_version = _positive_int_env(
        "FUNES_INDEX_EMBEDDING_SCHEMA_VERSION",
        default_schema if provider_changed else int(active["schema_version"]),
    )
    return _embedding_profile(provider, model, dimensions, schema_version)


def native_environment(
    home: Path = HOME,
    profile: dict[str, object] | None = None,
) -> dict[str, str]:
    """Build a secret-preserving native environment for one embedding profile."""
    env = os.environ.copy()
    profile = profile or embedding_profile()
    env["FUNES_HOME"] = str(home)
    env["FUNES_EMBEDDING_PROVIDER"] = str(profile["provider"])
    env["FUNES_EMBEDDING_MODEL"] = str(profile["model"])
    env["FUNES_EMBEDDING_DIMENSIONS"] = str(profile["dimensions"])
    env["FUNES_EMBEDDING_SCHEMA_VERSION"] = str(profile["schema_version"])
    env["FUNES_RERANK_PROVIDER"] = os.getenv("FUNES_RERANK_PROVIDER", "none") or "none"
    env["FUNES_NATIVE_FALLBACK"] = os.getenv("FUNES_NATIVE_FALLBACK", "false") or "false"
    env["FUNES_RETRIEVAL_LANGUAGE_MODE"] = (
        os.getenv("FUNES_RETRIEVAL_LANGUAGE_MODE", "raw") or "raw"
    )
    return env


def source_app():
    """Return the encrypted raw/source store, or None when not configured."""
    global SOURCE_APP
    if SOURCE_APP is not None:
        return SOURCE_APP
    if not os.getenv("FUNES_STORAGE_REPO"):
        return None
    with SOURCE_APP_LOCK:
        if SOURCE_APP is None:
            os.environ.setdefault("FUNES_DATA_DIR", str(HOME / "source-store"))
            os.environ.setdefault("FUNES_LAZY_RESTORE", "true")
            os.environ.setdefault("FUNES_REQUIRE_DURABLE_ACK", "true")
            SOURCE_APP = SourceApp()
            start_canonical_reconciler(SOURCE_APP)
    return SOURCE_APP


def source_state() -> dict[str, object]:
    app = source_app()
    if app is None:
        return {"configured": False, "ready": False, "documents": 0}
    if app.syncer.restoring:
        return {"configured": True, "ready": False, "restoring": True, "documents": app.store.count()}
    if app.syncer.restore_failed:
        return {"configured": True, "ready": False, "error": "restore_failed", "documents": app.store.count()}
    active_profile = embedding_profile()
    build_profile = index_embedding_profile()
    build_memory = index_memory()
    checkpoint = app.store.native_index_checkpoint(build_profile, build_memory)
    optimize = app.store.native_optimize_checkpoint()
    checkpoint["optimize"] = optimize
    checkpoint["memory"] = build_memory
    checkpoint["cutover_ready"] = bool(
        checkpoint.get("complete")
        and checkpoint.get("indexed")
        and optimize.get("status") == "optimized"
        and optimize.get("fingerprint") == build_profile["fingerprint"]
        and optimize.get("memory") == build_memory
        and optimize.get("index_fingerprint") == checkpoint.get("index_fingerprint")
    )
    return {
        "configured": True,
        "ready": True,
        "documents": app.store.count(),
        "restored": app.restore_result,
        "sync": app.store.sync_status(),
        "active_index": {"memory": REMOTE, "profile": active_profile},
        "build_index": {"memory": build_memory, "profile": build_profile},
        "canonical_index": checkpoint,
    }


def close_source_app() -> None:
    global SOURCE_APP
    with SOURCE_APP_LOCK:
        app = SOURCE_APP
        SOURCE_APP = None
    if app is not None:
        try:
            stop_canonical_reconciler(app)
            app.close()
        except Exception:
            pass


def prepare_source_documents(app, docs: list[dict]) -> list[dict]:
    """Build the durable raw-first representation without provider I/O."""
    return prepare_source_ingest_documents(app, docs)


def ingest_source_documents(docs: list[dict]) -> tuple[int, dict, list[dict]] | None:
    app = source_app()
    if app is None:
        return None
    if app.syncer.restoring or app.syncer.restore_failed:
        error = "restore_in_progress" if app.syncer.restoring else "restore_failed"
        return 503, {"ok": False, "durable": False, "error": error}, []
    # Serialize raw Hub durability with canonical commits. A failed raw delta is
    # marked waiting_durability before this lock is released, so native can
    # never publish a shadow whose sidecar source was not durable.
    with WRITE_LOCK:
        result = persist_source_ingest(app, docs)
    identities = [str(item.get("source_identity", "")) for item in result["items"]]
    canonical = app.store.get_many(identities)
    result["ok"] = bool(result["durable"])
    return (200 if result["durable"] else 503), result, canonical


def _ingest_operation_id(docs: list[dict]) -> str:
    """Return a stable identifier without retaining another full JSON copy."""
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for chunk in encoder.iterencode(docs):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _ingest_operation_limits() -> tuple[float, int, int]:
    try:
        ttl = max(1.0, float(os.getenv("FUNES_INGEST_OPERATION_TTL", "3600")))
    except ValueError:
        ttl = 3600.0
    try:
        completed = max(1, int(os.getenv("FUNES_INGEST_OPERATION_MAX_COMPLETED", "64")))
    except ValueError:
        completed = 64
    try:
        retry_after = max(1, int(os.getenv("FUNES_INGEST_RETRY_AFTER", "2")))
    except ValueError:
        retry_after = 2
    return ttl, completed, retry_after


def _ingest_operation_timeout() -> float:
    try:
        return max(0.1, float(os.getenv("FUNES_INGEST_OPERATION_TIMEOUT", "1800")))
    except ValueError:
        return 1800.0


def _cleanup_ingest_operations_locked(now: float, preserve: str | None = None) -> None:
    """Bound completed-operation metadata; active documents are never evicted."""
    ttl, max_completed, _retry_after = _ingest_operation_limits()
    completed = [
        (operation_id, operation)
        for operation_id, operation in INGEST_OPERATIONS.items()
        if operation.get("state") != "running"
    ]
    for operation_id, operation in completed:
        completed_at = float(operation.get("completed_at", now))
        if operation_id != preserve and now - completed_at >= ttl:
            INGEST_OPERATIONS.pop(operation_id, None)
    completed = sorted(
        (
            (operation_id, operation)
            for operation_id, operation in INGEST_OPERATIONS.items()
            if operation.get("state") != "running"
        ),
        key=lambda item: (
            float(item[1].get("completed_at", 0.0)),
            item[0] == preserve,
        ),
        reverse=True,
    )
    for operation_id, _operation in completed[max_completed:]:
        INGEST_OPERATIONS.pop(operation_id, None)


def _safe_ingest_error(value: object) -> str:
    error = str(value or "ingest_failed")
    return error if error in {
        "durability_pending",
        "ingest_failed",
        "ingest_operation_timeout",
        "ingest_unavailable",
        "restore_failed",
        "restore_in_progress",
    } else "ingest_failed"


def _public_ingest_result(result: dict) -> dict[str, object]:
    """Return durability/count metadata only, never raw source documents."""
    durable = result.get("durable") is True
    try:
        accepted = (
            int(result["accepted"])
            if "accepted" in result
            else int(result.get("created", 0))
            + int(result.get("updated", 0))
            + int(result.get("deduped", 0))
        )
    except (TypeError, ValueError):
        accepted = 0
    public: dict[str, object] = {
        "ok": durable,
        "durable": durable,
        "accepted": accepted if durable else 0,
    }
    for name in ("created", "updated", "deduped"):
        if name in result:
            try:
                public[name] = int(result[name])
            except (TypeError, ValueError):
                pass
    if not durable:
        public["error"] = _safe_ingest_error(result.get("error"))
    return public


def _ingest_operation_response(operation: dict[str, object]) -> tuple[int, dict[str, object]]:
    operation_id = str(operation["operation_id"])
    state = str(operation.get("state", "failed"))
    payload: dict[str, object] = {
        "ok": state == "succeeded",
        "operation_id": operation_id,
        "status": state,
        "status_url": f"/ingest/operations/{operation_id}",
        "durable": state == "succeeded",
    }
    if operation.get("timed_out") is True:
        payload.update(
            ok=False,
            status="timed_out",
            durable=False,
            error="ingest_operation_timeout",
        )
        return 503, payload
    if state == "running":
        _ttl, _completed, retry_after = _ingest_operation_limits()
        payload["retry_after"] = retry_after
        return 202, payload
    if state == "succeeded":
        payload["accepted"] = int(operation.get("accepted", 0))
        return 200, payload
    payload["error"] = _safe_ingest_error(operation.get("error"))
    return 503, payload


def schedule_ingest_process_restart() -> None:
    """Replace PID 1 after the timeout response has reached the caller."""
    def terminate() -> None:
        time.sleep(0.25)
        try:
            os.execv(sys.executable, [sys.executable, *sys.argv])
        except OSError:
            os._exit(75)

    thread = threading.Thread(
        target=terminate,
        name="funes-ingest-fail-stop",
        daemon=True,
    )
    try:
        thread.start()
    except RuntimeError:
        os._exit(75)


def _timed_out_active_operation_locked(
    now: float,
) -> tuple[dict[str, object] | None, bool]:
    """Mark a hung worker without freeing its slot; return whether to restart."""
    global INGEST_RESTART_SCHEDULED
    if INGEST_ACTIVE_OPERATION is None:
        return None, False
    operation = INGEST_OPERATIONS.get(INGEST_ACTIVE_OPERATION)
    if operation is None or operation.get("state") != "running":
        return None, False
    started_at = float(operation.get("started_at", now))
    if operation.get("timed_out") is not True and now - started_at < _ingest_operation_timeout():
        return None, False
    operation["timed_out"] = True
    operation["error"] = "ingest_operation_timeout"
    if INGEST_RESTART_SCHEDULED:
        return operation, False
    INGEST_RESTART_SCHEDULED = True
    return operation, True


def _restart_pending_response() -> tuple[int, dict[str, object]]:
    return 503, {
        "ok": False,
        "durable": False,
        "status": "restart_pending",
        "error": "ingest_operation_timeout",
    }


def _finish_ingest_operation(
    operation_id: str,
    *,
    durable: bool,
    accepted: int = 0,
    error: str = "ingest_failed",
) -> None:
    global INGEST_ACTIVE_OPERATION
    now = time.monotonic()
    with INGEST_OPERATION_LOCK:
        operation = INGEST_OPERATIONS.get(operation_id)
        if operation is None or operation.get("state") != "running":
            return
        operation.pop("documents", None)
        if operation.get("timed_out") is True:
            # A process restart is already committed.  Do not clear the active
            # slot or turn a post-timeout completion into a durable ACK.
            operation["worker_finished"] = True
            operation["completed_at"] = now
            return
        operation["state"] = "succeeded" if durable else "failed"
        operation["accepted"] = accepted if durable else 0
        operation["completed_at"] = now
        if not durable:
            operation["error"] = _safe_ingest_error(error)
        if INGEST_ACTIVE_OPERATION == operation_id:
            INGEST_ACTIVE_OPERATION = None
        _cleanup_ingest_operations_locked(now, preserve=operation_id)


def _run_ingest_operation(operation_id: str, docs: list[dict]) -> None:
    try:
        source_result = ingest_source_documents(docs)
        if source_result is None:
            _finish_ingest_operation(
                operation_id,
                durable=False,
                error="ingest_unavailable",
            )
            return
        _status, result, _canonical = source_result
        public = _public_ingest_result(result)
        if public["durable"] is True:
            _finish_ingest_operation(
                operation_id,
                durable=True,
                accepted=int(public["accepted"]),
            )
            return
        _finish_ingest_operation(
            operation_id,
            durable=False,
            error=str(public.get("error") or "ingest_failed"),
        )
    except Exception:
        # Status responses intentionally expose no provider error text, paths,
        # source contents, or other exception details.
        _finish_ingest_operation(operation_id, durable=False)


def start_ingest_operation(docs: list[dict]) -> tuple[int, dict[str, object]]:
    """Start, deduplicate, or reject a bounded asynchronous ingest."""
    global INGEST_ACTIVE_OPERATION
    operation_id = _ingest_operation_id(docs)
    now = time.monotonic()
    operation = None
    response = None
    schedule_restart = False
    with INGEST_OPERATION_LOCK:
        _cleanup_ingest_operations_locked(now)
        timed_out, schedule_restart = _timed_out_active_operation_locked(now)
        if timed_out is not None:
            response = _restart_pending_response()
        elif INGEST_RESTART_SCHEDULED:
            response = _restart_pending_response()
        else:
            existing = INGEST_OPERATIONS.get(operation_id)
            if INGEST_ACTIVE_OPERATION is not None:
                if INGEST_ACTIVE_OPERATION == operation_id and existing is not None:
                    response = _ingest_operation_response(existing)
                else:
                    _ttl, _completed, retry_after = _ingest_operation_limits()
                    response = 429, {
                        "ok": False,
                        "durable": False,
                        "error": "ingest_busy",
                        "retry_after": retry_after,
                    }
            elif existing is not None and existing.get("state") == "succeeded":
                response = _ingest_operation_response(existing)
            else:
                operation = {
                    "operation_id": operation_id,
                    "state": "running",
                    "documents": docs,
                    "accepted": 0,
                    "started_at": now,
                }
                INGEST_OPERATIONS[operation_id] = operation
                INGEST_ACTIVE_OPERATION = operation_id
    if schedule_restart:
        schedule_ingest_process_restart()
    if response is not None:
        return response
    assert operation is not None
    thread = threading.Thread(
        target=_run_ingest_operation,
        args=(operation_id, docs),
        name="funes-ingest-operation",
        daemon=True,
    )
    try:
        thread.start()
    except RuntimeError:
        _finish_ingest_operation(operation_id, durable=False)
    with INGEST_OPERATION_LOCK:
        return _ingest_operation_response(operation)


def get_ingest_operation(operation_id: str) -> tuple[int, dict[str, object]]:
    if not re.fullmatch(r"[0-9a-f]{64}", operation_id):
        return 404, {"error": "not found"}
    schedule_restart = False
    with INGEST_OPERATION_LOCK:
        now = time.monotonic()
        _cleanup_ingest_operations_locked(now)
        _timed_out, schedule_restart = _timed_out_active_operation_locked(now)
        operation = INGEST_OPERATIONS.get(operation_id)
        if operation is None:
            response = 404, {"error": "not found"}
        else:
            response = _ingest_operation_response(operation)
    if schedule_restart:
        schedule_ingest_process_restart()
    return response


def _normalized_harness_agent(value: object) -> str | None:
    normalized = str(value or "").strip().lower().replace("-", "_")
    normalized = "claude" if normalized == "claude_code" else normalized
    return normalized if normalized in {"codex", "pi", "claude", "hermes"} else None


def _partition_source_rankings(
    hit_rankings: tuple[list[dict], ...],
    filters: dict[str, object],
    harness: str | None,
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Keep raw public rows while preserving source-store rank order."""
    native_filterable = set(filters).issubset({"source_agent", "repo"})
    expected_agent = _normalized_harness_agent(harness) if harness else None
    fallback_enabled = not harness or expected_agent is not None
    source_rankings = []
    session_fallback_rankings = []
    for hits in hit_rankings:
        eligible_hits = []
        fallback_hits = []
        for item in hits:
            public_item = _public_source_item(item)
            is_session = str(item.get("source_type", "")).lower() in NATIVE_SESSION_TYPES
            if is_session:
                source_agent = _normalized_harness_agent(item.get("source_agent"))
                # The native index cannot authoritatively apply role/date
                # filters, so the sidecar must also honor harness when it
                # returns session rows for those requests.
                if harness and (
                    not fallback_enabled or source_agent != expected_agent
                ):
                    continue
                eligible_hits.append(public_item)
                if native_filterable:
                    fallback_hits.append(public_item)
                continue
            eligible_hits.append(public_item)
        if eligible_hits:
            # This single sequence preserves both the source store's cross-type
            # ranks and raw-query-before-rewrite ordering for hybrid fusion.
            source_rankings.append(eligible_hits)
        if fallback_hits:
            session_fallback_rankings.append(fallback_hits)
    return source_rankings, session_fallback_rankings


def search_source_rankings(
    query: str,
    limit: int,
    filters: dict[str, object],
    harness: str | None = None,
) -> tuple[str, list[list[dict]], list[list[dict]]]:
    app = source_app()
    if app is None or app.syncer.restoring or app.syncer.restore_failed:
        return query, [], []
    rewritten = app.translator.rewrite_query(query)
    candidate_limit = expanded_candidate_limit(limit)
    allow_broad_scan = embedding_profile()["provider"] != "voyage"
    raw_hits = app.store.search(
        query,
        candidate_limit,
        filters=filters,
        allow_broad_scan=allow_broad_scan,
    )
    rewritten_hits = (
        app.store.search(
            rewritten,
            candidate_limit,
            filters=filters,
            allow_broad_scan=allow_broad_scan,
        )
        if rewritten != query
        else []
    )
    source_rankings, session_fallback_rankings = _partition_source_rankings(
        (raw_hits, rewritten_hits), filters, harness
    )
    return rewritten, source_rankings, session_fallback_rankings


def search_source_bm25_rankings(
    query: str,
    limit: int,
    filters: dict[str, object],
    harness: str | None = None,
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Run one syntax-safe raw BM25 lookup for a degraded Voyage request."""
    app = source_app()
    if app is None or app.syncer.restoring or app.syncer.restore_failed:
        return [], []
    raw_hits = app.store.search(
        query,
        expanded_candidate_limit(limit),
        filters=filters,
        allow_broad_scan=False,
    )
    return _partition_source_rankings((raw_hits,), filters, harness)


def search_source_documents(query: str, limit: int, filters: dict[str, object]) -> tuple[str, list[dict]]:
    """Compatibility helper for callers that consume a fused sidecar ranking."""
    rewritten, rankings, _fallback_rankings = search_source_rankings(query, limit, filters)
    return rewritten, stable_rrf(rankings, limit)

# A remote Lance memory can take longer than the Space ingress timeout to open
# on the first recall (model + snapshot + ANN/FTS handles).  Warm it in the
# background at process start so the public search path is hot before an agent
# asks its first question.  Keep only coarse state here: never expose child
# stderr or query/source text in the readiness response.
_WARM_STATE_LOCK = threading.Lock()
_WARM_STATE = {
    "state": "not_started",
    "started_at": None,
    "finished_at": None,
    "refresh_pending": False,
}


def warm_state() -> dict[str, object]:
    with _WARM_STATE_LOCK:
        return dict(_WARM_STATE)


def _warm_native_memory(*, replace: bool = False) -> None:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _WARM_STATE_LOCK:
        if _WARM_STATE.get("state") != "warming":
            _WARM_STATE.update(state="warming", started_at=started, finished_at=None)
    try:
        # A minimal recall initializes the same remote dataset, embedding model,
        # text index, and reranker used by real requests.  It is intentionally
        # run outside the HTTP request lifecycle.  Refreshes use a second child
        # and swap it in only after it is ready, so an old worker can keep
        # serving while a newly-pushed remote snapshot is opened.
        # Always build a replacement child outside INDEX_LOCK.  The first warm
        # used to call the active worker while holding that lock, so the first
        # real HTTP recall could wait for the full cold-open duration.  A
        # replacement is safe for both startup and post-push refreshes: the
        # old worker remains available until the candidate is ready.
        _refresh_native_worker()
    except Exception:
        # Readiness remains useful when a provider/HF endpoint is temporarily
        # unavailable; the next request will retry through the normal worker
        # recovery path.  Do not retain or emit exception text.
        state = "error"
    else:
        state = "ready"
    with _WARM_STATE_LOCK:
        refresh_pending = bool(_WARM_STATE.get("refresh_pending"))
        _WARM_STATE.update(
            state=state,
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            refresh_pending=False,
        )
    if refresh_pending:
        request_warm(force=True)


def request_warm(*, force: bool = False) -> dict[str, object]:
    """Start one background refresh, optionally replacing an old remote worker."""
    with _WARM_STATE_LOCK:
        if _WARM_STATE.get("state") == "warming":
            if force:
                _WARM_STATE["refresh_pending"] = True
            return dict(_WARM_STATE)
        # Reserve the state before starting the thread.  Thread.start() may be
        # delayed, and a second /warm or ingest call must not launch another
        # native MCP child in that gap.
        _WARM_STATE.update(
            state="warming",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            finished_at=None,
            refresh_pending=False,
        )
    threading.Thread(
        target=_warm_native_memory,
        kwargs={"replace": force},
        name="funes-native-warm",
        daemon=True,
    ).start()
    return warm_state()

# The default Funes embedding model is English-oriented.  Keep this small,
# deterministic fallback for installations without a translation provider so
# Chinese queries do not enter the slow CJK tokenizer path.  Technical names
# and identifiers are always collected separately and remain verbatim.
CHINESE_RETRIEVAL_TERMS = (
    ("previous_response_id", "previous_response_id"),
    ("后台任务", "background task"),
    ("保持运行", "keep running"),
    ("如何保持", "how to keep"),
    ("不能上传", "must not upload"),
    ("硬件标识", "hardware identifier"),
    ("远程 MCP", "remote MCP"),
    ("上下文丢失", "context loss"),
    ("上下文丢了", "context loss"),
    ("重启后", "after restart"),
    ("自动恢复", "automatic recovery"),
    ("断网时", "when offline"),
    ("重复扫描", "repeated scan"),
    ("如何去重", "how to deduplicate"),
    ("换 embedding", "switch embedding"),
    ("远程", "remote"),
    ("日志", "logs"),
    ("时区", "timezone"),
    ("对齐", "align"),
    ("扫描", "scan"),
    ("文件", "file"),
    ("上传", "upload"),
    ("模型", "model"),
    ("重建", "rebuild"),
    ("挂了", "unavailable failed"),
    ("怎么办", "what to do"),
    ("共存", "coexistence"),
    ("注意", "considerations"),
    ("查询", "query"),
    ("字段", "fields schema"),
    ("检索", "retrieval"),
    ("处理", "handle"),
    ("关系", "relationship"),
    ("能不能", "can"),
    ("多台", "multiple"),
    ("避免", "avoid"),
    ("泄露", "leak"),
    ("重复", "duplicate"),
    ("搜", "search"),
    ("丢吗", "lost"),
    ("第二轮", "second turn"),
    ("第二次", "second turn"),
    ("丢上下文", "context loss"),
    ("上下文", "context"),
    ("高延迟", "high latency"),
    ("延迟", "latency"),
    ("连接", "connection"),
    ("配置", "configuration settings"),
    ("设置", "configuration settings"),
    ("默认", "default"),
    ("推理", "reasoning inference"),
    ("记忆", "memory"),
    ("会话", "conversation session"),
    ("历史", "history previous"),
    ("之前", "previous prior"),
    ("上次", "previous last time"),
    ("以前", "previous earlier"),
    ("为什么", "why cause"),
    ("为何", "why cause"),
    ("是什么", "what is"),
    ("讨论", "discussion"),
    ("决定", "decision"),
    ("测试", "test result"),
    ("偏好", "preference"),
    ("部署", "deployment deployed"),
    ("丢", "loss lost"),
    ("办公室", "office"),
)


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum("\u4e00" <= c <= "\u9fff" for c in text) / max(1, len(text))


def technical_query_entities(text: str) -> tuple[str, ...]:
    """Return distinctive mixed-language identifiers suitable for an exact hit."""
    entities = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_.*:/-]{1,}", text):
        normalized = token.lower().rstrip("*:-/")
        if (
            "_" in normalized
            or "/" in normalized
            or ":" in normalized
            or any(char.isdigit() for char in normalized)
        ):
            if normalized and normalized not in entities:
                entities.append(normalized)
    return tuple(entities)


def sidecar_has_exact_entities(query: str, results: list[dict]) -> bool:
    entities = technical_query_entities(query)
    if not entities:
        return False
    return any(
        all(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(entity)}(?![A-Za-z0-9_])",
                str(item.get("raw_text", "")),
                re.IGNORECASE,
            )
            for entity in entities
        )
        for item in results[:3]
    )


def query_text(raw: str) -> str:
    """Build an ASCII retrieval shadow while leaving the caller's raw query intact."""
    if LANGUAGE_MODE == "raw":
        return raw
    if LANGUAGE_MODE == "auto" and cjk_ratio(raw) < TRANSLATION_THRESHOLD:
        return raw
    # Keep technical entities verbatim even when the optional provider is absent.
    entities = re.findall(r"[A-Za-z][A-Za-z0-9_.*:/-]{1,}", raw)
    # Provider rewriting belongs exclusively to Translator.rewrite_query.
    # This deterministic map is used only after that single provider path falls
    # back to raw CJK.
    fallback = []
    for phrase, english in CHINESE_RETRIEVAL_TERMS:
        if phrase in raw:
            fallback.append(english)
    # Do not put a CJK-only shadow back into the native CLI.  The raw query is
    # still returned to clients and the original source text remains untouched.
    pieces = []
    for piece in (" ".join(fallback), " ".join(entities)):
        if piece and piece not in pieces:
            pieces.append(piece)
    shadow = " ".join(pieces).strip()
    return shadow or "memory context retrieval"


def run(
    *args: str,
    timeout: int = 180,
    profile: dict[str, object] | None = None,
) -> tuple[int, str, str]:
    env = native_environment(profile=profile)
    p = subprocess.run([FUNES_BIN, *args], text=True, capture_output=True, timeout=timeout, env=env)
    return p.returncode, p.stdout, p.stderr


CANONICAL_FACETS = (
    "source_agent",
    "source_type",
    "project",
    "repo",
    "device_id",
    "content_type",
    "source_missing",
    "role",
    "timestamp",
    "source_path",
    "worktree",
    "message_id",
)
NATIVE_GET_RE = re.compile(
    r"(?m)^\s*→\s*get\s+(.+?)(?=\s+--(?:from|to|memory)\b|$)"
)
NATIVE_REPORT_RE = re.compile(
    r"\bingested\s+sources=(\d+)\s+chunks=(\d+)\s+unchanged=(\d+)\s+stale=(\d+)\s+held=(\d+)(?:\s+commit=(\S+))?"
)
NATIVE_HELD_SOURCE_IDS_RE = re.compile(
    r"(?m)[ \t]held_source_ids=(\[[^\r\n]*\])[ \t]*$"
)
NATIVE_HELD_SOURCE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
NATIVE_RECORD_ERROR_RE = re.compile(
    r"invalid canonical JSONL record|must not be empty|must not contain NUL|invalid timestamp",
    re.IGNORECASE,
)
CANONICAL_REF_PREFIX = "funes-doc:"
HELD_SOURCE_ID_DOMAIN = b"funes-held-source-v1\0"


def _opaque_source_id(source_identity: str) -> str:
    digest = hashlib.sha256(HELD_SOURCE_ID_DOMAIN + source_identity.encode()).hexdigest()
    return "sha256:" + digest


def canonical_reference(source_identity: str) -> str:
    encoded = base64.urlsafe_b64encode(source_identity.encode()).decode().rstrip("=")
    return CANONICAL_REF_PREFIX + encoded


def canonical_reference_identity(reference: str) -> str | None:
    if not reference.startswith(CANONICAL_REF_PREFIX):
        return None
    encoded = reference.removeprefix(CANONICAL_REF_PREFIX)
    try:
        return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    except (ValueError, UnicodeDecodeError):
        return None


def _strip_raw_fields(value):
    """Remove raw-bearing keys recursively before writing canonical JSONL."""
    if isinstance(value, dict):
        return {
            key: _strip_raw_fields(item)
            for key, item in value.items()
            if key not in {"raw_text", "text"}
        }
    if isinstance(value, list):
        return [_strip_raw_fields(item) for item in value]
    return value


def canonical_source_version(item: dict, profile: dict[str, object] | None = None) -> str:
    profile = profile or embedding_profile()
    retrieval_hash = hashlib.sha256(str(item.get("retrieval_text", "")).encode()).hexdigest()
    parts = [
        str(item.get("source_version", "")),
        str(item.get("content_hash", "")),
        str(item.get("translation_hash", "")),
        str(item.get("translation_version", "")),
        retrieval_hash,
        "1" if item.get("source_missing") else "0",
        str(int(item.get("native_generation") or 0)),
        str(profile["fingerprint"]),
    ]
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_document(
    item: dict, profile: dict[str, object] | None = None
) -> dict:
    """Build the native canonical envelope from the durable raw source."""
    profile = profile or embedding_profile()
    metadata = _strip_raw_fields(dict(item.get("metadata") or {}))
    metadata["source_version"] = str(item.get("source_version", ""))
    if item.get("session_id") is not None:
        metadata.setdefault("session_id", item["session_id"])
    document = {
        "source_identity": str(item["source_identity"]),
        "source_version": canonical_source_version(item, profile),
        "raw_text": str(item["raw_text"]),
        "retrieval_text": str(item["retrieval_text"]),
        "content_hash": str(item["content_hash"]),
        "updated_at": str(item.get("retrieval_updated_at") or item.get("updated_at")),
        "metadata": metadata,
    }
    for name in CANONICAL_FACETS:
        if item.get(name) is not None:
            document[name] = item[name]
    # Native recall renders the canonical session coordinate in `→ get`.
    # Encode a reserved sidecar reference even when source metadata came from a
    # session, so a missing raw row can never fall through to native get.
    document["session_id"] = canonical_reference(document["source_identity"])
    return document


def _native_update(
    item: dict,
    status: str,
    version: str | None,
    error: str | None = None,
    *,
    profile: dict[str, object] | None = None,
    memory: str | None = None,
) -> dict:
    profile = profile or index_embedding_profile()
    return {
        "source_identity": str(item["source_identity"]),
        "source_version": str(item.get("source_version", "")),
        "content_hash": str(item["content_hash"]),
        "native_index_version": version,
        "native_index_status": status,
        "native_index_profile": profile["fingerprint"],
        "native_index_memory": memory or index_memory(),
        "native_indexed_at": (
            utc_now()
            if status in {"indexed", "held_secret", "held_invalid"}
            else None
        ),
        "native_index_error": error,
        "native_generation": int(item.get("native_generation") or 0),
    }


def _write_canonical_jsonl(path: Path, records: list[tuple[dict, dict]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for _, document in records:
            stream.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")


def _reported_held_source_ids(
    output: str,
    records: list[tuple[dict, dict]],
    held: int,
) -> set[str] | None:
    """Validate the opaque extended report field before trusting its batch mapping."""
    matches = NATIVE_HELD_SOURCE_IDS_RE.findall(output)
    if len(matches) != 1:
        return None
    try:
        source_ids = json.loads(matches[0])
    except (TypeError, ValueError):
        return None
    if not isinstance(source_ids, list) or len(source_ids) != held:
        return None
    if any(
        not isinstance(source_id, str)
        or NATIVE_HELD_SOURCE_ID_RE.fullmatch(source_id) is None
        for source_id in source_ids
    ) or len(set(source_ids)) != held:
        return None
    record_source_ids = {
        _opaque_source_id(str(item["source_identity"])) for item, _ in records
    }
    if len(record_source_ids) != len(records) or not set(source_ids) <= record_source_ids:
        return None
    return set(source_ids)


def _ingest_canonical_subset(
    records: list[tuple[dict, dict]],
    directory: Path,
    sequence: list[int],
    *,
    memory: str,
    profile: dict[str, object],
) -> tuple[list[dict], bool]:
    sequence[0] += 1
    path = directory / f"batch-{sequence[0]:06d}.jsonl"
    _write_canonical_jsonl(path, records)
    try:
        code, output, error_output = run(
            "ingest-docs",
            str(path),
            "--memory",
            memory,
            timeout=CANONICAL_INDEX_TIMEOUT,
            profile=profile,
        )
    except subprocess.TimeoutExpired:
        return [
            _native_update(
                item, "retry", None, "TimeoutExpired", profile=profile, memory=memory
            )
            for item, _ in records
        ], False
    except (OSError, subprocess.SubprocessError) as exc:
        return [
            _native_update(
                item, "retry", None, type(exc).__name__, profile=profile, memory=memory
            )
            for item, _ in records
        ], False
    if code != 0:
        if NATIVE_RECORD_ERROR_RE.search(error_output or ""):
            if len(records) == 1:
                item, _ = records[0]
                return [
                    _native_update(
                        item,
                        "held_invalid",
                        None,
                        "invalid_source",
                        profile=profile,
                        memory=memory,
                    )
                ], False
            middle = len(records) // 2
            left, left_commit = _ingest_canonical_subset(
                records[:middle], directory, sequence, memory=memory, profile=profile
            )
            right, right_commit = _ingest_canonical_subset(
                records[middle:], directory, sequence, memory=memory, profile=profile
            )
            return left + right, left_commit or right_commit
        return [
            _native_update(
                item, "retry", None, "native_exit", profile=profile, memory=memory
            )
            for item, _ in records
        ], False
    report = NATIVE_REPORT_RE.search(output)
    if report is None:
        return [
            _native_update(
                item, "retry", None, "invalid_report", profile=profile, memory=memory
            )
            for item, _ in records
        ], False
    sources = int(report.group(1))
    unchanged = int(report.group(3))
    stale = int(report.group(4))
    held = int(report.group(5))
    committed = bool(report.group(6))
    if sources + unchanged + stale + held != len(records):
        return [
            _native_update(
                item, "retry", None, "invalid_report", profile=profile, memory=memory
            )
            for item, _ in records
        ], committed
    if held == 0 and stale == 0:
        return [
            _native_update(
                item,
                "indexed",
                document["source_version"],
                profile=profile,
                memory=memory,
            )
            for item, document in records
        ], committed
    # The extension identifies held rows only.  A stale row has no identity mapping, so preserve
    # the legacy bisection fallback unless every non-held row is safe to mark indexed.
    if stale == 0:
        held_source_ids = _reported_held_source_ids(output, records, held)
        if held_source_ids is not None:
            return [
                _native_update(
                    item,
                    (
                        "held_secret"
                        if _opaque_source_id(str(item["source_identity"]))
                        in held_source_ids
                        else "indexed"
                    ),
                    document["source_version"],
                    profile=profile,
                    memory=memory,
                )
                for item, document in records
            ], committed
    if len(records) == 1:
        item, document = records[0]
        if held == 1:
            return [
                _native_update(
                    item,
                    "held_secret",
                    document["source_version"],
                    profile=profile,
                    memory=memory,
                )
            ], committed
        return [
            _native_update(
                item, "retry", None, "native_stale", profile=profile, memory=memory
            )
        ], committed
    middle = len(records) // 2
    left, left_commit = _ingest_canonical_subset(
        records[:middle], directory, sequence, memory=memory, profile=profile
    )
    right, right_commit = _ingest_canonical_subset(
        records[middle:], directory, sequence, memory=memory, profile=profile
    )
    return left + right, committed or left_commit or right_commit


def reconcile_canonical_index(app) -> dict[str, object]:
    """Index one restart-safe batch and persist only derived status in the sidecar."""
    memory = index_memory()
    if not memory or app.syncer.restoring or app.syncer.restore_failed:
        return {"attempted": 0, "indexed": 0, "held": 0, "durable": False}

    profile = index_embedding_profile()

    def select_and_ingest(directory: Path):
        if app.syncer.restoring or app.syncer.restore_failed:
            return [], [], False
        rows = app.store.canonical_index_candidates(
            CANONICAL_INDEX_BATCH,
            str(profile["fingerprint"]),
            memory,
        )
        selected = []
        for item in rows:
            document = canonical_document(item, profile)
            if (
                item.get("native_index_status") in {
                    "indexed", "held_secret", "held_invalid"
                }
                and item.get("native_index_version") == document["source_version"]
                and item.get("native_index_profile") == profile["fingerprint"]
                and item.get("native_index_memory") == memory
            ):
                continue
            selected.append((item, document))
        if not selected:
            return selected, [], False
        updates, committed = _ingest_canonical_subset(
            selected,
            directory,
            [0],
            memory=memory,
            profile=profile,
        )
        return selected, updates, committed

    with tempfile.TemporaryDirectory(prefix="funes-canonical-") as temporary:
        with WRITE_LOCK:
            translation_lock = getattr(app, "translation_lock", None)
            if translation_lock is None:
                records, updates, committed = select_and_ingest(Path(temporary))
            else:
                with translation_lock:
                    records, updates, committed = select_and_ingest(Path(temporary))
    if not records:
        durable = not app.syncer.restoring and not app.syncer.restore_failed
        if durable:
            optimize_canonical_index(app, profile, memory)
        return {"attempted": 0, "indexed": 0, "held": 0, "durable": durable}
    if (
        committed
        and memory == REMOTE
        and profile["fingerprint"] == embedding_profile()["fingerprint"]
    ):
        request_warm(force=True)
    status_documents = []
    valid_updates = []
    for update in updates:
        current = app.store.get(update["source_identity"])
        if (
            current is None
            or str(current.get("source_version", "")) != update["source_version"]
            or str(current.get("content_hash", "")) != update["content_hash"]
        ):
            continue
        status_documents.append({**current, **update})
        valid_updates.append(update)
    pending_marker = None
    if any(update["native_index_status"] == "indexed" for update in valid_updates):
        previous = app.store.native_optimize_checkpoint()
        pending_marker = native_optimize_marker(
            profile, "pending", previous, memory=memory
        )
        status_documents.append(pending_marker)
    if valid_updates:
        status_documents.append(app.store.native_index_state_record(valid_updates))
    sync = app.syncer.upload(status_documents) if status_documents else {"durable": False}
    durable = bool(sync.get("durable"))
    if durable:
        app.store.update_native_index(valid_updates)
        if pending_marker is not None:
            app.store.set_native_optimize_checkpoint(pending_marker)
    return {
        "attempted": len(records),
        "indexed": sum(item["native_index_status"] == "indexed" for item in updates) if durable else 0,
        "held": (
            sum(
                item["native_index_status"] in {"held_secret", "held_invalid"}
                for item in valid_updates
            )
            if durable
            else 0
        ),
        "durable": durable,
    }


def optimize_native_index(
    memory: str | None = None,
    profile: dict[str, object] | None = None,
) -> bool:
    """Optimize the committed remote index without rebuilding local fallback state."""
    memory = memory or index_memory()
    profile = profile or index_embedding_profile()
    try:
        code, _, _ = run(
            "optimize-index",
            memory,
            timeout=CANONICAL_OPTIMIZE_TIMEOUT,
            profile=profile,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return code == 0


def native_optimize_marker(
    profile: dict[str, object],
    status: str,
    previous: dict[str, object],
    index_fingerprint: str | None = None,
    *,
    memory: str | None = None,
) -> dict[str, object]:
    """Build one monotonic marker for content-addressed restore ordering."""
    return {
        **profile,
        "memory": memory or index_memory(),
        "index_fingerprint": index_fingerprint,
        "status": status,
        "optimized_at": utc_now(),
        "revision": int(previous.get("revision") or 0) + 1,
        "_funes_record": "native_optimize_checkpoint",
    }


def optimize_canonical_index(
    app,
    profile: dict[str, object],
    memory: str | None = None,
) -> bool:
    """Run and durably checkpoint one optimize after a profile backlog catches up."""
    memory = memory or index_memory()
    with WRITE_LOCK:
        previous = app.store.native_optimize_checkpoint()
        if (
            previous.get("status") == "optimized"
            and previous.get("fingerprint") == profile["fingerprint"]
            and previous.get("memory") == memory
            and int(previous.get("revision") or 0) > 0
        ):
            return False
        checkpoint = app.store.native_index_checkpoint(profile, memory)
        if not checkpoint["complete"] or not checkpoint["indexed"]:
            return False
        optimized = optimize_native_index(memory, profile)
        marker = native_optimize_marker(
            profile,
            "optimized" if optimized else "retry",
            previous,
            str(checkpoint["index_fingerprint"]),
            memory=memory,
        )
        sync = app.syncer.upload([marker])
        if not sync.get("durable"):
            return False
        app.store.set_native_optimize_checkpoint(marker)
    if (
        optimized
        and memory == REMOTE
        and profile["fingerprint"] == embedding_profile()["fingerprint"]
    ):
        request_warm(force=True)
    return optimized


def _canonical_reconcile_background(app) -> None:
    app.restore_done.wait()
    while not app.canonical_index_stop.is_set():
        try:
            reconcile_canonical_index(app)
        except Exception:
            # Retry state remains in the encrypted source store. Never log raw
            # rows, subprocess output, provider payloads, or credentials.
            pass
        if app.canonical_index_stop.wait(CANONICAL_INDEX_INTERVAL):
            break


def start_canonical_reconciler(app) -> None:
    if not index_memory() or getattr(app, "canonical_index_thread", None) is not None:
        return
    app.canonical_index_stop = threading.Event()
    app.canonical_index_thread = threading.Thread(
        target=_canonical_reconcile_background,
        args=(app,),
        name="funes-canonical-index-reconcile",
        daemon=True,
    )
    app.canonical_index_thread.start()


def stop_canonical_reconciler(app) -> None:
    stop = getattr(app, "canonical_index_stop", None)
    thread = getattr(app, "canonical_index_thread", None)
    if stop is not None:
        stop.set()
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=CANONICAL_INDEX_TIMEOUT + 5)


def native_result_ids(output: str) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in NATIVE_GET_RE.findall(output)))


def _public_source_item(item: dict) -> dict:
    public = dict(item)
    public.pop("retrieval_text", None)
    return public


def materialize_native_results(
    output: str,
    app,
    limit: int,
    *,
    deadline: float | None = None,
) -> list[dict]:
    """Resolve native rank coordinates to raw sidecar documents or sessions."""
    results = []
    for identity in native_result_ids(output)[:limit]:
        is_canonical_reference = identity.startswith(CANONICAL_REF_PREFIX)
        canonical_identity = canonical_reference_identity(identity)
        lookup_identity = canonical_identity or identity
        item = app.store.get(lookup_identity) if app is not None else None
        if item is not None:
            public = _public_source_item(item)
            public["retrieval_backend"] = "native_funes"
            results.append(public)
            continue
        if is_canonical_reference:
            # A canonical native reference without its encrypted sidecar raw is
            # unsafe to render: native get would expose retrieval_text.
            continue
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise NativeMcpTimeoutError("native MCP request timed out")
        session = get(identity, timeout=remaining)
        results.append(
            {
                "raw_text": session,
                "source_type": "session",
                "session_id": identity,
                "retrieval_backend": "native_funes",
            }
        )
    return results


class NativeMcpError(subprocess.SubprocessError):
    """A native MCP failure without carrying process output into logs or HTTP responses."""


class NativeMcpBusyError(NativeMcpError):
    """The single native read worker is occupied; callers should retry shortly."""


class NativeMcpTimeoutError(NativeMcpError):
    """A native request exhausted its complete caller-owned time budget."""


class NativeMcpWorker:
    """Keep one ``funes mcp`` process warm for model and remote-cache reuse."""

    def __init__(
        self,
        binary: str,
        remote: str,
        home: Path,
        *,
        timeout: float = MCP_TIMEOUT,
        handshake_timeout: float = MCP_HANDSHAKE_TIMEOUT,
    ) -> None:
        self.binary = binary
        self.remote = remote
        self.home = Path(home)
        self.timeout = float(timeout)
        self.handshake_timeout = float(handshake_timeout)
        self._process = None
        self._next_id = 1
        self._lock = threading.RLock()

    @property
    def process(self):
        """Expose the child for focused tests without making it part of the HTTP contract."""
        return self._process

    def close(self) -> None:
        with self._lock:
            self._stop_locked()

    def _environment(self) -> dict[str, str]:
        # Keep HF_TOKEN/HF_HOME and any other caller-provided Hub settings.  Only
        # FUNES_HOME is pinned to the Space's durable warm-cache directory.
        return native_environment(self.home)

    @staticmethod
    def _alive(process) -> bool:
        try:
            return process.poll() is None
        except AttributeError:
            return True

    @staticmethod
    def _close_stream(stream) -> None:
        if stream is None:
            return
        try:
            stream.close()
        except (OSError, ValueError):
            pass

    @staticmethod
    def _reap_process(process) -> None:
        try:
            process.wait()
        except (OSError, AttributeError, TypeError):
            pass

    def _stop_locked(self, deadline: float | None = None) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if self._alive(process):
            try:
                process.terminate()
            except (OSError, AttributeError):
                pass
            if deadline is not None:
                try:
                    process.kill()
                except (OSError, AttributeError):
                    pass
                threading.Thread(
                    target=self._reap_process,
                    args=(process,),
                    name="funes-native-mcp-reap",
                    daemon=True,
                ).start()
            else:
                try:
                    process.wait(timeout=0.5)
                except (OSError, subprocess.TimeoutExpired, AttributeError, TypeError):
                    try:
                        process.kill()
                    except (OSError, AttributeError):
                        pass
                    try:
                        process.wait(timeout=0.5)
                    except (OSError, subprocess.TimeoutExpired, AttributeError, TypeError):
                        pass
        self._close_stream(getattr(process, "stdin", None))
        self._close_stream(getattr(process, "stdout", None))

    def _start_locked(self, deadline: float | None = None) -> None:
        args = [self.binary, "mcp"]
        if self.remote:
            args.append(self.remote)
        try:
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=self._environment(),
            )
        except OSError as exc:
            raise NativeMcpError("native MCP unavailable") from exc
        self._process = process
        # JSON-RPC ids are local to one child.  Resetting here also makes a
        # restarted worker interoperable with strict fake/native servers.
        self._next_id = 1
        try:
            handshake_timeout = self.handshake_timeout
            if deadline is not None:
                handshake_timeout = min(
                    handshake_timeout,
                    max(0.001, deadline - time.monotonic()),
                )
            self._request_locked(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "funes-space-bridge", "version": "1"},
                },
                handshake_timeout,
            )
            self._send_locked({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        except NativeMcpError:
            self._stop_locked(deadline)
            raise

    def _ensure_started_locked(self, deadline: float | None = None) -> None:
        if self._process is not None and self._alive(self._process):
            return
        self._stop_locked()
        self._start_locked(deadline)

    def _send_locked(self, message: dict) -> None:
        process = self._process
        if process is None or not self._alive(process):
            raise NativeMcpError("native MCP process exited")
        stdin = getattr(process, "stdin", None)
        if stdin is None:
            raise NativeMcpError("native MCP stdin unavailable")
        try:
            stdin.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
            stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise NativeMcpError("native MCP write failed") from exc

    @staticmethod
    def _read_line(stream, deadline: float) -> str:
        """Read one newline-delimited frame without ever blocking past deadline."""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NativeMcpTimeoutError("native MCP request timed out")
            try:
                fd = stream.fileno()
            except (AttributeError, OSError, ValueError):
                # Small in-memory fakes used by tests do not expose a file
                # descriptor.  Their readline is non-blocking by contract.
                readable = True
            else:
                try:
                    readable = bool(select.select([fd], [], [], remaining)[0])
                except (OSError, ValueError):
                    readable = True
            if not readable:
                raise NativeMcpTimeoutError("native MCP request timed out")
            try:
                line = stream.readline()
            except (OSError, ValueError) as exc:
                raise NativeMcpError("native MCP read failed") from exc
            if line in ("", b""):
                raise NativeMcpError("native MCP process closed stdout")
            if isinstance(line, bytes):
                line = line.decode("utf-8", "replace")
            return line.strip()

    def _request_locked(self, method: str, params: dict, timeout: float) -> object:
        request_id = self._next_id
        self._next_id += 1
        self._send_locked({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        process = self._process
        stdout = getattr(process, "stdout", None) if process is not None else None
        if stdout is None:
            raise NativeMcpError("native MCP stdout unavailable")
        deadline = time.monotonic() + max(0.001, float(timeout))
        while True:
            line = self._read_line(stdout, deadline)
            if not line:
                continue
            try:
                message = json.loads(line)
            except (TypeError, ValueError):
                # A native server must keep stdout JSON-RPC, but ignoring a
                # stray line avoids reflecting it (which could contain data).
                continue
            if not isinstance(message, dict) or message.get("id") != request_id:
                continue
            if message.get("error") is not None:
                raise NativeMcpError("native MCP request failed")
            return message.get("result")

    def _call(self, method: str, params: dict, *, timeout: float | None = None) -> object:
        # A read-only recall/get can safely be retried once after a dead child;
        # the retry also covers a child that exits during initialization.
        last_error = None
        total_timeout = self.timeout if timeout is None else max(0.001, float(timeout))
        deadline = time.monotonic() + total_timeout
        with self._lock:
            for attempt in range(2):
                try:
                    self._ensure_started_locked(deadline)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise NativeMcpTimeoutError("native MCP request timed out")
                    return self._request_locked(method, params, remaining)
                except NativeMcpTimeoutError:
                    self._stop_locked(deadline)
                    raise
                except NativeMcpError as exc:
                    last_error = exc
                    self._stop_locked(deadline)
                    if attempt == 0 and time.monotonic() < deadline:
                        continue
                    raise
        raise last_error or NativeMcpError("native MCP request failed")

    @staticmethod
    def _text(result: object) -> str:
        if isinstance(result, str):
            return result
        if not isinstance(result, dict):
            raise NativeMcpError("native MCP response malformed")
        if result.get("isError"):
            raise NativeMcpError("native MCP tool failed")
        content = result.get("content")
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text", ""), str)
            )
        text = result.get("text")
        return text if isinstance(text, str) else ""

    def call_tool(self, name: str, arguments: dict, *, timeout: float | None = None) -> str:
        result = self._call("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)
        return self._text(result)

    def recall(
        self,
        query: str,
        *,
        k: int = 8,
        candidates: int | None = None,
        half_life: float | None = None,
        neighbors: int | None = None,
        block_type: str | None = None,
        harness: str | None = None,
        timeout: float | None = None,
        **extra,
    ) -> str:
        arguments = {"query": str(query), "k": int(k)}
        for name, value in (
            ("candidates", candidates),
            ("half_life", half_life),
            ("neighbors", neighbors),
            ("block_type", block_type),
            ("harness", harness),
        ):
            if value is not None:
                arguments[name] = value
        # Keep compatibility with callers that used the old HTTP filter names;
        # rmcp/serde ignores unknown optional fields while native recall still
        # receives the same query and bounded tuning values.
        arguments.update({name: value for name, value in extra.items() if value is not None})
        result = self.call_tool("recall", arguments, timeout=timeout)
        if result.startswith("recall error:"):
            raise NativeMcpError("native recall failed")
        return result

    def get(
        self,
        session_id: str,
        *,
        from_: int | None = None,
        to: int | None = None,
        timeout: float | None = None,
        **extra,
    ) -> str:
        if from_ is None and "from" in extra:
            from_ = extra.pop("from")
        arguments = {"session_id": str(session_id)}
        if from_ is not None:
            arguments["from"] = from_
        if to is not None:
            arguments["to"] = to
        arguments.update({name: value for name, value in extra.items() if value is not None})
        result = self.call_tool("get", arguments, timeout=timeout)
        if result.startswith("get error:"):
            raise NativeMcpError("native get failed")
        return result


MCP_WORKER = None
_MCP_WORKER_CONFIG = None
_MCP_WORKER_LOCK = threading.Lock()


def native_worker() -> NativeMcpWorker:
    """Return the process singleton, rebuilding it only when runtime config changes."""
    global MCP_WORKER, _MCP_WORKER_CONFIG
    # Tests and embedders may supply a small fake directly; do not replace it.
    if MCP_WORKER is not None and not isinstance(MCP_WORKER, NativeMcpWorker):
        return MCP_WORKER
    config = (
        FUNES_BIN, REMOTE, str(HOME), MCP_TIMEOUT, MCP_HANDSHAKE_TIMEOUT,
        embedding_profile()["fingerprint"],
        os.getenv("FUNES_RERANK_PROVIDER", "none"),
        os.getenv("FUNES_NATIVE_FALLBACK", "false"),
    )
    with _MCP_WORKER_LOCK:
        # During the initial background warm there is intentionally no active
        # worker yet.  Do not race it by spawning a second native MCP child;
        # callers receive a bounded retryable error instead.
        with _WARM_STATE_LOCK:
            warming = _WARM_STATE.get("state") == "warming"
        if MCP_WORKER is None and warming:
            raise NativeMcpError("native MCP warming")
        if isinstance(MCP_WORKER, NativeMcpWorker) and _MCP_WORKER_CONFIG == config:
            return MCP_WORKER
        if isinstance(MCP_WORKER, NativeMcpWorker):
            MCP_WORKER.close()
        MCP_WORKER = NativeMcpWorker(
            FUNES_BIN,
            REMOTE,
            HOME,
            timeout=MCP_TIMEOUT,
            handshake_timeout=MCP_HANDSHAKE_TIMEOUT,
        )
        _MCP_WORKER_CONFIG = config
        return MCP_WORKER


def close_native_worker() -> None:
    global MCP_WORKER, _MCP_WORKER_CONFIG
    with _MCP_WORKER_LOCK:
        if isinstance(MCP_WORKER, NativeMcpWorker):
            MCP_WORKER.close()
        MCP_WORKER = None
        _MCP_WORKER_CONFIG = None


def _refresh_native_worker() -> None:
    """Warm a replacement child and atomically swap it with the active one."""
    global MCP_WORKER, _MCP_WORKER_CONFIG
    config = (
        FUNES_BIN, REMOTE, str(HOME), MCP_TIMEOUT, MCP_HANDSHAKE_TIMEOUT,
        embedding_profile()["fingerprint"],
        os.getenv("FUNES_RERANK_PROVIDER", "none"),
        os.getenv("FUNES_NATIVE_FALLBACK", "false"),
    )
    candidate = NativeMcpWorker(
        FUNES_BIN,
        REMOTE,
        HOME,
        timeout=MCP_TIMEOUT,
        handshake_timeout=MCP_HANDSHAKE_TIMEOUT,
    )
    try:
        probe = candidate.recall("memory", k=1, candidates=1, half_life=0, neighbors=0)
        if probe.startswith("recall error:"):
            raise NativeMcpError("native recall failed")
    except Exception:
        candidate.close()
        raise
    with INDEX_LOCK:
        with _MCP_WORKER_LOCK:
            previous = MCP_WORKER
            MCP_WORKER = candidate
            _MCP_WORKER_CONFIG = config
        if isinstance(previous, NativeMcpWorker):
            previous.close()


atexit.register(close_native_worker)
atexit.register(close_source_app)


def _locked_native_call(invoke, timeout: float | None):
    deadline = None
    lock_timeout = RECALL_LOCK_TIMEOUT
    if timeout is not None:
        deadline = time.monotonic() + max(0.001, float(timeout))
        lock_timeout = min(lock_timeout, max(0.001, deadline - time.monotonic()))
    if not INDEX_LOCK.acquire(timeout=lock_timeout):
        raise NativeMcpBusyError("native MCP busy")
    try:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise NativeMcpTimeoutError("native MCP request timed out")
        return invoke(remaining)
    finally:
        INDEX_LOCK.release()


def recall(query: str, *, timeout: float | None = None, **kwargs) -> str:
    return _locked_native_call(
        lambda remaining: native_worker().recall(
            query,
            timeout=remaining,
            **kwargs,
        ),
        timeout,
    )


def get(session_id: str, *, timeout: float | None = None, **kwargs) -> str:
    return _locked_native_call(
        lambda remaining: native_worker().get(
            session_id,
            timeout=remaining,
            **kwargs,
        ),
        timeout,
    )


class _DeadlineRunBusyError(RuntimeError):
    """A deadline runner is occupied by a different request."""


_DEADLINE_RUN_LOCK = threading.Lock()
_DEADLINE_RUNS: dict[str, tuple[threading.Event, dict[str, object]]] = {}


def _run_before_deadline(invoke, deadline: float, *, thread_name: str):
    """Run a blocking dependency behind a caller-owned monotonic deadline.

    At most one ignored-timeout call may survive for each dependency name.
    Later callers fail fast rather than creating unbounded daemon threads or
    receiving another request's result while an upstream dependency is wedged.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("operation deadline exhausted")
    with _DEADLINE_RUN_LOCK:
        state = _DEADLINE_RUNS.get(thread_name)
        if state is not None:
            # Names identify a dependency, not a request. Waiting here could
            # return the first request's result to a later, different query.
            raise _DeadlineRunBusyError("operation already in progress")
        done: threading.Event = threading.Event()
        outcome: dict[str, object] = {}
        state = done, outcome
        _DEADLINE_RUNS[thread_name] = state

        def runner() -> None:
            try:
                outcome["value"] = invoke()
            except Exception as exc:
                outcome["error"] = exc
            finally:
                done.set()
                with _DEADLINE_RUN_LOCK:
                    if _DEADLINE_RUNS.get(thread_name) is state:
                        _DEADLINE_RUNS.pop(thread_name, None)

        # NativeMcpWorker receives the same deadline and terminates its
        # subprocess on timeout. This is the final HTTP safeguard for a
        # wedged Python call or dependency that ignores its timeout.
        threading.Thread(target=runner, name=thread_name, daemon=True).start()
    done, outcome = state
    if not done.wait(max(0.0, deadline - time.monotonic())):
        raise TimeoutError("operation deadline exhausted")
    error = outcome.get("error")
    if error is not None:
        raise error
    return outcome.get("value")


def _http_native_results(
    query: str,
    app,
    limit: int,
    tuning: dict[str, object],
    deadline: float,
) -> list[dict]:
    """Recall and materialize raw text within one hard HTTP deadline."""

    def invoke() -> list[dict]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NativeMcpTimeoutError("native MCP request timed out")
        output = recall(query, k=limit, timeout=remaining, **tuning)
        return (
            materialize_native_results(output, app, limit, deadline=deadline)
            if output
            else []
        )

    try:
        result = _run_before_deadline(
            invoke,
            deadline,
            thread_name="funes-http-native-recall",
        )
    except _DeadlineRunBusyError as exc:
        raise NativeMcpBusyError("native MCP busy") from exc
    except TimeoutError as exc:
        raise NativeMcpTimeoutError("native MCP request timed out") from exc
    return list(result or [])


def auth_ok(handler: BaseHTTPRequestHandler) -> bool:
    supplied = handler.headers.get("X-Funes-Authorization", "") or handler.headers.get("X-Funes-Token", "")
    if supplied:
        return bool(TOKEN) and supplied == "Bearer " + TOKEN
    # Public Spaces and local tests can use the normal Authorization header.
    # Private Spaces reserve that header for the Hub token and use the explicit
    # application header above.
    return bool(TOKEN) and handler.headers.get("Authorization", "") == "Bearer " + TOKEN


def ready_payload() -> tuple[int, dict[str, object]]:
    """Return the authenticated readiness/status payload shared by both APIs."""
    if not REMOTE:
        return 503, {"ok": False, "error": "FUNES_MEMORY is not configured", "native_warm": warm_state()}
    code, out, err = run("status", REMOTE, timeout=30)
    sources = source_state()
    source_ok = not sources.get("configured") or bool(sources.get("ready"))
    return (
        200 if code == 0 and source_ok else 503,
        {
            "ok": code == 0 and source_ok,
            "remote": REMOTE,
            "status": out[-2000:],
            "error": err[-500:],
            "native_warm": warm_state(),
            "source_store": sources,
            "embedding_profile": embedding_profile(),
        },
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "funes-http/1"

    def log_message(self, fmt: str, *args) -> None:
        # Never log request bodies, Authorization, or raw memory text.
        return

    def send_json(self, code: int, obj: object, headers: dict[str, str] | None = None) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def body(self) -> dict:
        try:
            max_bytes = max(1, int(os.getenv("FUNES_MAX_BODY_BYTES", "64000000")))
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if n < 0 or n > max_bytes:
            raise ValueError("request too large")
        raw = self.rfile.read(n)
        if len(raw) != n:
            raise ValueError("truncated request body")
        encoding = self.headers.get("Content-Encoding", "identity").strip().lower()
        if encoding in ("", "identity"):
            decoded = raw
        elif encoding == "gzip":
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as stream:
                    decoded = stream.read(max_bytes + 1)
            except (EOFError, OSError, zlib.error) as exc:
                raise ValueError("invalid gzip request body") from exc
        else:
            raise ValueError("unsupported Content-Encoding")
        if len(decoded) > max_bytes:
            raise ValueError("request too large")
        obj = json.loads(decoded or b"{}")
        if not isinstance(obj, dict):
            raise ValueError("JSON object required")
        return obj

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"ok": True, "service": "funes"})
            return
        operation_prefix = "/ingest/operations/"
        if self.path.startswith(operation_prefix):
            if not auth_ok(self):
                self.send_json(401, {"error": "unauthorized"})
                return
            operation_id = self.path[len(operation_prefix):]
            code, payload = get_ingest_operation(operation_id)
            headers = (
                {"Retry-After": str(payload["retry_after"])}
                if code == 202
                else None
            )
            self.send_json(code, payload, headers)
            return
        if self.path in ("/ready", "/sync/status"):
            if not auth_ok(self):
                self.send_json(401, {"error": "unauthorized"})
                return
            code, payload = ready_payload()
            self.send_json(code, payload)
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not auth_ok(self):
            self.send_json(401, {"error": "unauthorized"})
            return
        try:
            obj = self.body()
            if self.path == "/sync/status":
                code, payload = ready_payload()
                self.send_json(code, payload)
                return
            if self.path == "/sources/check":
                app = source_app()
                if app is None:
                    self.send_json(
                        503,
                        {"ok": False, "error": "FUNES_STORAGE_REPO is not configured"},
                    )
                    return
                if app.syncer.restoring or app.syncer.restore_failed:
                    error = "restore_in_progress" if app.syncer.restoring else "restore_failed"
                    self.send_json(503, {"ok": False, "error": error})
                    return
                identities = validate_source_identity_batch(obj.get("source_identities"))
                present = app.store.existing_identities(identities)
                present_set = set(present)
                missing = [identity for identity in identities if identity not in present_set]
                self.send_json(200, {"ok": True, "present": present, "missing": missing})
                return
            if self.path == "/sync":
                app = source_app()
                if app is not None:
                    if app.syncer.restoring or app.syncer.restore_failed:
                        error = "restore_in_progress" if app.syncer.restoring else "restore_failed"
                        self.send_json(503, {"ok": False, "durable": False, "error": error})
                        return
                    result = app.syncer.upload()
                    result["ok"] = bool(result.get("durable"))
                    self.send_json(200 if result["ok"] else 503, result)
                    return
                if not REMOTE:
                    self.send_json(503, {"ok": False, "durable": False, "error": "FUNES_MEMORY is not configured"})
                    return
                # Every successful /ingest already completes a native push.
                # This compatibility checkpoint therefore acknowledges that
                # durable state without rebuilding or pushing the index again.
                self.send_json(200, {"ok": True, "durable": True, "remote": REMOTE})
                return
            if self.path == "/reindex":
                scope = str(obj.get("scope", ""))
                if scope not in {"retrieval_text", "all"}:
                    self.send_json(400, {"error": "scope must be retrieval_text or all"})
                    return
                app = source_app()
                if app is None:
                    self.send_json(
                        503,
                        {
                            "queued": False,
                            "durable": False,
                            "error": "FUNES_STORAGE_REPO is not configured",
                        },
                    )
                    return
                with WRITE_LOCK:
                    result = queue_source_reindex(app, scope)
                self.send_json(202 if result.get("durable") else 503, result)
                return
            if self.path == "/warm":
                if not REMOTE:
                    self.send_json(503, {"ok": False, "error": "FUNES_MEMORY is not configured"})
                    return
                self.send_json(202, {"ok": True, "native_warm": request_warm(force=True)})
                return
            if self.path in ("/search", "/recall"):
                raw_query = str(obj.get("query", "")).strip()
                if not raw_query:
                    self.send_json(400, {"error": "query is required"})
                    return
                limit = max(1, min(int(obj.get("limit", obj.get("k", 8))), 50))
                facet_values = obj.get("facets") or {}
                if not isinstance(facet_values, dict):
                    raise ValueError("facets must be an object")
                filters = {
                    key: obj.get(key, facet_values.get(key))
                    for key in (
                        "source_agent",
                        "source_type",
                        "project",
                        "repo",
                        "device_id",
                        "role",
                        "content_type",
                        "source_missing",
                        "since",
                        "until",
                    )
                    if obj.get(key, facet_values.get(key)) is not None
                }
                app = source_app()
                if app is not None and (app.syncer.restoring or app.syncer.restore_failed):
                    error = "restore_in_progress" if app.syncer.restoring else "restore_failed"
                    self.send_json(503, {"ok": False, "results": [], "results_text": "", "error": error})
                    return
                harness = str(obj.get("harness", "")).strip() or None
                profile = embedding_profile()
                embedding_provider = str(profile["provider"])
                sidecar_authoritative = any(
                    key in filters for key in ("role", "since", "until")
                )
                voyage_hot_path = (
                    embedding_provider == "voyage" and not sidecar_authoritative
                )
                voyage_deadline = (
                    time.monotonic() + VOYAGE_HTTP_TIMEOUT
                    if embedding_provider == "voyage"
                    else None
                )
                if voyage_hot_path:
                    # Native Voyage recall already fuses its vector and Lance
                    # FTS/BM25 rankings. Do not duplicate that work in SQLite
                    # unless the provider/native worker actually fails.
                    source_query = raw_query
                    source_rankings = []
                    session_fallback_rankings = []
                elif embedding_provider == "voyage" and sidecar_authoritative:
                    # Role and date filters are sidecar-only. Keep this one
                    # raw BM25 lookup bounded by the same Voyage HTTP budget;
                    # a query translator/rewrite can otherwise consume it.
                    source_query = raw_query
                    try:
                        source_rankings, session_fallback_rankings = (
                            _run_before_deadline(
                                lambda: search_source_bm25_rankings(
                                    raw_query, limit, filters, harness
                                ),
                                voyage_deadline,
                                thread_name="funes-http-sidecar-bm25",
                            )
                        )
                    except Exception:
                        source_rankings = []
                        session_fallback_rankings = []
                else:
                    source_query, source_rankings, session_fallback_rankings = (
                        search_source_rankings(raw_query, limit, filters, harness)
                    )
                sidecar_results = stable_rrf(source_rankings, limit)
                has_cjk = (
                    LANGUAGE_MODE == "auto"
                    and any("\u4e00" <= char <= "\u9fff" for char in raw_query)
                )
                cjk_query = (
                    has_cjk
                    and cjk_ratio(raw_query) >= TRANSLATION_THRESHOLD
                )
                exact_sidecar = (
                    len(sidecar_results) >= min(limit, 3)
                    and sidecar_has_exact_entities(raw_query, sidecar_results)
                )
                # Voyage is multilingual: embed the user's original query.
                # Query shadows remain available only to legacy local mode and
                # never replace the returned or vectorized raw text.
                query = (
                    raw_query
                    if LANGUAGE_MODE == "raw" or embedding_provider == "voyage"
                    else source_query if source_query != raw_query else query_text(raw_query)
                )
                # CJK queries use the ASCII shadow above.  Keep the native
                # search bounded so the CPU Space does not spend its entire
                # request window reranking broad generic terms.
                tuning = {}
                if cjk_query:
                    tuning = {
                        "candidates": max(6, min(20, limit * 3)),
                        "neighbors": 0,
                        "half_life": 0,
                    }
                for name in ("harness",):
                    if obj.get(name):
                        tuning[name] = str(obj[name])
                for name in (
                    "source_agent",
                    "source_type",
                    "project",
                    "repo",
                    "device_id",
                    "content_type",
                    "source_missing",
                ):
                    if name in filters:
                        tuning[name] = filters[name]
                # Date/role filtering remains authoritative in the sidecar.
                native_allowed = (
                    (embedding_provider == "voyage" or not exact_sidecar)
                    and not any(key in filters for key in ("role", "since", "until"))
                )
                native_budget = (
                    CJK_NATIVE_TIMEOUT
                    if has_cjk and sidecar_results
                    else HTTP_NATIVE_TIMEOUT
                )
                if embedding_provider == "voyage":
                    native_budget = min(native_budget, VOYAGE_NATIVE_TIMEOUT)
                if voyage_deadline is not None:
                    native_budget = min(
                        native_budget,
                        max(0.001, voyage_deadline - time.monotonic()),
                    )
                native_deadline = time.monotonic() + native_budget
                native_failure = ""
                try:
                    # The native CLI defaults to 30 fused candidates, recency
                    # weighting, and neighbor expansion. Those defaults are
                    # useful interactively but can exceed a CPU Space ingress
                    # deadline after a large remote snapshot is opened. Keep
                    # the HTTP surface bounded while allowing operators to
                    # raise the cap with FUNES_HTTP_MAX_CANDIDATES.
                    requested_candidates = int(tuning.get("candidates", max(2, limit * 2)))
                    tuning["candidates"] = min(HTTP_MAX_CANDIDATES, requested_candidates)
                    tuning.setdefault("neighbors", 0)
                    tuning.setdefault("half_life", 0)
                    if native_allowed and voyage_hot_path:
                        results = _http_native_results(
                            query,
                            app,
                            limit,
                            tuning,
                            native_deadline,
                        )
                    else:
                        out = (
                            recall(
                                query,
                                k=limit,
                                timeout=max(
                                    0.001,
                                    native_deadline - time.monotonic(),
                                ),
                                **tuning,
                            )
                            if native_allowed
                            else ""
                        )
                        results = (
                            materialize_native_results(
                                out,
                                app,
                                limit,
                                deadline=native_deadline,
                            )
                            if out
                            else []
                        )
                except NativeMcpBusyError:
                    results = []
                    native_failure = "busy"
                    retrieval_degraded = (
                        "voyage_unavailable"
                        if embedding_provider == "voyage"
                        else "native_mcp_busy"
                    )
                except NativeMcpError:
                    results = []
                    native_failure = "unavailable"
                    retrieval_degraded = (
                        "voyage_unavailable"
                        if embedding_provider == "voyage"
                        else "native_mcp_unavailable"
                    )
                else:
                    retrieval_degraded = ""
                if native_failure and voyage_hot_path:
                    try:
                        source_rankings, session_fallback_rankings = (
                            _run_before_deadline(
                                lambda: search_source_bm25_rankings(
                                    raw_query, limit, filters, harness
                                ),
                                voyage_deadline,
                                thread_name="funes-http-sidecar-bm25",
                            )
                        )
                    except Exception:
                        # A degraded lookup must not turn a provider outage
                        # into an unbounded or disconnected HTTP request.
                        source_rankings = []
                        session_fallback_rankings = []
                    sidecar_results = stable_rrf(source_rankings, limit)
                if native_failure == "busy":
                    if not any(source_rankings) and not any(session_fallback_rankings):
                        self.send_json(
                            429,
                            {
                                "ok": False,
                                "results": [],
                                "results_text": "",
                                "error": "native_mcp_busy",
                                "retry_after": 3,
                            },
                            {"Retry-After": "3"},
                        )
                        return
                elif native_failure == "unavailable":
                    if not any(source_rankings) and not any(session_fallback_rankings):
                        unavailable = retrieval_degraded
                        self.send_json(
                            503,
                            {
                                "ok": False,
                                "results": [],
                                "results_text": "",
                                "error": "native_mcp_unavailable",
                                "retrieval_degraded": unavailable,
                                "retrieval_backend": "unavailable",
                                "embedding_profile": profile,
                            },
                        )
                        return
                if native_failure:
                    results = sidecar_results
                elif not voyage_hot_path:
                    # Legacy/local mode still fuses the sidecar rankings with
                    # native semantic hits. Voyage already performs this RRF
                    # internally and must not pay for duplicate Python FTS.
                    results = (
                        stable_rrf([*source_rankings, results], limit)
                        if results
                        else sidecar_results
                    )
                results_text = "\n\n".join(
                    str(item.get("raw_text", ""))
                    for item in results
                    if item.get("raw_text")
                )
                response = {
                    "ok": True,
                    "query": raw_query,
                    "retrieval_query": query,
                    "results": results,
                    "results_text": results_text,
                    "error": "",
                    "retrieval_backend": (
                        "bm25"
                        if retrieval_degraded or not native_allowed
                        else f"{embedding_provider}_lance_bm25_rrf"
                    ),
                    "embedding_profile": profile,
                }
                if retrieval_degraded:
                    response["retrieval_degraded"] = retrieval_degraded
                self.send_json(200, response)
                return
            if self.path == "/get":
                sid = str(obj.get("source_identity", obj.get("id", obj.get("session_id", "")))).strip()
                if not sid:
                    self.send_json(400, {"error": "session_id is required"})
                    return
                app = source_app()
                is_canonical_reference = sid.startswith(CANONICAL_REF_PREFIX)
                canonical_identity = canonical_reference_identity(sid)
                lookup_identity = canonical_identity or sid
                item = app.store.get(lookup_identity) if app is not None and not app.syncer.restoring and not app.syncer.restore_failed else None
                if item is not None:
                    self.send_json(200, {"ok": True, "result": _public_source_item(item), "error": ""})
                    return
                if is_canonical_reference:
                    self.send_json(404, {"ok": False, "result": "", "error": "not_found"})
                    return
                try:
                    out = get(
                        sid,
                        from_=obj.get("from"),
                        to=obj.get("to"),
                        timeout=HTTP_NATIVE_TIMEOUT,
                    )
                except NativeMcpBusyError:
                    self.send_json(429, {"ok": False, "result": "", "error": "native_mcp_busy", "retry_after": 3}, {"Retry-After": "3"})
                    return
                except NativeMcpError:
                    self.send_json(503, {"ok": False, "result": "", "error": "native_mcp_unavailable"})
                    return
                self.send_json(200, {"ok": True, "result": out, "error": ""})
                return
            if self.path == "/ingest":
                docs = obj.get("documents", obj.get("records", obj.get("items")))
                if docs is None:
                    docs = [obj]
                if not isinstance(docs, list) or not docs:
                    self.send_json(400, {"error": "documents must be a non-empty list"})
                    return
                if not all(isinstance(doc, dict) for doc in docs):
                    self.send_json(400, {"error": "each document must be an object"})
                    return
                prefer = self.headers.get("Prefer", "")
                respond_async = any(
                    item.strip().partition(";")[0].lower() == "respond-async"
                    for item in prefer.split(",")
                )
                if respond_async:
                    status, result = start_ingest_operation(docs)
                    headers = {}
                    if status == 202:
                        headers["Location"] = str(result["status_url"])
                        headers["Retry-After"] = str(result["retry_after"])
                    elif status == 429:
                        headers["Retry-After"] = str(result["retry_after"])
                    self.send_json(status, result, headers)
                    return
                source_result = ingest_source_documents(docs)
                if source_result is not None:
                    status, result, _canonical = source_result
                    self.send_json(status, _public_ingest_result(result))
                    return
                # The raw encrypted sidecar is mandatory for HTTP ingestion.
                # Never fall through to the legacy synchronous transcript
                # indexer: request threads must not run native embeddings.
                self.send_json(503, {"error": "FUNES_STORAGE_REPO is not configured", "durable": False})
                return
                if not REMOTE:
                    self.send_json(503, {"error": "FUNES_MEMORY is not configured", "durable": False})
                    return
                now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                source = Path(tempfile.mkdtemp(prefix="funes-ingest-", dir=HOME / "sources"))
                try:
                    harnesses = set()
                    session_ids = []
                    for index, doc in enumerate(docs):
                        if not isinstance(doc, dict):
                            self.send_json(400, {"error": "each document must be an object"})
                            return
                        raw = str(doc.get("raw_text", doc.get("text", "")))
                        if not raw:
                            self.send_json(400, {"error": "raw_text is required"})
                            return
                        sid = str(doc.get("source_identity") or doc.get("session_id") or hashlib.sha256(raw.encode()).hexdigest()[:32])
                        # Native Funes owns parsing, chunking, embedding, and the
                        # TruffleHog push gate.  Keep one synthetic transcript per
                        # source identity so retries remain idempotent.
                        agent = str(doc.get("source_agent", "codex")).lower()
                        harness = {"claude_code": "claude"}.get(agent, agent if agent in {"codex", "pi", "claude", "hermes"} else "codex")
                        harnesses.add(harness)
                        session_ids.append(sid)
                        metadata = {
                            k: doc[k]
                            for k in (
                                "source_identity",
                                "source_agent",
                                "source_type",
                                "device_id",
                                "project",
                                "repo",
                                "worktree",
                                "session_id",
                                "message_id",
                                "role",
                                "timestamp",
                                "source_path",
                                "content_hash",
                                "ingested_at",
                                "updated_at",
                                "source_missing",
                                "content_type",
                                "agent_type",
                                "parent_session_id",
                                "agent_id",
                                "translation_hash",
                                "translation_version",
                                "translation_status",
                            )
                            if doc.get(k) is not None
                        }
                        metadata.setdefault("source_agent", agent)
                        metadata.setdefault("ingested_at", now)
                        cwd = str(doc.get("worktree", doc.get("project", "remote")))
                        role = str(doc.get("role", "user"))
                        source_timestamp = str(doc.get("timestamp") or now)
                        if harness == "pi":
                            line = {"type": "session", "id": sid, "cwd": cwd, "timestamp": source_timestamp, "metadata": metadata}
                            msg = {"type": "message", "id": str(doc.get("message_id") or hashlib.sha256((sid + raw).encode()).hexdigest()[:24]), "timestamp": source_timestamp, "message": {"role": role, "content": [{"type": "text", "text": raw}]}}
                        elif harness == "claude":
                            line = {"type": role if role in {"user", "assistant"} else "user", "uuid": str(doc.get("message_id") or hashlib.sha256((sid + raw).encode()).hexdigest()[:24]), "timestamp": source_timestamp, "cwd": cwd, "metadata": metadata}
                            msg = {"type": line["type"], "uuid": line["uuid"], "timestamp": source_timestamp, "cwd": cwd, "message": {"role": role, "content": [{"type": "text", "text": raw}]}}
                        else:
                            line = {"type": "session_meta", "timestamp": source_timestamp, "payload": {"id": sid, "cwd": cwd, "metadata": metadata}}
                            msg = {"type": "response_item", "timestamp": source_timestamp, "payload": {"type": "message", "role": role, "content": [{"type": "input_text", "text": raw}]}}
                        # Keep each harness in its own directory.  A Pi/Claude
                        # parser must never rescan a Codex envelope from the
                        # same batch.
                        harness_dir = source / harness
                        harness_dir.mkdir(exist_ok=True)
                        (harness_dir / f"{index:08d}-{hashlib.sha256(sid.encode()).hexdigest()[:16]}.jsonl").write_text(json.dumps(line, ensure_ascii=False) + "\n" + json.dumps(msg, ensure_ascii=False) + "\n", encoding="utf-8")
                    outputs = []
                    errors = []
                    with WRITE_LOCK:
                        # The long-lived MCP process keeps its own model/index handles; leave it
                        # alive while Lance appends. Readers use consistent snapshots and can
                        # continue serving the previous remote head while this durable write runs.
                        for harness in sorted(harnesses):
                            code, out, err = run("index", str(source / harness), "--harness", harness, "--yes", timeout=INGEST_INDEX_TIMEOUT)
                            outputs.append(out)
                            errors.append(err)
                            if code != 0:
                                self.send_json(503, {"ok": False, "durable": False, "session_ids": session_ids, "error": "native_index_failed"})
                                return
                        # Keep the durable chunk push incremental.  Forcing a
                        # full remote reindex for every tiny memory-file batch
                        # repeatedly invalidates the read worker; native Funes
                        # can search newly-pushed deltas until its normal index
                        # threshold is reached.
                        code, pout, perr = run("push", REMOTE, "--yes", timeout=INGEST_PUSH_TIMEOUT)
                    outputs.append(pout)
                    errors.append(perr)
                    durable = code == 0
                    self.send_json(200 if durable else 503, {"ok": durable, "durable": durable, "accepted": len(docs) if durable else 0, "session_ids": session_ids, "output": "".join(outputs)[-3000:] if durable else "", "error": "" if durable else "native_push_failed"})
                finally:
                    shutil.rmtree(source, ignore_errors=True)
                return
            self.send_json(404, {"error": "not found"})
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
        except (OSError, subprocess.SubprocessError) as exc:
            self.send_json(500, {"error": str(exc)})


def serve(host: str = "0.0.0.0", port: int = PORT) -> None:
    (HOME / "sources").mkdir(parents=True, exist_ok=True)
    source_app()
    request_warm()
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    serve()
