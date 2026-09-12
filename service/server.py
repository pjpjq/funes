#!/usr/bin/env python3
"""Small durable HTTP service for remote Funes memory retrieval.

The service deliberately uses only the Python standard library at runtime.  The
optional ``huggingface_hub`` package is used for snapshot transport when the
corresponding environment variables are configured.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor


FIELDS = (
    "source_agent", "source_type", "device_id", "project", "repo", "worktree",
    "session_id", "message_id", "role", "timestamp", "source_path", "content_hash",
    "ingested_at", "updated_at", "content_type", "source_missing", "agent_type",
    "parent_session_id", "agent_id",
)
CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
PROMPT_VERSION = "funes-retrieval-v2"
RETRIEVAL_PROMPT = """You are a retrieval normalization engine.
Convert the natural-language Chinese portions of the input into concise English optimized for semantic retrieval.
Rules:
1. Preserve all technical entities verbatim.
2. Preserve code, commands, URLs, file paths, API names, environment variables, model names, product names, repository names, IDs, error messages, quoted strings and version numbers exactly.
3. Never translate identifiers such as previous_response_id, Codex, Northflank, Funes, MCP, CPA, Gemini, Claude, OpenAI.
4. Preserve decisions, constraints, causes, outcomes, preferences and factual details.
5. Do not summarize away important information.
6. Add concise retrieval keywords/entities when useful.
7. Never invent facts.
8. Output retrieval text only."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def cjk_ratio(value: str) -> float:
    if not value:
        return 0.0
    chars = [c for c in value if not c.isspace()]
    return sum(bool(CJK_RE.match(c)) for c in chars) / max(1, len(chars))


class Store:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "funes.sqlite3"
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_identity TEXT NOT NULL,
                    source_version TEXT NOT NULL DEFAULT '',
                    raw_text TEXT NOT NULL,
                    retrieval_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    source_agent TEXT, source_type TEXT, device_id TEXT, project TEXT,
                    repo TEXT, worktree TEXT, session_id TEXT, message_id TEXT, role TEXT,
                    timestamp TEXT, source_path TEXT, content_hash TEXT NOT NULL,
                    ingested_at TEXT NOT NULL, updated_at TEXT NOT NULL, content_type TEXT,
                    source_missing INTEGER NOT NULL DEFAULT 0, agent_type TEXT,
                    parent_session_id TEXT, agent_id TEXT,
                    translation_hash TEXT, translation_version TEXT, translation_status TEXT,
                    UNIQUE(source_identity)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    retrieval_text, content='memories', content_rowid='id', tokenize='unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, retrieval_text) VALUES (new.id, new.retrieval_text);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, retrieval_text) VALUES('delete', old.id, old.retrieval_text);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, retrieval_text) VALUES('delete', old.id, old.retrieval_text);
                    INSERT INTO memories_fts(rowid, retrieval_text) VALUES (new.id, new.retrieval_text);
                END;
                CREATE TABLE IF NOT EXISTS translation_cache (
                    query TEXT PRIMARY KEY, rewritten TEXT NOT NULL, created_at TEXT NOT NULL,
                    translation_hash TEXT, translation_version TEXT, translation_status TEXT
                );
                CREATE TABLE IF NOT EXISTS sync_state (
                    id INTEGER PRIMARY KEY CHECK(id=1), last_sync TEXT, last_error TEXT,
                    snapshot_path TEXT, restored_at TEXT
                );
                INSERT OR IGNORE INTO sync_state(id) VALUES(1);
                """
            )
            # Upgrades from the first HTTP prototype are additive and safe on a
            # restarted Space; derived translation fields never replace raw_text.
            columns = {row[1] for row in self.conn.execute("PRAGMA table_info(memories)")}
            for name in ("translation_hash", "translation_version", "translation_status"):
                if name not in columns:
                    self.conn.execute(f"ALTER TABLE memories ADD COLUMN {name} TEXT")
            cache_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(translation_cache)")}
            for name in ("translation_hash", "translation_version", "translation_status"):
                if name not in cache_columns:
                    self.conn.execute(f"ALTER TABLE translation_cache ADD COLUMN {name} TEXT")

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def ingest(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        created = updated = deduped = 0
        results = []
        with self.lock, self.conn:
            for doc in docs:
                raw = str(doc.get("raw_text", doc.get("text", "")))
                if not raw:
                    raise ValueError("raw_text/text is required")
                retrieval = str(doc.get("retrieval_text") or normalize_text(raw))
                metadata = dict(doc.get("metadata") or {})
                source_identity = str(doc.get("source_identity") or self._identity(doc, metadata, raw))
                source_version = str(doc.get("source_version", metadata.get("source_version", "")))
                content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                now = utc_now()
                row = self.conn.execute(
                    "SELECT id, content_hash, source_version FROM memories WHERE source_identity=?",
                    (source_identity,),
                ).fetchone()
                values = {k: doc.get(k, metadata.get(k)) for k in FIELDS if k not in ("content_hash", "ingested_at", "updated_at")}
                values["source_missing"] = int(bool(values.get("source_missing", False)))
                if row and row["content_hash"] == content_hash and row["source_version"] == source_version:
                    deduped += 1
                    results.append({"id": row["id"], "status": "deduped", "source_identity": source_identity})
                    continue
                if row:
                    self.conn.execute(
                        """UPDATE memories SET source_version=?, raw_text=?, retrieval_text=?, metadata_json=?,
                        source_agent=?, source_type=?, device_id=?, project=?, repo=?, worktree=?, session_id=?,
                        message_id=?, role=?, timestamp=?, source_path=?, content_hash=?, updated_at=?, content_type=?,
                        source_missing=?, agent_type=?, parent_session_id=?, agent_id=?, translation_hash=?, translation_version=?, translation_status=? WHERE id=?""",
                        (source_version, raw, retrieval, json.dumps(metadata, ensure_ascii=False),
                         values.get("source_agent"), values.get("source_type"), values.get("device_id"),
                         values.get("project"), values.get("repo"), values.get("worktree"), values.get("session_id"),
                         values.get("message_id"), values.get("role"), values.get("timestamp"), values.get("source_path"),
                         content_hash, now, values.get("content_type"), values["source_missing"], values.get("agent_type"),
                         values.get("parent_session_id"), values.get("agent_id"), values.get("translation_hash"), values.get("translation_version"), values.get("translation_status"), row["id"]),
                    )
                    updated += 1
                    results.append({"id": row["id"], "status": "updated", "source_identity": source_identity})
                else:
                    cur = self.conn.execute(
                        """INSERT INTO memories(source_identity,source_version,raw_text,retrieval_text,metadata_json,
                        source_agent,source_type,device_id,project,repo,worktree,session_id,message_id,role,timestamp,
                        source_path,content_hash,ingested_at,updated_at,content_type,source_missing,agent_type,parent_session_id,agent_id,translation_hash,translation_version,translation_status)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (source_identity, source_version, raw, retrieval, json.dumps(metadata, ensure_ascii=False),
                         values.get("source_agent"), values.get("source_type"), values.get("device_id"), values.get("project"),
                         values.get("repo"), values.get("worktree"), values.get("session_id"), values.get("message_id"),
                         values.get("role"), values.get("timestamp"), values.get("source_path"), content_hash, now, now,
                         values.get("content_type"), values["source_missing"], values.get("agent_type"),
                         values.get("parent_session_id"), values.get("agent_id"), values.get("translation_hash"), values.get("translation_version"), values.get("translation_status")),
                    )
                    created += 1
                    results.append({"id": cur.lastrowid, "status": "created", "source_identity": source_identity})
        return {"created": created, "updated": updated, "deduped": deduped, "items": results}

    @staticmethod
    def _identity(doc: dict[str, Any], metadata: dict[str, Any], raw: str) -> str:
        explicit = doc.get("source_identity") or metadata.get("source_identity")
        if explicit:
            return str(explicit)
        source_path = doc.get("source_path", metadata.get("source_path"))
        # A source file can contain many independently retrievable chunks.  Keep
        # those identities distinct while retaining source_path in metadata.
        chunk_id = next((doc.get(k, metadata.get(k)) for k in ("chunk_id", "chunk_index", "ordinal", "part", "message_id") if doc.get(k, metadata.get(k)) is not None), None)
        if source_path and chunk_id is not None:
            return f"{source_path}#chunk:{chunk_id}"
        for key in ("source_path", "message_id", "session_id", "source_id"):
            value = doc.get(key, metadata.get(key))
            if value:
                return str(value)
        stable = json.dumps({k: doc.get(k, metadata.get(k)) for k in ("source_agent", "source_type", "project", "repo", "worktree")}, sort_keys=True)
        return "generated:" + hashlib.sha256((stable + raw[:256]).encode()).hexdigest()

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["source_missing"] = bool(out.get("source_missing"))
        try:
            out["metadata"] = json.loads(out.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            out["metadata"] = {}
        return out

    def get(self, ident: str | int) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM memories WHERE id=? OR source_identity=?", (str(ident), str(ident))).fetchone()
            return self._row(row) if row else None

    def search(self, query: str, limit: int = 20, rerank: Any = None, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        filters = filters or {}
        clauses = []
        params: list[Any] = []
        for key in ("source_agent", "source_type", "project", "repo", "device_id", "role", "content_type"):
            value = filters.get(key)
            if value:
                clauses.append(f"m.{key} = ?")
                params.append(str(value))
        if filters.get("since"):
            clauses.append("m.timestamp >= ?"); params.append(str(filters["since"]))
        if filters.get("until"):
            clauses.append("m.timestamp <= ?"); params.append(str(filters["until"]))
        facet = (" AND " + " AND ".join(clauses)) if clauses else ""
        with self.lock:
            try:
                rows = self.conn.execute(
                    """SELECT m.*, bm25(memories_fts) AS score FROM memories_fts
                    JOIN memories m ON m.id=memories_fts.rowid WHERE memories_fts MATCH ?""" + facet +
                    " ORDER BY score LIMIT ?", [query, *params, limit]
                ).fetchall()
            except sqlite3.OperationalError:
                # FTS MATCH is intentionally strict; a plain substring fallback keeps recall useful.
                like = "%" + query.replace("%", "\\%") + "%"
                rows = self.conn.execute("SELECT m.*, 0.0 AS score FROM memories m WHERE (m.retrieval_text LIKE ? ESCAPE '\\' OR m.raw_text LIKE ? ESCAPE '\\')" + facet.replace("m.", "m.") + " ORDER BY m.updated_at DESC LIMIT ?", [like, like, *params, limit]).fetchall()
            if not rows and cjk_ratio(query) > 0:
                # unicode61 does not segment every CJK script consistently;
                # retain a character-level shadow fallback for Chinese recall.
                chars = list(dict.fromkeys(c for c in query if CJK_RE.match(c)))
                if chars:
                    text_clauses = " OR ".join("(m.retrieval_text LIKE ? OR m.raw_text LIKE ?)" for _ in chars)
                    text_params = [v for c in chars for v in (f"%{c}%", f"%{c}%")]
                    rows = self.conn.execute(f"SELECT m.*, 0.0 AS score FROM memories m WHERE ({text_clauses}){facet} ORDER BY m.updated_at DESC LIMIT ?", [*text_params, *params, limit]).fetchall()
            results = [self._row(r) for r in rows]
            # Optional embedding/rerank integrations can be injected by callers
            # without making the durable store depend on a model runtime.
            if rerank is not None:
                results = list(rerank(query, results))
            return results

    def sources(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute("SELECT source_identity, source_version, source_path, source_type, content_hash, updated_at, source_missing FROM memories ORDER BY updated_at DESC").fetchall()
            return [dict(r) for r in rows]

    def reindex(self) -> int:
        with self.lock, self.conn:
            self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            return int(self.conn.execute("SELECT count(*) FROM memories").fetchone()[0])

    def count(self) -> int:
        with self.lock:
            return int(self.conn.execute("SELECT count(*) FROM memories").fetchone()[0])

    def translation_get(self, query: str) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT rewritten FROM translation_cache WHERE query=?", (query,)).fetchone()
            return row[0] if row else None

    def translation_put(self, query: str, rewritten: str, status: str = "ok", translation_hash: str = "") -> None:
        with self.lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO translation_cache(query,rewritten,created_at,translation_hash,translation_version,translation_status) VALUES(?,?,?,?,?,?)", (query, rewritten, utc_now(), translation_hash, PROMPT_VERSION, status))

    def set_sync(self, **kwargs: Any) -> None:
        fields = ", ".join(f"{k}=?" for k in kwargs)
        with self.lock, self.conn:
            self.conn.execute(f"UPDATE sync_state SET {fields} WHERE id=1", tuple(kwargs.values()))

    def sync_status(self) -> dict[str, Any]:
        with self.lock:
            row = self.conn.execute("SELECT * FROM sync_state WHERE id=1").fetchone()
            return {**dict(row), "documents": self.count()}

    def snapshot(self, path: Path) -> None:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM memories ORDER BY id").fetchall()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    d = self._row(row)
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def restore(self, path: Path) -> int:
        if not path.exists():
            return 0
        docs = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    d["metadata"] = d.pop("metadata", {})
                    docs.append(d)
        result = self.ingest(docs)
        return result["created"] + result["updated"]


class Translator:
    def __init__(self, store: Store):
        self.store = store
        self.base = os.getenv("TRANSLATION_BASE_URL", "").rstrip("/")
        self.key = os.getenv("TRANSLATION_API_KEY", "")
        self.model = os.getenv("TRANSLATION_MODEL", "")
        self.threshold = float(os.getenv("TRANSLATE_CHINESE_THRESHOLD", os.getenv("TRANSLATION_CJK_THRESHOLD", "0.15")))
        self.batch_size = max(1, int(os.getenv("TRANSLATION_BATCH_SIZE", "16")))
        self.concurrency = max(1, int(os.getenv("TRANSLATION_CONCURRENCY", "4")))
        self.retries = max(1, int(os.getenv("TRANSLATION_RETRIES", "2")))
        self.mode = os.getenv("FUNES_RETRIEVAL_LANGUAGE_MODE", "auto").lower()

    def _cache_key(self, query: str) -> str:
        return json.dumps({"raw": query, "model": self.model, "prompt_version": PROMPT_VERSION}, ensure_ascii=False, sort_keys=True)

    def rewrite(self, query: str) -> str:
        raw_query = query
        query = normalize_text(query)
        if self.mode == "raw" or not self.base or not self.key or not self.model or (self.mode == "auto" and cjk_ratio(query) < self.threshold):
            return query
        cache_key = self._cache_key(raw_query)
        cached = self.store.translation_get(cache_key)
        if cached:
            return cached
        payload = json.dumps({"model": self.model, "temperature": 0, "messages": [
            {"role": "system", "content": RETRIEVAL_PROMPT},
            {"role": "user", "content": query},
        ]}).encode()
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=payload, headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=8) as response:
                    data = json.loads(response.read())
                rewritten = normalize_text(data["choices"][0]["message"]["content"])
                if rewritten:
                    self.store.translation_put(cache_key, rewritten)
                    return rewritten
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
                if attempt + 1 < self.retries:
                    time.sleep(0.2 * (2**attempt))
        return query

    def normalize_document(self, raw: str) -> tuple[str, str, str, str]:
        """Return derived retrieval text plus hash/version/status metadata."""
        normalized = normalize_text(raw)
        translation_hash = hashlib.sha256((raw + self.model + PROMPT_VERSION).encode("utf-8")).hexdigest()
        if self.mode == "raw":
            return normalized, translation_hash, PROMPT_VERSION, "skipped_raw_mode"
        if self.mode == "auto" and cjk_ratio(normalized) < self.threshold:
            return normalized, translation_hash, PROMPT_VERSION, "skipped_non_cjk"
        if not self.base or not self.key or not self.model:
            return normalized, translation_hash, PROMPT_VERSION, "fallback_no_provider"
        rewritten = self.rewrite(normalized)
        if rewritten == normalized:
            return normalized, translation_hash, PROMPT_VERSION, "fallback_provider_error"
        return " ".join(dict.fromkeys((normalized, rewritten))), translation_hash, PROMPT_VERSION, "ok"

    def normalize_many(self, raws: list[str]) -> list[tuple[str, str, str, str]]:
        """Normalize a batch with bounded concurrency; one failure never blocks ingest."""
        if not raws:
            return []
        # The provider API is intentionally called through the same cached single-
        # item path.  Batching here bounds worker fan-out and avoids an unbounded
        # request storm during historical backfill.
        out: list[tuple[str, str, str, str] | None] = [None] * len(raws)
        for begin in range(0, len(raws), self.batch_size):
            end = min(len(raws), begin + self.batch_size)
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                values = list(pool.map(self.normalize_document, raws[begin:end]))
            out[begin:end] = values
        return [x for x in out if x is not None]

    def retrieval_shadow(self, raw: str) -> str:
        """Normalize at write time while preserving raw CJK terms for fallback."""
        return self.normalize_document(raw)[0]


class SnapshotSync:
    def __init__(self, store: Store):
        self.store = store
        # Snapshot creation and Hub upload must be one serialized operation.
        # Without this lock, concurrent /sync requests can upload an older
        # snapshot after a newer one and roll the durable dataset backwards.
        self.upload_lock = threading.Lock()
        self.repo = os.getenv("FUNES_STORAGE_REPO", "")
        self.token = os.getenv("HF_TOKEN", "")
        self.filename = os.getenv("FUNES_SNAPSHOT_FILE", "funes-snapshot.jsonl")
        self.restore_failed = False
        self.restore_error = None
        self.restored = False

    @staticmethod
    def _truth(name: str, default: bool = False) -> bool:
        return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}

    def _fail_restore(self, exc: Exception) -> int:
        # A failed remote restore is a safety boundary: serving an empty/partial
        # database and later uploading it could destroy the only durable copy.
        self.restore_failed = True
        self.restore_error = type(exc).__name__
        self.store.set_sync(last_error=self.restore_error)
        return -1

    def snapshot_path(self) -> Path:
        return self.store.data_dir / self.filename

    def restore(self) -> int:
        if self.repo and self.token:
            try:
                from huggingface_hub import hf_hub_download
                downloaded = hf_hub_download(repo_id=self.repo, repo_type="dataset", filename=self.filename, token=self.token, local_dir=str(self.store.data_dir))
                restored = self.store.restore(Path(downloaded))
                self.restored = True
                return restored
            except Exception as exc:  # optional recovery must never stop serving
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 404 and self._truth("FUNES_ALLOW_EMPTY_REMOTE"):
                    self.restored = True
                    return 0
                return self._fail_restore(exc)
        local = self.snapshot_path()
        if not local.exists():
            self.restored = True
            return 0
        try:
            restored = self.store.restore(local)
            self.restored = True
            return restored
        except Exception as exc:
            return self._fail_restore(exc)

    def _secret_gate(self, path: Path) -> tuple[bool, str]:
        """Run the same TruffleHog CLI contract used by native `funes push`.

        The gate is deliberately fail-closed.  Its stdout/stderr are never
        logged because TruffleHog may include secret material in findings.
        """
        binary = os.getenv("FUNES_TRUFFLEHOG") or shutil.which("trufflehog")
        if not binary:
            return False, "secret_scanner_unavailable"
        try:
            result = subprocess.run(
                [binary, "filesystem", str(path.parent), "--json", "--no-verification",
                 "--no-update", "--fail", "--fail-on-scan-errors",
                 "--results=verified,unknown,unverified"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=int(os.getenv("FUNES_SECRET_SCAN_TIMEOUT", "120")),
            )
        except (OSError, subprocess.SubprocessError):
            return False, "secret_scan_error"
        return (result.returncode == 0), ("clean" if result.returncode == 0 else "secret_detected")

    def upload(self) -> dict[str, Any]:
        with self.upload_lock:
            if self.restore_failed:
                return {"uploaded": False, "durable": False, "reason": "restore_failed"}
            path = self.snapshot_path()
            self.store.snapshot(path)
            if not self.repo or not self.token:
                self.store.set_sync(last_sync=utc_now(), snapshot_path=str(path), last_error=None)
                if self._truth("FUNES_REQUIRE_DURABLE_ACK"):
                    return {"uploaded": False, "durable": False, "path": str(path), "reason": "HF storage not configured"}
                return {"uploaded": False, "durable": True, "path": str(path), "reason": "local durable store"}
            clean, gate_reason = self._secret_gate(path)
            if not clean:
                self.store.set_sync(last_error=gate_reason, snapshot_path=str(path))
                return {"uploaded": False, "durable": False, "path": str(path), "reason": gate_reason}
            try:
                from huggingface_hub import HfApi
                HfApi(token=self.token).upload_file(path_or_fileobj=str(path), path_in_repo=self.filename, repo_id=self.repo, repo_type="dataset", commit_message="funes snapshot")
                self.store.set_sync(last_sync=utc_now(), snapshot_path=str(path), last_error=None)
                return {"uploaded": True, "durable": True, "path": str(path)}
            except Exception as exc:
                self.store.set_sync(last_error=type(exc).__name__, snapshot_path=str(path))
                return {"uploaded": False, "durable": False, "path": str(path), "reason": type(exc).__name__}


class App:
    def __init__(self):
        # Free Gradio Spaces do not expose /data.  The Hub snapshot remains the
        # durable source of truth; operators can override this with a writable
        # mounted volume when one is available.
        self.store = Store(os.getenv("FUNES_DATA_DIR", "/tmp/funes-data"))
        self.translator = Translator(self.store)
        self.syncer = SnapshotSync(self.store)
        self.restore_result = self.syncer.restore()
        # FTS5 external-content tables need an explicit rebuild after restoring
        # rows from a JSONL snapshot (the insert triggers only cover new writes).
        self.store.reindex()

    def close(self):
        self.store.close()


PROTECTED = {"/ingest", "/search", "/recall", "/get", "/sources", "/sync/status", "/reindex", "/sync"}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FunesHTTP/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            # Never log request bodies, query text, or authorization headers.
            return

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            token = os.getenv("FUNES_AUTH_TOKEN") or os.getenv("FUNES_API_TOKEN", "")
            if not token:
                self._json(503, {"error": "auth_not_configured"})
                return False
            supplied = self.headers.get("Authorization", "")
            if supplied != "Bearer " + token:
                self._json(401, {"error": "unauthorized"})
                return False
            return True

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 20_000_000:
                raise ValueError("request too large")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("JSON object required")
            return data

        @staticmethod
        def _public(item: dict[str, Any]) -> dict[str, Any]:
            # Raw text is the source of truth.  Derived retrieval text is opt-in
            # for diagnostics and never replaces the context an agent receives.
            if os.getenv("RETURN_RETRIEVAL_TEXT", "false").lower() not in {"1", "true", "yes"}:
                item = dict(item)
                item.pop("retrieval_text", None)
            return item

        def do_GET(self) -> None:
            from urllib.parse import parse_qs, urlparse
            route = urlparse(self.path).path
            if route == "/health":
                return self._json(200, {"status": "ok", "service": "funes"})
            if route == "/ready":
                try:
                    if app.syncer.restore_failed:
                        return self._json(503, {"status": "not_ready", "error": "restore_failed"})
                    count = app.store.count()
                    return self._json(200, {"status": "ready", "documents": count, "restored": app.restore_result})
                except Exception as exc:
                    return self._json(503, {"status": "not_ready", "error": type(exc).__name__})
            if route in PROTECTED and self._authorized():
                if route == "/sources":
                    return self._json(200, {"sources": app.store.sources()})
                if route == "/sync/status":
                    return self._json(200, app.store.sync_status())
                if route == "/get":
                    ident = parse_qs(urlparse(self.path).query).get("id", [""])[0]
                    item = app.store.get(ident)
                    return self._json(200 if item else 404, self._public(item) if item else {"error": "not_found"})
            return self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path not in PROTECTED or not self._authorized():
                if self.path not in PROTECTED:
                    return self._json(404, {"error": "not_found"})
                return
            try:
                body = self._body()
                if self.path == "/ingest":
                    if app.syncer.restore_failed:
                        return self._json(503, {"error": "restore_failed", "durable": False})
                    docs = body.get("documents", body.get("records", body.get("items")))
                    if docs is None:
                        docs = [body]
                    if not isinstance(docs, list):
                        raise ValueError("documents must be a list")
                    # Keep raw_text untouched; retrieval_text is a normalized,
                    # optional translation shadow used solely by FTS.
                    prepared = []
                    normalized = app.translator.normalize_many([
                        str(doc.get("raw_text", doc.get("text", ""))) for doc in docs
                    ])
                    for doc, derived in zip(docs, normalized):
                        item = dict(doc)
                        shadow, translation_hash, translation_version, translation_status = derived
                        if not item.get("retrieval_text"):
                            item["retrieval_text"] = shadow
                        item.setdefault("translation_hash", translation_hash)
                        item.setdefault("translation_version", translation_version)
                        item.setdefault("translation_status", translation_status)
                        prepared.append(item)
                    result = app.store.ingest(prepared)
                    result["accepted"] = result["created"] + result["updated"] + result["deduped"]
                    # A remote caller must not receive an ACK that can be lost
                    # between SQLite commit and HF persistence.  The upload is
                    # idempotent; retrying the same source identities is safe.
                    result["sync"] = app.syncer.upload()
                    if not result["sync"].get("durable"):
                        return self._json(503, {"error": "durability_pending", **result})
                    return self._json(200, result)
                if self.path in ("/search", "/recall"):
                    query = normalize_text(str(body.get("query", body.get("q", ""))))
                    if not query:
                        raise ValueError("query is required")
                    rewritten = app.translator.rewrite(query)
                    filters = {key: body.get(key) for key in ("source_agent", "source_type", "project", "repo", "device_id", "role", "content_type", "since", "until") if body.get(key)}
                    hits = app.store.search(rewritten, int(body.get("limit", 20)), filters=filters)
                    if rewritten != query and not hits:
                        hits = app.store.search(query, int(body.get("limit", 20)), filters=filters)
                    return self._json(200, {"query": query, "rewritten_query": rewritten, "results": [self._public(x) for x in hits]})
                if self.path == "/get":
                    item = app.store.get(body.get("id", body.get("source_identity", "")))
                    return self._json(200 if item else 404, self._public(item) if item else {"error": "not_found"})
                if self.path == "/reindex":
                    return self._json(200, {"reindexed": app.store.reindex()})
                if self.path == "/sync":
                    result = app.syncer.upload()
                    return self._json(200 if result.get("durable") else 503, result)
                if self.path == "/sync/status":
                    return self._json(200, app.store.sync_status())
            except (ValueError, json.JSONDecodeError) as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:
                return self._json(500, {"error": type(exc).__name__})

    return Handler


def serve(host: str = "0.0.0.0", port: int = 7860) -> None:
    app = App()
    server = ThreadingHTTPServer((host, port), make_handler(app))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    serve(os.getenv("FUNES_HOST", "0.0.0.0"), int(os.getenv("PORT", "7860")))
