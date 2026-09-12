#!/usr/bin/env python3
"""Small durable HTTP service for remote Funes memory retrieval.

The service deliberately uses only the Python standard library at runtime.  The
optional ``huggingface_hub`` package is used for snapshot transport when the
corresponding environment variables are configured.
"""
from __future__ import annotations

import hashlib
import gzip
import io
import json
import os
import re
import sqlite3
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import uuid
import urllib.error
import urllib.request
from collections import Counter
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
    "translation_hash", "translation_version", "translation_status",
    "retrieval_updated_at", "native_index_version", "native_index_status",
    "native_indexed_at", "native_index_error", "retrieval_generation",
    "native_generation",
)
CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
PROMPT_VERSION = "funes-retrieval-v2"
QUERY_PROMPT_VERSION = "funes-query-retrieval-v1"
ENCRYPTED_MAGIC = b"FUNES-SOURCE-V1\0"
ENCRYPTED_AAD = b"funes-source-snapshot-v1"
REINDEX_SCOPES = {"retrieval_text", "all"}
NATIVE_SESSION_TYPES = {
    "session", "codex", "codex_session", "pi", "pi_session",
    "claude", "claude_session",
}
LOW_VALUE_CONTENT_TYPES = {"tool_call", "tool_result", "shell_output", "progress"}
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
QUERY_RETRIEVAL_PROMPT = """You are a retrieval query normalization engine.
Rewrite the user's retrieval query into short English text optimized for semantic retrieval.
Apply the same preservation rules used for document normalization:
1. Preserve all technical entities verbatim.
2. Preserve code, commands, URLs, file paths, API names, environment variables, model names, product names, repository names, IDs, error messages, quoted strings and version numbers exactly.
3. Never translate identifiers such as previous_response_id, Codex, Northflank, Funes, MCP, CPA, Gemini, Claude, OpenAI.
4. Preserve the user's intent, constraints and factual details.
5. This is a retrieval query: only rewrite its intent. Do not answer the question.
6. Do not add configuration values, solutions, recommendations, facts, numbers or versions that are not present in the query.
7. Keep the output concise and never invent facts.
8. Output retrieval text only."""
TECHNICAL_ENTITY_RE = re.compile(
    r"https?://\S+|(?:[~/]|\.{1,2}/)[^\s]+|[A-Za-z][A-Za-z0-9_.*:/-]*"
)
ARABIC_NUMBER_RE = re.compile(r"\d+(?:\.\d+)*")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def cjk_ratio(value: str) -> float:
    if not value:
        return 0.0
    chars = [c for c in value if not c.isspace()]
    return sum(bool(CJK_RE.match(c)) for c in chars) / max(1, len(chars))


def expanded_candidate_limit(limit: int) -> int:
    """Bound per-route recall while leaving room for cross-route consensus."""
    limit = max(1, int(limit))
    return min(100, limit * 3)


def result_identity(item: dict[str, Any]) -> str:
    for name in ("source_identity", "session_id", "id"):
        value = str(item.get(name, "")).strip()
        if value:
            return value
    return ""


def stable_rrf(
    rankings: list[list[dict[str, Any]]],
    limit: int,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """Fuse rankings with stable tie-breaking and cross-route identity dedupe."""
    scores: dict[str, float] = {}
    first_seen: dict[str, tuple[int, int, int]] = {}
    items: dict[str, dict[str, Any]] = {}
    sequence = 0
    for list_index, ranking in enumerate(rankings):
        seen_in_ranking = set()
        for rank, item in enumerate(ranking, start=1):
            identity = result_identity(item)
            key = identity or f"anonymous:{list_index}:{rank}"
            if key in seen_in_ranking:
                continue
            seen_in_ranking.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
            if key not in items:
                items[key] = item
                first_seen[key] = (list_index, rank, sequence)
                sequence += 1
    ordered = sorted(items, key=lambda key: (-scores[key], *first_seen[key]))
    return [items[key] for key in ordered[:limit]]


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
                    ingested_at TEXT NOT NULL, updated_at TEXT NOT NULL, retrieval_updated_at TEXT,
                    content_type TEXT,
                    source_missing INTEGER NOT NULL DEFAULT 0, agent_type TEXT,
                    parent_session_id TEXT, agent_id TEXT,
                    translation_hash TEXT, translation_version TEXT, translation_status TEXT,
                    native_index_version TEXT, native_index_status TEXT,
                    native_indexed_at TEXT, native_index_error TEXT,
                    retrieval_generation INTEGER NOT NULL DEFAULT 0,
                    native_generation INTEGER NOT NULL DEFAULT 0,
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
                CREATE TABLE IF NOT EXISTS reindex_controls (
                    generation INTEGER PRIMARY KEY,
                    scope TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    row_cursor INTEGER NOT NULL DEFAULT 0,
                    applied_at TEXT
                );
                """
            )
            # Upgrades from the first HTTP prototype are additive and safe on a
            # restarted Space; derived translation fields never replace raw_text.
            columns = {row[1] for row in self.conn.execute("PRAGMA table_info(memories)")}
            for name in (
                "translation_hash",
                "translation_version",
                "translation_status",
                "retrieval_updated_at",
                "native_index_version",
                "native_index_status",
                "native_indexed_at",
                "native_index_error",
            ):
                if name not in columns:
                    self.conn.execute(f"ALTER TABLE memories ADD COLUMN {name} TEXT")
            for name in ("retrieval_generation", "native_generation"):
                if name not in columns:
                    self.conn.execute(
                        f"ALTER TABLE memories ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
                    )
            cache_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(translation_cache)")}
            for name in ("translation_hash", "translation_version", "translation_status"):
                if name not in cache_columns:
                    self.conn.execute(f"ALTER TABLE translation_cache ADD COLUMN {name} TEXT")
            control_columns = {
                row[1] for row in self.conn.execute("PRAGMA table_info(reindex_controls)")
            }
            if "row_cursor" not in control_columns:
                self.conn.execute(
                    """ALTER TABLE reindex_controls ADD COLUMN row_cursor
                    INTEGER NOT NULL DEFAULT 0"""
                )

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
                incoming_updated = doc.get("updated_at", metadata.get("updated_at")) or now
                row = self.conn.execute(
                    """SELECT id, content_hash, source_version, updated_at, retrieval_text,
                    translation_hash, translation_version, translation_status,
                    retrieval_updated_at, native_index_version, native_index_status,
                    native_indexed_at, native_index_error, source_missing,
                    retrieval_generation, native_generation
                    FROM memories WHERE source_identity=?""",
                    (source_identity,),
                ).fetchone()
                values = {k: doc.get(k, metadata.get(k)) for k in FIELDS if k not in ("content_hash", "ingested_at", "updated_at")}
                values["source_missing"] = int(bool(values.get("source_missing", False)))
                values["retrieval_generation"] = int(values.get("retrieval_generation") or 0)
                values["native_generation"] = int(values.get("native_generation") or 0)
                if row and row["content_hash"] == content_hash and row["source_version"] == source_version:
                    derived_supplied = any(
                        name in doc
                        for name in (
                            "retrieval_text",
                            "translation_hash",
                            "translation_version",
                            "translation_status",
                            "retrieval_updated_at",
                            "native_index_version",
                            "native_index_status",
                            "native_indexed_at",
                            "native_index_error",
                            "retrieval_generation",
                            "native_generation",
                            "source_missing",
                        )
                    )
                    incoming_status = values.get("translation_status") if "translation_status" in doc else row["translation_status"]
                    current_status = row["translation_status"]
                    retryable = {"pending_provider", "fallback_provider_error", "fallback_no_provider"}
                    final = {
                        "ok",
                        "skipped_raw_mode",
                        "skipped_non_cjk",
                        "skipped_native_session",
                        "skipped_low_value",
                    }
                    would_regress = current_status in final and incoming_status in retryable
                    incoming_retrieval = retrieval if "retrieval_text" in doc else row["retrieval_text"]
                    incoming_translation_hash = values.get("translation_hash") if "translation_hash" in doc else row["translation_hash"]
                    incoming_translation_version = values.get("translation_version") if "translation_version" in doc else row["translation_version"]
                    incoming_retrieval_updated_at = (
                        values.get("retrieval_updated_at")
                        if "retrieval_updated_at" in doc
                        else row["retrieval_updated_at"]
                    )
                    # Legacy deltas have no generation and therefore belong to
                    # generation zero; they must not overwrite a queued reset.
                    incoming_retrieval_generation = values["retrieval_generation"]
                    would_regress = (
                        would_regress
                        and incoming_retrieval_generation <= row["retrieval_generation"]
                    )
                    if incoming_retrieval_generation < row["retrieval_generation"]:
                        incoming_retrieval = row["retrieval_text"]
                        incoming_translation_hash = row["translation_hash"]
                        incoming_translation_version = row["translation_version"]
                        incoming_status = row["translation_status"]
                        incoming_retrieval_updated_at = row["retrieval_updated_at"]
                        incoming_retrieval_generation = row["retrieval_generation"]
                    translation_changed = (
                        incoming_retrieval,
                        incoming_translation_hash,
                        incoming_translation_version,
                        incoming_status,
                    ) != (
                        row["retrieval_text"],
                        row["translation_hash"],
                        row["translation_version"],
                        current_status,
                    )
                    incoming_source_missing = (
                        values["source_missing"]
                        if "source_missing" in doc
                        else row["source_missing"]
                    )
                    source_missing_changed = incoming_source_missing != row["source_missing"]
                    incoming_native = (
                        values.get("native_index_version") if "native_index_version" in doc else row["native_index_version"],
                        values.get("native_index_status") if "native_index_status" in doc else row["native_index_status"],
                        values.get("native_indexed_at") if "native_indexed_at" in doc else row["native_indexed_at"],
                        values.get("native_index_error") if "native_index_error" in doc else row["native_index_error"],
                    )
                    incoming_native_generation = values["native_generation"]
                    if incoming_native_generation < row["native_generation"]:
                        incoming_native = (
                            row["native_index_version"],
                            row["native_index_status"],
                            row["native_indexed_at"],
                            row["native_index_error"],
                        )
                        incoming_native_generation = row["native_generation"]
                    if source_missing_changed and not any(
                        name in doc
                        for name in (
                            "native_index_version",
                            "native_index_status",
                            "native_indexed_at",
                            "native_index_error",
                        )
                    ):
                        incoming_native = (None, None, None, None)
                    if (
                        row["native_index_status"] in {"indexed", "held_secret"}
                        and incoming_native[1] not in {"indexed", "held_secret"}
                        and not translation_changed
                        and not source_missing_changed
                    ):
                        # Immutable Hub deltas restore in filename order. An
                        # older raw/retry delta for this exact source+shadow must
                        # not regress a terminal native checkpoint.
                        incoming_native = (
                            row["native_index_version"],
                            row["native_index_status"],
                            row["native_indexed_at"],
                            row["native_index_error"],
                        )
                    incoming_derived = (
                        incoming_retrieval,
                        incoming_translation_hash,
                        incoming_translation_version,
                        incoming_status,
                        incoming_retrieval_updated_at,
                        incoming_retrieval_generation,
                        incoming_source_missing,
                        *incoming_native,
                        incoming_native_generation,
                    )
                    current_derived = (
                        row["retrieval_text"],
                        row["translation_hash"],
                        row["translation_version"],
                        current_status,
                        row["retrieval_updated_at"],
                        row["retrieval_generation"],
                        row["source_missing"],
                        row["native_index_version"],
                        row["native_index_status"],
                        row["native_indexed_at"],
                        row["native_index_error"],
                        row["native_generation"],
                    )
                    if derived_supplied and not would_regress and incoming_derived != current_derived:
                        # Raw revisions and derived retrieval shadows have separate
                        # lifecycles.  A reconciled shadow must update in place even
                        # when the source bytes/version are unchanged.
                        self.conn.execute(
                            """UPDATE memories SET retrieval_text=?, translation_hash=?,
                            translation_version=?, translation_status=?, retrieval_updated_at=?,
                            retrieval_generation=?,
                            source_missing=?,
                            native_index_version=?, native_index_status=?, native_indexed_at=?,
                            native_index_error=?, native_generation=? WHERE id=?""",
                            (*incoming_derived, row["id"]),
                        )
                        updated += 1
                        results.append({"id": row["id"], "status": "derived_updated", "source_identity": source_identity})
                    else:
                        deduped += 1
                        results.append({"id": row["id"], "status": "deduped", "source_identity": source_identity})
                    continue
                if row and self._older(incoming_updated, row["updated_at"]):
                    # Delta files can arrive out of order after a retry.  Never
                    # let an older local version roll a durable source back.
                    deduped += 1
                    results.append({"id": row["id"], "status": "stale", "source_identity": source_identity})
                    continue
                if row:
                    self.conn.execute(
                        """UPDATE memories SET source_version=?, raw_text=?, retrieval_text=?, metadata_json=?,
                        source_agent=?, source_type=?, device_id=?, project=?, repo=?, worktree=?, session_id=?,
                        message_id=?, role=?, timestamp=?, source_path=?, content_hash=?, updated_at=?, retrieval_updated_at=?, content_type=?,
                        source_missing=?, agent_type=?, parent_session_id=?, agent_id=?, translation_hash=?, translation_version=?, translation_status=?,
                        native_index_version=?, native_index_status=?, native_indexed_at=?, native_index_error=?,
                        retrieval_generation=?, native_generation=? WHERE id=?""",
                        (source_version, raw, retrieval, json.dumps(metadata, ensure_ascii=False),
                         values.get("source_agent"), values.get("source_type"), values.get("device_id"),
                         values.get("project"), values.get("repo"), values.get("worktree"), values.get("session_id"),
                         values.get("message_id"), values.get("role"), values.get("timestamp"), values.get("source_path"),
                         content_hash, incoming_updated, values.get("retrieval_updated_at"), values.get("content_type"), values["source_missing"], values.get("agent_type"),
                         values.get("parent_session_id"), values.get("agent_id"), values.get("translation_hash"), values.get("translation_version"), values.get("translation_status"),
                         values.get("native_index_version"), values.get("native_index_status"), values.get("native_indexed_at"), values.get("native_index_error"),
                         values["retrieval_generation"], values["native_generation"], row["id"]),
                    )
                    updated += 1
                    results.append({"id": row["id"], "status": "updated", "source_identity": source_identity})
                else:
                    cur = self.conn.execute(
                        """INSERT INTO memories(source_identity,source_version,raw_text,retrieval_text,metadata_json,
                        source_agent,source_type,device_id,project,repo,worktree,session_id,message_id,role,timestamp,
                        source_path,content_hash,ingested_at,updated_at,retrieval_updated_at,content_type,source_missing,agent_type,parent_session_id,agent_id,translation_hash,translation_version,translation_status,
                        native_index_version,native_index_status,native_indexed_at,native_index_error,
                        retrieval_generation,native_generation)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (source_identity, source_version, raw, retrieval, json.dumps(metadata, ensure_ascii=False),
                         values.get("source_agent"), values.get("source_type"), values.get("device_id"), values.get("project"),
                         values.get("repo"), values.get("worktree"), values.get("session_id"), values.get("message_id"),
                         values.get("role"), values.get("timestamp"), values.get("source_path"), content_hash, values.get("ingested_at") or now, incoming_updated, values.get("retrieval_updated_at"),
                         values.get("content_type"), values["source_missing"], values.get("agent_type"),
                         values.get("parent_session_id"), values.get("agent_id"), values.get("translation_hash"), values.get("translation_version"), values.get("translation_status"),
                         values.get("native_index_version"), values.get("native_index_status"), values.get("native_indexed_at"), values.get("native_index_error"),
                         values["retrieval_generation"], values["native_generation"]),
                    )
                    created += 1
                    results.append({"id": cur.lastrowid, "status": "created", "source_identity": source_identity})
        return {"created": created, "updated": updated, "deduped": deduped, "items": results}

    @staticmethod
    def _older(candidate: Any, current: Any) -> bool:
        if not candidate or not current:
            return False
        def key(value: Any):
            text = str(value)
            try:
                return (0, float(text))
            except ValueError:
                return (1, text)
        return key(candidate) < key(current)

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
            row = self.conn.execute(
                "SELECT * FROM memories WHERE source_identity=? ORDER BY id LIMIT 1",
                (str(ident),),
            ).fetchone()
            if row is None:
                row = self.conn.execute("SELECT * FROM memories WHERE id=?", (str(ident),)).fetchone()
            return self._row(row) if row else None

    def get_many(self, identities: list[str]) -> list[dict[str, Any]]:
        """Return canonical stored rows without exceeding SQLite's bind limit."""
        unique = list(dict.fromkeys(str(value) for value in identities if value))
        found: dict[str, dict[str, Any]] = {}
        with self.lock:
            for begin in range(0, len(unique), 500):
                current = unique[begin : begin + 500]
                placeholders = ",".join("?" for _ in current)
                rows = self.conn.execute(
                    f"SELECT * FROM memories WHERE source_identity IN ({placeholders})",
                    current,
                ).fetchall()
                for row in rows:
                    item = self._row(row)
                    found[item["source_identity"]] = item
        return [found[value] for value in unique if value in found]

    def pending_translations(self, limit: int) -> list[dict[str, Any]]:
        """Return a bounded restart-safe reconciliation batch."""
        with self.lock:
            rows = self.conn.execute(
                """SELECT * FROM memories WHERE translation_status='pending_provider'
                AND COALESCE(native_index_status, '') != 'waiting_durability'
                ORDER BY updated_at, id LIMIT ?""",
                (max(1, int(limit)),),
            ).fetchall()
            return [self._row(row) for row in rows]

    def latest_reindex_generation(self) -> int:
        with self.lock:
            return int(
                self.conn.execute(
                    """SELECT max(value) FROM (
                    SELECT COALESCE(max(generation), 0) AS value FROM reindex_controls
                    UNION ALL SELECT COALESCE(max(retrieval_generation), 0) FROM memories
                    UNION ALL SELECT COALESCE(max(native_generation), 0) FROM memories
                    )"""
                ).fetchone()[0]
            )

    def next_reindex_control(self, scope: str) -> dict[str, Any]:
        if scope not in REINDEX_SCOPES:
            raise ValueError("scope must be retrieval_text or all")
        return {
            "_funes_record": "reindex_control",
            "generation": self.latest_reindex_generation() + 1,
            "scope": scope,
            "created_at": utc_now(),
        }

    def record_reindex_control(self, control: dict[str, Any]) -> bool:
        scope = str(control.get("scope", ""))
        generation = int(control.get("generation", 0))
        created_at = str(control.get("created_at") or utc_now())
        if scope not in REINDEX_SCOPES or generation < 1:
            raise ValueError("invalid reindex control")
        with self.lock, self.conn:
            cursor = self.conn.execute(
                """INSERT OR IGNORE INTO reindex_controls(
                generation, scope, created_at, row_cursor, applied_at
                ) VALUES(?,?,?,0,NULL)""",
                (generation, scope, created_at),
            )
            return cursor.rowcount == 1

    @staticmethod
    def _provider_reindex_eligible(item: dict[str, Any]) -> bool:
        source_type = str(item.get("source_type") or "").lower()
        content_type = str(item.get("content_type") or "").lower()
        return (
            source_type not in NATIVE_SESSION_TYPES
            and content_type not in LOW_VALUE_CONTENT_TYPES
        )

    @staticmethod
    def _canonical_reindex_eligible(item: dict[str, Any]) -> bool:
        return (
            str(item.get("source_type") or "").lower() not in NATIVE_SESSION_TYPES
            and str(item.get("content_type") or "").lower()
            not in LOW_VALUE_CONTENT_TYPES
        )

    def apply_pending_reindex_controls(self, batch_size: int = 500) -> dict[str, int]:
        """Apply one restart-safe row batch without changing raw source truth."""
        applied = retrieval_reset = native_reset = scanned = 0
        with self.lock, self.conn:
            control = self.conn.execute(
                """SELECT generation, scope, row_cursor FROM reindex_controls
                WHERE applied_at IS NULL ORDER BY generation LIMIT 1"""
            ).fetchone()
            if control is None:
                return {
                    "applied": 0,
                    "scanned": 0,
                    "retrieval_reset": 0,
                    "native_reset": 0,
                }
            generation = int(control["generation"])
            scope = str(control["scope"])
            row_cursor = int(control["row_cursor"] or 0)
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE id>? ORDER BY id LIMIT ?",
                (row_cursor, max(1, int(batch_size))),
            ).fetchall()
            for row in rows:
                item = dict(row)
                retrieval_generation = int(item.get("retrieval_generation") or 0)
                native_generation = int(item.get("native_generation") or 0)
                retrieval_values = (
                    item.get("retrieval_text"), item.get("translation_hash"),
                    item.get("translation_version"), item.get("translation_status"),
                    item.get("retrieval_updated_at"),
                )
                native_values = (
                    item.get("native_index_version"), item.get("native_index_status"),
                    item.get("native_indexed_at"), item.get("native_index_error"),
                )
                if generation > retrieval_generation:
                    retrieval_generation = generation
                    if self._provider_reindex_eligible(item):
                        retrieval_values = (
                            item["raw_text"], None, PROMPT_VERSION,
                            "pending_provider", None,
                        )
                        retrieval_reset += 1
                        if generation > native_generation:
                            native_generation = generation
                            if item.get("native_index_status") != "waiting_durability":
                                native_values = (None, None, None, None)
                                native_reset += 1
                if scope == "all" and generation > native_generation:
                    native_generation = generation
                    if (
                        self._canonical_reindex_eligible(item)
                        and item.get("native_index_status") != "waiting_durability"
                    ):
                        native_values = (None, None, None, None)
                        native_reset += 1
                self.conn.execute(
                    """UPDATE memories SET retrieval_text=?, translation_hash=?,
                    translation_version=?, translation_status=?, retrieval_updated_at=?,
                    retrieval_generation=?, native_index_version=?, native_index_status=?,
                    native_indexed_at=?, native_index_error=?, native_generation=?
                    WHERE id=?""",
                    (
                        *retrieval_values, retrieval_generation, *native_values,
                        native_generation, row["id"],
                    ),
                )
            scanned = len(rows)
            next_cursor = int(rows[-1]["id"]) if rows else row_cursor
            has_more = self.conn.execute(
                "SELECT 1 FROM memories WHERE id>? LIMIT 1", (next_cursor,)
            ).fetchone()
            if has_more is None:
                self.conn.execute(
                    """UPDATE reindex_controls SET row_cursor=?, applied_at=?
                    WHERE generation=?""",
                    (next_cursor, utc_now(), generation),
                )
                applied += 1
            else:
                self.conn.execute(
                    "UPDATE reindex_controls SET row_cursor=? WHERE generation=?",
                    (next_cursor, generation),
                )
        return {
            "applied": applied,
            "scanned": scanned,
            "retrieval_reset": retrieval_reset,
            "native_reset": native_reset,
        }

    def drain_reindex_controls(self, batch_size: int = 500) -> dict[str, int]:
        """Drain controls using bounded transactions; intended for restore only."""
        totals = {"applied": 0, "scanned": 0, "retrieval_reset": 0, "native_reset": 0}
        while True:
            result = self.apply_pending_reindex_controls(batch_size)
            for name in totals:
                totals[name] += result[name]
            if not result["scanned"] and not result["applied"]:
                return totals

    def reset_reindex_control_cursors(self) -> None:
        """Restart every control after a complete unordered Hub restore."""
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE reindex_controls SET row_cursor=0, applied_at=NULL"
            )

    def canonical_index_candidates(self, limit: int) -> list[dict[str, Any]]:
        """Return final non-session retrieval shadows for native reconciliation."""
        with self.lock:
            rows = self.conn.execute(
                """SELECT * FROM memories
                WHERE translation_status IN ('ok','skipped_non_cjk','skipped_raw_mode')
                AND lower(COALESCE(source_type, '')) NOT IN
                    ('session','codex','codex_session','pi','pi_session','claude','claude_session')
                AND lower(COALESCE(content_type, '')) NOT IN
                    ('tool_call','tool_result','shell_output','progress')
                AND COALESCE(native_index_status, '') NOT IN
                    ('indexed','held_secret','waiting_durability')
                ORDER BY COALESCE(retrieval_updated_at, updated_at), id LIMIT ?""",
                (max(1, int(limit)),),
            ).fetchall()
            return [self._row(row) for row in rows]

    def update_native_index(self, updates: list[dict[str, Any]]) -> int:
        """Persist derived native state only when the raw source revision still matches."""
        changed = 0
        with self.lock, self.conn:
            for item in updates:
                cursor = self.conn.execute(
                    """UPDATE memories SET native_index_version=?, native_index_status=?,
                    native_indexed_at=?, native_index_error=?
                    WHERE source_identity=? AND source_version=? AND content_hash=?
                    AND native_generation=?""",
                    (
                        item.get("native_index_version"),
                        item.get("native_index_status"),
                        item.get("native_indexed_at"),
                        item.get("native_index_error"),
                        str(item.get("source_identity", "")),
                        str(item.get("source_version", "")),
                        str(item.get("content_hash", "")),
                        int(item.get("native_generation") or 0),
                    ),
                )
                changed += cursor.rowcount
        return changed

    def mark_translations_pending(self, documents: list[dict[str, Any]]) -> None:
        """Requeue derived writes whose encrypted delta did not become durable."""
        with self.lock, self.conn:
            for item in documents:
                self.conn.execute(
                    """UPDATE memories SET retrieval_text=raw_text, translation_status='pending_provider'
                    WHERE source_identity=? AND content_hash=? AND source_version=?
                    AND retrieval_text=? AND translation_status=?
                    AND retrieval_generation=?""",
                    (
                        item.get("source_identity"),
                        item.get("content_hash"),
                        str(item.get("source_version", "")),
                        item.get("retrieval_text"),
                        item.get("translation_status"),
                        int(item.get("retrieval_generation") or 0),
                    ),
                )

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
        if filters.get("source_missing") is not None:
            clauses.append("m.source_missing = ?")
            params.append(int(bool(filters["source_missing"])))
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

    def translation_put(self, query: str, rewritten: str, status: str = "ok", translation_hash: str = "", translation_version: str = PROMPT_VERSION) -> None:
        with self.lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO translation_cache(query,rewritten,created_at,translation_hash,translation_version,translation_status) VALUES(?,?,?,?,?,?)", (query, rewritten, utc_now(), translation_hash, translation_version, status))

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
            cache = self.conn.execute(
                "SELECT query,rewritten,created_at,translation_hash,translation_version,translation_status "
                "FROM translation_cache ORDER BY query"
            ).fetchall()
            controls = self.conn.execute(
                """SELECT generation,scope,created_at FROM reindex_controls
                ORDER BY generation"""
            ).fetchall()
            path.parent.mkdir(parents=True, exist_ok=True)
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "wt", encoding="utf-8") as f:
                for row in rows:
                    d = self._row(row)
                    d["_funes_record"] = "memory"
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
                for row in cache:
                    d = dict(row)
                    d["_funes_record"] = "translation_cache"
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
                for row in controls:
                    d = dict(row)
                    d["_funes_record"] = "reindex_control"
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def iter_documents(self, batch_size: int = 500):
        """Yield durable rows in bounded batches for snapshot/delta transport."""
        batch: list[dict[str, Any]] = []
        with self.lock:
            cursor = self.conn.execute("SELECT * FROM memories ORDER BY id")
            for row in cursor:
                batch.append(self._row(row))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def restore_documents(
        self, documents: Any, batch_size: int = 500, *, apply_controls: bool = True
    ) -> int:
        """Restore a stream without materialising a multi-gigabyte snapshot."""
        total = 0
        batch: list[dict[str, Any]] = []
        for item in documents:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            record_type = item.pop("_funes_record", "memory")
            if record_type == "reindex_control":
                if batch:
                    result = self.ingest(batch)
                    total += result["created"] + result["updated"]
                    batch = []
                self.record_reindex_control(item)
                continue
            if record_type == "translation_cache":
                query = str(item.get("query", ""))
                rewritten = str(item.get("rewritten", ""))
                if query and rewritten:
                    self.translation_put(
                        query,
                        rewritten,
                        status=str(item.get("translation_status") or "ok"),
                        translation_hash=str(item.get("translation_hash") or ""),
                        translation_version=str(item.get("translation_version") or PROMPT_VERSION),
                    )
                continue
            if record_type != "memory":
                continue
            item["metadata"] = item.pop("metadata", item.pop("metadata_json", {}))
            batch.append(item)
            if len(batch) >= batch_size:
                result = self.ingest(batch)
                total += result["created"] + result["updated"]
                batch = []
        if batch:
            result = self.ingest(batch)
            total += result["created"] + result["updated"]
        if apply_controls:
            self.drain_reindex_controls(batch_size)
        return total

    def restore(self, path: Path, *, apply_controls: bool = True) -> int:
        if not path.exists():
            return 0
        opener = gzip.open if path.name.endswith(".gz") else open
        def rows():
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        return self.restore_documents(rows(), apply_controls=apply_controls)


class Translator:
    def __init__(self, store: Store):
        self.store = store
        self.base = os.getenv("TRANSLATION_BASE_URL", "").rstrip("/")
        self.key = os.getenv("TRANSLATION_API_KEY", "")
        self.model = os.getenv("TRANSLATION_MODEL", "")
        self.threshold = float(os.getenv("TRANSLATE_CHINESE_THRESHOLD", os.getenv("TRANSLATION_CJK_THRESHOLD", "0.15")))
        self.batch_size = max(1, int(os.getenv("TRANSLATION_BATCH_SIZE", "16")))
        self.concurrency = max(1, int(os.getenv("TRANSLATION_CONCURRENCY", "4")))
        # Kept for environment compatibility; HTTP ingest never consumes this
        # budget because all provider work belongs to the background reconciler.
        self.max_per_ingest = max(0, int(os.getenv("TRANSLATION_MAX_PER_INGEST", "0")))
        self.retries = max(1, int(os.getenv("TRANSLATION_RETRIES", "2")))
        self.timeout = max(1.0, float(os.getenv("TRANSLATION_TIMEOUT", "12")))
        self.query_max_tokens = max(1, int(os.getenv("TRANSLATION_QUERY_MAX_TOKENS", "128")))
        self.mode = os.getenv("FUNES_RETRIEVAL_LANGUAGE_MODE", "auto").lower()
        self._provider_lock = threading.Lock()
        self._provider_disabled_until = 0.0

    def _cache_key(self, query: str, prompt_version: str = PROMPT_VERSION) -> str:
        return hashlib.sha256(
            (query + self.model + prompt_version).encode("utf-8")
        ).hexdigest()

    def rewrite(self, query: str) -> str:
        return self._rewrite(query, RETRIEVAL_PROMPT, PROMPT_VERSION)

    def rewrite_query(self, query: str) -> str:
        return self._rewrite(
            query,
            QUERY_RETRIEVAL_PROMPT,
            QUERY_PROMPT_VERSION,
            max_tokens=self.query_max_tokens,
            validate=self._valid_query_rewrite,
        )

    @staticmethod
    def _valid_query_rewrite(raw_query: str, rewritten: str) -> bool:
        max_length = max(256, len(re.sub(r"\s+", "", raw_query)) * 6)
        if len(rewritten) > max_length:
            return False
        entities = set(TECHNICAL_ENTITY_RE.findall(raw_query))
        if any(entity not in rewritten for entity in entities):
            return False
        source_numbers = Counter(ARABIC_NUMBER_RE.findall(raw_query))
        return Counter(ARABIC_NUMBER_RE.findall(rewritten)) == source_numbers

    def _rewrite(
        self,
        query: str,
        prompt: str,
        prompt_version: str,
        *,
        max_tokens: int | None = None,
        validate: Any = None,
    ) -> str:
        raw_query = query
        query = normalize_text(query)
        if self.mode == "raw" or not self.base or not self.key or not self.model or (self.mode == "auto" and cjk_ratio(query) < self.threshold):
            return query
        cache_key = self._cache_key(raw_query, prompt_version)
        cached = self.store.translation_get(cache_key)
        if cached:
            return cached
        with self._provider_lock:
            if time.monotonic() < self._provider_disabled_until:
                return query
        request_body = {"model": self.model, "temperature": 0, "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]}
        if max_tokens is not None:
            request_body["max_tokens"] = max_tokens
        payload = json.dumps(request_body).encode()
        endpoint = self.base + ("/chat/completions" if self.base.endswith("/v1") else "/v1/chat/completions")
        req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    data = json.loads(response.read())
                rewritten = normalize_text(data["choices"][0]["message"]["content"])
                if rewritten and (validate is None or validate(query, rewritten)):
                    self.store.translation_put(
                        cache_key,
                        rewritten,
                        translation_version=prompt_version,
                    )
                    return rewritten
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 425, 429) and exc.code < 500:
                    with self._provider_lock:
                        self._provider_disabled_until = time.monotonic() + 300
                    break
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        disabled_for = float(retry_after) if retry_after else 30.0
                    except ValueError:
                        disabled_for = 30.0
                    with self._provider_lock:
                        self._provider_disabled_until = time.monotonic() + min(300.0, max(1.0, disabled_for))
                if attempt + 1 < self.retries:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        delay = float(retry_after) if retry_after else 0.2 * (2**attempt)
                    except ValueError:
                        delay = 0.2 * (2**attempt)
                    time.sleep(min(30.0, max(0.0, delay)))
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
                if attempt + 1 < self.retries:
                    time.sleep(0.2 * (2**attempt))
        return query

    def normalize_document(self, raw: str) -> tuple[str, str, str, str]:
        """Return derived retrieval text plus hash/version/status metadata."""
        normalized, translation_hash, translation_version, translation_status = self.pending_document(raw)
        if translation_status != "pending_provider":
            return normalized, translation_hash, translation_version, translation_status
        if not self.base or not self.key or not self.model:
            return normalized, translation_hash, translation_version, "pending_provider"
        rewritten = self.rewrite(normalized)
        if rewritten == normalized:
            return normalized, translation_hash, translation_version, "pending_provider"
        return rewritten, translation_hash, translation_version, "ok"

    def pending_document(self, raw: str) -> tuple[str, str, str, str]:
        """Describe the durable pre-provider state for a source revision."""
        normalized = normalize_text(raw)
        translation_hash = hashlib.sha256((raw + self.model + PROMPT_VERSION).encode("utf-8")).hexdigest()
        if self.mode == "raw":
            return normalized, translation_hash, PROMPT_VERSION, "skipped_raw_mode"
        if self.mode == "auto" and cjk_ratio(normalized) < self.threshold:
            return normalized, translation_hash, PROMPT_VERSION, "skipped_non_cjk"
        return normalized, translation_hash, PROMPT_VERSION, "pending_provider"

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
        self.repo = os.getenv("FUNES_STORAGE_REPO") or os.getenv("FUNES_MEMORY", "")
        self.token = os.getenv("HF_TOKEN", "")
        self.filename = os.getenv("FUNES_SNAPSHOT_FILE", "funes-snapshot.jsonl.gz")
        self.prefix = os.getenv("FUNES_SNAPSHOT_PREFIX", "funes-snapshot-")
        self.delta_prefix = os.getenv("FUNES_DELTA_PREFIX", "funes-delta-")
        self.control_prefix = os.getenv("FUNES_REINDEX_PREFIX", "funes-reindex-")
        self.restore_batch = max(50, int(os.getenv("FUNES_RESTORE_BATCH", "500")))
        self.restore_failed = False
        self.restore_error = None
        self.restored = False
        self.restoring = False
        self.storage_key = (
            os.getenv("FUNES_STORAGE_KEY")
            or os.getenv("FUNES_API_TOKEN")
            or os.getenv("FUNES_AUTH_TOKEN", "")
        )

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

    def _repo_files(self) -> list[str]:
        """List snapshot and delta objects without exposing repository contents."""
        from huggingface_hub import HfApi
        api = HfApi(token=self.token)
        files = []
        for item in api.list_repo_tree(self.repo, repo_type="dataset", recursive=True, token=self.token):
            name = getattr(item, "path", "")
            if (
                name == self.filename + ".enc"
                or name.startswith(self.prefix)
                or name.startswith(self.delta_prefix)
                or name.startswith(self.control_prefix)
            ):
                if name.endswith((".jsonl.enc", ".jsonl.gz.enc")):
                    files.append(name)
        snapshots = sorted(name for name in files if name == self.filename + ".enc" or name.startswith(self.prefix))
        deltas = sorted(name for name in files if name.startswith(self.delta_prefix))
        controls = sorted(name for name in files if name.startswith(self.control_prefix))
        # Controls replay last and carry monotonic per-derived-field generations.
        # This makes immutable hash-named deltas safe regardless of their order.
        return snapshots + deltas + controls

    def _restore_file(self, filename: str) -> int:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=self.repo,
            repo_type="dataset",
            filename=filename,
            token=self.token,
            local_dir=str(self.store.data_dir / "remote"),
        )
        encrypted = Path(downloaded)
        if not filename.endswith(".enc"):
            if not self._truth("FUNES_ALLOW_PLAINTEXT_SOURCE_RESTORE"):
                raise RuntimeError("plaintext source snapshot restore is disabled")
            return self.store.restore_documents(
                self._iter_file(encrypted), self.restore_batch, apply_controls=False
            )
        with tempfile.TemporaryDirectory(prefix="funes-source-restore-") as directory:
            plaintext = Path(directory) / Path(filename).name.removesuffix(".enc")
            self._decrypt_file(encrypted, plaintext)
            return self.store.restore_documents(
                self._iter_file(plaintext), self.restore_batch, apply_controls=False
            )

    @staticmethod
    def _iter_file(path: Path):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    def restore(self) -> int:
        if self.repo and self.token:
            try:
                files = self._repo_files()
                if not files:
                    # Backwards-compatible single-file snapshot lookup.
                    files = [self.filename + ".enc"]
                restored = 0
                for filename in files:
                    restored += self._restore_file(filename)
                # A complete Hub restore can replay an old generation-zero
                # revision into an id below a partially persisted row cursor.
                # Rewind only here; ordinary local restarts keep their cursor.
                self.store.reset_reindex_control_cursors()
                self.store.drain_reindex_controls(self.restore_batch)
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
            restored = self.store.restore(local, apply_controls=False)
            self.store.drain_reindex_controls(self.restore_batch)
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
                [binary, "filesystem", str(path), "--json", "--no-verification",
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

    def _write_jsonl_gzip(self, docs: list[dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as stream:
            for doc in docs:
                stream.write(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")

    def _encryption_key(self) -> bytes:
        if not self.storage_key:
            raise RuntimeError("FUNES_STORAGE_KEY is not configured")
        return hashlib.sha256(
            b"funes-source-storage-v1\0" + self.storage_key.encode("utf-8")
        ).digest()

    def _encrypt_file(self, source: Path, target: Path) -> None:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        nonce = os.urandom(12)
        encryptor = Cipher(algorithms.AES(self._encryption_key()), modes.GCM(nonce)).encryptor()
        encryptor.authenticate_additional_data(ENCRYPTED_AAD)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("wb") as outgoing:
            outgoing.write(ENCRYPTED_MAGIC)
            outgoing.write(struct.pack(">B", len(nonce)))
            outgoing.write(nonce)
            while chunk := incoming.read(1024 * 1024):
                outgoing.write(encryptor.update(chunk))
            outgoing.write(encryptor.finalize())
            outgoing.write(encryptor.tag)

    def _decrypt_file(self, source: Path, target: Path) -> None:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        size = source.stat().st_size
        with source.open("rb") as incoming:
            magic = incoming.read(len(ENCRYPTED_MAGIC))
            if magic != ENCRYPTED_MAGIC:
                raise RuntimeError("invalid encrypted source snapshot header")
            nonce_size_raw = incoming.read(1)
            if len(nonce_size_raw) != 1:
                raise RuntimeError("truncated encrypted source snapshot")
            nonce_size = struct.unpack(">B", nonce_size_raw)[0]
            nonce = incoming.read(nonce_size)
            header_size = len(ENCRYPTED_MAGIC) + 1 + nonce_size
            if len(nonce) != nonce_size or size < header_size + 16:
                raise RuntimeError("truncated encrypted source snapshot")
            incoming.seek(-16, os.SEEK_END)
            tag = incoming.read(16)
            ciphertext_size = size - header_size - len(tag)
            incoming.seek(header_size)
            decryptor = Cipher(
                algorithms.AES(self._encryption_key()), modes.GCM(nonce, tag)
            ).decryptor()
            decryptor.authenticate_additional_data(ENCRYPTED_AAD)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            try:
                with temporary.open("wb") as outgoing:
                    remaining = ciphertext_size
                    while remaining:
                        chunk = incoming.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise RuntimeError("truncated encrypted source snapshot")
                        remaining -= len(chunk)
                        outgoing.write(decryptor.update(chunk))
                    outgoing.write(decryptor.finalize())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

    def upload_reindex_control(self, control: dict[str, Any]) -> dict[str, Any]:
        """Persist one encrypted reindex command before it becomes runnable."""
        generation = int(control.get("generation", 0))
        scope = str(control.get("scope", ""))
        if generation < 1 or scope not in REINDEX_SCOPES:
            raise ValueError("invalid reindex control")
        with self.upload_lock:
            if self.restore_failed:
                return {"uploaded": False, "durable": False, "reason": "restore_failed"}
            digest = hashlib.sha256(
                json.dumps(
                    control, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()[:16]
            name = f"{self.control_prefix}{generation:020d}-{digest}.jsonl.gz.enc"
            encrypted = self.store.data_dir / "reindex-queue" / name
            try:
                with tempfile.TemporaryDirectory(prefix="funes-reindex-control-") as directory:
                    plaintext = Path(directory) / name.removesuffix(".enc")
                    temporary = encrypted.with_name(encrypted.name + ".partial")
                    self._write_jsonl_gzip([control], plaintext)
                    self._encrypt_file(plaintext, temporary)
                    with temporary.open("rb") as stream:
                        os.fsync(stream.fileno())
                    os.replace(temporary, encrypted)
                    directory_fd = os.open(encrypted.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            except Exception as exc:
                encrypted.with_name(encrypted.name + ".partial").unlink(missing_ok=True)
                reason = type(exc).__name__
                self.store.set_sync(last_error=reason)
                return {"uploaded": False, "durable": False, "reason": reason}
            if self.repo:
                if not self.token:
                    return {
                        "uploaded": False,
                        "durable": False,
                        "reason": "HF storage not configured",
                    }
                try:
                    from huggingface_hub import HfApi
                    HfApi(token=self.token).upload_file(
                        path_or_fileobj=str(encrypted),
                        path_in_repo=name,
                        repo_id=self.repo,
                        repo_type="dataset",
                        commit_message="funes encrypted reindex control",
                    )
                except Exception as exc:
                    reason = type(exc).__name__
                    self.store.set_sync(last_error=reason)
                    return {"uploaded": False, "durable": False, "reason": reason}
            self.store.set_sync(last_sync=utc_now(), snapshot_path=str(encrypted), last_error=None)
            return {
                "uploaded": bool(self.repo),
                "durable": True,
                "generation": generation,
                "scope": scope,
            }

    def upload(self, docs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        with self.upload_lock:
            if self.restore_failed:
                return {"uploaded": False, "durable": False, "reason": "restore_failed"}
            # Normal ingest uses an immutable, content-addressed delta.  This
            # avoids rewriting a multi-gigabyte snapshot for every new turn and
            # makes retries idempotent.  `/sync` without documents still emits a
            # compact full snapshot for operators.
            if docs is not None:
                digest = hashlib.sha256(json.dumps(docs, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
                # Content addressing makes a retry upload the same immutable
                # object. Restore is order-independent because Store.ingest
                # applies updated_at LWW rather than trusting filename order.
                path = self.store.data_dir / f"{self.delta_prefix}{digest}.jsonl.gz"
                self._write_jsonl_gzip(docs, path)
            else:
                path = self.snapshot_path()
                self.store.snapshot(path)
            if not self.repo or not self.token:
                self.store.set_sync(last_sync=utc_now(), snapshot_path=str(path), last_error=None)
                if self.repo or self._truth("FUNES_REQUIRE_DURABLE_ACK"):
                    return {"uploaded": False, "durable": False, "path": str(path), "reason": "HF storage not configured"}
                return {"uploaded": False, "durable": True, "path": str(path), "reason": "local durable store"}
            try:
                encrypted = path.with_name(path.name + ".enc")
                self._encrypt_file(path, encrypted)
            except Exception as exc:
                reason = type(exc).__name__
                self.store.set_sync(last_error=reason, snapshot_path=str(path))
                return {"uploaded": False, "durable": False, "path": str(path), "reason": reason}
            try:
                from huggingface_hub import HfApi
                target = (path.name if docs is not None else self.filename) + ".enc"
                HfApi(token=self.token).upload_file(path_or_fileobj=str(encrypted), path_in_repo=target, repo_id=self.repo, repo_type="dataset", commit_message="funes encrypted source delta" if docs is not None else "funes encrypted source snapshot")
                if docs is not None:
                    path.unlink(missing_ok=True)
                encrypted.unlink(missing_ok=True)
                self.store.set_sync(last_sync=utc_now(), snapshot_path=str(path), last_error=None)
                return {"uploaded": True, "durable": True, "path": str(path)}
            except Exception as exc:
                self.store.set_sync(last_error=type(exc).__name__, snapshot_path=str(path))
                return {"uploaded": False, "durable": False, "path": str(path), "reason": type(exc).__name__}


FINAL_TRANSLATION_STATUSES = {
    "ok",
    "skipped_raw_mode",
    "skipped_non_cjk",
    "skipped_native_session",
    "skipped_low_value",
}


def prepare_ingest_documents(app: Any, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the raw-first durable representation without calling a provider."""
    prepared = []
    for doc in docs:
        if not isinstance(doc, dict):
            raise ValueError("each document must be an object")
        item = dict(doc)
        raw = str(item.get("raw_text", item.get("text", "")))
        if not raw:
            raise ValueError("raw_text/text is required")
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        identity = str(item.get("source_identity") or app.store._identity(item, metadata, raw))
        item["source_identity"] = identity
        generation = app.store.latest_reindex_generation()
        item["retrieval_generation"] = generation
        item["native_generation"] = generation
        source_version = str(item.get("source_version", metadata.get("source_version", "")))
        content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        existing = app.store.get(identity)
        same_final_revision = bool(
            existing
            and existing.get("content_hash") == content_hash
            and str(existing.get("source_version", "")) == source_version
            and existing.get("translation_status") in FINAL_TRANSLATION_STATUSES
        )
        if same_final_revision and not item.get("retrieval_text"):
            item["retrieval_text"] = existing.get("retrieval_text") or normalize_text(raw)
            for name in ("translation_hash", "translation_version", "translation_status"):
                if existing.get(name) is not None:
                    item.setdefault(name, existing[name])
            prepared.append(item)
            continue
        if item.get("retrieval_text"):
            prepared.append(item)
            continue
        source_type = str(item.get("source_type", item.get("kind", ""))).lower()
        content_type = str(item.get("content_type", "")).lower()
        native_session = source_type in {
            "session",
            "codex",
            "codex_session",
            "pi",
            "pi_session",
            "claude",
            "claude_session",
        }
        low_value = content_type in {"tool_call", "tool_result", "shell_output", "progress"}
        if native_session or low_value:
            shadow = normalize_text(raw)
            translation_hash = hashlib.sha256(
                (raw + app.translator.model + PROMPT_VERSION).encode("utf-8")
            ).hexdigest()
            derived = (
                shadow,
                translation_hash,
                PROMPT_VERSION,
                "skipped_native_session" if native_session else "skipped_low_value",
            )
        else:
            pending_document = getattr(app.translator, "pending_document", None)
            if callable(pending_document):
                derived = pending_document(raw)
            else:
                translation_hash = hashlib.sha256(
                    (raw + app.translator.model + PROMPT_VERSION).encode("utf-8")
                ).hexdigest()
                derived = normalize_text(raw), translation_hash, PROMPT_VERSION, "pending_provider"
        shadow, translation_hash, translation_version, translation_status = derived
        item["retrieval_text"] = shadow
        item.setdefault("translation_hash", translation_hash)
        item.setdefault("translation_version", translation_version)
        item.setdefault("translation_status", translation_status)
        if translation_status in FINAL_TRANSLATION_STATUSES:
            item.setdefault(
                "retrieval_updated_at",
                item.get("updated_at", metadata.get("updated_at")) or utc_now(),
            )
        prepared.append(item)
    return prepared


def _persist_translation_documents(app: Any, documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Translate and durably write derived fields without ever replacing raw text."""
    if not documents:
        return {"attempted": 0, "updated": 0, "durable": True}
    active = []
    for selected in documents:
        current = app.store.get(str(selected["source_identity"]))
        if (
            not current
            or current.get("translation_status") != "pending_provider"
            or current.get("native_index_status") == "waiting_durability"
        ):
            continue
        if (
            current.get("content_hash") == selected.get("content_hash")
            and str(current.get("source_version", "")) == str(selected.get("source_version", ""))
            and int(current.get("retrieval_generation") or 0)
            == int(selected.get("retrieval_generation") or 0)
        ):
            active.append(current)
    if not active:
        return {"attempted": 0, "updated": 0, "durable": True}
    derived_values = app.translator.normalize_many([str(item["raw_text"]) for item in active])
    updates = []
    for selected, derived in zip(active, derived_values):
        current = app.store.get(str(selected["source_identity"]))
        if not current:
            continue
        if (
            current.get("content_hash") != selected.get("content_hash")
            or str(current.get("source_version", "")) != str(selected.get("source_version", ""))
            or int(current.get("retrieval_generation") or 0)
            != int(selected.get("retrieval_generation") or 0)
        ):
            # A newer source revision won the race while the provider was in
            # flight.  Never attach an old shadow to new raw content.
            continue
        shadow, translation_hash, translation_version, translation_status = derived
        if translation_status == "pending_provider":
            continue
        updated = dict(current)
        updated["retrieval_text"] = shadow
        updated["translation_hash"] = translation_hash
        updated["translation_version"] = translation_version
        updated["translation_status"] = translation_status
        updated["retrieval_updated_at"] = utc_now()
        updated["native_index_version"] = None
        updated["native_index_status"] = None
        updated["native_indexed_at"] = None
        updated["native_index_error"] = None
        updates.append(updated)
    if not updates:
        return {"attempted": len(active), "updated": 0, "durable": True}
    result = app.store.ingest(updates)
    changed = [
        item["source_identity"]
        for item in result["items"]
        if item["status"] in {"updated", "derived_updated"}
    ]
    canonical = app.store.get_many(changed)
    if not canonical:
        return {"attempted": len(active), "updated": 0, "durable": True}
    sync = app.syncer.upload(canonical)
    if not sync.get("durable"):
        app.store.mark_translations_pending(canonical)
    return {
        "attempted": len(active),
        "updated": len(canonical) if sync.get("durable") else 0,
        "durable": bool(sync.get("durable")),
        "sync": sync,
    }


def persist_translation_documents(app: Any, documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Serialize provider work and leave every failed item restart-safe."""
    lock = getattr(app, "translation_lock", None)
    try:
        if lock is None:
            return _persist_translation_documents(app, documents)
        with lock:
            return _persist_translation_documents(app, documents)
    except Exception:
        current = app.store.get_many(
            [str(item.get("source_identity", "")) for item in documents]
        )
        app.store.mark_translations_pending(current)
        return {
            "attempted": len(documents),
            "updated": 0,
            "durable": True,
            "retry_pending": True,
        }


def ingest_documents(app: Any, docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist and upload raw-first rows without provider or native work."""
    prepared = prepare_ingest_documents(app, docs)
    for item in prepared:
        existing = app.store.get(str(item["source_identity"]))
        if existing and existing.get("native_index_status") == "waiting_durability":
            item["native_index_version"] = None
            item["native_index_status"] = "retry"
            item["native_indexed_at"] = None
            item["native_index_error"] = None
    result = app.store.ingest(prepared)
    identities = [str(item["source_identity"]) for item in result["items"]]
    canonical = app.store.get_many(identities)
    raw_sync = app.syncer.upload(canonical)
    result.update(
        accepted=result["created"] + result["updated"] + result["deduped"],
        durable=bool(raw_sync.get("durable")),
        sync=raw_sync,
    )
    result["translation"] = {"attempted": 0, "updated": 0, "durable": True}
    if not result["durable"]:
        app.store.update_native_index(
            [
                {
                    "source_identity": item["source_identity"],
                    "source_version": item.get("source_version", ""),
                    "content_hash": item["content_hash"],
                    "native_index_version": None,
                    "native_index_status": "waiting_durability",
                    "native_indexed_at": None,
                    "native_index_error": "durability_pending",
                    "native_generation": int(item.get("native_generation") or 0),
                }
                for item in canonical
            ]
        )
        result["error"] = "durability_pending"
        return result
    return result


def queue_reindex(app: Any, scope: str) -> dict[str, Any]:
    """Durably enqueue a reindex control without provider or native work."""
    if scope not in REINDEX_SCOPES:
        raise ValueError("scope must be retrieval_text or all")
    def persist_control() -> dict[str, Any]:
        if app.syncer.restoring or app.syncer.restore_failed:
            return {
                "queued": False,
                "durable": False,
                "error": "restore_in_progress" if app.syncer.restoring else "restore_failed",
            }
        control = app.store.next_reindex_control(scope)
        sync = app.syncer.upload_reindex_control(control)
        if not sync.get("durable"):
            return {
                "queued": False,
                "durable": False,
                "scope": scope,
                "error": str(sync.get("reason") or "durability_pending"),
            }
        app.store.record_reindex_control(control)
        wake = getattr(app, "reindex_wake", None)
        if wake is not None:
            wake.set()
        return {
            "queued": True,
            "durable": True,
            "scope": scope,
            "generation": int(control["generation"]),
        }
    lock = getattr(app, "reindex_lock", None) or threading.Lock()
    with lock:
        return persist_control()


class App:
    def __init__(self):
        # Free Gradio Spaces do not expose /data.  The Hub snapshot remains the
        # durable source of truth; operators can override this with a writable
        # mounted volume when one is available.
        self.store = Store(os.getenv("FUNES_DATA_DIR", "/tmp/funes-data"))
        self.translator = Translator(self.store)
        self.syncer = SnapshotSync(self.store)
        self.restore_result = 0
        self.restore_done = threading.Event()
        self.reconcile_stop = threading.Event()
        self.reconcile_wake = threading.Event()
        self.reindex_stop = threading.Event()
        self.reindex_wake = threading.Event()
        self.reindex_lock = threading.Lock()
        self.reindex_batch_size = max(
            1, int(os.getenv("FUNES_REINDEX_BATCH_SIZE", "500"))
        )
        self.translation_lock = threading.Lock()
        self.reconcile_interval = max(0.01, float(os.getenv("TRANSLATION_RECONCILE_INTERVAL", "300")))
        self.restore_thread = None
        if os.getenv("FUNES_LAZY_RESTORE", "false").lower() in {"1", "true", "yes", "on"}:
            self.syncer.restoring = True
            self.restore_thread = threading.Thread(target=self._restore_background, name="funes-restore", daemon=True)
            self.restore_thread.start()
        else:
            self._restore_background()
        self.reconcile_thread = threading.Thread(
            target=self._reconcile_background,
            name="funes-translation-reconcile",
            daemon=True,
        )
        self.reconcile_thread.start()
        self.reindex_thread = threading.Thread(
            target=self._reindex_background,
            name="funes-reindex-control",
            daemon=True,
        )
        self.reindex_thread.start()

    def _restore_background(self) -> None:
        try:
            self.restore_result = self.syncer.restore()
            if not self.syncer.restore_failed:
                # FTS5 external-content tables need an explicit rebuild after
                # restoring rows from a JSONL snapshot.
                self.store.reindex()
        finally:
            self.syncer.restoring = False
            self.restore_done.set()

    def ingest_documents(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        return ingest_documents(self, docs)

    def reconcile_pending(self) -> dict[str, Any]:
        if self.syncer.restoring or self.syncer.restore_failed:
            return {"attempted": 0, "updated": 0, "durable": False}
        limit = self.translator.batch_size * self.translator.concurrency
        return persist_translation_documents(self, self.store.pending_translations(limit))

    def _reconcile_background(self) -> None:
        self.restore_done.wait()
        while not self.reconcile_stop.is_set():
            try:
                self.reconcile_pending()
            except Exception:
                # Pending state is already durable.  A later interval retries;
                # never log raw text, provider payloads, or credentials here.
                pass
            wake = getattr(self, "reconcile_wake", None)
            if wake is None:
                self.reconcile_stop.wait(self.reconcile_interval)
            else:
                wake.wait(self.reconcile_interval)
                wake.clear()
            if self.reconcile_stop.is_set():
                break

    def _reindex_background(self) -> None:
        self.restore_done.wait()
        while not self.reindex_stop.is_set():
            result = self.store.apply_pending_reindex_controls(
                getattr(self, "reindex_batch_size", 500)
            )
            if result["applied"]:
                self.reconcile_wake.set()
            if result["scanned"] or result["applied"]:
                continue
            self.reindex_wake.wait(self.reconcile_interval)
            self.reindex_wake.clear()

    def close(self):
        self.reconcile_stop.set()
        if getattr(self, "reconcile_wake", None) is not None:
            self.reconcile_wake.set()
        if getattr(self, "reindex_stop", None) is not None:
            self.reindex_stop.set()
        if getattr(self, "reindex_wake", None) is not None:
            self.reindex_wake.set()
        if self.reconcile_thread is not threading.current_thread():
            self.reconcile_thread.join()
        if self.restore_thread is not None and self.restore_thread is not threading.current_thread():
            self.restore_thread.join()
        if (
            getattr(self, "reindex_thread", None) is not None
            and self.reindex_thread is not threading.current_thread()
        ):
            self.reindex_thread.join()
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
            supplied = self.headers.get("X-Funes-Authorization", "") or self.headers.get("Authorization", "")
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
            # HTTP always returns raw source truth, never its English shadow.
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
                    if app.syncer.restoring:
                        return self._json(503, {"status": "restoring"})
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
                    params = parse_qs(urlparse(self.path).query)
                    ident = params.get("source_identity", params.get("id", [""]))[0]
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
                    if app.syncer.restoring or app.syncer.restore_failed:
                        return self._json(503, {"error": "restore_in_progress" if app.syncer.restoring else "restore_failed", "durable": False})
                    docs = body.get("documents", body.get("records", body.get("items")))
                    if docs is None:
                        docs = [body]
                    if not isinstance(docs, list):
                        raise ValueError("documents must be a list")
                    result = app.ingest_documents(docs)
                    if not result["durable"]:
                        return self._json(503, {"error": "durability_pending", **result})
                    return self._json(200, result)
                if self.path in ("/search", "/recall"):
                    if app.syncer.restoring:
                        return self._json(503, {"error": "restore_in_progress", "results": []})
                    query = normalize_text(str(body.get("query", body.get("q", ""))))
                    if not query:
                        raise ValueError("query is required")
                    rewritten = app.translator.rewrite_query(query)
                    facet_values = body.get("facets") or {}
                    if not isinstance(facet_values, dict):
                        raise ValueError("facets must be an object")
                    filters = {
                        key: body.get(key, facet_values.get(key))
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
                        if body.get(key, facet_values.get(key)) is not None
                    }
                    limit = max(1, min(int(body.get("limit", 20)), 100))
                    candidate_limit = expanded_candidate_limit(limit)
                    raw_hits = app.store.search(query, candidate_limit, filters=filters)
                    rewritten_hits = (
                        app.store.search(rewritten, candidate_limit, filters=filters)
                        if rewritten != query
                        else []
                    )
                    hits = stable_rrf([raw_hits, rewritten_hits], limit)
                    return self._json(200, {"query": query, "rewritten_query": rewritten, "results": [self._public(x) for x in hits]})
                if self.path == "/get":
                    item = app.store.get(body.get("source_identity", body.get("id", "")))
                    return self._json(200 if item else 404, self._public(item) if item else {"error": "not_found"})
                if self.path == "/reindex":
                    result = queue_reindex(app, str(body.get("scope", "")))
                    return self._json(202 if result.get("durable") else 503, result)
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
