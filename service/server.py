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
import zlib
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
    "native_index_profile", "native_index_memory", "native_indexed_at",
    "native_index_error", "retrieval_generation",
    "native_generation", "embedding_generation",
)
SOURCE_METADATA_FIELDS = (
    "device_id",
    "project",
    "repo",
    "worktree",
    "source_agent",
    "source_type",
    "session_id",
    "message_id",
    "role",
    "timestamp",
    "source_path",
    "agent_type",
    "parent_session_id",
    "agent_id",
)
SOURCE_METADATA_CLOCK_FIELDS = (*SOURCE_METADATA_FIELDS, "metadata_json")
SOURCE_METADATA_CLOCK_KEY = "_source_metadata_clocks"
DERIVED_INPUT_FIELDS = (
    "retrieval_text",
    "translation_hash",
    "translation_version",
    "translation_status",
    "retrieval_updated_at",
    "native_index_version",
    "native_index_status",
    "native_index_profile",
    "native_index_memory",
    "native_indexed_at",
    "native_index_error",
    "retrieval_generation",
    "native_generation",
    "embedding_generation",
    "source_missing",
)
CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
PROMPT_VERSION = "funes-retrieval-v2"
QUERY_PROMPT_VERSION = "funes-query-retrieval-v1"
ENCRYPTED_MAGIC = b"FUNES-SOURCE-V1\0"
ENCRYPTED_AAD = b"funes-source-snapshot-v1"
REINDEX_SCOPES = {"retrieval_text", "all"}
MAX_SOURCE_CHECK_IDENTITIES = 5000
NATIVE_SESSION_TYPES = {
    "session", "codex", "codex_session", "pi", "pi_session",
    "claude", "claude_session",
}
LOW_VALUE_CONTENT_TYPES = {"tool_call", "tool_result", "shell_output", "progress"}
NATIVE_TERMINAL_STATUSES = {"indexed", "held_secret", "held_invalid"}
NATIVE_CHECKPOINT_STATE_VERSION = 2
FTS_SCHEMA_VERSION = 3


def _is_legacy_codex_automation_output(
    source_agent: object, source_type: object, source_path: object
) -> bool:
    """Identify old automation run output that was once imported as memory."""
    path = str(source_path or "").replace("\\", "/").lower()
    return (
        str(source_agent or "").lower() == "codex"
        and str(source_type or "").lower() == "memory"
        and "/.codex/automations/" in path
        and not path.endswith((".md", ".toml"))
    )


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
ASCII_FTS_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,63}")
CJK_SEQUENCE_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+")
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


def technical_fts_terms(value: str, limit: int = 12) -> list[str]:
    """Extract bounded, syntax-safe ASCII identifiers from mixed CJK text."""
    terms = []
    seen = set()
    for term in ASCII_FTS_TERM_RE.findall(value):
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def technical_fts_query(terms: list[str]) -> str:
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def technical_index_text(raw_text: Any, retrieval_text: Any) -> str:
    """Materialize bounded ASCII identifiers so mixed CJK tokens stay indexable."""
    terms = []
    seen = set()
    for value in (raw_text, retrieval_text):
        for term in ASCII_FTS_TERM_RE.findall(str(value or "")):
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
            if len(terms) >= 256:
                return " ".join(terms)
    return " ".join(terms)


def cjk_retrieval_terms(value: str, limit: int = 24) -> list[str]:
    """Return bounded CJK tri/bi-grams for in-memory candidate reranking."""
    terms = []
    seen = set()
    sequences = CJK_SEQUENCE_RE.findall(value)
    for width in (3, 2):
        for sequence in sequences:
            for index in range(max(0, len(sequence) - width + 1)):
                term = sequence[index:index + width]
                if term in seen:
                    continue
                seen.add(term)
                terms.append(term)
                if len(terms) >= limit:
                    return terms
    return terms


def expanded_candidate_limit(limit: int) -> int:
    """Bound per-route recall while leaving room for cross-route consensus."""
    limit = max(1, int(limit))
    return min(100, limit * 3)


def validate_source_identity_batch(value: Any) -> list[str]:
    """Validate and order-dedupe one bounded source reconciliation batch."""
    if not isinstance(value, list):
        raise ValueError("source_identities must be a list")
    if len(value) > MAX_SOURCE_CHECK_IDENTITIES:
        raise ValueError(
            f"source_identities must contain at most {MAX_SOURCE_CHECK_IDENTITIES} items"
        )
    if any(not isinstance(identity, str) or not identity.strip() for identity in value):
        raise ValueError("source_identities must contain only non-empty strings")
    return list(dict.fromkeys(value))


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


MEMORIES_SECONDARY_INDEXES: tuple[tuple[str, str], ...] = (
    (
        "memories_source_agent_role_idx",
        "CREATE INDEX IF NOT EXISTS memories_source_agent_role_idx ON memories(source_agent, role)",
    ),
    (
        "memories_source_agent_type_idx",
        "CREATE INDEX IF NOT EXISTS memories_source_agent_type_idx ON memories(source_agent, source_type)",
    ),
    (
        "memories_translation_pending_idx",
        """CREATE INDEX IF NOT EXISTS memories_translation_pending_idx
        ON memories(updated_at,id)
        WHERE translation_status='pending_provider'
        AND COALESCE(native_index_status, '') != 'waiting_durability'""",
    ),
    (
        "memories_canonical_pending_idx",
        """CREATE INDEX IF NOT EXISTS memories_canonical_pending_idx
        ON memories(
            CASE WHEN native_index_status='retry' THEN 1 ELSE 0 END,
            COALESCE(retrieval_updated_at,updated_at),id)
        WHERE native_index_pending=1""",
    ),
)


class Store:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "funes.sqlite3"
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.create_function(
            "funes_identifiers", 2, technical_index_text, deterministic=True
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._bulk_restore_depth = 0
        self._bulk_restore_prev_pragmas: dict[str, Any] | None = None
        self._init_schema()

    def _read_connection(self) -> sqlite3.Connection:
        """Open a short-lived WAL reader that never waits on ``self.lock``.

        The main connection and Python lock serialize durable writes.  Recall
        and get are read-only and SQLite WAL can safely serve their last
        committed snapshot while an ingest transaction is active.
        """
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
            timeout=5,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

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
                    search_identifiers TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    source_metadata_clock_json TEXT NOT NULL DEFAULT '{}',
                    source_agent TEXT, source_type TEXT, device_id TEXT, project TEXT,
                    repo TEXT, worktree TEXT, session_id TEXT, message_id TEXT, role TEXT,
                    timestamp TEXT, source_path TEXT, content_hash TEXT NOT NULL,
                    ingested_at TEXT NOT NULL, updated_at TEXT NOT NULL, retrieval_updated_at TEXT,
                    content_type TEXT,
                    source_missing INTEGER NOT NULL DEFAULT 0, agent_type TEXT,
                    parent_session_id TEXT, agent_id TEXT,
                    translation_hash TEXT, translation_version TEXT, translation_status TEXT,
                    native_index_version TEXT, native_index_status TEXT, native_index_profile TEXT,
                    native_index_memory TEXT,
                    native_indexed_at TEXT, native_index_error TEXT,
                    native_index_pending INTEGER NOT NULL DEFAULT 1,
                    retrieval_generation INTEGER NOT NULL DEFAULT 0,
                    native_generation INTEGER NOT NULL DEFAULT 0,
                    embedding_generation INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(source_identity)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    raw_text, retrieval_text, search_identifiers,
                    content='memories', content_rowid='id', tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS translation_cache (
                    query TEXT PRIMARY KEY, rewritten TEXT NOT NULL, created_at TEXT NOT NULL,
                    translation_hash TEXT, translation_version TEXT, translation_status TEXT
                );
                CREATE TABLE IF NOT EXISTS sync_state (
                    id INTEGER PRIMARY KEY CHECK(id=1), last_sync TEXT, last_error TEXT,
                    snapshot_path TEXT, restored_at TEXT,
                    native_optimize_provider TEXT, native_optimize_model TEXT,
                    native_optimize_dimensions INTEGER, native_optimize_schema_version INTEGER,
                    native_optimize_layout_version INTEGER NOT NULL DEFAULT 0,
                    native_optimize_memory TEXT,
                    native_optimize_fingerprint TEXT, native_optimize_index_fingerprint TEXT,
                    native_optimize_status TEXT, native_optimized_at TEXT,
                    native_optimize_revision INTEGER NOT NULL DEFAULT 0,
                    native_checkpoint_profile TEXT NOT NULL DEFAULT '',
                    native_checkpoint_memory TEXT NOT NULL DEFAULT '',
                    native_index_revision INTEGER NOT NULL DEFAULT 0,
                    native_eligible_count INTEGER NOT NULL DEFAULT 0,
                    native_indexed_count INTEGER NOT NULL DEFAULT 0,
                    native_held_count INTEGER NOT NULL DEFAULT 0,
                    native_invalid_count INTEGER NOT NULL DEFAULT 0,
                    native_checkpoint_state_version INTEGER NOT NULL DEFAULT 0,
                    fts_schema_version INTEGER NOT NULL DEFAULT 0
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
                "native_index_profile",
                "native_index_memory",
                "native_indexed_at",
                "native_index_error",
            ):
                if name not in columns:
                    self.conn.execute(f"ALTER TABLE memories ADD COLUMN {name} TEXT")
            if "search_identifiers" not in columns:
                self.conn.execute(
                    "ALTER TABLE memories ADD COLUMN search_identifiers TEXT NOT NULL DEFAULT ''"
                )
            if "source_metadata_clock_json" not in columns:
                self.conn.execute(
                    """ALTER TABLE memories ADD COLUMN source_metadata_clock_json
                    TEXT NOT NULL DEFAULT '{}'"""
                )
            if "native_index_pending" not in columns:
                self.conn.execute(
                    "ALTER TABLE memories ADD COLUMN native_index_pending INTEGER NOT NULL DEFAULT 1"
                )
            for name in (
                "retrieval_generation",
                "native_generation",
                "embedding_generation",
            ):
                if name not in columns:
                    self.conn.execute(
                        f"ALTER TABLE memories ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
                    )
            cache_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(translation_cache)")}
            for name in ("translation_hash", "translation_version", "translation_status"):
                if name not in cache_columns:
                    self.conn.execute(f"ALTER TABLE translation_cache ADD COLUMN {name} TEXT")
            sync_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(sync_state)")}
            for name, sql_type in (
                ("native_optimize_provider", "TEXT"),
                ("native_optimize_model", "TEXT"),
                ("native_optimize_dimensions", "INTEGER"),
                ("native_optimize_schema_version", "INTEGER"),
                ("native_optimize_layout_version", "INTEGER NOT NULL DEFAULT 0"),
                ("native_optimize_memory", "TEXT"),
                ("native_optimize_fingerprint", "TEXT"),
                ("native_optimize_index_fingerprint", "TEXT"),
                ("native_optimize_status", "TEXT"),
                ("native_optimized_at", "TEXT"),
                ("native_optimize_revision", "INTEGER NOT NULL DEFAULT 0"),
                ("native_checkpoint_profile", "TEXT NOT NULL DEFAULT ''"),
                ("native_checkpoint_memory", "TEXT NOT NULL DEFAULT ''"),
                ("native_index_revision", "INTEGER NOT NULL DEFAULT 0"),
                ("native_eligible_count", "INTEGER NOT NULL DEFAULT 0"),
                ("native_indexed_count", "INTEGER NOT NULL DEFAULT 0"),
                ("native_held_count", "INTEGER NOT NULL DEFAULT 0"),
                ("native_invalid_count", "INTEGER NOT NULL DEFAULT 0"),
                ("native_checkpoint_state_version", "INTEGER NOT NULL DEFAULT 0"),
                ("fts_schema_version", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if name not in sync_columns:
                    self.conn.execute(f"ALTER TABLE sync_state ADD COLUMN {name} {sql_type}")
            control_columns = {
                row[1] for row in self.conn.execute("PRAGMA table_info(reindex_controls)")
            }
            if "row_cursor" not in control_columns:
                self.conn.execute(
                    """ALTER TABLE reindex_controls ADD COLUMN row_cursor
                    INTEGER NOT NULL DEFAULT 0"""
                )
            fts_columns = [
                row[1] for row in self.conn.execute("PRAGMA table_info(memories_fts)")
            ]
            fts_version = int(
                self.conn.execute(
                    "SELECT fts_schema_version FROM sync_state WHERE id=1"
                ).fetchone()[0]
                or 0
            )
            if (
                fts_columns != ["raw_text", "retrieval_text", "search_identifiers"]
                or fts_version < FTS_SCHEMA_VERSION
            ):
                # The first sidecar indexed only the derived retrieval shadow.
                # Rebuild exactly once at startup so existing durable raw rows
                # become the primary lexical source without client re-upload.
                self._drop_fts_triggers_locked()
                self.conn.execute(
                    """UPDATE memories SET search_identifiers=
                    funes_identifiers(raw_text, retrieval_text)"""
                )
                if fts_columns != [
                    "raw_text", "retrieval_text", "search_identifiers"
                ]:
                    self.conn.execute("DROP TABLE memories_fts")
                    self.conn.execute(
                        """CREATE VIRTUAL TABLE memories_fts USING fts5(
                        raw_text, retrieval_text, search_identifiers,
                        content='memories', content_rowid='id', tokenize='unicode61')"""
                    )
                self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                self.conn.execute(
                    "UPDATE sync_state SET fts_schema_version=? WHERE id=1",
                    (FTS_SCHEMA_VERSION,),
                )
            self._create_fts_triggers_locked()

            native_state_version = int(
                self.conn.execute(
                    "SELECT native_checkpoint_state_version FROM sync_state WHERE id=1"
                ).fetchone()[0]
                or 0
            )
            if native_state_version < NATIVE_CHECKPOINT_STATE_VERSION:
                self._drop_native_state_triggers_locked()
                # Early unified-memory builds imported every Codex automation
                # run/evaluation file as durable memory. Discovery has since
                # been narrowed to instructions and explicit memory files, but
                # restored source snapshots can still contain those large
                # legacy rows. Keep their raw text and lexical availability;
                # classify them as existing low-value progress so checkpoint
                # rebuilds never send them through a paid embedder again.
                self.conn.execute(
                    """UPDATE memories SET content_type='progress'
                    WHERE lower(COALESCE(source_agent,''))='codex'
                    AND lower(COALESCE(source_type,''))='memory'
                    AND replace(lower(COALESCE(source_path,'')),char(92),'/')
                        LIKE '%/.codex/automations/%'
                    AND lower(COALESCE(source_path,'')) NOT LIKE '%.md'
                    AND lower(COALESCE(source_path,'')) NOT LIKE '%.toml'"""
                )
                state = self.conn.execute(
                    """SELECT native_checkpoint_profile,native_checkpoint_memory,
                    native_optimize_fingerprint,native_optimize_memory
                    FROM sync_state WHERE id=1"""
                ).fetchone()
                profile = str(
                    state["native_checkpoint_profile"]
                    or state["native_optimize_fingerprint"]
                    or ""
                )
                memory = str(
                    state["native_checkpoint_memory"]
                    or state["native_optimize_memory"]
                    or ""
                )
                if not profile:
                    terminal = self.conn.execute(
                        """SELECT COALESCE(native_index_profile, '') AS profile,
                        COALESCE(native_index_memory, '') AS memory, count(*) AS amount
                        FROM memories
                        WHERE native_index_status IN
                            ('indexed','held_secret','held_invalid')
                        GROUP BY profile,memory ORDER BY amount DESC,profile,memory LIMIT 1"""
                    ).fetchone()
                    if terminal is not None:
                        profile = str(terminal["profile"])
                        memory = str(terminal["memory"])
                self._rebuild_native_checkpoint_state_locked(
                    profile, memory, initialize=True
                )
            self._create_native_state_triggers_locked()
            self._create_secondary_indexes_locked()

    def _drop_secondary_indexes_locked(self) -> None:
        for name, _ in MEMORIES_SECONDARY_INDEXES:
            self.conn.execute(f"DROP INDEX IF EXISTS {name}")

    def _create_secondary_indexes_locked(self) -> None:
        for _name, sql in MEMORIES_SECONDARY_INDEXES:
            self.conn.execute(sql)

    def _save_and_apply_bulk_pragmas_locked(self) -> None:
        if self._bulk_restore_prev_pragmas is None:
            prev_sync = self.conn.execute("PRAGMA synchronous").fetchone()[0]
            prev_cache = self.conn.execute("PRAGMA cache_size").fetchone()[0]
            prev_mmap = self.conn.execute("PRAGMA mmap_size").fetchone()[0]
            self._bulk_restore_prev_pragmas = {
                "synchronous": prev_sync,
                "cache_size": prev_cache,
                "mmap_size": prev_mmap,
            }

        sync_val = os.getenv("FUNES_BULK_RESTORE_SYNCHRONOUS", "NORMAL").strip().upper()
        if sync_val in ("OFF", "0"):
            self.conn.execute("PRAGMA synchronous = OFF")
        elif sync_val in ("NORMAL", "1"):
            self.conn.execute("PRAGMA synchronous = NORMAL")
        elif sync_val in ("FULL", "2"):
            self.conn.execute("PRAGMA synchronous = FULL")
        elif sync_val:
            self.conn.execute(f"PRAGMA synchronous = {sync_val}")

        cache_val = os.getenv("FUNES_BULK_RESTORE_CACHE_SIZE", "-64000").strip()
        try:
            self.conn.execute(f"PRAGMA cache_size = {int(cache_val)}")
        except (ValueError, sqlite3.OperationalError):
            self.conn.execute("PRAGMA cache_size = -64000")

        mmap_val = os.getenv("FUNES_BULK_RESTORE_MMAP_SIZE", str(256 * 1024 * 1024)).strip()
        try:
            self.conn.execute(f"PRAGMA mmap_size = {int(mmap_val)}")
        except (ValueError, sqlite3.OperationalError):
            self.conn.execute("PRAGMA mmap_size = 268435456")

    def _restore_pragmas_locked(self) -> None:
        prev = self._bulk_restore_prev_pragmas
        self._bulk_restore_prev_pragmas = None
        if not prev:
            return
        if "synchronous" in prev:
            self.conn.execute(f"PRAGMA synchronous = {prev['synchronous']}")
        if "cache_size" in prev:
            self.conn.execute(f"PRAGMA cache_size = {prev['cache_size']}")
        if "mmap_size" in prev:
            self.conn.execute(f"PRAGMA mmap_size = {prev['mmap_size']}")
        self.conn.execute("PRAGMA shrink_memory")

    def _drop_fts_triggers_locked(self) -> None:
        for trigger in ("memories_ai", "memories_ad", "memories_au"):
            self.conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")

    def _create_fts_triggers_locked(self) -> None:
        self.conn.execute(
            """CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid,raw_text,retrieval_text,search_identifiers)
            VALUES(new.id,new.raw_text,new.retrieval_text,new.search_identifiers); END"""
        )
        self.conn.execute(
            """CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(
                memories_fts,rowid,raw_text,retrieval_text,search_identifiers)
            VALUES(
                'delete',old.id,old.raw_text,old.retrieval_text,old.search_identifiers); END"""
        )
        self.conn.execute(
            """CREATE TRIGGER IF NOT EXISTS memories_au
            AFTER UPDATE OF raw_text,retrieval_text,search_identifiers ON memories
            WHEN old.raw_text IS NOT new.raw_text
                OR old.retrieval_text IS NOT new.retrieval_text
                OR old.search_identifiers IS NOT new.search_identifiers
            BEGIN
            INSERT INTO memories_fts(
                memories_fts,rowid,raw_text,retrieval_text,search_identifiers)
            VALUES(
                'delete',old.id,old.raw_text,old.retrieval_text,old.search_identifiers);
            INSERT INTO memories_fts(rowid,raw_text,retrieval_text,search_identifiers)
            VALUES(new.id,new.raw_text,new.retrieval_text,new.search_identifiers); END"""
        )

    def _drop_native_state_triggers_locked(self) -> None:
        for trigger in (
            "memories_native_ai",
            "memories_native_ad",
            "memories_native_au",
        ):
            self.conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")

    def _create_native_state_triggers_locked(self) -> None:
        eligible_new = (
            "lower(COALESCE(new.content_type,'')) NOT IN "
            "('tool_call','tool_result','shell_output','progress')"
        )
        eligible_old = (
            "lower(COALESCE(old.content_type,'')) NOT IN "
            "('tool_call','tool_result','shell_output','progress')"
        )
        current_new = (
            "COALESCE(new.native_index_profile,'')=native_checkpoint_profile "
            "AND COALESCE(new.native_index_memory,'')=native_checkpoint_memory"
        )
        current_old = (
            "COALESCE(old.native_index_profile,'')=native_checkpoint_profile "
            "AND COALESCE(old.native_index_memory,'')=native_checkpoint_memory"
        )
        pending_current_new = (
            "COALESCE(new.native_index_profile,'')="
            "(SELECT native_checkpoint_profile FROM sync_state WHERE id=1) "
            "AND COALESCE(new.native_index_memory,'')="
            "(SELECT native_checkpoint_memory FROM sync_state WHERE id=1)"
        )
        terminal_new = (
            "COALESCE(new.native_index_status,'') IN "
            "('indexed','held_secret','held_invalid')"
        )
        self.conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS memories_native_ai
            AFTER INSERT ON memories BEGIN
            UPDATE sync_state SET
                native_index_revision=native_index_revision+
                    CASE WHEN {eligible_new} THEN 1 ELSE 0 END,
                native_eligible_count=native_eligible_count+
                    CASE WHEN {eligible_new} THEN 1 ELSE 0 END,
                native_indexed_count=native_indexed_count+
                    CASE WHEN {eligible_new} AND new.native_index_status='indexed'
                        AND {current_new} THEN 1 ELSE 0 END,
                native_held_count=native_held_count+
                    CASE WHEN {eligible_new}
                        AND new.native_index_status IN ('held_secret','held_invalid')
                        AND {current_new} THEN 1 ELSE 0 END,
                native_invalid_count=native_invalid_count+
                    CASE WHEN {eligible_new}
                        AND new.native_index_status='held_invalid'
                        AND {current_new} THEN 1 ELSE 0 END
            WHERE id=1;
            UPDATE memories SET native_index_pending=CASE
                WHEN {eligible_new}
                    AND COALESCE(new.native_index_status,'')!='waiting_durability'
                    AND NOT ({terminal_new} AND {pending_current_new})
                THEN 1 ELSE 0 END
            WHERE id=new.id; END"""
        )
        self.conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS memories_native_ad
            AFTER DELETE ON memories BEGIN
            UPDATE sync_state SET
                native_index_revision=native_index_revision+
                    CASE WHEN {eligible_old} THEN 1 ELSE 0 END,
                native_eligible_count=native_eligible_count-
                    CASE WHEN {eligible_old} THEN 1 ELSE 0 END,
                native_indexed_count=native_indexed_count-
                    CASE WHEN {eligible_old} AND old.native_index_status='indexed'
                        AND {current_old} THEN 1 ELSE 0 END,
                native_held_count=native_held_count-
                    CASE WHEN {eligible_old}
                        AND old.native_index_status IN ('held_secret','held_invalid')
                        AND {current_old} THEN 1 ELSE 0 END,
                native_invalid_count=native_invalid_count-
                    CASE WHEN {eligible_old}
                        AND old.native_index_status='held_invalid'
                        AND {current_old} THEN 1 ELSE 0 END
            WHERE id=1; END"""
        )
        self.conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS memories_native_au
            AFTER UPDATE OF source_identity,source_version,raw_text,retrieval_text,
                content_hash,content_type,source_missing,translation_hash,
                translation_version,native_index_version,native_index_status,
                native_index_profile,native_index_memory,native_indexed_at,
                native_generation,embedding_generation ON memories
            WHEN (old.source_identity IS NOT new.source_identity
                OR old.source_version IS NOT new.source_version
                OR old.raw_text IS NOT new.raw_text
                OR old.retrieval_text IS NOT new.retrieval_text
                OR old.content_hash IS NOT new.content_hash
                OR old.content_type IS NOT new.content_type
                OR old.source_missing IS NOT new.source_missing
                OR old.translation_hash IS NOT new.translation_hash
                OR old.translation_version IS NOT new.translation_version
                OR old.native_index_version IS NOT new.native_index_version
                OR old.native_index_status IS NOT new.native_index_status
                OR old.native_index_profile IS NOT new.native_index_profile
                OR old.native_index_memory IS NOT new.native_index_memory
                OR old.native_indexed_at IS NOT new.native_indexed_at
                OR old.native_generation IS NOT new.native_generation
                OR old.embedding_generation IS NOT new.embedding_generation)
            BEGIN
            UPDATE sync_state SET
                native_index_revision=native_index_revision+
                    CASE WHEN {eligible_old} OR {eligible_new} THEN 1 ELSE 0 END,
                native_eligible_count=native_eligible_count
                    +CASE WHEN {eligible_new} THEN 1 ELSE 0 END
                    -CASE WHEN {eligible_old} THEN 1 ELSE 0 END,
                native_indexed_count=native_indexed_count
                    +CASE WHEN {eligible_new} AND new.native_index_status='indexed'
                        AND {current_new} THEN 1 ELSE 0 END
                    -CASE WHEN {eligible_old} AND old.native_index_status='indexed'
                        AND {current_old} THEN 1 ELSE 0 END,
                native_held_count=native_held_count
                    +CASE WHEN {eligible_new}
                        AND new.native_index_status IN ('held_secret','held_invalid')
                        AND {current_new} THEN 1 ELSE 0 END
                    -CASE WHEN {eligible_old}
                        AND old.native_index_status IN ('held_secret','held_invalid')
                        AND {current_old} THEN 1 ELSE 0 END,
                native_invalid_count=native_invalid_count
                    +CASE WHEN {eligible_new}
                        AND new.native_index_status='held_invalid'
                        AND {current_new} THEN 1 ELSE 0 END
                    -CASE WHEN {eligible_old}
                        AND old.native_index_status='held_invalid'
                        AND {current_old} THEN 1 ELSE 0 END
            WHERE id=1;
            UPDATE memories SET native_index_pending=CASE
                WHEN {eligible_new}
                    AND COALESCE(new.native_index_status,'')!='waiting_durability'
                    AND NOT ({terminal_new} AND {pending_current_new})
                THEN 1 ELSE 0 END
            WHERE id=new.id; END"""
        )

    def _rebuild_native_checkpoint_state_locked(
        self, profile_fingerprint: str, memory: str, *, initialize: bool = False
    ) -> None:
        profile_fingerprint = str(profile_fingerprint or "")
        memory = str(memory or "")
        self.conn.execute(
            """UPDATE memories SET native_index_pending=CASE
            WHEN lower(COALESCE(content_type,'')) NOT IN
                ('tool_call','tool_result','shell_output','progress')
                AND COALESCE(native_index_status,'')!='waiting_durability'
                AND NOT (
                    COALESCE(native_index_status,'') IN
                        ('indexed','held_secret','held_invalid')
                    AND COALESCE(native_index_profile,'')=?
                    AND COALESCE(native_index_memory,'')=?)
            THEN 1 ELSE 0 END""",
            (profile_fingerprint, memory),
        )
        counts = self.conn.execute(
            """SELECT count(*) AS eligible,
            sum(CASE WHEN native_index_status='indexed'
                AND COALESCE(native_index_profile,'')=?
                AND COALESCE(native_index_memory,'')=? THEN 1 ELSE 0 END) AS indexed,
            sum(CASE WHEN native_index_status IN ('held_secret','held_invalid')
                AND COALESCE(native_index_profile,'')=?
                AND COALESCE(native_index_memory,'')=? THEN 1 ELSE 0 END) AS held,
            sum(CASE WHEN native_index_status='held_invalid'
                AND COALESCE(native_index_profile,'')=?
                AND COALESCE(native_index_memory,'')=? THEN 1 ELSE 0 END) AS invalid
            FROM memories WHERE lower(COALESCE(content_type,'')) NOT IN
                ('tool_call','tool_result','shell_output','progress')""",
            (
                profile_fingerprint, memory,
                profile_fingerprint, memory,
                profile_fingerprint, memory,
            ),
        ).fetchone()
        current_revision = int(
            self.conn.execute(
                "SELECT native_index_revision FROM sync_state WHERE id=1"
            ).fetchone()[0]
            or 0
        )
        eligible = int(counts["eligible"] or 0)
        if initialize and current_revision == 0:
            current_revision = eligible
        self.conn.execute(
            """UPDATE sync_state SET native_checkpoint_profile=?,
            native_checkpoint_memory=?, native_index_revision=?,
            native_eligible_count=?,native_indexed_count=?,native_held_count=?,
            native_invalid_count=?,native_checkpoint_state_version=? WHERE id=1""",
            (
                profile_fingerprint, memory, current_revision, eligible,
                int(counts["indexed"] or 0), int(counts["held"] or 0),
                int(counts["invalid"] or 0), NATIVE_CHECKPOINT_STATE_VERSION,
            ),
        )

    def _ensure_native_checkpoint_profile_locked(
        self, profile_fingerprint: str, memory: str
    ) -> None:
        state = self.conn.execute(
            """SELECT native_checkpoint_profile,native_checkpoint_memory
            FROM sync_state WHERE id=1"""
        ).fetchone()
        if (
            str(state["native_checkpoint_profile"] or "") != str(profile_fingerprint or "")
            or str(state["native_checkpoint_memory"] or "") != str(memory or "")
        ):
            self._rebuild_native_checkpoint_state_locked(profile_fingerprint, memory)

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def ingest(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        created = updated = deduped = 0
        results = []
        with self.lock, self.conn:
            default_embedding_generation: int | None = None

            def get_default_embedding_generation() -> int:
                nonlocal default_embedding_generation
                if default_embedding_generation is None:
                    default_embedding_generation = (
                        self._latest_embedding_generation_locked()
                    )
                return default_embedding_generation

            for doc in docs:
                raw = str(doc.get("raw_text", doc.get("text", "")))
                if not raw:
                    raise ValueError("raw_text/text is required")
                retrieval = str(doc.get("retrieval_text") or normalize_text(raw))
                search_identifiers = technical_index_text(raw, retrieval)
                metadata = dict(doc.get("metadata") or {})

                def supplied(name: str) -> bool:
                    return name in doc or name in metadata

                def incoming(name: str) -> Any:
                    return doc[name] if name in doc else metadata.get(name)

                source_identity = str(doc.get("source_identity") or self._identity(doc, metadata, raw))
                source_version = str(doc.get("source_version", metadata.get("source_version", "")))
                content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                now = utc_now()
                incoming_updated_explicit = (
                    "updated_at" in doc or "updated_at" in metadata
                )
                incoming_metadata_clocks_explicit = (
                    SOURCE_METADATA_CLOCK_KEY in doc
                )
                incoming_updated = doc.get("updated_at", metadata.get("updated_at")) or now
                incoming_metadata_clocks = self._incoming_source_metadata_clocks(
                    doc.get(SOURCE_METADATA_CLOCK_KEY), incoming_updated
                )
                incoming_metadata_clock_json = (
                    self._serialize_source_metadata_clocks(
                        incoming_metadata_clocks
                    )
                )
                row = self.conn.execute(
                    """SELECT id, content_hash, source_version, updated_at, retrieval_text,
                    metadata_json, source_metadata_clock_json,
                    device_id, project, repo, worktree, source_agent,
                    source_type, session_id, message_id, role, timestamp, source_path,
                    agent_type, parent_session_id, agent_id,
                    translation_hash, translation_version, translation_status,
                    retrieval_updated_at, native_index_version, native_index_status,
                    native_index_profile, native_index_memory, native_indexed_at,
                    native_index_error, source_missing,
                    retrieval_generation, native_generation, embedding_generation
                    FROM memories WHERE source_identity=?""",
                    (source_identity,),
                ).fetchone()
                values = {k: doc.get(k, metadata.get(k)) for k in FIELDS if k not in ("content_hash", "ingested_at", "updated_at")}
                if _is_legacy_codex_automation_output(
                    values.get("source_agent"),
                    values.get("source_type"),
                    values.get("source_path"),
                ):
                    # Pre-filter releases briefly discovered automation run
                    # logs as persistent memory. Retain their raw source while
                    # keeping operational output out of paid embeddings.
                    values["content_type"] = "progress"
                values["source_missing"] = int(bool(values.get("source_missing", False)))
                values["retrieval_generation"] = int(values.get("retrieval_generation") or 0)
                values["native_generation"] = int(values.get("native_generation") or 0)
                embedding_generation_supplied = (
                    supplied("embedding_generation")
                )
                incoming_embedding_generation = int(
                    values.get("embedding_generation") or 0
                )
                if row is not None:
                    incoming_embedding_generation = max(
                        incoming_embedding_generation
                        if embedding_generation_supplied
                        else int(row["embedding_generation"] or 0),
                        int(row["embedding_generation"] or 0),
                    )
                elif self._bulk_restore_depth and embedding_generation_supplied:
                    pass
                else:
                    incoming_embedding_generation = max(
                        incoming_embedding_generation,
                        get_default_embedding_generation(),
                    )
                values["embedding_generation"] = incoming_embedding_generation
                if row and row["content_hash"] == content_hash and row["source_version"] == source_version:
                    derived_supplied = any(
                        supplied(name) for name in DERIVED_INPUT_FIELDS
                    )
                    incoming_status = (
                        incoming("translation_status")
                        if supplied("translation_status")
                        else row["translation_status"]
                    )
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
                    incoming_retrieval = (
                        str(incoming("retrieval_text") or normalize_text(raw))
                        if supplied("retrieval_text")
                        else row["retrieval_text"]
                    )
                    incoming_translation_hash = (
                        incoming("translation_hash")
                        if supplied("translation_hash")
                        else row["translation_hash"]
                    )
                    incoming_translation_version = (
                        incoming("translation_version")
                        if supplied("translation_version")
                        else row["translation_version"]
                    )
                    incoming_retrieval_updated_at = (
                        incoming("retrieval_updated_at")
                        if supplied("retrieval_updated_at")
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
                        if supplied("source_missing")
                        else row["source_missing"]
                    )
                    source_missing_changed = incoming_source_missing != row["source_missing"]
                    incoming_native = (
                        incoming("native_index_version") if supplied("native_index_version") else row["native_index_version"],
                        incoming("native_index_status") if supplied("native_index_status") else row["native_index_status"],
                        incoming("native_index_profile") if supplied("native_index_profile") else row["native_index_profile"],
                        incoming("native_index_memory") if supplied("native_index_memory") else row["native_index_memory"],
                        incoming("native_indexed_at") if supplied("native_indexed_at") else row["native_indexed_at"],
                        incoming("native_index_error") if supplied("native_index_error") else row["native_index_error"],
                    )
                    incoming_native_generation = values["native_generation"]
                    if incoming_native_generation < row["native_generation"]:
                        incoming_native = (
                            row["native_index_version"],
                            row["native_index_status"],
                            row["native_index_profile"],
                            row["native_index_memory"],
                            row["native_indexed_at"],
                            row["native_index_error"],
                        )
                        incoming_native_generation = row["native_generation"]
                    if source_missing_changed and not any(
                        name in doc
                        or name in metadata
                        for name in (
                            "native_index_version",
                            "native_index_status",
                            "native_index_profile",
                            "native_index_memory",
                            "native_indexed_at",
                            "native_index_error",
                        )
                    ):
                        incoming_native = (None, None, None, None, None, None)
                    terminal_checkpoint_regression = (
                        row["native_index_status"] in NATIVE_TERMINAL_STATUSES
                        and incoming_native[1] in NATIVE_TERMINAL_STATUSES
                        and incoming_native_generation <= row["native_generation"]
                        and (
                            row["native_index_status"] == "held_invalid"
                            or incoming_native[2] != row["native_index_profile"]
                            or incoming_native[3] != row["native_index_memory"]
                            or incoming_native[4] is not None
                        )
                        and not self._newer_timestamp(
                            incoming_native[4], row["native_indexed_at"]
                        )
                        and not source_missing_changed
                    )
                    if terminal_checkpoint_regression or (
                        row["native_index_status"] in NATIVE_TERMINAL_STATUSES
                        and incoming_native[1] not in NATIVE_TERMINAL_STATUSES
                        and not translation_changed
                        and not source_missing_changed
                    ):
                        # Immutable Hub deltas restore in filename order. An
                        # older raw/retry delta for this exact source+shadow must
                        # not regress a terminal native checkpoint.
                        incoming_native = (
                            row["native_index_version"],
                            row["native_index_status"],
                            row["native_index_profile"],
                            row["native_index_memory"],
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
                        incoming_embedding_generation,
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
                        row["native_index_profile"],
                        row["native_index_memory"],
                        row["native_indexed_at"],
                        row["native_index_error"],
                        row["native_generation"],
                        row["embedding_generation"],
                    )

                    def source_metadata_value(name: str) -> str | None:
                        value = incoming(name) if supplied(name) else row[name]
                        return None if value is None else str(value)

                    candidate_metadata_values = tuple(
                        source_metadata_value(name) for name in SOURCE_METADATA_FIELDS
                    )
                    current_metadata_values = tuple(
                        row[name] for name in SOURCE_METADATA_FIELDS
                    )
                    current_metadata_json = self._canonical_metadata_json(
                        row["metadata_json"]
                    )
                    candidate_metadata_json = (
                        self._canonical_metadata_json(metadata)
                        if "metadata" in doc
                        else current_metadata_json
                    )
                    metadata_values = current_metadata_values
                    metadata_json = current_metadata_json
                    current_metadata_clocks = self._source_metadata_clocks(
                        row["source_metadata_clock_json"], row["updated_at"]
                    )
                    metadata_clocks = dict(current_metadata_clocks)
                    joined_values = list(current_metadata_values)
                    for index, (name, current, candidate) in enumerate(
                        zip(
                            SOURCE_METADATA_FIELDS,
                            current_metadata_values,
                            candidate_metadata_values,
                            strict=True,
                        )
                    ):
                        if not supplied(name):
                            continue
                        if (
                            not incoming_updated_explicit
                            and not incoming_metadata_clocks_explicit
                            and candidate == current
                        ):
                            continue
                        incoming_clock = incoming_metadata_clocks[name]
                        clock_order = self._updated_at_key(incoming_clock)
                        current_clock = current_metadata_clocks[name]
                        current_clock_order = self._updated_at_key(current_clock)
                        if clock_order > current_clock_order:
                            joined_values[index] = candidate
                            metadata_clocks[name] = incoming_clock
                        elif clock_order == current_clock_order:
                            joined_values[index] = self._source_metadata_join(
                                current, candidate
                            )
                            metadata_clocks[name] = max(
                                current_clock, incoming_clock
                            )
                    metadata_values = tuple(joined_values)
                    if "metadata" in doc:
                        implicit_unchanged_metadata = (
                            not incoming_updated_explicit
                            and not incoming_metadata_clocks_explicit
                            and candidate_metadata_json == current_metadata_json
                        )
                        incoming_clock = incoming_metadata_clocks["metadata_json"]
                        clock_order = self._updated_at_key(incoming_clock)
                        current_clock = current_metadata_clocks["metadata_json"]
                        current_clock_order = self._updated_at_key(current_clock)
                        if implicit_unchanged_metadata:
                            pass
                        elif clock_order > current_clock_order:
                            metadata_json = candidate_metadata_json
                            metadata_clocks["metadata_json"] = incoming_clock
                        elif clock_order == current_clock_order:
                            metadata_json = max(
                                current_metadata_json, candidate_metadata_json
                            )
                            metadata_clocks["metadata_json"] = max(
                                current_clock, incoming_clock
                            )
                    metadata_updated_at = max(
                        [str(row["updated_at"]), *metadata_clocks.values()],
                        key=lambda value: (self._updated_at_key(value), value),
                    )
                    metadata_clock_json = self._serialize_source_metadata_clocks(
                        metadata_clocks
                    )
                    current_metadata_clock_json = (
                        self._serialize_source_metadata_clocks(
                            current_metadata_clocks
                        )
                    )
                    metadata_changed = (
                        metadata_values != current_metadata_values
                        or metadata_json != current_metadata_json
                        or metadata_clock_json != current_metadata_clock_json
                    )
                    derived_changed = False
                    if derived_supplied and not would_regress and incoming_derived != current_derived:
                        # Raw revisions and derived retrieval shadows have separate
                        # lifecycles.  A reconciled shadow must update in place even
                        # when the source bytes/version are unchanged.
                        self.conn.execute(
                            """UPDATE memories SET retrieval_text=?, search_identifiers=?,
                            translation_hash=?,
                            translation_version=?, translation_status=?, retrieval_updated_at=?,
                            retrieval_generation=?,
                            source_missing=?,
                            native_index_version=?, native_index_status=?, native_index_profile=?,
                            native_index_memory=?, native_indexed_at=?,
                            native_index_error=?, native_generation=?,
                            embedding_generation=? WHERE id=?""",
                            (
                                incoming_derived[0],
                                technical_index_text(raw, incoming_derived[0]),
                                *incoming_derived[1:],
                                row["id"],
                            ),
                        )
                        derived_changed = True
                    if metadata_changed:
                        # Source attribution can move independently of immutable
                        # source bytes and their derived translation/native state.
                        # Keep its timestamp monotonic so an older restored delta
                        # cannot roll the attribution back.
                        self.conn.execute(
                            """UPDATE memories SET
                            device_id=?,project=?,repo=?,worktree=?,source_agent=?,
                            source_type=?,session_id=?,message_id=?,role=?,timestamp=?,
                            source_path=?,agent_type=?,parent_session_id=?,agent_id=?,
                            metadata_json=?,source_metadata_clock_json=?,updated_at=?
                            WHERE id=?""",
                            (
                                *metadata_values,
                                metadata_json,
                                metadata_clock_json,
                                metadata_updated_at,
                                row["id"],
                            ),
                        )
                    if derived_changed or metadata_changed:
                        updated += 1
                        results.append(
                            {
                                "id": row["id"],
                                "status": (
                                    "derived_updated"
                                    if derived_changed
                                    else "metadata_updated"
                                ),
                                "source_identity": source_identity,
                            }
                        )
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
                        """UPDATE memories SET source_version=?, raw_text=?, retrieval_text=?,
                        search_identifiers=?, metadata_json=?, source_metadata_clock_json=?,
                        source_agent=?, source_type=?, device_id=?, project=?, repo=?, worktree=?, session_id=?,
                        message_id=?, role=?, timestamp=?, source_path=?, content_hash=?, updated_at=?, retrieval_updated_at=?, content_type=?,
                        source_missing=?, agent_type=?, parent_session_id=?, agent_id=?, translation_hash=?, translation_version=?, translation_status=?,
                        native_index_version=?, native_index_status=?, native_index_profile=?,
                        native_index_memory=?, native_indexed_at=?, native_index_error=?,
                        retrieval_generation=?, native_generation=?,
                        embedding_generation=? WHERE id=?""",
                        (source_version, raw, retrieval, search_identifiers, json.dumps(metadata, ensure_ascii=False), incoming_metadata_clock_json,
                         values.get("source_agent"), values.get("source_type"), values.get("device_id"),
                         values.get("project"), values.get("repo"), values.get("worktree"), values.get("session_id"),
                         values.get("message_id"), values.get("role"), values.get("timestamp"), values.get("source_path"),
                         content_hash, incoming_updated, values.get("retrieval_updated_at"), values.get("content_type"), values["source_missing"], values.get("agent_type"),
                         values.get("parent_session_id"), values.get("agent_id"), values.get("translation_hash"), values.get("translation_version"), values.get("translation_status"),
                         values.get("native_index_version"), values.get("native_index_status"), values.get("native_index_profile"), values.get("native_index_memory"), values.get("native_indexed_at"), values.get("native_index_error"),
                         values["retrieval_generation"], values["native_generation"],
                         values["embedding_generation"], row["id"]),
                    )
                    updated += 1
                    results.append({"id": row["id"], "status": "updated", "source_identity": source_identity})
                else:
                    cur = self.conn.execute(
                        """INSERT INTO memories(source_identity,source_version,raw_text,retrieval_text,search_identifiers,metadata_json,source_metadata_clock_json,
                        source_agent,source_type,device_id,project,repo,worktree,session_id,message_id,role,timestamp,
                        source_path,content_hash,ingested_at,updated_at,retrieval_updated_at,content_type,source_missing,agent_type,parent_session_id,agent_id,translation_hash,translation_version,translation_status,
                        native_index_version,native_index_status,native_index_profile,native_index_memory,native_indexed_at,native_index_error,
                        retrieval_generation,native_generation,embedding_generation)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (source_identity, source_version, raw, retrieval, search_identifiers, json.dumps(metadata, ensure_ascii=False), incoming_metadata_clock_json,
                         values.get("source_agent"), values.get("source_type"), values.get("device_id"), values.get("project"),
                         values.get("repo"), values.get("worktree"), values.get("session_id"), values.get("message_id"),
                         values.get("role"), values.get("timestamp"), values.get("source_path"), content_hash, values.get("ingested_at") or now, incoming_updated, values.get("retrieval_updated_at"),
                         values.get("content_type"), values["source_missing"], values.get("agent_type"),
                         values.get("parent_session_id"), values.get("agent_id"), values.get("translation_hash"), values.get("translation_version"), values.get("translation_status"),
                         values.get("native_index_version"), values.get("native_index_status"), values.get("native_index_profile"), values.get("native_index_memory"), values.get("native_indexed_at"), values.get("native_index_error"),
                         values["retrieval_generation"], values["native_generation"],
                         values["embedding_generation"]),
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
    def _updated_at_key(value: Any) -> tuple[int, Any]:
        """Order numeric and RFC3339 timestamps on one stable time axis."""
        text = str(value or "")
        try:
            return (2, datetime.fromtimestamp(float(text), timezone.utc).timestamp())
        except (ValueError, OverflowError, OSError):
            pass
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return (2, parsed.timestamp())
        except (ValueError, OverflowError, OSError):
            return (1, text)

    @staticmethod
    def _source_metadata_clocks(value: Any, fallback: Any) -> dict[str, str]:
        if isinstance(value, dict):
            parsed = value
        else:
            try:
                parsed = json.loads(str(value or "{}"))
            except json.JSONDecodeError:
                parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        fallback_text = str(fallback or "")
        return {
            name: str(parsed.get(name) or fallback_text)
            for name in SOURCE_METADATA_CLOCK_FIELDS
        }

    @classmethod
    def _incoming_source_metadata_clocks(
        cls, value: Any, updated_at: Any
    ) -> dict[str, str]:
        clocks = cls._source_metadata_clocks(value, updated_at)
        upper = cls._updated_at_key(updated_at)
        updated_text = str(updated_at or "")
        return {
            name: updated_text if cls._updated_at_key(clock) > upper else clock
            for name, clock in clocks.items()
        }

    @staticmethod
    def _serialize_source_metadata_clocks(clocks: dict[str, str]) -> str:
        return json.dumps(
            clocks, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @staticmethod
    def _canonical_metadata_json(value: Any) -> str:
        try:
            metadata = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError:
            metadata = str(value or "")
        return json.dumps(
            metadata if metadata is not None else {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @staticmethod
    def _source_metadata_join(current: str | None, incoming: str | None) -> str | None:
        """Join one equal-time source field with a stable None/string order."""
        current_key = (0, "") if current is None else (1, str(current))
        incoming_key = (0, "") if incoming is None else (1, str(incoming))
        return incoming if incoming_key > current_key else current

    @staticmethod
    def _newer_timestamp(candidate: Any, current: Any) -> bool:
        if not candidate:
            return False
        if not current:
            return True

        def key(value: Any):
            text = str(value)
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return (1, parsed.timestamp())
            except ValueError:
                return (0, text)

        return key(candidate) > key(current)

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
        out.pop("search_identifiers", None)
        out.pop("native_index_pending", None)
        out[SOURCE_METADATA_CLOCK_KEY] = self._source_metadata_clocks(
            out.pop("source_metadata_clock_json", "{}"), out.get("updated_at")
        )
        out["source_missing"] = bool(out.get("source_missing"))
        try:
            out["metadata"] = json.loads(out.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            out["metadata"] = {}
        return out

    def get(self, ident: str | int) -> dict[str, Any] | None:
        reader = self._read_connection()
        try:
            row = reader.execute(
                "SELECT * FROM memories WHERE source_identity=? ORDER BY id LIMIT 1",
                (str(ident),),
            ).fetchone()
            if row is None:
                row = reader.execute(
                    "SELECT * FROM memories WHERE id=?", (str(ident),)
                ).fetchone()
            return self._row(row) if row else None
        finally:
            reader.close()

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

    def existing_identities(self, identities: list[str]) -> list[str]:
        """Return existing identities in request order without loading raw payloads."""
        unique = list(dict.fromkeys(str(value) for value in identities if value))
        found: set[str] = set()
        with self.lock:
            for begin in range(0, len(unique), 500):
                current = unique[begin : begin + 500]
                placeholders = ",".join("?" for _ in current)
                rows = self.conn.execute(
                    f"SELECT source_identity FROM memories WHERE source_identity IN ({placeholders})",
                    current,
                ).fetchall()
                found.update(str(row["source_identity"]) for row in rows)
        return [value for value in unique if value in found]

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
                    UNION ALL SELECT COALESCE(max(embedding_generation), 0) FROM memories
                    )"""
                ).fetchone()[0]
            )

    def _latest_embedding_generation_locked(self) -> int:
        return int(
            self.conn.execute(
                """SELECT max(value) FROM (
                SELECT COALESCE(max(generation), 0) AS value
                FROM reindex_controls WHERE scope='all'
                UNION ALL
                SELECT COALESCE(max(embedding_generation), 0) FROM memories
                )"""
            ).fetchone()[0]
        )

    def latest_embedding_generation(self) -> int:
        with self.lock:
            return self._latest_embedding_generation_locked()

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
        return str(item.get("content_type") or "").lower() not in LOW_VALUE_CONTENT_TYPES

    def apply_pending_reindex_controls(self, batch_size: int = 500) -> dict[str, int]:
        """Apply one restart-safe row batch without changing raw source truth."""
        applied = retrieval_reset = native_reset = scanned = updated = 0
        with self.lock, self.conn:
            control = self.conn.execute(
                """SELECT generation, scope, row_cursor FROM reindex_controls
                WHERE applied_at IS NULL ORDER BY generation DESC LIMIT 1"""
            ).fetchone()
            if control is None:
                return {
                    "applied": 0,
                    "scanned": 0,
                    "updated": 0,
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
                embedding_generation = int(item.get("embedding_generation") or 0)
                retrieval_values = (
                    item.get("retrieval_text"), item.get("translation_hash"),
                    item.get("translation_version"), item.get("translation_status"),
                    item.get("retrieval_updated_at"),
                )
                native_values = (
                    item.get("native_index_version"), item.get("native_index_status"),
                    item.get("native_index_profile"),
                    item.get("native_index_memory"),
                    item.get("native_indexed_at"), item.get("native_index_error"),
                )
                current_values = (
                    *retrieval_values, retrieval_generation, *native_values,
                    native_generation, embedding_generation,
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
                                native_values = (None, None, None, None, None, None)
                                native_reset += 1
                if scope == "all" and generation > native_generation:
                    native_generation = generation
                    if (
                        self._canonical_reindex_eligible(item)
                        and item.get("native_index_status") != "waiting_durability"
                    ):
                        native_values = (None, None, None, None, None, None)
                        native_reset += 1
                if scope == "all" and generation > embedding_generation:
                    embedding_generation = generation
                    if (
                        self._canonical_reindex_eligible(item)
                        and item.get("native_index_status") != "waiting_durability"
                        and any(value is not None for value in native_values)
                    ):
                        native_values = (None, None, None, None, None, None)
                        native_reset += 1
                next_values = (
                    *retrieval_values, retrieval_generation, *native_values,
                    native_generation, embedding_generation,
                )
                if next_values != current_values:
                    self.conn.execute(
                        """UPDATE memories SET retrieval_text=?, search_identifiers=?,
                        translation_hash=?,
                        translation_version=?, translation_status=?, retrieval_updated_at=?,
                        retrieval_generation=?, native_index_version=?, native_index_status=?,
                        native_index_profile=?, native_index_memory=?, native_indexed_at=?,
                        native_index_error=?, native_generation=?,
                        embedding_generation=?
                        WHERE id=?""",
                        (
                            next_values[0],
                            technical_index_text(item["raw_text"], next_values[0]),
                            *next_values[1:],
                            row["id"],
                        ),
                    )
                    updated += 1
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
            "updated": updated,
            "retrieval_reset": retrieval_reset,
            "native_reset": native_reset,
        }

    def drain_reindex_controls(self, batch_size: int = 500) -> dict[str, int]:
        """Drain controls using bounded transactions; intended for restore only."""
        totals = {
            "applied": 0,
            "scanned": 0,
            "updated": 0,
            "retrieval_reset": 0,
            "native_reset": 0,
        }
        while True:
            result = self.apply_pending_reindex_controls(batch_size)
            for name in totals:
                totals[name] += result[name]
            if not result["scanned"] and not result["applied"]:
                return totals

    def compact_reindex_controls(self, *, replay: bool = False) -> dict[str, int]:
        """Keep only the latest retrieval effect and the latest all effect."""
        with self.lock, self.conn:
            latest = self.conn.execute(
                """SELECT generation, scope FROM reindex_controls
                ORDER BY generation DESC LIMIT 1"""
            ).fetchone()
            if latest is None:
                return {"kept": 0, "deleted": 0}
            keep = {int(latest["generation"])}
            if str(latest["scope"]) != "all":
                latest_all = self.conn.execute(
                    """SELECT generation FROM reindex_controls WHERE scope='all'
                    ORDER BY generation DESC LIMIT 1"""
                ).fetchone()
                if latest_all is not None:
                    keep.add(int(latest_all["generation"]))
            placeholders = ",".join("?" for _ in keep)
            deleted = self.conn.execute(
                f"DELETE FROM reindex_controls WHERE generation NOT IN ({placeholders})",
                tuple(sorted(keep)),
            ).rowcount
            if replay:
                self.conn.execute(
                    "UPDATE reindex_controls SET row_cursor=0, applied_at=NULL"
                )
            return {"kept": len(keep), "deleted": deleted}

    def canonical_index_candidates(
        self, limit: int, profile_fingerprint: str = "", memory: str = ""
    ) -> list[dict[str, Any]]:
        """Return durable raw rows whose native checkpoint is not current."""
        with self.lock, self.conn:
            self._ensure_native_checkpoint_profile_locked(
                str(profile_fingerprint), str(memory)
            )
            rows = self.conn.execute(
                """SELECT * FROM memories WHERE native_index_pending=1
                ORDER BY CASE WHEN native_index_status='retry' THEN 1 ELSE 0 END,
                COALESCE(retrieval_updated_at,updated_at),id LIMIT ?""",
                (max(1, int(limit)),),
            ).fetchall()
            return [self._row(row) for row in rows]

    def native_index_failure_counts(self) -> dict[str, int]:
        """Return allowlisted pending diagnostics without exposing row data."""
        counts = {
            "timeout": 0,
            "native_exit": 0,
            "invalid_report": 0,
            "stale": 0,
            "durability_pending": 0,
            "other": 0,
        }
        categories = {
            "TimeoutExpired": "timeout",
            "native_exit": "native_exit",
            "invalid_report": "invalid_report",
            "native_stale": "stale",
            "durability_pending": "durability_pending",
        }
        with self.lock:
            rows = self.conn.execute(
                """SELECT native_index_error,count(*) AS amount FROM memories
                WHERE native_index_error IS NOT NULL AND (
                    native_index_pending=1 OR native_index_status='waiting_durability'
                ) GROUP BY native_index_error"""
            ).fetchall()
        for row in rows:
            category = categories.get(str(row["native_index_error"]), "other")
            counts[category] += int(row["amount"])
        return counts

    @staticmethod
    def _native_state_fingerprint(state: dict[str, Any]) -> str:
        profile_fingerprint = state.get("profile")
        if profile_fingerprint is None:
            profile_fingerprint = state.get("fingerprint")
        payload = {
            "profile": str(profile_fingerprint or ""),
            "memory": str(state.get("memory") or ""),
            "revision": int(state.get("revision") or 0),
            "eligible": int(state.get("eligible") or 0),
            "indexed": int(state.get("indexed") or 0),
            "held": int(state.get("held") or 0),
            "invalid": int(state.get("invalid") or 0),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def native_index_checkpoint(
        self, profile: dict[str, Any], memory: str = ""
    ) -> dict[str, Any]:
        """Return an O(1) durable checkpoint for one embedding target."""
        fingerprint = str(profile.get("fingerprint", ""))
        memory = str(memory or profile.get("memory") or "")
        with self.lock, self.conn:
            self._ensure_native_checkpoint_profile_locked(fingerprint, memory)
            row = self.conn.execute(
                """SELECT native_index_revision,native_eligible_count,
                native_indexed_count,native_held_count,native_invalid_count
                FROM sync_state WHERE id=1"""
            ).fetchone()
        eligible = int(row["native_eligible_count"] or 0)
        indexed = int(row["native_indexed_count"] or 0)
        held = int(row["native_held_count"] or 0)
        invalid = int(row["native_invalid_count"] or 0)
        revision = int(row["native_index_revision"] or 0)
        pending = max(0, eligible - indexed - held)
        checkpoint = {
            **profile,
            "memory": memory,
            "revision": revision,
            "eligible": eligible,
            "indexed": indexed,
            "held": held,
            "invalid": invalid,
            "pending": pending,
            "complete": pending == 0,
        }
        checkpoint["index_fingerprint"] = self._native_state_fingerprint(checkpoint)
        return checkpoint

    @staticmethod
    def _native_terminal_counts(
        item: dict[str, Any], profile_fingerprint: str, memory: str
    ) -> tuple[int, int, int]:
        if (
            str(item.get("native_index_profile") or "") != profile_fingerprint
            or str(item.get("native_index_memory") or "") != memory
        ):
            return 0, 0, 0
        status = str(item.get("native_index_status") or "")
        return (
            int(status == "indexed"),
            int(status in {"held_secret", "held_invalid"}),
            int(status == "held_invalid"),
        )

    def native_index_state_record(
        self, updates: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Return a raw-free, versioned state record, optionally projected over updates."""
        with self.lock:
            state = self.conn.execute(
                """SELECT native_checkpoint_profile,native_checkpoint_memory,
                native_index_revision,native_eligible_count,native_indexed_count,
                native_held_count,native_invalid_count FROM sync_state WHERE id=1"""
            ).fetchone()
            profile_fingerprint = str(state["native_checkpoint_profile"] or "")
            memory = str(state["native_checkpoint_memory"] or "")
            revision = int(state["native_index_revision"] or 0)
            eligible = int(state["native_eligible_count"] or 0)
            indexed = int(state["native_indexed_count"] or 0)
            held = int(state["native_held_count"] or 0)
            invalid = int(state["native_invalid_count"] or 0)
            projected: dict[str, dict[str, Any]] = {}
            for update in updates or []:
                if update.get("_funes_record") not in (None, "memory"):
                    continue
                identity = str(update.get("source_identity") or "")
                if not identity:
                    continue
                current = projected.get(identity)
                if current is None:
                    row = self.conn.execute(
                        """SELECT source_identity,source_version,content_hash,content_type,
                        native_generation,native_index_version,native_index_status,
                        native_index_profile,native_index_memory,native_indexed_at
                        FROM memories WHERE source_identity=?""",
                        (identity,),
                    ).fetchone()
                    if row is None:
                        continue
                    current = dict(row)
                if (
                    ("source_version" in update and str(update["source_version"] or "") != str(current["source_version"] or ""))
                    or ("content_hash" in update and str(update["content_hash"] or "") != str(current["content_hash"] or ""))
                    or ("native_generation" in update and int(update["native_generation"] or 0) != int(current["native_generation"] or 0))
                ):
                    continue
                desired = dict(current)
                for name in (
                    "native_index_version", "native_index_status",
                    "native_index_profile", "native_index_memory",
                    "native_indexed_at",
                ):
                    if name in update:
                        desired[name] = update.get(name)
                current_native = tuple(
                    current.get(name)
                    for name in (
                        "native_index_version", "native_index_status",
                        "native_index_profile", "native_index_memory",
                        "native_indexed_at",
                    )
                )
                desired_native = tuple(
                    desired.get(name)
                    for name in (
                        "native_index_version", "native_index_status",
                        "native_index_profile", "native_index_memory",
                        "native_indexed_at",
                    )
                )
                if current_native != desired_native and self._canonical_reindex_eligible(current):
                    old_counts = self._native_terminal_counts(
                        current, profile_fingerprint, memory
                    )
                    new_counts = self._native_terminal_counts(
                        desired, profile_fingerprint, memory
                    )
                    revision += 1
                    indexed += new_counts[0] - old_counts[0]
                    held += new_counts[1] - old_counts[1]
                    invalid += new_counts[2] - old_counts[2]
                projected[identity] = desired
        record = {
            "_funes_record": "native_index_state",
            "state_version": NATIVE_CHECKPOINT_STATE_VERSION,
            "profile": profile_fingerprint,
            "memory": memory,
            "revision": revision,
            "eligible": eligible,
            "indexed": indexed,
            "held": held,
            "invalid": invalid,
        }
        record["index_fingerprint"] = self._native_state_fingerprint(record)
        return record

    def set_native_index_state(self, state: dict[str, Any]) -> bool:
        """Monotonically merge raw-free checkpoint metadata during restore."""
        if int(state.get("state_version") or 0) != NATIVE_CHECKPOINT_STATE_VERSION:
            return False
        incoming_revision = int(state.get("revision") or 0)
        counts = [int(state.get(name) or 0) for name in ("eligible", "indexed", "held", "invalid")]
        if incoming_revision < 0 or any(value < 0 for value in counts):
            return False
        profile_fingerprint = str(state.get("profile") or "")
        memory = str(state.get("memory") or "")
        expected = self._native_state_fingerprint(state)
        if state.get("index_fingerprint") not in (None, expected):
            return False
        with self.lock, self.conn:
            current = self.conn.execute(
                """SELECT native_index_revision,native_checkpoint_profile,
                native_checkpoint_memory FROM sync_state WHERE id=1"""
            ).fetchone()
            current_key = (
                int(current["native_index_revision"] or 0),
                str(current["native_checkpoint_profile"] or ""),
                str(current["native_checkpoint_memory"] or ""),
            )
            incoming_key = (incoming_revision, profile_fingerprint, memory)
            if incoming_key < current_key:
                return False
            if self._bulk_restore_depth:
                self.conn.execute(
                    """UPDATE sync_state SET native_checkpoint_profile=?,
                    native_checkpoint_memory=?,native_index_revision=?,
                    native_eligible_count=?,native_indexed_count=?,native_held_count=?,
                    native_invalid_count=? WHERE id=1""",
                    (profile_fingerprint, memory, incoming_revision, *counts),
                )
            elif incoming_revision > current_key[0]:
                self.conn.execute(
                    "UPDATE sync_state SET native_index_revision=? WHERE id=1",
                    (incoming_revision,),
                )
        return True

    def native_optimize_checkpoint(self) -> dict[str, Any]:
        """Return the last durable optimize marker without provider secrets."""
        with self.lock:
            row = self.conn.execute("SELECT * FROM sync_state WHERE id=1").fetchone()
        return {
            "provider": row["native_optimize_provider"],
            "model": row["native_optimize_model"],
            "dimensions": row["native_optimize_dimensions"],
            "schema_version": row["native_optimize_schema_version"],
            "index_layout_version": int(row["native_optimize_layout_version"] or 0),
            "memory": row["native_optimize_memory"],
            "fingerprint": row["native_optimize_fingerprint"],
            "index_fingerprint": row["native_optimize_index_fingerprint"],
            "status": row["native_optimize_status"],
            "optimized_at": row["native_optimized_at"],
            "revision": int(row["native_optimize_revision"] or 0),
        }

    def set_native_optimize_checkpoint(self, checkpoint: dict[str, Any]) -> bool:
        """Apply a monotonic durable optimize marker restored from snapshots/deltas."""
        incoming_at = str(checkpoint.get("optimized_at") or "")
        incoming_revision = int(checkpoint.get("revision") or 0)
        incoming_layout_version = int(checkpoint.get("index_layout_version") or 0)
        if not incoming_at or not checkpoint.get("fingerprint"):
            return False
        with self.lock, self.conn:
            current = self.conn.execute(
                """SELECT native_optimized_at,native_optimize_revision,
                native_optimize_status,native_optimize_memory,
                native_optimize_fingerprint,native_optimize_layout_version
                FROM sync_state WHERE id=1"""
            ).fetchone()
            if current:
                current_revision = int(current["native_optimize_revision"] or 0)
                current_layout_version = int(
                    current["native_optimize_layout_version"] or 0
                )
                current_at = str(current["native_optimized_at"] or "")
                current_namespace = (
                    str(current["native_optimize_memory"] or ""),
                    str(current["native_optimize_fingerprint"] or ""),
                )
                incoming_namespace = (
                    str(checkpoint.get("memory") or ""),
                    str(checkpoint.get("fingerprint") or ""),
                )
                if incoming_layout_version < current_layout_version:
                    return False
                if current_namespace == incoming_namespace:
                    if incoming_revision < current_revision:
                        return False
                    if (
                        incoming_revision == current_revision
                        and current["native_optimize_status"] == "optimized"
                        and checkpoint.get("status") != "optimized"
                    ):
                        return False
                    if incoming_revision == current_revision and current_at > incoming_at:
                        return False
                elif (
                    current_at > incoming_at
                    or (current_at == incoming_at and current_namespace > incoming_namespace)
                ):
                    return False
            self.conn.execute(
                """UPDATE sync_state SET native_optimize_provider=?, native_optimize_model=?,
                native_optimize_dimensions=?, native_optimize_schema_version=?,
                native_optimize_layout_version=?,
                native_optimize_memory=?,
                native_optimize_fingerprint=?, native_optimize_index_fingerprint=?,
                native_optimize_status=?, native_optimized_at=?,
                native_optimize_revision=? WHERE id=1""",
                (
                    checkpoint.get("provider"), checkpoint.get("model"),
                    checkpoint.get("dimensions"), checkpoint.get("schema_version"),
                    incoming_layout_version,
                    checkpoint.get("memory"),
                    checkpoint.get("fingerprint"), checkpoint.get("index_fingerprint"),
                    checkpoint.get("status"), incoming_at, incoming_revision,
                ),
            )
        return True

    def update_native_index(self, updates: list[dict[str, Any]]) -> int:
        """Persist derived native state only when the raw source revision still matches."""
        changed = 0
        with self.lock, self.conn:
            for item in updates:
                cursor = self.conn.execute(
                    """UPDATE memories SET native_index_version=?, native_index_status=?,
                    native_index_profile=?, native_index_memory=?, native_indexed_at=?,
                    native_index_error=?
                    WHERE source_identity=? AND source_version=? AND content_hash=?
                    AND native_generation=?""",
                    (
                        item.get("native_index_version"),
                        item.get("native_index_status"),
                        item.get("native_index_profile"),
                        item.get("native_index_memory"),
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
                    """UPDATE memories SET retrieval_text=raw_text,
                    search_identifiers=funes_identifiers(raw_text,raw_text),
                    translation_status='pending_provider'
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

    def search(
        self,
        query: str,
        limit: int = 20,
        rerank: Any = None,
        filters: dict[str, Any] | None = None,
        *,
        allow_broad_scan: bool = True,
    ) -> list[dict[str, Any]]:
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
        reader = self._read_connection()
        try:
            strict_fts_failed = False
            try:
                rows = reader.execute(
                    """SELECT m.*, bm25(memories_fts, 5.0, 1.0, 2.0) AS score FROM memories_fts
                    JOIN memories m ON m.id=memories_fts.rowid WHERE memories_fts MATCH ?""" + facet +
                    " ORDER BY score LIMIT ?", [query, *params, limit]
                ).fetchall()
            except sqlite3.OperationalError:
                # Let a syntax-safe technical query run before the more
                # expensive substring compatibility path below.
                rows = []
                strict_fts_failed = True
            if cjk_ratio(query) > 0:
                # Mixed Chinese/technical queries are common in agent history.
                # Try their safe ASCII identifiers through FTS before the
                # character-level compatibility scan touches every raw row.
                technical_terms = technical_fts_terms(query)
                facet_terms = {
                    str(value).casefold()
                    for value in filters.values()
                    if value is not None
                }
                content_terms = [
                    term for term in technical_terms if term.casefold() not in facet_terms
                ] or technical_terms
                if technical_terms:
                    technical_rows = reader.execute(
                        """SELECT m.*, bm25(memories_fts, 5.0, 1.0, 2.0) AS score FROM memories_fts
                        JOIN memories m ON m.id=memories_fts.rowid WHERE memories_fts MATCH ?"""
                        + facet
                        + " ORDER BY score LIMIT ?",
                        [
                            technical_fts_query(content_terms),
                            *params,
                            min(100, max(32, limit * 4)),
                        ],
                    ).fetchall()
                    combined = []
                    seen = set()
                    for row in [*rows, *technical_rows]:
                        if row["id"] in seen:
                            continue
                        seen.add(row["id"])
                        combined.append(row)
                    rows = combined
                covered = any(
                    all(
                        term.casefold()
                        in (
                            str(row["retrieval_text"] or "")
                            + " "
                            + str(row["raw_text"] or "")
                        ).casefold()
                        for term in content_terms
                    )
                    for row in rows
                )
                if allow_broad_scan and content_terms and not covered:
                    # unicode61 can merge an identifier with adjacent Chinese
                    # characters. Facet indexes keep this compatibility lookup
                    # bounded for source-filtered agent recalls.
                    technical_clauses = " OR ".join(
                        "(m.retrieval_text LIKE ? ESCAPE '\\' OR m.raw_text LIKE ? ESCAPE '\\')"
                        for _ in content_terms
                    )
                    technical_params = []
                    technical_score = " + ".join(
                        "CASE WHEN (m.retrieval_text LIKE ? ESCAPE '\\' OR m.raw_text LIKE ? ESCAPE '\\') THEN ? ELSE 0 END"
                        for _ in content_terms
                    )
                    score_params = []
                    for term in content_terms:
                        escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                        pattern = f"%{escaped}%"
                        technical_params.extend((pattern, pattern))
                        score_params.extend((pattern, pattern, min(32, len(term))))
                    technical_rows = reader.execute(
                        f"SELECT m.*, 0.0 AS score FROM memories m WHERE ({technical_clauses}){facet} ORDER BY ({technical_score}) DESC, m.updated_at DESC LIMIT ?",
                        [
                            *technical_params,
                            *params,
                            *score_params,
                            min(100, limit * 4),
                        ],
                    ).fetchall()
                    combined = []
                    seen = set()
                    for row in [*technical_rows, *rows]:
                        if row["id"] in seen:
                            continue
                        seen.add(row["id"])
                        combined.append(row)
                    rows = combined
                cjk_terms = cjk_retrieval_terms(query)
                if rows and cjk_terms:
                    rows.sort(
                        key=lambda row: sum(
                            len(term)
                            for term in cjk_terms
                            if term
                            in (
                                str(row["retrieval_text"] or "")
                                + " "
                                + str(row["raw_text"] or "")
                            )
                        ),
                        reverse=True,
                    )
                rows = rows[:limit]
            if allow_broad_scan and not rows and strict_fts_failed:
                # FTS MATCH is intentionally strict; a plain substring fallback keeps recall useful.
                like = "%" + query.replace("%", "\\%") + "%"
                rows = reader.execute("SELECT m.*, 0.0 AS score FROM memories m WHERE (m.retrieval_text LIKE ? ESCAPE '\\' OR m.raw_text LIKE ? ESCAPE '\\')" + facet.replace("m.", "m.") + " ORDER BY m.updated_at DESC LIMIT ?", [like, like, *params, limit]).fetchall()
            if allow_broad_scan and not rows and cjk_ratio(query) > 0:
                # unicode61 does not segment every CJK script consistently;
                # retain a character-level shadow fallback for Chinese recall.
                chars = list(dict.fromkeys(c for c in query if CJK_RE.match(c)))
                if chars:
                    text_clauses = " OR ".join("(m.retrieval_text LIKE ? OR m.raw_text LIKE ?)" for _ in chars)
                    text_params = [v for c in chars for v in (f"%{c}%", f"%{c}%")]
                    rows = reader.execute(f"SELECT m.*, 0.0 AS score FROM memories m WHERE ({text_clauses}){facet} ORDER BY m.updated_at DESC LIMIT ?", [*text_params, *params, limit]).fetchall()
            results = [self._row(r) for r in rows]
        finally:
            reader.close()
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
            optimize = self.native_optimize_checkpoint()
            native_state = self.native_index_state_record()
            path.parent.mkdir(parents=True, exist_ok=True)
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "wt", encoding="utf-8") as f:
                for row in self.conn.execute(
                    "SELECT * FROM memories ORDER BY id"
                ):
                    d = self._row(row)
                    d["_funes_record"] = "memory"
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
                for row in self.conn.execute(
                    "SELECT query,rewritten,created_at,translation_hash,translation_version,translation_status "
                    "FROM translation_cache ORDER BY query"
                ):
                    d = dict(row)
                    d["_funes_record"] = "translation_cache"
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
                for row in self.conn.execute(
                    """SELECT generation,scope,created_at FROM reindex_controls
                    ORDER BY generation"""
                ):
                    d = dict(row)
                    d["_funes_record"] = "reindex_control"
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
                if optimize.get("fingerprint"):
                    optimize["_funes_record"] = "native_optimize_checkpoint"
                    f.write(json.dumps(optimize, ensure_ascii=False) + "\n")
                f.write(json.dumps(native_state, ensure_ascii=False) + "\n")

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
        self.begin_bulk_restore()
        try:
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
                if record_type == "native_optimize_checkpoint":
                    if batch:
                        result = self.ingest(batch)
                        total += result["created"] + result["updated"]
                        batch = []
                    self.set_native_optimize_checkpoint(item)
                    continue
                if record_type == "native_index_state":
                    if batch:
                        result = self.ingest(batch)
                        total += result["created"] + result["updated"]
                        batch = []
                    self.set_native_index_state(item)
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
        finally:
            self.finish_bulk_restore()

    def restore(self, path: Path, *, apply_controls: bool = True) -> int:
        if not path.exists():
            return 0
        opener = gzip.open if path.name.endswith(".gz") else open
        def rows():
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        self.begin_bulk_restore()
        try:
            return self.restore_documents(rows(), apply_controls=apply_controls)
        finally:
            self.finish_bulk_restore()

    def begin_bulk_restore(self) -> None:
        """Suspend per-row indexes while one logical restore is in flight."""
        with self.lock:
            if self._bulk_restore_depth == 0:
                self._save_and_apply_bulk_pragmas_locked()
                try:
                    with self.conn:
                        self._drop_fts_triggers_locked()
                        self._drop_native_state_triggers_locked()
                        self._drop_secondary_indexes_locked()
                except Exception:
                    try:
                        with self.conn:
                            self._create_secondary_indexes_locked()
                            self._create_fts_triggers_locked()
                            self._create_native_state_triggers_locked()
                    finally:
                        self._restore_pragmas_locked()
                    raise
            self._bulk_restore_depth += 1

    def finish_bulk_restore(self) -> None:
        """Rebuild derived indexes once after the outermost restore."""
        with self.lock:
            if self._bulk_restore_depth <= 0:
                raise RuntimeError("bulk restore is not active")
            self._bulk_restore_depth -= 1
            if self._bulk_restore_depth:
                return
            try:
                with self.conn:
                    try:
                        state = self.conn.execute(
                            """SELECT native_checkpoint_profile,native_checkpoint_memory
                            FROM sync_state WHERE id=1"""
                        ).fetchone()
                        if state is not None:
                            self._rebuild_native_checkpoint_state_locked(
                                str(state["native_checkpoint_profile"] or ""),
                                str(state["native_checkpoint_memory"] or ""),
                            )
                        # FTS5 external-content tables need an explicit rebuild after
                        # restoring rows from a JSONL snapshot.
                        self.conn.execute(
                            "INSERT INTO memories_fts(memories_fts) VALUES('rebuild')"
                        )
                    finally:
                        try:
                            self._create_secondary_indexes_locked()
                        finally:
                            try:
                                self._create_fts_triggers_locked()
                            finally:
                                self._create_native_state_triggers_locked()
            finally:
                self._restore_pragmas_locked()


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
        self.mode = os.getenv("FUNES_RETRIEVAL_LANGUAGE_MODE", "raw").lower()
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
        self.upload_lock = threading.RLock()
        self.repo = os.getenv("FUNES_STORAGE_REPO") or os.getenv("FUNES_MEMORY", "")
        self.token = os.getenv("HF_TOKEN", "")
        self.filename = os.getenv("FUNES_SNAPSHOT_FILE", "funes-snapshot.jsonl.gz")
        self.prefix = os.getenv("FUNES_SNAPSHOT_PREFIX", "funes-snapshot-")
        self.delta_prefix = os.getenv("FUNES_DELTA_PREFIX", "funes-delta-")
        self.delta_dir = (os.getenv("FUNES_DELTA_DIR", "deltas") or "").strip("/")
        self.control_prefix = os.getenv("FUNES_REINDEX_PREFIX", "funes-reindex-")
        self.manifest_filename = os.getenv(
            "FUNES_RESTORE_MANIFEST_FILE", "funes-restore-manifest-v1.json"
        )
        self.restore_batch = max(50, int(os.getenv("FUNES_RESTORE_BATCH", "500")))
        try:
            restore_download_workers = int(
                os.getenv("FUNES_RESTORE_DOWNLOAD_WORKERS", "16")
            )
        except ValueError:
            restore_download_workers = 16
        self.restore_download_workers = max(1, min(restore_download_workers, 32))
        self._restore_prefetch_root: Path | None = None
        self._restore_revision: str | None = None
        self.covered_revision: str | None = None
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

    def delta_target(self, digest: str) -> str:
        shard = digest[:2] if len(digest) >= 2 else digest.zfill(2)
        filename = f"{self.delta_prefix}{digest}.jsonl.gz.enc"
        if self.delta_dir:
            return f"{self.delta_dir}/{shard}/{filename}"
        return filename

    @staticmethod
    def _safe_repo_filename(value: Any) -> bool:
        return (
            isinstance(value, str)
            and value not in {".", ".."}
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", value)
            is not None
        )

    @staticmethod
    def _safe_repo_path(value: Any) -> bool:
        if not isinstance(value, str) or not value:
            return False
        parts = value.split("/")
        return all(
            part not in {".", ".."}
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", part) is not None
            for part in parts
        )

    def _is_delta_name(self, name: str) -> bool:
        encrypted_suffixes = (".jsonl.enc", ".jsonl.gz.enc")
        if not (
            isinstance(name, str)
            and self._safe_repo_path(name)
            and name.endswith(encrypted_suffixes)
            and bool(self.delta_prefix)
        ):
            return False
        if "/" not in name:
            return name.startswith(self.delta_prefix)
        if name.count("/") < 2:
            return False
        prefix, shard, filename = name.rsplit("/", 2)
        valid_dirs = {
            d.strip("/")
            for d in (self.delta_dir, "deltas")
            if d and d.strip("/")
        }
        return prefix in valid_dirs and filename.startswith(self.delta_prefix)

    def _validate_restore_manifest(
        self, value: Any, repo_files: set[str]
    ) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != {
            "version",
            "snapshot",
            "deltas",
            "controls",
        }:
            raise ValueError("invalid restore manifest schema")
        if type(value["version"]) is not int or value["version"] != 1:
            raise ValueError("unsupported restore manifest version")
        snapshot = value["snapshot"]
        deltas = value["deltas"]
        controls = value["controls"]
        if not isinstance(deltas, list) or not isinstance(controls, list):
            raise ValueError("invalid restore manifest entries")
        encrypted_suffixes = (".jsonl.enc", ".jsonl.gz.enc")
        if not (
            self._safe_repo_filename(snapshot)
            and snapshot.endswith(encrypted_suffixes)
            and (
                snapshot == self.filename + ".enc"
                or (bool(self.prefix) and snapshot.startswith(self.prefix))
            )
        ):
            raise ValueError("invalid restore manifest snapshot")
        for name in deltas:
            if not self._is_delta_name(name):
                raise ValueError("invalid restore manifest delta")
        if len(deltas) != len(set(deltas)):
            raise ValueError("duplicate restore manifest delta")

        for name in controls:
            if not (
                self._safe_repo_filename(name)
                and bool(self.control_prefix)
                and name.startswith(self.control_prefix)
                and name.endswith(encrypted_suffixes)
            ):
                raise ValueError("invalid restore manifest control")
        if len(controls) != len(set(controls)):
            raise ValueError("duplicate restore manifest control")
        referenced = [snapshot, *deltas, *controls]
        if len(referenced) != len(set(referenced)):
            raise ValueError("duplicate restore manifest entry")
        if any(name not in repo_files for name in referenced):
            raise FileNotFoundError("restore manifest references missing object")
        return {
            "version": 1,
            "snapshot": snapshot,
            "deltas": list(deltas),
            "controls": list(controls),
        }

    def _download_restore_manifest(
        self, repo_files: set[str], revision: str
    ) -> dict[str, Any]:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=self.repo,
            repo_type="dataset",
            filename=self.manifest_filename,
            revision=revision,
            token=self.token,
            local_dir=str(self.store.data_dir / "remote"),
        )
        path = Path(downloaded)
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("invalid restore manifest file")
        return self._validate_restore_manifest(
            json.loads(path.read_text(encoding="utf-8")), repo_files
        )

    def _manifest_bytes(self, manifest: dict[str, Any]) -> bytes:
        if not self._safe_repo_filename(self.manifest_filename):
            raise ValueError("invalid restore manifest filename")
        references = {
            manifest["snapshot"],
            *manifest["deltas"],
            *manifest["controls"],
        }
        validated = self._validate_restore_manifest(manifest, references)
        return (
            json.dumps(
                validated,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    def _commit_with_manifest(
        self,
        api: Any,
        encrypted: Path,
        target: str,
        manifest: dict[str, Any],
        commit_message: str,
        parent_commit: str,
    ) -> Any:
        from huggingface_hub import CommitOperationAdd

        return api.create_commit(
            repo_id=self.repo,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(
                    path_in_repo=target, path_or_fileobj=str(encrypted)
                ),
                CommitOperationAdd(
                    path_in_repo=self.manifest_filename,
                    path_or_fileobj=self._manifest_bytes(manifest),
                ),
            ],
            commit_message=commit_message,
            parent_commit=parent_commit,
        )

    @staticmethod
    def _commit_oid(result: Any) -> str | None:
        oid = getattr(result, "oid", "")
        return oid if isinstance(oid, str) and oid else None

    def _source_artifact_names(self, repo_files: set[str]) -> set[str]:
        encrypted_suffixes = (".jsonl.enc", ".jsonl.gz.enc")
        return {
            name
            for name in repo_files
            if name.endswith(encrypted_suffixes)
            and (
                name == self.filename + ".enc"
                or (bool(self.prefix) and name.startswith(self.prefix))
                or self._is_delta_name(name)
                or (bool(self.control_prefix) and name.startswith(self.control_prefix))
            )
        }

    def _advance_coverage(
        self, state: dict[str, Any], commit_result: Any
    ) -> None:
        oid = self._commit_oid(commit_result)
        if oid and (
            self.covered_revision == state["head"]
            or not self._source_artifact_names(state["repo_files"])
        ):
            self.covered_revision = oid

    def _repo_tree_files(
        self, api: Any, revision: str
    ) -> tuple[set[str], dict[str, str]]:
        repo_files = set()
        blob_ids = {}
        for item in api.list_repo_tree(
            self.repo,
            repo_type="dataset",
            recursive=True,
            revision=revision,
            token=self.token,
        ):
            name = getattr(item, "path", "")
            if isinstance(name, str):
                repo_files.add(name)
                blob_id = getattr(item, "blob_id", "")
                if isinstance(blob_id, str) and blob_id:
                    blob_ids[name] = blob_id
        return repo_files, blob_ids

    def _remote_restore_state(self, api: Any) -> dict[str, Any]:
        info = api.repo_info(
            repo_id=self.repo,
            repo_type="dataset",
            token=self.token,
        )
        head = getattr(info, "sha", "")
        if not isinstance(head, str) or not head:
            raise RuntimeError("Hub repository head is unavailable")
        repo_files, blob_ids = self._repo_tree_files(api, head)
        manifest = None
        if self.manifest_filename in repo_files:
            manifest = self._download_restore_manifest(repo_files, head)
        return {
            "head": head,
            "manifest": manifest,
            "repo_files": repo_files,
            "blob_ids": blob_ids,
        }

    def _commit_manifest_append(
        self,
        api: Any,
        encrypted: Path,
        target: str,
        field: str,
        commit_message: str,
        state: dict[str, Any],
    ) -> tuple[bool, bool]:
        last_error = None
        legacy_target = (
            Path(target).name if field == "deltas" and "/" in target else None
        )
        for _ in range(3):
            manifest = state["manifest"]
            if manifest is None:
                raise RuntimeError("active restore manifest disappeared")
            if target in manifest[field] or (
                legacy_target is not None and legacy_target in manifest[field]
            ):
                return False, True
            updated_manifest = {
                **manifest,
                field: [*manifest[field], target],
            }
            try:
                commit_result = self._commit_with_manifest(
                    api,
                    encrypted,
                    target,
                    updated_manifest,
                    commit_message,
                    state["head"],
                )
                self._advance_coverage(state, commit_result)
                return True, False
            except Exception as exc:
                last_error = exc
                latest = self._remote_restore_state(api)
                if latest["head"] == state["head"]:
                    raise
                state = latest
        manifest = state["manifest"]
        if manifest is not None and (
            target in manifest[field]
            or (legacy_target is not None and legacy_target in manifest[field])
        ):
            return False, True
        raise RuntimeError("restore manifest changed repeatedly") from last_error

    @staticmethod
    def _manifest_suffix(base: list[str], current: list[str]) -> list[str]:
        if current[: len(base)] != base:
            raise RuntimeError("restore manifest history is not append-only")
        return current[len(base) :]

    def _commit_compact_snapshot(
        self,
        api: Any,
        encrypted: Path,
        target: str,
        commit_message: str,
        base_state: dict[str, Any],
    ) -> tuple[Any, bool]:
        base_manifest = base_state["manifest"]
        base_snapshot_blob = None
        if base_manifest is not None:
            base_snapshot_blob = base_state["blob_ids"].get(
                base_manifest["snapshot"]
            )
        state = base_state
        desired_manifest = {
            "version": 1,
            "snapshot": target,
            "deltas": [],
            "controls": [],
        }
        last_error = None
        for _ in range(3):
            try:
                commit_result = self._commit_with_manifest(
                    api,
                    encrypted,
                    target,
                    desired_manifest,
                    commit_message,
                    state["head"],
                )
                retained_suffix = bool(
                    desired_manifest["deltas"] or desired_manifest["controls"]
                )
                return commit_result, retained_suffix
            except Exception as exc:
                last_error = exc
                latest = self._remote_restore_state(api)
                if latest["head"] == state["head"]:
                    raise
                latest_manifest = latest["manifest"]
                if base_manifest is None or latest_manifest is None:
                    raise RuntimeError(
                        "cannot safely merge concurrent snapshot compaction"
                    ) from exc
                latest_snapshot_blob = latest["blob_ids"].get(
                    latest_manifest["snapshot"]
                )
                if (
                    latest_manifest["snapshot"] != base_manifest["snapshot"]
                    or not base_snapshot_blob
                    or latest_snapshot_blob != base_snapshot_blob
                ):
                    raise RuntimeError("concurrent snapshot compaction won") from exc
                desired_manifest = {
                    "version": 1,
                    "snapshot": target,
                    "deltas": self._manifest_suffix(
                        base_manifest["deltas"], latest_manifest["deltas"]
                    ),
                    "controls": self._manifest_suffix(
                        base_manifest["controls"], latest_manifest["controls"]
                    ),
                }
                state = latest
        raise RuntimeError("snapshot compaction changed repeatedly") from last_error

    def _repo_files(self) -> list[str]:
        """List snapshot and delta objects without exposing repository contents."""
        from huggingface_hub import HfApi
        api = HfApi(token=self.token)
        state = self._remote_restore_state(api)
        self._restore_revision = state["head"]
        manifest = state["manifest"]
        repo_files = state["repo_files"]
        if manifest is not None:
            return [
                manifest["snapshot"],
                *manifest["deltas"],
                *manifest["controls"],
            ]
        files = list(self._source_artifact_names(repo_files))
        snapshots = sorted(name for name in files if name == self.filename + ".enc" or name.startswith(self.prefix))
        deltas = sorted(
            (name for name in files if self._is_delta_name(name)),
            key=lambda name: (1 if "/" in name else 0, name),
        )
        controls = sorted(name for name in files if name.startswith(self.control_prefix))
        # Controls replay last and carry monotonic per-derived-field generations.
        # This makes immutable hash-named deltas safe regardless of their order.
        return snapshots + deltas + controls

    def _restore_file(self, filename: str) -> int:
        encrypted = None
        if self._restore_prefetch_root is not None:
            relative = Path(filename)
            candidate = self._restore_prefetch_root / relative
            if (
                not relative.is_absolute()
                and ".." not in relative.parts
                and candidate.is_file()
            ):
                encrypted = candidate
        if encrypted is None:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=self.repo,
                repo_type="dataset",
                filename=filename,
                revision=self._restore_revision,
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

    def _prefetch_restore_files(self, filenames: list[str]) -> Path | None:
        """Download immutable restore inputs concurrently, with serial fallback."""
        if len(filenames) < 2 or self.restore_download_workers <= 1:
            return None
        try:
            from huggingface_hub import snapshot_download

            root = snapshot_download(
                repo_id=self.repo,
                repo_type="dataset",
                token=self.token,
                allow_patterns=filenames,
                revision=self._restore_revision,
                local_dir=str(self.store.data_dir / "remote"),
                max_workers=self.restore_download_workers,
            )
        except Exception:
            # Prefetch is only a latency optimization. The ordered per-file
            # path below remains authoritative and preserves fail-closed restore.
            return None
        return Path(root)

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
                self._restore_revision = None
                self.covered_revision = None
                restored_revision = None
                self.store.begin_bulk_restore()
                try:
                    files = self._repo_files()
                    restored_revision = self._restore_revision
                    if not files:
                        # Backwards-compatible single-file snapshot lookup.
                        files = [self.filename + ".enc"]
                    restored = 0
                    self._restore_prefetch_root = self._prefetch_restore_files(files)
                    try:
                        for filename in files:
                            restored += self._restore_file(filename)
                    finally:
                        self._restore_prefetch_root = None
                    # A complete Hub restore can replay an old generation-zero
                    # revision into an id below a partially persisted row cursor.
                    # Rewind only here; ordinary local restarts keep their cursor.
                    self.store.compact_reindex_controls(replay=True)
                    self.store.drain_reindex_controls(self.restore_batch)
                finally:
                    self.store.finish_bulk_restore()
                self.covered_revision = restored_revision
                self._restore_revision = None
                self.restored = True
                return restored
            except Exception as exc:  # optional recovery must never stop serving
                self._restore_revision = None
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
            self.store.compact_reindex_controls()
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
                    api = HfApi(token=self.token)
                    state = self._remote_restore_state(api)
                    manifest = state["manifest"]
                    if manifest is not None:
                        _, already_uploaded = self._commit_manifest_append(
                            api,
                            encrypted,
                            name,
                            "controls",
                            "funes encrypted reindex control",
                            state,
                        )
                        if already_uploaded:
                            self.store.set_sync(
                                last_sync=utc_now(),
                                snapshot_path=str(encrypted),
                                last_error=None,
                            )
                            return {
                                "uploaded": False,
                                "durable": True,
                                "already_uploaded": True,
                                "generation": generation,
                                "scope": scope,
                            }
                    else:
                        commit_result = api.upload_file(
                            path_or_fileobj=str(encrypted),
                            path_in_repo=name,
                            repo_id=self.repo,
                            repo_type="dataset",
                            commit_message="funes encrypted reindex control",
                        )
                        self._advance_coverage(state, commit_result)
                        latest = self._remote_restore_state(api)
                        if latest["manifest"] is not None:
                            self._commit_manifest_append(
                                api,
                                encrypted,
                                name,
                                "controls",
                                "funes encrypted reindex control",
                                latest,
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
            api = None
            state = None
            if docs is not None:
                durable_docs = list(docs)
                if not any(
                    item.get("_funes_record") == "native_index_state"
                    for item in durable_docs
                ):
                    durable_docs.append(self.store.native_index_state_record())
                digest = hashlib.sha256(json.dumps(durable_docs, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
                # Content addressing makes a retry upload the same immutable
                # object. Restore does not trust filename order: source revisions
                # use their row high-water while source metadata uses persisted
                # per-field clocks with deterministic equal-time joins.
                path = self.store.data_dir / f"{self.delta_prefix}{digest}.jsonl.gz"
                self._write_jsonl_gzip(durable_docs, path)
            else:
                path = self.snapshot_path()
                if self.repo and self.token:
                    try:
                        self._encryption_key()
                        from huggingface_hub import HfApi
                        api = HfApi(token=self.token)
                        state = self._remote_restore_state(api)
                        if (
                            self._source_artifact_names(state["repo_files"])
                            and self.covered_revision != state["head"]
                        ):
                            reason = "remote_history_not_restored"
                            self.store.set_sync(
                                last_error=reason, snapshot_path=str(path)
                            )
                            return {
                                "uploaded": False,
                                "durable": False,
                                "path": str(path),
                                "reason": reason,
                            }
                    except Exception as exc:
                        reason = type(exc).__name__
                        self.store.set_sync(
                            last_error=reason, snapshot_path=str(path)
                        )
                        return {
                            "uploaded": False,
                            "durable": False,
                            "path": str(path),
                            "reason": reason,
                        }
                self.store.snapshot(path)
            if not self.repo or not self.token:
                self.store.set_sync(last_sync=utc_now(), snapshot_path=str(path), last_error=None)
                if self.repo or self._truth("FUNES_REQUIRE_DURABLE_ACK"):
                    return {"uploaded": False, "durable": False, "path": str(path), "reason": "HF storage not configured"}
                return {"uploaded": False, "durable": True, "path": str(path), "reason": "local durable store"}
            if docs is not None:
                try:
                    self._encryption_key()
                except Exception as exc:
                    reason = type(exc).__name__
                    self.store.set_sync(last_error=reason, snapshot_path=str(path))
                    return {
                        "uploaded": False,
                        "durable": False,
                        "path": str(path),
                        "reason": reason,
                    }
            target = (
                self.delta_target(digest)
                if docs is not None
                else self.filename + ".enc"
            )
            legacy_target = (
                f"{path.name}.enc"
                if docs is not None and "/" in target
                else None
            )
            manifest = None
            if docs is not None:
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=self.token)
                    state = self._remote_restore_state(api)
                    manifest = state["manifest"]
                    if manifest is not None:
                        already_durable = (
                            target in manifest["deltas"]
                            or (
                                legacy_target is not None
                                and legacy_target in manifest["deltas"]
                            )
                        )
                    else:
                        already_durable = api.file_exists(
                            repo_id=self.repo,
                            filename=target,
                            repo_type="dataset",
                            revision=state["head"],
                            token=self.token,
                        )
                        if not already_durable and legacy_target is not None:
                            already_durable = api.file_exists(
                                repo_id=self.repo,
                                filename=legacy_target,
                                repo_type="dataset",
                                revision=state["head"],
                                token=self.token,
                            )
                        if already_durable:
                            latest = self._remote_restore_state(api)
                            if latest["manifest"] is not None:
                                state = latest
                                manifest = latest["manifest"]
                                already_durable = (
                                    target in manifest["deltas"]
                                    or (
                                        legacy_target is not None
                                        and legacy_target in manifest["deltas"]
                                    )
                                )
                    if already_durable:
                        path.unlink(missing_ok=True)
                        self.store.set_sync(last_sync=utc_now(), snapshot_path=str(path), last_error=None)
                        return {
                            "uploaded": False,
                            "durable": True,
                            "path": str(path),
                            "already_uploaded": True,
                        }
                except Exception as exc:
                    reason = type(exc).__name__
                    self.store.set_sync(last_error=reason, snapshot_path=str(path))
                    return {
                        "uploaded": False,
                        "durable": False,
                        "path": str(path),
                        "reason": reason,
                    }
            try:
                encrypted = path.with_name(path.name + ".enc")
                self._encrypt_file(path, encrypted)
            except Exception as exc:
                reason = type(exc).__name__
                self.store.set_sync(last_error=reason, snapshot_path=str(path))
                return {"uploaded": False, "durable": False, "path": str(path), "reason": reason}
            try:
                if api is None:
                    from huggingface_hub import HfApi
                    api = HfApi(token=self.token)
                commit_message = (
                    "funes encrypted source delta"
                    if docs is not None
                    else "funes encrypted source snapshot"
                )
                if docs is None:
                    commit_result, retained_suffix = self._commit_compact_snapshot(
                        api,
                        encrypted,
                        target,
                        commit_message,
                        state,
                    )
                    oid = self._commit_oid(commit_result)
                    if oid and not retained_suffix:
                        self.covered_revision = oid
                elif manifest is not None:
                    _, already_uploaded = self._commit_manifest_append(
                        api,
                        encrypted,
                        target,
                        "deltas",
                        commit_message,
                        state,
                    )
                    if already_uploaded:
                        path.unlink(missing_ok=True)
                        encrypted.unlink(missing_ok=True)
                        self.store.set_sync(
                            last_sync=utc_now(),
                            snapshot_path=str(path),
                            last_error=None,
                        )
                        return {
                            "uploaded": False,
                            "durable": True,
                            "path": str(path),
                            "already_uploaded": True,
                        }
                else:
                    commit_result = api.upload_file(
                        path_or_fileobj=str(encrypted),
                        path_in_repo=target,
                        repo_id=self.repo,
                        repo_type="dataset",
                        commit_message=commit_message,
                    )
                    self._advance_coverage(state, commit_result)
                    latest = self._remote_restore_state(api)
                    if latest["manifest"] is not None:
                        self._commit_manifest_append(
                            api,
                            encrypted,
                            target,
                            "deltas",
                            commit_message,
                            latest,
                        )
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


def _same_revision_derived_change(
    item: dict[str, Any],
    metadata: dict[str, Any],
    existing: dict[str, Any],
    raw: str,
) -> bool:
    """Return whether supplied derived state can change the stored revision."""
    def supplied(name: str) -> bool:
        return name in item or name in metadata

    def incoming(name: str) -> Any:
        return item[name] if name in item else metadata.get(name)

    if supplied("source_missing") and bool(incoming("source_missing")) != bool(
        existing.get("source_missing")
    ):
        return True

    current_retrieval_generation = int(existing.get("retrieval_generation") or 0)
    incoming_retrieval_generation = (
        int(incoming("retrieval_generation") or 0)
        if supplied("retrieval_generation")
        else current_retrieval_generation
    )
    if incoming_retrieval_generation > current_retrieval_generation:
        return True
    if incoming_retrieval_generation >= current_retrieval_generation:
        retrieval_values = {
            "retrieval_text": str(incoming("retrieval_text") or normalize_text(raw))
            if supplied("retrieval_text")
            else existing.get("retrieval_text"),
            "translation_hash": incoming("translation_hash")
            if supplied("translation_hash")
            else existing.get("translation_hash"),
            "translation_version": incoming("translation_version")
            if supplied("translation_version")
            else existing.get("translation_version"),
            "translation_status": incoming("translation_status")
            if supplied("translation_status")
            else existing.get("translation_status"),
            "retrieval_updated_at": incoming("retrieval_updated_at")
            if supplied("retrieval_updated_at")
            else existing.get("retrieval_updated_at"),
        }
        if any(retrieval_values[name] != existing.get(name) for name in retrieval_values):
            return True

    current_native_generation = int(existing.get("native_generation") or 0)
    incoming_native_generation = (
        int(incoming("native_generation") or 0)
        if supplied("native_generation")
        else current_native_generation
    )
    if incoming_native_generation > current_native_generation:
        return True
    if incoming_native_generation >= current_native_generation:
        for name in (
            "native_index_version",
            "native_index_status",
            "native_index_profile",
            "native_index_memory",
            "native_indexed_at",
            "native_index_error",
        ):
            if supplied(name) and incoming(name) != existing.get(name):
                return True

    if supplied("embedding_generation"):
        return int(incoming("embedding_generation") or 0) > int(
            existing.get("embedding_generation") or 0
        )
    return False


def prepare_ingest_documents(app: Any, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the raw-first durable representation without calling a provider."""
    prepared = []
    # Reindex generation is constant for this serialized ingest batch.  Query it
    # once: each lookup computes MAX values over the current source table, so a
    # per-document lookup turns a historical backfill into O(batch * rows).
    generation = app.store.latest_reindex_generation()
    embedding_generation = app.store.latest_embedding_generation()
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

        def supplied(name: str) -> bool:
            return name in item or name in metadata

        def incoming(name: str) -> Any:
            return item[name] if name in item else metadata.get(name)

        identity = str(item.get("source_identity") or app.store._identity(item, metadata, raw))
        item["source_identity"] = identity
        source_version = str(item.get("source_version", metadata.get("source_version", "")))
        content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        existing = app.store.get(identity)
        same_revision = bool(
            existing
            and existing.get("content_hash") == content_hash
            and str(existing.get("source_version", "")) == source_version
        )
        derived_change = bool(
            same_revision
            and _same_revision_derived_change(item, metadata, existing, raw)
        )
        if same_revision and not derived_change:
            # Source attribution refreshes are independent of the immutable raw
            # revision. Let Store.ingest preserve every derived generation and
            # checkpoint instead of stamping the latest global reindex epoch.
            prepared.append(item)
            continue
        if same_revision:
            for name in (
                "retrieval_generation",
                "native_generation",
                "embedding_generation",
            ):
                if not supplied(name):
                    item[name] = int(existing.get(name) or 0)
        else:
            item["retrieval_generation"] = generation
            item["native_generation"] = generation
            item["embedding_generation"] = embedding_generation
        same_final_revision = bool(
            same_revision
            and existing.get("translation_status") in FINAL_TRANSLATION_STATUSES
        )
        supplied_retrieval = (
            incoming("retrieval_text") if supplied("retrieval_text") else None
        )
        if same_final_revision and not supplied_retrieval:
            item["retrieval_text"] = existing.get("retrieval_text") or normalize_text(raw)
            for name in ("translation_hash", "translation_version", "translation_status"):
                if not supplied(name) and existing.get(name) is not None:
                    item[name] = existing[name]
            prepared.append(item)
            continue
        if supplied_retrieval:
            prepared.append(item)
            continue
        source_type = str(
            incoming("source_type")
            if supplied("source_type")
            else item.get("kind", "")
        ).lower()
        content_type = str(
            incoming("content_type") if supplied("content_type") else ""
        ).lower()
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
        updated["native_index_profile"] = None
        updated["native_index_memory"] = None
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
            item["native_index_profile"] = None
            item["native_index_memory"] = None
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
        durability_sensitive = {
            str(item["source_identity"])
            for item in result["items"]
            if item["status"] in {"created", "updated", "derived_updated"}
        }
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
                if str(item["source_identity"]) in durability_sensitive
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
        upload_lock = getattr(app.syncer, "upload_lock", None) or threading.RLock()
        with upload_lock:
            sync = app.syncer.upload_reindex_control(control)
            if not sync.get("durable"):
                return {
                    "queued": False,
                    "durable": False,
                    "scope": scope,
                    "error": str(sync.get("reason") or "durability_pending"),
                }
            app.store.record_reindex_control(control)
            app.store.compact_reindex_controls()
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
            self.store.compact_reindex_controls()
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


PROTECTED = {"/ingest", "/search", "/recall", "/get", "/sources", "/sources/check", "/sync/status", "/reindex", "/sync"}


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
            try:
                max_bytes = max(1, int(os.getenv("FUNES_MAX_BODY_BYTES", "20000000")))
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if length < 0 or length > max_bytes:
                raise ValueError("request too large")
            raw = self.rfile.read(length)
            if len(raw) != length:
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
            data = json.loads(decoded or b"{}")
            if not isinstance(data, dict):
                raise ValueError("JSON object required")
            return data

        @staticmethod
        def _public(item: dict[str, Any]) -> dict[str, Any]:
            # HTTP always returns raw source truth, never its English shadow.
            item = dict(item)
            item.pop("retrieval_text", None)
            item.pop(SOURCE_METADATA_CLOCK_KEY, None)
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
                if self.path == "/sources/check":
                    if app.syncer.restoring or app.syncer.restore_failed:
                        error = "restore_in_progress" if app.syncer.restoring else "restore_failed"
                        return self._json(503, {"ok": False, "error": error})
                    identities = validate_source_identity_batch(body.get("source_identities"))
                    present = app.store.existing_identities(identities)
                    present_set = set(present)
                    missing = [identity for identity in identities if identity not in present_set]
                    return self._json(200, {"ok": True, "present": present, "missing": missing})
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
