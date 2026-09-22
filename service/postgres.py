"""PostgreSQL source store; the merge/checkpoint algorithms stay in ``Store``.

Only an explicit migration may initialize/finalize a database. Normal startup
verifies schema, triggers, indexes and a durable ready marker without scanning
source rows. There is no SQLite fallback and no automatic transaction replay.

This is the B-baseline source store: raw text, metadata, identifiers and small
B-tree indexes only. Voyage/Lance owns semantic and lexical retrieval. There is
no PostgreSQL tsvector, GIN, pgvector, content scan, or startup index rebuild.
Equal retrieval/raw text is physically stored once (retrieval_text IS NULL)
and expanded on logical reads, snapshots and native-index handoff.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import threading
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from service.server import (
    MEMORIES_GENERATION_INDEXES,
    MEMORIES_SECONDARY_INDEXES,
    NATIVE_CHECKPOINT_STATE_VERSION,
    Store,
    technical_index_text,
    utc_now,
)

POSTGRES_SCHEMA_VERSION = 2
POSTGRES_STREAM_BATCH = 500
_TEXT_ESCAPE = "\ue000"
_TEXT_DECODE_RE = re.compile(_TEXT_ESCAPE + "([e0])")


def encode_pg_text(value: str) -> str:
    """Reversibly encode NUL (illegal in PG TEXT), including marker collisions.

    Migration COPY must apply this to EVERY source text column, exactly once.
    Ordinary Store calls must pass unencoded values: Connection handles them.
    """
    return value.replace(_TEXT_ESCAPE, _TEXT_ESCAPE + "e").replace("\x00", _TEXT_ESCAPE + "0")


def decode_pg_text(value: str) -> str:
    """Inverse of encode_pg_text; do not apply twice to already-decoded rows."""
    return _TEXT_DECODE_RE.sub(lambda match: _TEXT_ESCAPE if match[1] == "e" else "\x00", value)


def _encode_parameters(value):
    if isinstance(value, str):
        return encode_pg_text(value)
    if isinstance(value, (tuple, list)):
        return type(value)(_encode_parameters(item) for item in value)
    if isinstance(value, dict):
        return {key: _encode_parameters(item) for key, item in value.items()}
    return value


class PostgresNotReady(RuntimeError):
    """Database missing, incomplete, incompatible, or unavailable; never fallback."""


class Row:
    """The name/index/keys protocol Store uses from sqlite3.Row."""

    __slots__ = ("_names", "_positions", "_values")

    def __init__(self, names, values, positions=None):
        self._names = names
        self._positions = positions or {name: i for i, name in enumerate(names)}
        self._values = tuple(values)

    def keys(self):
        return list(self._names)

    def __getitem__(self, key):
        return self._values[self._positions[key] if isinstance(key, str) else key]

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)


def _row_factory(cursor):
    names = tuple(column.name for column in cursor.description or ())
    positions = {name: i for i, name in enumerate(names)}
    def make_row(values):
        decoded = [decode_pg_text(value) if isinstance(value, str) else value for value in values]
        # Only NULL means "same as raw". An intentional empty shadow stays empty.
        if "raw_text" in positions and "retrieval_text" in positions:
            retrieval = positions["retrieval_text"]
            if decoded[retrieval] is None:
                decoded[retrieval] = decoded[positions["raw_text"]]
        return Row(names, decoded, positions)
    return make_row


def _bind_sql(sql: str, *, parameters: bool) -> str:
    """Convert qmarks outside SQL literals/comments; preserve every literal %.

    Psycopg's parameter parser interprets percent signs even inside SQL strings,
    so existing percents must be doubled when a parameter sequence is supplied.
    This is a lexer, not str.replace: quoted ?, dollar bodies, and comments must
    not consume bind parameters. Generated ``%s`` placeholders are never escaped.
    """
    result = []
    i = 0
    quote = None
    escape_quote = False
    dollar = None
    block_depth = 0
    line_comment = False
    while i < len(sql):
        char = sql[i]
        pair = sql[i:i + 2]
        if line_comment:
            if char == "\n":
                line_comment = False
        elif block_depth:
            if pair == "/*":
                block_depth += 1
                result.append(pair)
                i += 2
                continue
            if pair == "*/":
                block_depth -= 1
                result.append(pair)
                i += 2
                continue
        elif dollar:
            if sql.startswith(dollar, i):
                result.append(dollar)
                i += len(dollar)
                dollar = None
                continue
        elif quote:
            if escape_quote and char == "\\" and i + 1 < len(sql):
                result.append("\\" + ("%%" if parameters and sql[i + 1] == "%" else sql[i + 1]))
                i += 2
                continue
            if char == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    result.append(char * 2)
                    i += 2
                    continue
                quote = None
        elif pair == "--":
            line_comment = True
        elif pair == "/*":
            block_depth = 1
            result.append(pair)
            i += 2
            continue
        elif char in ("'", '"'):
            quote = char
            escape_quote = char == "'" and i > 0 and sql[i - 1] in "eE" and (i < 2 or not (sql[i - 2].isalnum() or sql[i - 2] in "_$"))
        elif char == "$":
            match = re.match(r"\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$", sql[i:])
            if match:
                dollar = match.group()
                result.append(dollar)
                i += len(dollar)
                continue
        elif char == "?" and parameters:
            result.append("%s")
            i += 1
            continue
        result.append("%%" if parameters and char == "%" else char)
        i += 1
    return "".join(result)


def _translate_sql(sql: str, *, parameters: bool) -> tuple[str, bool]:
    statement = sql.strip().rstrip(";")
    if re.match(r"INSERT\s+OR\s+IGNORE\s+INTO\s+", statement, re.I):
        statement = re.sub(r"^INSERT\s+OR\s+IGNORE", "INSERT", statement, flags=re.I)
        statement += " ON CONFLICT DO NOTHING"
    elif re.match(r"INSERT\s+OR\s+REPLACE\s+", statement, re.I):
        if not re.match(r"INSERT\s+OR\s+REPLACE\s+INTO\s+translation_cache\s*\(", statement, re.I):
            raise ValueError("unsupported SQLite REPLACE statement")
        statement = re.sub(r"^INSERT\s+OR\s+REPLACE", "INSERT", statement, flags=re.I)
        statement += " ON CONFLICT (query) DO UPDATE SET " + ",".join(
            f"{name}=EXCLUDED.{name}" for name in (
                "rewritten", "created_at", "translation_hash",
                "translation_version", "translation_status",
            )
        )
    # PostgreSQL 14 requires a name for the two generation UNION subqueries.
    if re.match(r"SELECT\s+max\(value\)\s+FROM\s*\(", statement, re.I):
        statement += " AS generations"
    returns_id = bool(re.match(r"INSERT\s+INTO\s+memories\s*\(", statement, re.I))
    if returns_id:
        if re.search(r"\bRETURNING\b", statement, re.I):
            raise ValueError("Store adapter owns INSERT memories RETURNING id")
        statement += " RETURNING id"
    return _bind_sql(statement, parameters=parameters), returns_id


class Cursor:
    def __init__(self, cursor, *, returns_id: bool = False):
        self._cursor = cursor
        self.rowcount = cursor.rowcount
        self.lastrowid = None
        if returns_id:
            row = cursor.fetchone()
            self.lastrowid = row[0] if row else None
            cursor.close()
        elif cursor.description is None:
            cursor.close()

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        try:
            return self._cursor.fetchall()
        finally:
            self._cursor.close()

    def fetchmany(self, size: int = POSTGRES_STREAM_BATCH):
        return self._cursor.fetchmany(size)

    def close(self):
        self._cursor.close()

    def __iter__(self):
        try:
            yield from self._cursor
        finally:
            self._cursor.close()


class Connection:
    """SQLite-style transaction context that NEVER closes the psycopg connection.

    Autocommit outside this context avoids leaked idle read transactions. Writers
    share a schema-scoped advisory lock, preserving Store's read/merge/write
    semantics even across processes. Nested contexts are psycopg savepoints.
    """

    def __init__(self, raw, *, require_ready: bool = True):
        self.raw = raw
        self.require_ready = require_ready
        self._transactions = []
        self._trace = None

    def execute(self, sql: str, parameters=None):
        if self.raw.closed:
            raise PostgresNotReady("PostgreSQL connection closed; explicit reconnect required")
        if self._trace:
            self._trace(sql)
        sql, returns_id = _translate_sql(sql, parameters=parameters is not None)
        return Cursor(self.raw.execute(sql, _encode_parameters(parameters)), returns_id=returns_id)

    def __enter__(self):
        transaction = self.raw.transaction()
        transaction.__enter__()
        try:
            if not self._transactions:
                # Do not put identities, text, passwords or DSNs into lock keys.
                self.raw.execute(
                    "SELECT pg_advisory_xact_lock("
                    "hashtextextended(current_schema() || ':funes-store-writer', 0))"
                ).close()
                if self.require_ready:
                    _verify_marker(self.raw)
        except BaseException as error:
            transaction.__exit__(type(error), error, error.__traceback__)
            raise
        self._transactions.append(transaction)
        return self

    def __exit__(self, kind, value, traceback):
        return self._transactions.pop().__exit__(kind, value, traceback)

    def commit(self):
        self.raw.commit()

    def rollback(self):
        self.raw.rollback()

    def close(self):
        self.raw.close()

    def set_trace_callback(self, callback):
        self._trace = callback


def _verify_marker(raw, *, require_ready: bool = True):
    try:
        with raw.cursor() as cursor:
            cursor.execute(
                "SELECT schema_version,migration_ready,document_count FROM funes_schema_state WHERE id=1"
            )
            state = cursor.fetchone()
    except Exception:
        raise PostgresNotReady("PostgreSQL source schema is missing or unavailable; run explicit migration") from None
    if not state or int(state[0]) != POSTGRES_SCHEMA_VERSION:
        raise PostgresNotReady("PostgreSQL source schema version is incompatible")
    if require_ready and not state[1]:
        raise PostgresNotReady("PostgreSQL source migration is not finalized")


# These are source columns, not a second implementation of Store's merge rules.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    source_identity TEXT NOT NULL UNIQUE,
    source_version TEXT NOT NULL DEFAULT '',
    raw_text TEXT NOT NULL, retrieval_text TEXT,
    search_identifiers TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    source_metadata_clock_json TEXT NOT NULL DEFAULT '{}',
    source_agent TEXT, source_type TEXT, device_id TEXT, project TEXT,
    repo TEXT, worktree TEXT, session_id TEXT, message_id TEXT, role TEXT,
    timestamp TEXT, source_path TEXT, content_hash TEXT NOT NULL,
    ingested_at TEXT NOT NULL, updated_at TEXT NOT NULL, retrieval_updated_at TEXT,
    content_type TEXT, source_missing INTEGER NOT NULL DEFAULT 0,
    agent_type TEXT, parent_session_id TEXT, agent_id TEXT,
    translation_hash TEXT, translation_version TEXT, translation_status TEXT,
    native_index_version TEXT, native_index_status TEXT, native_index_profile TEXT,
    native_index_memory TEXT, native_indexed_at TEXT, native_index_error TEXT,
    native_index_pending INTEGER NOT NULL DEFAULT 1,
    retrieval_generation BIGINT NOT NULL DEFAULT 0,
    native_generation BIGINT NOT NULL DEFAULT 0,
    embedding_generation BIGINT NOT NULL DEFAULT 0
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
    native_optimize_memory TEXT, native_optimize_fingerprint TEXT,
    native_optimize_index_fingerprint TEXT, native_optimize_status TEXT,
    native_optimized_at TEXT, native_optimize_revision BIGINT NOT NULL DEFAULT 0,
    native_checkpoint_profile TEXT NOT NULL DEFAULT '',
    native_checkpoint_memory TEXT NOT NULL DEFAULT '',
    native_index_revision BIGINT NOT NULL DEFAULT 0,
    native_eligible_count BIGINT NOT NULL DEFAULT 0,
    native_indexed_count BIGINT NOT NULL DEFAULT 0,
    native_held_count BIGINT NOT NULL DEFAULT 0,
    native_invalid_count BIGINT NOT NULL DEFAULT 0,
    native_checkpoint_state_version INTEGER NOT NULL DEFAULT 0,
    fts_schema_version INTEGER NOT NULL DEFAULT 0,
    fts_ready INTEGER NOT NULL DEFAULT 0
);
INSERT INTO sync_state(id) VALUES(1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS reindex_controls (
    generation BIGINT PRIMARY KEY, scope TEXT NOT NULL, created_at TEXT NOT NULL,
    row_cursor BIGINT NOT NULL DEFAULT 0, applied_at TEXT
);
CREATE TABLE IF NOT EXISTS funes_schema_state (
    id INTEGER PRIMARY KEY CHECK(id=1), schema_version INTEGER NOT NULL,
    migration_ready BOOLEAN NOT NULL DEFAULT FALSE,
    document_count BIGINT NOT NULL DEFAULT 0,
    migration_source TEXT, migrated_at TEXT
);
"""

_NATIVE_FIELDS = (
    "source_identity", "source_version", "raw_text", "retrieval_text",
    "content_hash", "content_type", "source_missing", "translation_hash",
    "translation_version", "native_index_version", "native_index_status",
    "native_index_profile", "native_index_memory", "native_indexed_at",
    "native_generation", "embedding_generation",
)


def _native_trigger_sql() -> str:
    changed = " OR ".join(
        "COALESCE(OLD.retrieval_text,OLD.raw_text) IS DISTINCT FROM "
        "COALESCE(NEW.retrieval_text,NEW.raw_text)" if key == "retrieval_text"
        else f"OLD.{key} IS DISTINCT FROM NEW.{key}"
        for key in _NATIVE_FIELDS
    )
    return f"""
CREATE OR REPLACE FUNCTION funes_native_state_change()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    checkpoint_profile TEXT;
    checkpoint_memory TEXT;
    old_eligible BOOLEAN := FALSE;
    new_eligible BOOLEAN := FALSE;
    old_current BOOLEAN := FALSE;
    new_current BOOLEAN := FALSE;
    old_status TEXT := '';
    new_status TEXT := '';
BEGIN
    IF TG_OP = 'UPDATE' AND NOT ({changed}) THEN
        RETURN NEW;
    END IF;
    SELECT native_checkpoint_profile,native_checkpoint_memory
    INTO STRICT checkpoint_profile,checkpoint_memory FROM sync_state WHERE id=1 FOR UPDATE;
    IF TG_OP = 'INSERT' THEN
        UPDATE funes_schema_state SET document_count=document_count+1 WHERE id=1;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE funes_schema_state SET document_count=document_count-1 WHERE id=1;
    END IF;
    IF TG_OP != 'INSERT' THEN
        old_eligible := lower(COALESCE(OLD.content_type,'')) NOT IN
            ('tool_call','tool_result','shell_output','progress');
        old_current := COALESCE(OLD.native_index_profile,'')=checkpoint_profile
            AND COALESCE(OLD.native_index_memory,'')=checkpoint_memory;
        old_status := COALESCE(OLD.native_index_status,'');
    END IF;
    IF TG_OP != 'DELETE' THEN
        new_eligible := lower(COALESCE(NEW.content_type,'')) NOT IN
            ('tool_call','tool_result','shell_output','progress');
        new_current := COALESCE(NEW.native_index_profile,'')=checkpoint_profile
            AND COALESCE(NEW.native_index_memory,'')=checkpoint_memory;
        new_status := COALESCE(NEW.native_index_status,'');
    END IF;
    UPDATE sync_state SET
        native_index_revision=native_index_revision+(old_eligible OR new_eligible)::int,
        native_eligible_count=native_eligible_count+new_eligible::int-old_eligible::int,
        native_indexed_count=native_indexed_count
            +(new_eligible AND new_current AND new_status='indexed')::int
            -(old_eligible AND old_current AND old_status='indexed')::int,
        native_held_count=native_held_count
            +(new_eligible AND new_current AND new_status IN ('held_secret','held_invalid'))::int
            -(old_eligible AND old_current AND old_status IN ('held_secret','held_invalid'))::int,
        native_invalid_count=native_invalid_count
            +(new_eligible AND new_current AND new_status='held_invalid')::int
            -(old_eligible AND old_current AND old_status='held_invalid')::int
    WHERE id=1;
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
END
$$;
CREATE OR REPLACE FUNCTION funes_native_pending_change()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    checkpoint_profile TEXT;
    checkpoint_memory TEXT;
BEGIN
    NEW.retrieval_text := NULLIF(NEW.retrieval_text, NEW.raw_text);
    IF TG_OP = 'UPDATE' AND NOT ({changed}) THEN RETURN NEW; END IF;
    SELECT native_checkpoint_profile,native_checkpoint_memory
    INTO STRICT checkpoint_profile,checkpoint_memory FROM sync_state WHERE id=1 FOR UPDATE;
    NEW.native_index_pending := CASE WHEN
        lower(COALESCE(NEW.content_type,'')) NOT IN
            ('tool_call','tool_result','shell_output','progress')
        AND COALESCE(NEW.native_index_status,'') != 'waiting_durability'
        AND NOT (COALESCE(NEW.native_index_status,'') IN ('indexed','held_secret','held_invalid')
            AND COALESCE(NEW.native_index_profile,'')=checkpoint_profile
            AND COALESCE(NEW.native_index_memory,'')=checkpoint_memory)
        THEN 1 ELSE 0 END;
    RETURN NEW;
END
$$;
DROP TRIGGER IF EXISTS memories_native_state ON memories;
CREATE TRIGGER memories_native_state AFTER INSERT OR UPDATE OR DELETE ON memories
FOR EACH ROW EXECUTE FUNCTION funes_native_state_change();
DROP TRIGGER IF EXISTS memories_native_pending ON memories;
CREATE TRIGGER memories_native_pending BEFORE INSERT OR UPDATE ON memories
FOR EACH ROW EXECUTE FUNCTION funes_native_pending_change();
"""


class PostgresStore(Store):
    """Reuse Store business rules against an explicitly migrated PostgreSQL DB.

    ``initialize=True`` is reserved for an offline migration/test. It creates an
    unready schema; call ``finalize_migration`` only after import and validation.
    DSN options/search_path may select an isolated schema for tests. Credentials
    are never logged, included in repr, or used as a fallback backend selector.
    """

    backend = "postgres"

    def __init__(self, data_dir: str, dsn: str | None = None, initialize: bool = False):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._dsn = dsn or os.getenv("FUNES_POSTGRES_DSN", "")
        if not self._dsn:
            raise PostgresNotReady("FUNES_POSTGRES_DSN is required for PostgreSQL source storage")
        self._migration_mode = bool(initialize)
        self._bulk_restore_depth = 0
        self._bulk_restore_prev_pragmas = None
        raw = self._connect()
        self.conn = Connection(raw, require_ready=not initialize)
        try:
            if initialize:
                self.initialize_schema()
            self.verify_schema(require_ready=not initialize)
        except BaseException:
            raw.close()
            raise

    def _connect(self):
        try:
            import psycopg
        except ImportError:
            raise PostgresNotReady("PostgreSQL source storage requires psycopg 3") from None
        try:
            return psycopg.connect(
                self._dsn, autocommit=True, row_factory=_row_factory,
                connect_timeout=10, application_name="funes-source-store",
            )
        except Exception:
            raise PostgresNotReady("PostgreSQL source connection unavailable") from None

    def initialize_schema(self) -> None:
        """Explicit DDL only; never marks a new/partial import ready."""
        if not self._migration_mode:
            raise PostgresNotReady("Schema initialization requires explicit migration mode")
        with self.lock, self.conn:
            self.conn.raw.execute(_SCHEMA).close()
            self.conn.execute(
                "INSERT INTO funes_schema_state(id,schema_version) VALUES(1,?) ON CONFLICT DO NOTHING",
                (POSTGRES_SCHEMA_VERSION,),
            )
            _verify_marker(self.conn.raw, require_ready=False)
            self._create_generation_indexes_locked()
            self._create_secondary_indexes_locked()
            self.conn.raw.execute(_native_trigger_sql()).close()

    def verify_schema(self, *, require_ready: bool = True) -> None:
        """Catalog/one-row checks only. No startup DDL, rebuild, COUNT or row scan."""
        with self.lock:
            raw = self.conn.raw
            _verify_marker(raw, require_ready=require_ready)
            expected_columns = {
                "memories": {
                    "id", "source_identity", "source_version", "raw_text", "retrieval_text",
                    "search_identifiers", "metadata_json", "source_metadata_clock_json",
                    "native_index_pending",
                },
                "translation_cache": {"query", "rewritten", "created_at", "translation_hash", "translation_version", "translation_status"},
                "sync_state": {"id", "fts_ready", "fts_schema_version", "native_checkpoint_state_version"},
                "reindex_controls": {"generation", "scope", "created_at", "row_cursor", "applied_at"},
            }
            # Schema declarations, rather than live row inspection, define all
            # fields needed by inherited ingest/checkpoint methods.
            from service.server import FIELDS
            expected_columns["memories"].update(FIELDS)
            sync_definition = _SCHEMA.split("CREATE TABLE IF NOT EXISTS sync_state (", 1)[1].split("\n);", 1)[0]
            expected_columns["sync_state"].update(re.findall(r"\b([a-z_]+)\s+(?:TEXT|INTEGER|BIGINT)\b", sync_definition))
            with raw.cursor() as cursor:
                cursor.execute(
                    "SELECT table_name,column_name FROM information_schema.columns "
                    "WHERE table_schema=current_schema() AND table_name = ANY(%s)",
                    (list(expected_columns),),
                )
                actual = {}
                for row in cursor:
                    actual.setdefault(row[0], set()).add(row[1])
                if any(not columns <= actual.get(table, set()) for table, columns in expected_columns.items()):
                    raise PostgresNotReady("PostgreSQL source schema is incomplete")
                cursor.execute(
                    "SELECT column_name,data_type,is_nullable,is_generated FROM information_schema.columns "
                    "WHERE table_schema=current_schema() AND table_name='memories'"
                )
                definitions = list(cursor)
                if any(row[1] == "tsvector" or row[3] != "NEVER" for row in definitions):
                    raise PostgresNotReady("PostgreSQL v2 source must not contain derived search vectors")
                if not any(row[0] == "retrieval_text" and row[2] == "YES" for row in definitions):
                    raise PostgresNotReady("PostgreSQL v2 requires nullable retrieval_text")
                cursor.execute(
                    "SELECT a.amname FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
                    "JOIN pg_am a ON a.oid=c.relam WHERE i.indrelid='memories'::regclass"
                )
                if any(row[0] != "btree" for row in cursor):
                    raise PostgresNotReady("PostgreSQL v2 source supports only B-tree indexes")
                cursor.execute(
                    "SELECT indexrelid::regclass::text FROM pg_index "
                    "WHERE indrelid='memories'::regclass AND indisvalid AND indisready"
                )
                indexes = {str(row[0]).split(".")[-1].strip('"') for row in cursor}
                required = {"memories_pkey", "memories_source_identity_key"}
                if require_ready:
                    required.update(name for name, _ in MEMORIES_GENERATION_INDEXES)
                    required.update(name for name, _ in MEMORIES_SECONDARY_INDEXES)
                if not required <= indexes:
                    raise PostgresNotReady("PostgreSQL source indexes are incomplete")
                cursor.execute(
                    "SELECT tgname,tgenabled FROM pg_trigger WHERE tgrelid='memories'::regclass "
                    "AND NOT tgisinternal"
                )
                triggers = {row[0]: row[1] for row in cursor}
                valid_modes = {"O", "A"} if require_ready else {"O", "A", "D"}
                if (set(triggers) != {"memories_native_state", "memories_native_pending"}
                        or len(set(triggers.values())) != 1
                        or not set(triggers.values()) <= valid_modes):
                    raise PostgresNotReady("PostgreSQL source native trigger is missing or disabled")
                cursor.execute("SELECT fts_ready,fts_schema_version,native_checkpoint_state_version FROM sync_state WHERE id=1")
                state = cursor.fetchone()
                if not state or (require_ready and (
                    bool(state[0]) or int(state[1]) != 0
                    or int(state[2]) != NATIVE_CHECKPOINT_STATE_VERSION
                )):
                    raise PostgresNotReady("PostgreSQL source checkpoint or disabled-FTS marker is invalid")

    def prepare_bulk_migration(self, *, defer_secondary_indexes: bool = True) -> None:
        """Offline COPY preparation, strictly limited to an UNREADY database.

        Keep PK/unique constraints; defer optional nonunique indexes and disable
        native row triggers to avoid millions of sync_state updates/WAL dead
        tuples. No derived full-text data is created during or after COPY.

        COPY callers must collapse equal retrieval/raw text to NULL before
        encoding each text field with encode_pg_text, copy all
        sync_state and control fields, then call finalize_migration. Interrupted
        imports remain unready; no runtime path may use this optimization.
        """
        if not self._migration_mode:
            raise PostgresNotReady("Bulk preparation requires explicit migration mode")
        with self.lock, self.conn:
            row = self.conn.execute("SELECT migration_ready FROM funes_schema_state WHERE id=1").fetchone()
            if row is None or row[0]:
                raise PostgresNotReady("Bulk preparation requires an unready source database")
            self.conn.execute("ALTER TABLE memories DISABLE TRIGGER memories_native_state")
            self.conn.execute("ALTER TABLE memories DISABLE TRIGGER memories_native_pending")
            if defer_secondary_indexes:
                for name, _ in (*MEMORIES_GENERATION_INDEXES, *MEMORIES_SECONDARY_INDEXES):
                    self.conn.execute(f"DROP INDEX IF EXISTS {name}")

    def _finalize_native_migration_locked(self) -> None:
        """Explicit scan, but update only wrong pending flags (avoid table bloat).

        This is migration-specific derived-state reconstruction, not an ingest
        merge implementation. Preserve the imported native revision exactly.
        """
        pending = """CASE WHEN
            lower(COALESCE(m.content_type,'')) NOT IN
                ('tool_call','tool_result','shell_output','progress')
            AND COALESCE(m.native_index_status,'')!='waiting_durability'
            AND NOT (COALESCE(m.native_index_status,'') IN
                ('indexed','held_secret','held_invalid')
                AND COALESCE(m.native_index_profile,'')=s.native_checkpoint_profile
                AND COALESCE(m.native_index_memory,'')=s.native_checkpoint_memory)
            THEN 1 ELSE 0 END"""
        self.conn.execute(f"""UPDATE memories m SET native_index_pending=({pending})
            FROM sync_state s WHERE s.id=1
            AND m.native_index_pending IS DISTINCT FROM ({pending})""")
        self.conn.execute("""WITH counts AS (
            SELECT count(*) AS eligible,
                count(*) FILTER (WHERE m.native_index_status='indexed'
                    AND COALESCE(m.native_index_profile,'')=s.native_checkpoint_profile
                    AND COALESCE(m.native_index_memory,'')=s.native_checkpoint_memory) AS indexed,
                count(*) FILTER (WHERE m.native_index_status IN ('held_secret','held_invalid')
                    AND COALESCE(m.native_index_profile,'')=s.native_checkpoint_profile
                    AND COALESCE(m.native_index_memory,'')=s.native_checkpoint_memory) AS held,
                count(*) FILTER (WHERE m.native_index_status='held_invalid'
                    AND COALESCE(m.native_index_profile,'')=s.native_checkpoint_profile
                    AND COALESCE(m.native_index_memory,'')=s.native_checkpoint_memory) AS invalid
            FROM memories m CROSS JOIN sync_state s WHERE s.id=1
                AND lower(COALESCE(m.content_type,'')) NOT IN
                    ('tool_call','tool_result','shell_output','progress'))
            UPDATE sync_state SET native_eligible_count=counts.eligible,
                native_indexed_count=counts.indexed,native_held_count=counts.held,
                native_invalid_count=counts.invalid,native_checkpoint_state_version=?
            FROM counts WHERE sync_state.id=1""", (NATIVE_CHECKPOINT_STATE_VERSION,))

    def finalize_migration(self, expected_count: int | None = None, *, source: str = "") -> dict[str, Any]:
        """Explicit final scans/index build; preserve revisions and optimize state.

        Importers must copy *all* sync_state columns before this call. Only
        incorrect pending flags are updated (unchanged source rows are not
        rewritten). Counts are recomputed for the imported profile. Deferred
        indexes/triggers are restored before the ready marker is committed.
        Any failure leaves the previously-unready database closed to runtime.
        """
        if not self._migration_mode:
            raise PostgresNotReady("Finalization requires explicit migration mode")
        with self.lock, self.conn:
            actual_count = int(self.conn.execute("SELECT count(*) FROM memories").fetchone()[0])
            if expected_count is not None and actual_count != int(expected_count):
                raise PostgresNotReady("PostgreSQL migration row count does not match source")
            self._finalize_native_migration_locked()
            self._create_generation_indexes_locked()
            self._create_secondary_indexes_locked()
            self.conn.execute("ALTER TABLE memories ENABLE TRIGGER memories_native_state")
            self.conn.execute("ALTER TABLE memories ENABLE TRIGGER memories_native_pending")
            # Explicit-ID imports must advance the identity sequence before the
            # next inherited Store.ingest INSERT asks for RETURNING id.
            self.conn.execute(
                "SELECT setval(pg_get_serial_sequence('memories','id'),"
                "GREATEST(COALESCE((SELECT max(id) FROM memories),0),1),"
                "EXISTS(SELECT 1 FROM memories))"
            )
            self.conn.execute("UPDATE sync_state SET fts_ready=0,fts_schema_version=0 WHERE id=1")
            self.conn.execute(
                "UPDATE funes_schema_state SET migration_ready=TRUE,migration_source=?,migrated_at=?,document_count=? WHERE id=1",
                (source, utc_now(), actual_count),
            )
            self.verify_schema()
        return {"documents": actual_count, "schema_version": POSTGRES_SCHEMA_VERSION, "ready": True}

    def reconnect(self) -> None:
        """Explicit fail-closed reconnection; no interrupted write is replayed."""
        with self.lock:
            old = self.conn
            replacement = Connection(self._connect(), require_ready=not self._migration_mode)
            self.conn = replacement
            try:
                self.verify_schema(require_ready=not self._migration_mode)
            except BaseException:
                replacement.close()
                self.conn = old
                raise
            old.close()

    def _read_connection(self):
        raw = self._connect()
        try:
            _verify_marker(raw, require_ready=not self._migration_mode)
            raw.execute("SET default_transaction_read_only=on").close()
            return Connection(raw, require_ready=not self._migration_mode)
        except BaseException:
            raw.close()
            raise

    def _row(self, row):
        result = super()._row(row)
        if result.get("retrieval_text") is None and "raw_text" in result:
            result["retrieval_text"] = result["raw_text"]
        return result

    def get(self, ident: str | int) -> dict[str, Any] | None:
        with closing(self._read_connection()) as reader:
            identity = str(ident)
            row = reader.execute("SELECT * FROM memories WHERE source_identity=? ORDER BY id LIMIT 1", (identity,)).fetchone()
            # A filename/UUID must never be bound against bigint (PostgreSQL
            # rejects it, unlike SQLite). Guard both numeric syntax and range.
            if row is None and re.fullmatch(r"[+]?\d+", identity, flags=re.ASCII) and len(identity) <= 20:
                numeric = int(identity)
                if 0 <= numeric <= 9223372036854775807:
                    row = reader.execute("SELECT * FROM memories WHERE id=?", (numeric,)).fetchone()
            return self._row(row) if row else None

    def _create_secondary_indexes_locked(self):
        for name, sql in MEMORIES_SECONDARY_INDEXES:
            if name == "memories_canonical_pending_idx":
                sql = sql.replace("CASE WHEN native_index_status='retry' THEN 1 ELSE 0 END,", "(CASE WHEN native_index_status='retry' THEN 1 ELSE 0 END),")
            self.conn.execute(sql)

    def mark_translations_pending(self, documents: list[dict[str, Any]]) -> None:
        """Requeue without SQLite functions or comparing compact NULL as text.

        Lock the matching revision and derive identifiers from its actual raw
        value, never the caller's possibly stale payload. Preserve Store's CAS
        guards so an older durability failure cannot undo a newer translation.
        """
        predicate = """source_identity=? AND content_hash=? AND source_version=?
            AND COALESCE(retrieval_text,raw_text)=? AND translation_status=?
            AND retrieval_generation=?"""
        with self.lock, self.conn:
            for item in documents:
                parameters = (
                    item.get("source_identity"), item.get("content_hash"),
                    str(item.get("source_version", "")), item.get("retrieval_text"),
                    item.get("translation_status"), int(item.get("retrieval_generation") or 0),
                )
                row = self.conn.execute(
                    "SELECT raw_text FROM memories WHERE " + predicate + " FOR UPDATE",
                    parameters,
                ).fetchone()
                if row is None:
                    continue
                identifiers = technical_index_text(row["raw_text"], row["raw_text"])
                self.conn.execute(
                    "UPDATE memories SET retrieval_text=NULL,search_identifiers=?,"
                    "translation_status='pending_provider' WHERE " + predicate,
                    (identifiers, *parameters),
                )

    def begin_bulk_restore(self):
        """Keep source indexes/triggers live; FTS remains owned by Lance."""
        with self.lock:
            self._bulk_restore_depth += 1

    def finish_bulk_restore(self, *, rebuild_fts: bool | None = None):
        with self.lock, self.conn:
            if self._bulk_restore_depth <= 0:
                raise RuntimeError("bulk restore is not active")
            self._bulk_restore_depth -= 1
            if not self._bulk_restore_depth:
                state = self.conn.execute("SELECT native_checkpoint_profile,native_checkpoint_memory FROM sync_state WHERE id=1").fetchone()
                self._rebuild_native_checkpoint_state_locked(str(state[0] or ""), str(state[1] or ""))

    def count(self) -> int:
        """O(1) committed row count; readiness never scans the memories table."""
        with closing(self._read_connection()) as reader:
            row = reader.execute("SELECT document_count FROM funes_schema_state WHERE id=1").fetchone()
            if row is None:
                raise PostgresNotReady("PostgreSQL source count marker is missing")
            return int(row[0])

    def fts_ready(self) -> bool:
        """Source readiness is independent of the deliberately disabled PG FTS."""
        return False

    def reindex(self) -> int:
        """No PG content index to rebuild; native reindex controls own retrieval."""
        return self.count()

    def iter_documents(self, batch_size: int = POSTGRES_STREAM_BATCH):
        size = max(1, min(int(batch_size), 5000))
        with closing(self._read_connection()) as reader, reader.raw.transaction():
            with reader.raw.cursor(name="funes_documents_" + uuid.uuid4().hex) as cursor:
                cursor.itersize = size
                cursor.execute("SELECT * FROM memories ORDER BY id")
                while rows := cursor.fetchmany(size):
                    yield [self._row(row) for row in rows]

    def snapshot(self, path: Path) -> None:
        """Repeatable-read snapshot, bounded server cursors, no source buffering."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if path.name.endswith(".gz") else open
        with closing(self._read_connection()) as reader, reader.raw.transaction():
            reader.raw.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY").close()
            # Reuse the exact Store serializers against the same MVCC snapshot.
            view = object.__new__(PostgresStore)
            view.conn = reader
            view.lock = threading.RLock()
            optimize = view.native_optimize_checkpoint()
            native_state = view.native_index_state_record()
            with opener(path, "wt", encoding="utf-8") as output:
                for table, order, kind in (
                    ("memories", "id", "memory"),
                    ("translation_cache", "query", "translation_cache"),
                    ("reindex_controls", "generation", "reindex_control"),
                ):
                    with reader.raw.cursor(name="funes_snapshot_" + uuid.uuid4().hex) as cursor:
                        cursor.itersize = POSTGRES_STREAM_BATCH
                        cursor.execute(f"SELECT * FROM {table} ORDER BY {order}")
                        while rows := cursor.fetchmany(POSTGRES_STREAM_BATCH):
                            for row in rows:
                                document = self._row(row) if kind == "memory" else dict(row)
                                document["_funes_record"] = kind
                                output.write(json.dumps(document, ensure_ascii=False) + "\n")
                if optimize.get("fingerprint"):
                    optimize["_funes_record"] = "native_optimize_checkpoint"
                    output.write(json.dumps(optimize, ensure_ascii=False) + "\n")
                output.write(json.dumps(native_state, ensure_ascii=False) + "\n")

    def search(self, query: str, limit: int = 20, rerank=None, filters=None, *, allow_broad_scan: bool = True):
        """Exact source identity/row-ID lookup only, never content search.

        Content and technical-identifier queries must use Voyage/Lance BM25 in
        the native route. This compatibility method never falls back to LIKE,
        scans raw payloads, or claims to implement a PostgreSQL lexical index.
        """
        query = str(query or "")
        result = self.get(query)
        if result is None:
            return []
        filters = filters or {}
        for key in ("source_agent", "source_type", "project", "repo", "device_id", "role", "content_type"):
            if filters.get(key) and result.get(key) != str(filters[key]):
                return []
        if filters.get("source_missing") is not None:
            if bool(result.get("source_missing")) != bool(filters["source_missing"]):
                return []
        timestamp = result.get("timestamp")
        if filters.get("since") and (timestamp is None or timestamp < str(filters["since"])):
            return []
        if filters.get("until") and (timestamp is None or timestamp > str(filters["until"])):
            return []
        result["score"] = 0.0
        results = [result]
        return list(rerank(query, results)) if rerank is not None else results
