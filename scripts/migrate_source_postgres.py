#!/usr/bin/env python3
"""Offline, resumable, byte-preserving SQLite -> PostgreSQL source migration.

The source must be a frozen SQLite backup, never the application's live WAL.
Every COPY batch and its cursor commit together. Neither initialization nor a
successful import makes the application ready: finalization is a separate,
explicit operation after the caller freezes/catches up the final source tail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from service.postgres import (  # noqa: E402
    POSTGRES_SCHEMA_VERSION, PostgresStore, decode_pg_text, encode_pg_text,
)
from service.server import (  # noqa: E402
    LOW_VALUE_CONTENT_TYPES, MEMORIES_GENERATION_INDEXES, MEMORIES_SECONDARY_INDEXES,
    NATIVE_CHECKPOINT_STATE_VERSION, utc_now,
)

TABLES = ("memories", "translation_cache", "reindex_controls", "sync_state")
KEYS = {"memories": "id", "translation_cache": "query",
        "reindex_controls": "generation", "sync_state": "id"}
MIGRATION_VERSION = 2
TRIGGERS = ("memories_native_state", "memories_native_pending")
LEDGER = "funes_source_migration"
DERIVED_FIELDS = {
    "memories": ["native_index_pending"],
    "sync_state": ["fts_ready", "fts_schema_version", "native_checkpoint_state_version",
                   "native_eligible_count", "native_indexed_count", "native_held_count", "native_invalid_count"],
}


class MigrationError(RuntimeError):
    """Safe operator-facing error: never contains DSN or source values."""


def quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":")).encode("ascii")


def file_signature(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class Snapshot:
    """Read-only frozen source with a mandatory full-file identity pin."""

    def __init__(self, path: str | Path, expected_sha256: str | None = None):
        self.path = Path(path).resolve(strict=True)
        self.signature = file_signature(self.path)
        self.conn = None
        self.assert_unchanged()
        digest = self.sha256()
        if expected_sha256 and digest != expected_sha256.lower():
            raise MigrationError("source SHA-256 does not match the expected snapshot")
        try:
            self.conn = sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True)
            self.conn.execute("PRAGMA query_only=ON")
            self.conn.execute("PRAGMA temp_store=FILE")
            self.conn.execute("PRAGMA cache_size=-8192")
            self.conn.create_function("funes_pg_key", 1,
                                      lambda value: encode_pg_text(value).encode("utf-8"),
                                      deterministic=True)
            self.schema = {}
            self.columns = {}
            self.counts = {}
            for table in TABLES:
                info = self.conn.execute(f"PRAGMA table_info({quoted(table)})").fetchall()
                if not info or KEYS[table] not in {row[1] for row in info}:
                    raise MigrationError(f"source schema is missing required table/key: {table}")
                if any(str(row[2]).upper() not in ("TEXT", "INTEGER") for row in info):
                    raise MigrationError(f"source schema has unsupported column types: {table}")
                self.schema[table] = [list(row) for row in info]
                self.columns[table] = [row[1] for row in info]
                # rowid, not the encoded text key, is the durable COPY cursor.
                self.conn.execute(f"SELECT rowid FROM {quoted(table)} LIMIT 1").fetchone()
                self.counts[table] = self.conn.execute(
                    f"SELECT count(*) FROM {quoted(table)}").fetchone()[0]
            if self.counts["sync_state"] != 1 or self.conn.execute(
                    "SELECT id FROM sync_state").fetchone() != (1,):
                raise MigrationError("source must contain exactly sync_state id=1")
            self.identity = {"sha256": digest, "size": self.signature[2],
                             "schema_sha256": hashlib.sha256(json_bytes(self.schema)).hexdigest(),
                             "counts": self.counts}
            self.assert_unchanged()
        except BaseException:
            self.close()
            raise

    def assert_unchanged(self, *, rehash: bool = False) -> None:
        if file_signature(self.path) != self.signature:
            raise MigrationError("source snapshot changed; create a new frozen backup, do not resume")
        for suffix in ("-wal", "-journal"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists() and sidecar.stat().st_size:
                raise MigrationError("source has a WAL/journal; use a closed, frozen SQLite backup")
        if rehash and self.sha256() != self.identity["sha256"]:
            raise MigrationError("source snapshot SHA-256 changed")

    def sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as source:
            while block := source.read(4 * 1024 * 1024):
                digest.update(block)
        self.assert_unchanged()
        return digest.hexdigest()

    def rows(self, table: str, *, through: int | None = None):
        columns = ",".join(map(quoted, self.columns[table]))
        key = quoted(KEYS[table])
        order = f"funes_pg_key({key})" if table == "translation_cache" else key
        where, params = (" WHERE rowid<=?", (through,)) if through is not None else ("", ())
        return self.conn.execute(f"SELECT {columns} FROM {quoted(table)}{where} ORDER BY {order}", params)

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def digest_rows(table: str, columns: list[str], rows) -> dict[str, Any]:
    """Length-framed typed SHA-256, preserving UTF-8 bytes and NULL vs empty.

    Both sides sort by integer PK, or *encoded* UTF-8 key bytes for query TEXT.
    PG rows must be decoded exactly once before reaching this function.
    """
    digest = hashlib.sha256(b"funes-source-table-v1\0" + json_bytes([table, columns]))
    count = 0
    nul_fields = {}
    for row in rows:
        if len(row) != len(columns):
            raise MigrationError(f"row shape differs: {table}")
        digest.update(b"R")
        for name, value in zip(columns, row):
            if value is None:
                tag, data = b"N", b""
            elif isinstance(value, str):
                tag, data = b"S", value.encode("utf-8")
                if "\0" in value:
                    nul_fields[name] = nul_fields.get(name, 0) + 1
            elif type(value) is int:
                tag, data = b"I", str(value).encode("ascii")
            else:
                raise MigrationError(f"unsupported SQLite/PG value type: {table}.{name}")
            digest.update(tag + struct.pack("!Q", len(data)))
            digest.update(data)
        count += 1
    return {"rows": count, "sha256": digest.hexdigest(), "nul_fields": nul_fields}


class Migration:
    def __init__(self, snapshot: Snapshot, dsn: str | None = None, *, event_callback=None):
        import psycopg
        self.source = snapshot
        self.event_callback = event_callback
        self.dsn = dsn or os.environ.get("FUNES_POSTGRES_DSN")
        if not self.dsn:
            raise MigrationError("FUNES_POSTGRES_DSN is required; credentials are never CLI arguments")
        try:
            self.pg = psycopg.connect(self.dsn, autocommit=True, connect_timeout=15,
                                     application_name="funes-source-migration")
        except Exception:
            raise MigrationError("PostgreSQL connection failed; check the secret environment/network") from None
        try:
            locked = self.pg.execute(
                "SELECT pg_try_advisory_lock(hashtextextended("
                "'funes-source-migration:'||current_database()||':'||current_schema(),0))"
            ).fetchone()[0]
            if not locked:
                raise MigrationError("another source migration owns this PostgreSQL schema")
        except BaseException:
            self.pg.close()
            raise

    def close(self):
        self.pg.close()  # Releases the session advisory lock, including on interrupt.

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _exists(self, table: str, connection=None) -> bool:
        return (connection or self.pg).execute(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
            "WHERE table_schema=current_schema() AND table_name=%s)", (table,)
        ).fetchone()[0]

    def _bootstrap_state(self):
        """Accept only the exact row created by schema initialization, not data."""
        defaults = self.pg.execute(
            "SELECT column_name,column_default FROM information_schema.columns "
            "WHERE table_schema=current_schema() AND table_name='sync_state' ORDER BY ordinal_position"
        ).fetchall()
        columns = ",".join(quoted(name) for name, _ in defaults)
        values = ",".join("1" if name == "id" else (default or "NULL") for name, default in defaults)
        actual = self.pg.execute(f"SELECT {columns} FROM sync_state").fetchall()
        expected = self.pg.execute(f"SELECT {values}").fetchall()
        if actual != expected:
            raise MigrationError("target has existing sync_state data; refusing initialization")

    def initialize(self):
        if self._exists(LEDGER):
            raise MigrationError("target already has a migration; use --resume, not --initialize")
        existing = [table for table in TABLES if self._exists(table)]
        if existing:
            if set(existing) != set(TABLES) or not self._exists("funes_schema_state"):
                raise MigrationError("target has an incomplete/different schema; refusing initialization")
            self.check_schema()
            marker = self.pg.execute("SELECT migration_ready,migration_source,document_count FROM funes_schema_state WHERE id=1").fetchone()
            if marker != (False, None, 0):
                raise MigrationError("target is already initialized/finalized for a source")
            for table in TABLES[:-1]:
                if self.pg.execute(f"SELECT EXISTS(SELECT 1 FROM {quoted(table)})").fetchone()[0]:
                    raise MigrationError(f"target has existing data: {table}; refusing initialization")
            self._bootstrap_state()
        elif self._exists("funes_schema_state"):
            raise MigrationError("target marker exists without its schema")
        # Only pristine targets reach DDL. A crash before the ledger is created
        # leaves the exact empty bootstrap shape accepted above, still unready.
        with tempfile.TemporaryDirectory(prefix="funes-pg-migration-") as directory:
            store = PostgresStore(directory, dsn=self.dsn, initialize=True)
            try:
                store.prepare_bulk_migration()
            finally:
                store.close()
        self.check_schema()
        progress = {table: {"last_rowid": None, "rows": 0, "complete": False} for table in TABLES}
        with self.pg.transaction():
            self.pg.execute(f"""CREATE TABLE {LEDGER} (
                id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL,
                source_identity JSONB NOT NULL, phase TEXT NOT NULL,
                progress JSONB NOT NULL, verification JSONB, final_verification JSONB,
                started_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            self.pg.execute(
                f"INSERT INTO {LEDGER}(id,version,source_identity,phase,progress,started_at,updated_at) "
                "VALUES(1,%s,%s::jsonb,'copying',%s::jsonb,%s,%s)",
                (MIGRATION_VERSION, json.dumps(self.source.identity), json.dumps(progress), utc_now(), utc_now()),
            )
        return self.load()

    def check_schema(self, *, require_runtime: bool = False, connection=None):
        pg = connection or self.pg
        if not self._exists("funes_schema_state", pg):
            raise MigrationError("target has no schema marker; explicit initialization is required")
        marker = pg.execute("SELECT schema_version,document_count FROM funes_schema_state WHERE id=1").fetchone()
        if not marker or marker[0] != POSTGRES_SCHEMA_VERSION or marker[1] < 0:
            raise MigrationError("target schema version does not match this importer")
        for table in TABLES:
            actual = pg.execute(
                "SELECT column_name,data_type,is_generated,is_nullable FROM information_schema.columns "
                "WHERE table_schema=current_schema() AND table_name=%s", (table,)
            ).fetchall()
            target = {name: kind for name, kind, generated, nullable in actual if generated == "NEVER"}
            generated = {name: kind for name, kind, generated, nullable in actual if generated != "NEVER"}
            expected_generated = {}
            expected = {row[1]: str(row[2]).upper() for row in self.source.schema[table]}
            if generated != expected_generated or set(target) != set(expected) or any(
                target[name] not in ({"text"} if kind == "TEXT" else {"integer", "bigint"})
                for name, kind in expected.items()
            ):
                raise MigrationError(f"source/target column schema differs: {table}")
            if table == "memories" and not any(
                name == "retrieval_text" and nullable == "YES"
                for name, kind, generated, nullable in actual
            ):
                raise MigrationError("target v2 requires nullable retrieval_text")
        triggers = pg.execute(
            "SELECT c.relname,t.tgname,t.tgenabled FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=current_schema() "
            "AND c.relname=ANY(%s) AND NOT t.tgisinternal", (list(TABLES),)
        ).fetchall()
        enabled = {("memories", name, "O") for name in TRIGGERS}
        disabled = {("memories", name, "D") for name in TRIGGERS}
        observed = {tuple(row) for row in triggers}
        if observed != enabled and (require_runtime or observed != disabled):
            raise MigrationError("target triggers are missing, disabled, or unexpected")
        indexes = pg.execute(
            "SELECT c.relname,a.amname,i.indisvalid,i.indisready FROM pg_index i "
            "JOIN pg_class c ON c.oid=i.indexrelid JOIN pg_am a ON a.oid=c.relam "
            "WHERE i.indrelid='memories'::regclass"
        ).fetchall()
        if any(row[1] != "btree" for row in indexes):
            raise MigrationError("target v2 permits only B-tree source indexes")
        required = {"memories_pkey", "memories_source_identity_key"}
        if require_runtime:
            required.update(name for name, _ in MEMORIES_GENERATION_INDEXES)
            required.update(name for name, _ in MEMORIES_SECONDARY_INDEXES)
        if not required <= {row[0] for row in indexes if row[2] and row[3]}:
            raise MigrationError("target indexes are missing or invalid")

    def load(self):
        self.check_schema()
        if not self._exists(LEDGER):
            raise MigrationError("target has no resumable migration; use --initialize only for a fresh target")
        row = self.pg.execute(
            f"SELECT version,source_identity,phase,progress,verification FROM {LEDGER} WHERE id=1"
        ).fetchone()
        if not row or row[0] != MIGRATION_VERSION or row[1] != self.source.identity:
            raise MigrationError("migration source identity/version differs; refusing resume")
        self.phase, self.progress, self.verification = row[2:]
        if set(self.progress) != set(TABLES):
            raise MigrationError("migration checkpoint table set differs")
        for table, state in self.progress.items():
            if (set(state) != {"last_rowid", "rows", "complete"}
                    or not isinstance(state["rows"], int)
                    or not 0 <= state["rows"] <= self.source.counts[table]
                    or (state["rows"] > 0 and not isinstance(state["last_rowid"], int))
                    or (state["complete"] and state["rows"] != self.source.counts[table])):
                raise MigrationError(f"invalid durable migration checkpoint: {table}")
        return self.phase

    def _lock(self, connection=None):
        pg = connection or self.pg
        pg.execute("LOCK TABLE " + ",".join(map(quoted, (*TABLES, "funes_schema_state", LEDGER)))
                        + " IN ACCESS EXCLUSIVE MODE")
        marker = pg.execute("SELECT migration_ready FROM funes_schema_state WHERE id=1").fetchone()
        if marker is None or tuple(marker) != (False,):
            raise MigrationError("target is already ready; migration cannot modify a live source")

    def _pg_digest(self, snapshot: Snapshot, table: str, connection=None):
        from psycopg.rows import tuple_row
        # Physical NULL is only a compact representation of raw_text, not a
        # change in the logical source. Expand BEFORE decoding the NUL codec.
        columns = ",".join(
            "COALESCE(retrieval_text,raw_text) AS retrieval_text"
            if table == "memories" and name == "retrieval_text" else quoted(name)
            for name in snapshot.columns[table]
        )
        key = quoted(KEYS[table])
        order = f"convert_to({key},'UTF8')" if table == "translation_cache" else key
        with (connection or self.pg).cursor(name="funes_source_verify", row_factory=tuple_row) as cursor:
            cursor.itersize = 128
            cursor.execute(f"SELECT {columns} FROM {quoted(table)} ORDER BY {order}")
            decoded = ((decode_pg_text(value) if isinstance(value, str) else value for value in row)
                       for row in cursor)
            return digest_rows(table, snapshot.columns[table], (tuple(row) for row in decoded))

    def verify_progress(self):
        """Re-read committed prefixes before resuming, not just importer counters."""
        with self.pg.transaction():
            self._lock()
            for table in TABLES:
                state = self.progress[table]
                if table == "sync_state" and not state["rows"]:
                    self._bootstrap_state()
                    continue
                actual = self._pg_digest(self.source, table)
                through = state["last_rowid"]
                expected = digest_rows(table, self.source.columns[table],
                                       self.source.rows(table, through=through) if state["rows"] else ())
                if expected != actual or actual["rows"] != state["rows"]:
                    raise MigrationError(f"committed target differs from resume checkpoint: {table}")
        self.source.assert_unchanged()

    def copy(self, *, batch_rows: int = 500, batch_bytes: int = 8 * 1024 * 1024,
             max_batches: int | None = None, before_commit=None, progress=None):
        if batch_rows < 1 or batch_bytes < 1 or (max_batches is not None and max_batches < 1):
            raise MigrationError("batch limits must be positive")
        self.load()
        if self.phase not in ("copying", "copied", "verified"):
            raise MigrationError("target is already finalized or its phase is invalid")
        self.verify_progress()
        batches = 0
        for table in TABLES:
            while not self.progress[table]["complete"]:
                self.source.assert_unchanged()
                state = dict(self.progress[table])
                columns = ",".join(map(quoted, self.source.columns[table]))
                where, params = ((" WHERE rowid>?", (state["last_rowid"], batch_rows))
                                 if state["last_rowid"] is not None else ("", (batch_rows,)))
                rows = self.source.conn.execute(
                    f"SELECT rowid,{columns} FROM {quoted(table)}{where} ORDER BY rowid LIMIT ?", params)
                copied, size = 0, 0
                with self.pg.transaction():
                    self._lock()
                    if table == "memories":
                        trigger_states = self.pg.execute(
                            "SELECT tgname,tgenabled FROM pg_trigger WHERE tgrelid='memories'::regclass "
                            "AND NOT tgisinternal").fetchall()
                        for trigger in TRIGGERS:
                            self.pg.execute(f"ALTER TABLE memories DISABLE TRIGGER {quoted(trigger)}")
                    target = table
                    if table == "sync_state":
                        target = "funes_migration_sync"
                        self.pg.execute(f"CREATE TEMP TABLE {target} (LIKE sync_state INCLUDING DEFAULTS) ON COMMIT DROP")
                    with self.pg.cursor() as cursor:
                        with cursor.copy(f"COPY {quoted(target)} ({columns}) FROM STDIN") as copier:
                            for row in rows:
                                values = list(row[1:])
                                if table == "memories":
                                    raw_pos = self.source.columns[table].index("raw_text")
                                    shadow_pos = self.source.columns[table].index("retrieval_text")
                                    # Triggers are disabled during COPY: apply the
                                    # exact-equality storage rule explicitly.
                                    if values[shadow_pos] == values[raw_pos]:
                                        values[shadow_pos] = None
                                encoded = []
                                for value in values:
                                    if isinstance(value, str):
                                        value = encode_pg_text(value)
                                        size += len(value.encode("utf-8"))
                                    elif value is not None and type(value) is not int:
                                        raise MigrationError(f"unsupported SQLite value type: {table}")
                                    encoded.append(value)
                                copier.write_row(encoded)
                                state["last_rowid"] = row[0]
                                copied += 1
                                if size >= batch_bytes:
                                    break
                    rows.close()
                    if table == "sync_state":
                        assignments = ",".join(f"{quoted(name)}=EXCLUDED.{quoted(name)}"
                                               for name in self.source.columns[table] if name != "id")
                        self.pg.execute(f"INSERT INTO sync_state ({columns}) SELECT {columns} FROM {target} "
                                        f"ON CONFLICT(id) DO UPDATE SET {assignments}")
                    if table == "memories":
                        for trigger, original in trigger_states:
                            action = "ENABLE" if original == "O" else "DISABLE"
                            self.pg.execute(f"ALTER TABLE memories {action} TRIGGER {quoted(trigger)}")
                    state["rows"] += copied
                    state["complete"] = state["rows"] == self.source.counts[table]
                    if not copied and not state["complete"]:
                        raise MigrationError(f"source ended before the pinned count: {table}")
                    next_progress = {**self.progress, table: state}
                    self.source.assert_unchanged()
                    if before_commit:
                        before_commit(table, copied)
                    self.pg.execute(f"UPDATE {LEDGER} SET progress=%s::jsonb,updated_at=%s WHERE id=1",
                                    (json.dumps(next_progress), utc_now()))
                self.progress = next_progress
                batches += 1
                if progress:
                    progress({"phase": "copying", "table": table, "rows": state["rows"],
                              "total": self.source.counts[table]})
                if max_batches is not None and batches >= max_batches:
                    return {"phase": "paused", "progress": self.progress, "ready": False}
        self.pg.execute(f"UPDATE {LEDGER} SET phase='copied',updated_at=%s WHERE id=1", (utc_now(),))
        self.phase = "copied"
        return self.verify()

    @staticmethod
    def _derived_state(snapshot: Snapshot):
        state = dict(zip(snapshot.columns["sync_state"], next(iter(snapshot.rows("sync_state")))))
        counts = {"native_eligible_count": 0, "native_indexed_count": 0,
                  "native_held_count": 0, "native_invalid_count": 0}
        profile, memory = state["native_checkpoint_profile"], state["native_checkpoint_memory"]
        for content_type, status, row_profile, row_memory in snapshot.conn.execute(
                "SELECT content_type,native_index_status,native_index_profile,native_index_memory FROM memories"):
            if str(content_type or "").lower() in LOW_VALUE_CONTENT_TYPES:
                continue
            counts["native_eligible_count"] += 1
            if (row_profile or "") == profile and (row_memory or "") == memory:
                counts["native_indexed_count"] += int(status == "indexed")
                counts["native_held_count"] += int(status in ("held_secret", "held_invalid"))
                counts["native_invalid_count"] += int(status == "held_invalid")
        counts.update(fts_ready=0, fts_schema_version=0,
                      native_checkpoint_state_version=NATIVE_CHECKPOINT_STATE_VERSION)
        return counts, profile, memory

    @staticmethod
    def _expected_rows(snapshot: Snapshot, table: str, derived):
        rows = snapshot.rows(table)
        if derived is None or table not in DERIVED_FIELDS:
            yield from rows
            return
        counts, profile, memory = derived
        positions = {name: index for index, name in enumerate(snapshot.columns[table])}
        for row in rows:
            values = list(row)
            if table == "sync_state":
                for name, value in counts.items():
                    values[positions[name]] = value
            else:
                def field(name):
                    return row[positions[name]] or ""
                eligible = str(field("content_type")).lower() not in LOW_VALUE_CONTENT_TYPES
                status = field("native_index_status")
                current = field("native_index_profile") == profile and field("native_index_memory") == memory
                pending = eligible and status != "waiting_durability" and not (
                    status in ("indexed", "held_secret", "held_invalid") and current)
                values[positions["native_index_pending"]] = int(pending)
            yield tuple(values)

    def _verify_tables(self, snapshot: Snapshot, *, connection=None, rebuilt: bool = False):
        report = {}
        derived = self._derived_state(snapshot) if rebuilt else None
        for table in TABLES:
            if self.event_callback:
                self.event_callback({"phase": "verifying", "table": table, "state": "started"})
            expected = digest_rows(table, snapshot.columns[table], self._expected_rows(snapshot, table, derived))
            actual = self._pg_digest(snapshot, table, connection)
            if actual != expected or actual["rows"] != snapshot.counts[table]:
                raise MigrationError(f"full decoded row digest/count mismatch: {table}")
            report[table] = actual
            if self.event_callback:
                self.event_callback({"phase": "verifying", "table": table, "state": "matched", **actual})
        if self.event_callback:
            self.event_callback({"phase": "rehashing_source"})
        snapshot.assert_unchanged(rehash=True)
        result = {"source": snapshot.identity, "tables": report}
        if derived is not None:
            result.update(rebuilt_fields=DERIVED_FIELDS, expected_derived_sync_state=derived[0])
        return result

    def _sequence(self):
        self.pg.execute("SELECT setval(pg_get_serial_sequence('memories','id'),"
                        "GREATEST(COALESCE((SELECT max(id) FROM memories),0),1),"
                        "EXISTS(SELECT 1 FROM memories))")

    def verify(self):
        self.load()
        if not all(state["complete"] for state in self.progress.values()):
            raise MigrationError("migration is incomplete; resume COPY before full verification")
        with self.pg.transaction():
            self._lock()
            report = self._verify_tables(self.source)
            self._sequence()  # Now safe for an explicitly offline incremental tail.
            self.pg.execute("UPDATE funes_schema_state SET document_count=%s WHERE id=1",
                            (report["tables"]["memories"]["rows"],))
            self.pg.execute(f"UPDATE {LEDGER} SET phase='verified',verification=%s::jsonb,updated_at=%s WHERE id=1",
                            (json.dumps(report), utc_now()))
        self.phase = "verified"
        return {"phase": "verified", "ready": False, **report}

    def finalize(self, *, tail_confirmed: bool = False, tail_source: Snapshot | None = None):
        self.load()
        if self.phase != "verified" or not self.verification:
            raise MigrationError("baseline import must pass full verification before finalization")
        if not tail_confirmed:
            raise MigrationError("finalization requires --tail-confirmed after writers stop and the tail is synchronized")
        final = tail_source or self.source
        if final.schema != self.source.schema:
            raise MigrationError("final tail source schema differs from the pinned baseline")
        self.source.assert_unchanged()
        if self.event_callback:
            self.event_callback({"phase": "building_postgres_indexes"})
        # This explicit final-stage constructor restores deferred indexes and
        # triggers but DOES NOT set ready or alter source rows. Resume/COPY never
        # calls it. All subsequent data validation and ready changes share the
        # store's SAME transaction; a failed post-finalize digest rolls it back.
        with tempfile.TemporaryDirectory(prefix="funes-pg-finalize-") as directory:
            store = PostgresStore(directory, dsn=self.dsn, initialize=True)
            try:
                with store.lock, store.conn:
                    connection = store.conn.raw
                    self._lock(connection)
                    before = self._verify_tables(final, connection=connection)
                    store.finalize_migration(expected_count=final.counts["memories"],
                                             source=json.dumps(final.identity, sort_keys=True))
                    report = self._verify_tables(final, connection=connection, rebuilt=True)
                    self.check_schema(require_runtime=True, connection=connection)
                    report["before_finalize_tables"] = before["tables"]
                    connection.execute(f"UPDATE {LEDGER} SET phase='ready',final_verification=%s::jsonb,updated_at=%s WHERE id=1",
                                       (json.dumps(report), utc_now()))
            finally:
                store.close()
        return {"phase": "ready", "ready": True, **report}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Frozen SQLite backup, not live WAL database")
    parser.add_argument("--expected-sha256", help="Optional externally recorded SHA-256; every run computes and pins SHA-256")
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "initialize", "resume", "verify", "finalize"):
        action.add_argument("--" + name, action="store_true")
    parser.add_argument("--defer-ready", action="store_true", help="Required for import; application remains fail-closed")
    parser.add_argument("--batch-rows", type=int, default=500)
    parser.add_argument("--batch-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-batches", type=int, help="Stop safely after this many committed COPY batches")
    parser.add_argument("--tail-confirmed", action="store_true", help="Operator confirms old writers are stopped and the current tail is synchronized")
    parser.add_argument("--tail-source", help="Optional independently frozen final SQLite state after incremental tail catchup")
    parser.add_argument("--tail-expected-sha256", help="Optional expected SHA-256 for --tail-source")
    args = parser.parse_args(argv)
    if (args.initialize or args.resume) and not args.defer_ready:
        parser.error("--initialize/--resume requires --defer-ready; finalization is a separate operation")
    if args.tail_source and not args.finalize:
        parser.error("--tail-source is only valid with --finalize")
    if args.tail_expected_sha256 and not args.tail_source:
        parser.error("--tail-expected-sha256 requires --tail-source")
    if args.finalize and not args.tail_confirmed:
        parser.error("--finalize requires --tail-confirmed after freezing/catching up the source")
    try:
        print(json.dumps({"phase": "hashing_source"}), flush=True)
        with Snapshot(args.source, args.expected_sha256) as source:
            if args.dry_run:
                tables = {table: digest_rows(table, source.columns[table], source.rows(table)) for table in TABLES}
                source.assert_unchanged()
                result = {"phase": "dry_run", "writes": False, "source": source.identity, "tables": tables}
            else:
                with Migration(source, event_callback=lambda state: print(json.dumps(state), flush=True)) as migration:
                    if args.initialize:
                        migration.initialize()
                    if args.initialize or args.resume:
                        result = migration.copy(batch_rows=args.batch_rows, batch_bytes=args.batch_bytes,
                                                max_batches=args.max_batches,
                                                progress=lambda state: print(json.dumps(state), flush=True))
                    elif args.verify:
                        result = migration.verify()
                    elif args.tail_source:
                        with Snapshot(args.tail_source, args.tail_expected_sha256) as final:
                            result = migration.finalize(tail_confirmed=args.tail_confirmed, tail_source=final)
                    else:
                        result = migration.finalize(tail_confirmed=args.tail_confirmed)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except KeyboardInterrupt:
        print(json.dumps({"error": "interrupted; readiness not confirmed; inspect the durable marker before resuming the same snapshot"}), file=sys.stderr)
        return 130
    except MigrationError as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1
    except Exception:
        # Database exception text can include the DSN, SQL values, or COPY row.
        print(json.dumps({"error": "migration failed; readiness not confirmed; inspect the durable marker and source/schema/network before resuming the same snapshot"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
