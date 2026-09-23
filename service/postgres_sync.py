"""PostgreSQL durability boundary for the source store, without Hub replay.

The native Lance memory still lives at FUNES_MEMORY/FUNES_INDEX_MEMORY.  This
adapter only commits bounded source/control batches and checks an already
migrated PostgreSQL source database.  It never bootstraps or rebuilds it.
"""
from __future__ import annotations

import threading
from typing import Any

from service.server import NATIVE_CHECKPOINT_STATE_VERSION, PROMPT_VERSION, REINDEX_SCOPES, utc_now


class PostgresSync:
    """The SnapshotSync interface backed by acknowledged database commits."""

    backend = "postgres"

    def __init__(self, store: Any, *, rebuild_fts: bool | None = None):
        self.store = store
        self.upload_lock = threading.RLock()
        self.restoring = False
        self.restored = False
        self.restore_failed = False
        self.restore_error: str | None = None
        self._progress = {"phase": "idle", "current": None, "completed": 0,
                          "total": 0, "rows": 0, "bytes": 0}

    @property
    def progress(self) -> dict[str, Any]:
        with self.upload_lock:
            return dict(self._progress)

    def _unavailable(self) -> None:
        # No exception text, DSN, SQL parameter, or source payload is public.
        # Do not write sync_state here: the database is precisely what failed.
        self.restore_failed = True
        self.restore_error = "postgres_unavailable"
        self._progress.update(phase="failed", error=self.restore_error)

    def check_ready(self) -> bool:
        """Reconnect/check the migrated database, without enumerating sources."""
        with self.upload_lock:
            try:
                # Probe the existing connection first. Reconnect explicitly on
                # failure, but never replay an interrupted transaction.
                try:
                    self.store.verify_schema()
                except Exception:
                    self.store.reconnect()
                    self.store.verify_schema()
            except Exception:
                self._unavailable()
                return False
            self.restore_failed = False
            self.restore_error = None
            self.restored = True
            self._progress.update(phase="complete")
            self._progress.pop("error", None)
            return True

    def restore(self) -> int:
        """Validate the existing source database; there are no rows to replay."""
        with self.upload_lock:
            self.restoring = True
            self._progress.update(phase="connecting")
            try:
                return 0 if self.check_ready() else -1
            finally:
                self.restoring = False

    def _restore_record(self, record: dict[str, Any]) -> None:
        """Reuse Store merge/guard methods, without its bulk-restore rebuild."""
        kind = record.pop("_funes_record", "memory")
        if kind == "memory":
            record["metadata"] = record.pop("metadata", record.pop("metadata_json", {}))
            self.store.ingest([record])
        elif kind == "translation_cache":
            query, rewritten = str(record.get("query") or ""), str(record.get("rewritten") or "")
            if not query or not rewritten:
                raise ValueError("invalid translation cache record")
            self.store.translation_put(
                query, rewritten,
                status=str(record.get("translation_status") or "ok"),
                translation_hash=str(record.get("translation_hash") or ""),
                translation_version=str(record.get("translation_version") or PROMPT_VERSION),
            )
        elif kind == "reindex_control":
            if not self.store.record_reindex_control(record):
                row = self.store.conn.execute(
                    "SELECT scope FROM reindex_controls WHERE generation=?",
                    (int(record.get("generation") or 0),),
                ).fetchone()
                if row is None or row["scope"] != record.get("scope"):
                    raise ValueError("conflicting reindex control")
        elif kind == "native_index_state":
            current = self.store.native_index_state_record()
            keys = ("profile", "memory", "revision", "eligible", "indexed", "held", "invalid")
            if int(record.get("state_version") or 0) != NATIVE_CHECKPOINT_STATE_VERSION:
                raise ValueError("invalid native state version")
            if any(int(record.get(key) or 0) < 0 for key in keys[2:]):
                raise ValueError("invalid native state")
            # Triggers, not a caller-supplied summary, own runtime row counts.
            # An old marker is harmless; a new/mismatched marker cannot claim
            # rows that were not part of the same successful write transaction.
            if int(record.get("revision") or 0) >= int(current["revision"]):
                if any(record.get(key) != current.get(key) for key in keys):
                    raise ValueError("native state does not match source rows")
                if record.get("index_fingerprint") not in (None, current["index_fingerprint"]):
                    raise ValueError("invalid native state fingerprint")
            self.store.set_native_index_state(record)
        elif kind == "native_optimize_checkpoint":
            if record.get("status") == "optimized":
                current = self.store.native_index_state_record()
                if (
                    record.get("fingerprint") != current["profile"]
                    or record.get("memory") != current["memory"]
                    or record.get("index_fingerprint") != current["index_fingerprint"]
                ):
                    raise ValueError("optimize marker does not match source state")
            self.store.set_native_optimize_checkpoint(record)
        else:
            raise ValueError("unknown source record type")

    def upload(self, docs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Commit every supplied record before issuing the durable ACK.

        Store methods enter their own transaction contexts; PostgresStore's
        connection facade nests those contexts so this outer transaction is the
        only commit boundary.  A failed record or COMMIT rolls back the batch.
        `/sync` without records acknowledges a real sync_state commit, not an
        export, full-table scan, or native-index rebuild.
        """
        with self.upload_lock:
            if not self.check_ready():
                return {"uploaded": False, "durable": False,
                        "reason": "postgres_unavailable"}
            try:
                records = list(docs or [])
                with self.store.lock, self.store.conn:
                    # A DSN/server default of synchronous_commit=off must not
                    # weaken an HTTP durable ACK into an asynchronous WAL write.
                    self.store.conn.execute("SET LOCAL synchronous_commit=on").close()
                    for record in records:
                        if not isinstance(record, dict):
                            raise ValueError("invalid source record")
                        self._restore_record(dict(record))
                    self.store.set_sync(last_sync=utc_now(), last_error=None)
                # This is deliberately outside the outer context: a deferred
                # COMMIT failure must never produce a positive durable result.
                return {"uploaded": True, "durable": True,
                        "backend": self.backend, "records": len(records)}
            except Exception:
                self._unavailable()
                return {"uploaded": False, "durable": False,
                        "reason": "postgres_unavailable"}

    def upload_reindex_control(self, control: dict[str, Any]) -> dict[str, Any]:
        """Persist the control before the caller makes it runnable/wakes workers."""
        generation = int(control.get("generation", 0))
        scope = str(control.get("scope", ""))
        if generation < 1 or scope not in REINDEX_SCOPES:
            raise ValueError("invalid reindex control")
        result = self.upload([{**control, "_funes_record": "reindex_control"}])
        if result.get("durable"):
            result.update(generation=generation, scope=scope)
        return result
