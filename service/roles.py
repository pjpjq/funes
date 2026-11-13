"""Split service lifecycle boundaries. No schema creation or remote mutation."""
from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass


def service_role() -> str:
    role = os.getenv("FUNES_SERVICE_ROLE", "combined")
    if role not in {"combined", "query", "writer"}:
        raise ValueError("invalid FUNES_SERVICE_ROLE")
    return role


def require_role_storage() -> str:
    role = service_role()
    if role != "combined" and not os.getenv("FUNES_POSTGRES_DSN"):
        raise ValueError("split roles require FUNES_POSTGRES_DSN")
    return role


def writer_lock_key(memory: str) -> int:
    memory = memory.strip().removeprefix("hf://datasets/").rstrip("/")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", memory, re.ASCII):
        raise ValueError("writer lease requires an explicit index memory")
    return int.from_bytes(hashlib.sha256(
        ("funes:writer-lifecycle:v1:" + memory).encode()
    ).digest()[:8], "big", signed=True)


class WriterLease:
    """Dedicated PG session, never pooled/reconnected or automatically reacquired.

    Session locks are not HF storage fencing: parent-CAS and draining the old
    writer remain mandatory during handoff, including network partitions.
    """

    def __init__(self, dsn, memory, *, connect=None, on_loss=None, interval=2.0):
        self.key = writer_lock_key(memory)
        self.dsn = dsn
        self._connect = connect
        self.on_loss = on_loss
        self.interval = interval
        self._session = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._ready = False
        self._attempted = False

    @property
    def ready(self):
        return self._ready and not self._stop.is_set()

    def acquire(self, *, monitor=True):
        with self._lock:
            if self._attempted:
                raise RuntimeError("writer lease cannot be reacquired")
            self._attempted = True
            try:
                if self._connect is None:
                    import psycopg
                    self._session = psycopg.connect(
                        self.dsn, autocommit=True, connect_timeout=3,
                        application_name="funes-writer-lease",
                        options="-c statement_timeout=3000",
                        keepalives=1, keepalives_idle=5, keepalives_interval=2,
                        keepalives_count=2, tcp_user_timeout=5000,
                    )
                else:
                    self._session = self._connect()
                cursor = self._session.execute("SELECT pg_try_advisory_lock(%s)", (self.key,))
                granted = cursor.fetchone()[0]
                cursor.close()
                if not granted:
                    raise RuntimeError("writer lease unavailable")
                self._ready = True
            except BaseException:
                if self._session is not None:
                    self._session.close()
                self._ready = False
                raise
        if monitor:
            self.start_monitor()
        return self

    def start_monitor(self):
        with self._lock:
            self.require()
            if self._thread is None:
                self._thread = threading.Thread(target=self._monitor, name="funes-writer-lease", daemon=True)
                self._thread.start()

    def start_thread(self, thread):
        with self._lock:
            self.require()
            thread.start()

    def require(self):
        if not self.ready:
            raise RuntimeError("writer lease unavailable")

    def check(self):
        lost = False
        with self._lock:
            if not self.ready:
                return False
            try:
                if self._session.closed:
                    raise RuntimeError("closed writer lease")
                # Probe the SAME session; a different connection proves nothing.
                self._session.execute("SELECT 1").close()
            except Exception:
                self._ready = False
                self._stop.set()
                lost = True
        if lost and self.on_loss is not None:
            self.on_loss()
        return self.ready

    def _monitor(self):
        while not self._stop.wait(self.interval):
            if not self.check():
                break

    def close(self):
        # Caller must first stop/reap all writer work; never release early.
        self._ready = False
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=8)
            if self._thread.is_alive():
                # Retain the session rather than release under running work.
                raise RuntimeError("writer lease monitor did not stop")
        if not self._lock.acquire(timeout=8):
            raise RuntimeError("writer lease session is busy; retaining lease")
        try:
            if self._session is not None:
                try:
                    if not self._session.closed:
                        self._session.execute("SELECT pg_advisory_unlock(%s)", (self.key,)).close()
                except Exception:
                    pass
                finally:
                    self._session.close()
                    self._session = None
        finally:
            self._lock.release()


@dataclass(frozen=True)
class QuerySnapshot:
    memory: str
    hf_sha: str
    lance_version: int

    @classmethod
    def parse(cls, value, memory):
        if (not isinstance(value, dict)
                or type(value.get("schema_version")) is not int
                or value["schema_version"] != 1
                or value.get("memory") != memory
                or not isinstance(value.get("hf_sha"), str)
                or re.fullmatch(r"[0-9a-f]{40}", value["hf_sha"]) is None
                or type(value.get("lance_version")) is not int
                or not 0 < value["lance_version"] <= (1 << 64) - 1):
            raise ValueError("invalid query-snapshot descriptor")
        return cls(memory, value["hf_sha"], value["lance_version"])

    def environment(self, env):
        return {**env, "FUNES_QUERY_HF_REVISION": self.hf_sha,
                "FUNES_QUERY_LANCE_VERSION": str(self.lance_version)}


class QuerySnapshotRefresher:
    """Single polling thread: coalesce updates without concurrent candidates."""

    def __init__(self, describe, install, *, interval=60.0):
        self.describe = describe
        self.install = install
        self.interval = max(1.0, interval)
        self.active = None
        self.error = None
        self.stop_event = threading.Event()
        self.wake = threading.Event()
        self.refresh_requested = threading.Event()
        self._poll_lock = threading.Lock()
        self.thread = None

    def poll_once(self):
        with self._poll_lock:
            if self.stop_event.is_set():
                return False
            try:
                candidate = self.describe()
                if (self.active is not None
                        and candidate.memory == self.active.memory
                        and candidate.lance_version == self.active.lance_version
                        and not self.refresh_requested.is_set()):
                    return False
                self.install(candidate)
                self.active = candidate
                self.refresh_requested.clear()
                self.error = None
                return True
            except Exception as exc:
                self.error = type(exc).__name__
                return False

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name="funes-query-snapshot", daemon=True)
            self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            self.poll_once()
            self.wake.wait(self.interval)
            self.wake.clear()

    def close(self):
        self.stop_event.set()
        self.wake.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=8)
            if self.thread.is_alive():
                raise RuntimeError("query snapshot thread did not stop")
