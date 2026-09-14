from __future__ import annotations
import hashlib, logging, os, signal, threading, time
from pathlib import Path
from .config import Config
from .discovery import _project_roots, discover_sources
from .parsers import complete_line_offset, parse_file
from .store import Store
from .client import SyncClient
from .native import NativeFunes
log=logging.getLogger("funes.sync")


def _is_appendable_source(source) -> bool:
    return source.kind in {
        "codex", "pi", "claude", "codex_session", "pi_session", "claude_session",
    } or source.kind.endswith("session")


def _cursor_offset(source, size: int) -> int:
    return complete_line_offset(source.path, size) if _is_appendable_source(source) else size

class SyncDaemon:
    def __init__(self, config=None, store=None, client=None):
        self.config=config or Config.load(); self.store=store or Store(config=self.config); self.client=client or SyncClient(self.config)
        # Memory-only companion mode must drain the HTTP queue even when the
        # shared config enables the native primary path for interactive hooks.
        self.native=NativeFunes(self.config) if self.config.native_primary and not self.config.memory_only else None
        self.running=False; self._wake=threading.Event(); self._observer=None
        self._source_cache=None; self._source_cache_at=0.0; self._source_cache_lock=threading.Lock()
        self._backfill_marker = self.config.state_dir / "initial-backfill.complete"
        self._source_schema_marker = self.config.state_dir / "source-schema-v2.complete"
        self._zero_record_marker = self.config.state_dir / "zero-record-repair-v1.complete"
        self._automation_identity_marker = self.config.state_dir / "automation-identity-v2.complete"
        remote_fingerprint=hashlib.sha256(self.config.remote_url.rstrip("/").encode()).hexdigest()[:16]
        self._remote_source_marker = self.config.state_dir / f"remote-source-v1-{remote_fingerprint}.complete"
        self._remote_source_cursor_key = f"remote_source_v1_after_{remote_fingerprint}"
    def _discover_sources(self):
        # Watchers wake on every transcript append. Re-walking all project
        # directories for each turn is wasteful; cache the bounded discovery
        # set until the periodic reconciliation interval expires. Creates,
        # moves, and deletes invalidate it immediately below.
        with self._source_cache_lock:
            now=time.monotonic()
            if self._source_cache is not None and now-self._source_cache_at < max(1,self.config.interval):
                return list(self._source_cache)
            sources=discover_sources(self.config)
            self._source_cache=tuple(sources); self._source_cache_at=now
            return list(sources)
    def _invalidate_source_cache(self):
        with self._source_cache_lock:
            self._source_cache=None; self._source_cache_at=0.0
    def _handle_filesystem_event(self, event):
        paths=[str(getattr(event,"src_path",""))]
        destination=str(getattr(event,"dest_path",""))
        if destination:
            paths.append(destination)
        excluded={".git","target","logs",".venv","node_modules","__pycache__"}
        paths=[path for path in paths if not any(part in excluded for part in Path(path).parts)]
        if not paths:
            return
        event_type=str(getattr(event,"event_type",""))
        structural=event_type in {"created","moved","deleted"}
        source_file=any(Path(path).suffix.lower() in {".jsonl",".ndjson",".json",".md",".txt"} for path in paths)
        if not source_file and not (structural and bool(getattr(event,"is_directory",False))):
            return
        if structural:
            self._invalidate_source_cache()
        self._wake.set()
    def scan_once(self):
        if not self.config.enabled or not self.config.auto_discover:
            return 0
        source_schema_missing=not self._source_schema_marker.exists()
        zero_record_repair_missing=not self._zero_record_marker.exists()
        automation_identity_missing=not self._automation_identity_marker.exists()
        legacy_state=bool(self.store.db.execute(
            "SELECT EXISTS(SELECT 1 FROM records) OR EXISTS(SELECT 1 FROM sources)"
        ).fetchone()[0])
        refresh_source_schema=source_schema_missing and (
            self._backfill_marker.exists() or legacy_state
        )
        if not refresh_source_schema and not self.config.initial_backfill and not self._backfill_marker.exists():
            # Seed cursors at EOF once so existing history is left untouched,
            # while files created or appended after installation still flow
            # through the normal incremental path.
            sources = self._discover_sources()
            present = {s.source_key for s in sources}
            self.store.mark_missing(present)
            for source in sources:
                try:
                    stat = source.path.stat()
                    offset = _cursor_offset(source, stat.st_size)
                except OSError:
                    continue
                self.store.register_source(source, stat)
                self.store.set_cursor(source.source_key, offset, stat.st_ino, stat.st_size)
            self._backfill_marker.parent.mkdir(parents=True, exist_ok=True)
            self._backfill_marker.write_text(str(time.time()), encoding="utf-8")
            self._source_schema_marker.write_text(str(time.time()), encoding="utf-8")
            return 0
        sources=self._discover_sources(); present={s.source_key for s in sources}
        self.store.mark_missing(present)
        total=0
        for s in sources:
            try: st=s.path.stat()
            except OSError: continue
            old=self.store.db.execute("SELECT size,mtime,inode FROM sources WHERE source_key=?", (s.source_key,)).fetchone()
            self.store.register_source(s,st)
            cur=self.store.cursor(s.source_key); start=0
            has_records=bool(self.store.db.execute(
                "SELECT EXISTS(SELECT 1 FROM records WHERE source_key=? LIMIT 1)",
                (s.source_key,),
            ).fetchone()[0])
            seeded_without_backfill=not self.config.initial_backfill and self._backfill_marker.exists()
            repair_zero_record=zero_record_repair_missing and not seeded_without_backfill and not has_records
            repair_automation=automation_identity_missing and not seeded_without_backfill and s.kind=="codex_memory" and "/automations/" in s.source_key.replace("\\","/")
            # No full reparse for unchanged files; append-only growth resumes at the byte cursor.
            if not refresh_source_schema and not repair_zero_record and not repair_automation and old and old[0] == st.st_size and old[1] == st.st_mtime and old[2] == st.st_ino:
                continue
            appendable = _is_appendable_source(s)
            try:
                offset = _cursor_offset(s, st.st_size)
                can_append = not refresh_source_schema and not repair_zero_record and not repair_automation and (has_records or seeded_without_backfill) and appendable and cur and cur.get("inode")==st.st_ino and st.st_size>=cur.get("size",0)
                if can_append:
                    candidate = int(cur.get("offset", 0))
                    # Legacy versions advanced cursors past an unterminated
                    # JSONL record. A valid cursor normally returns after one
                    # byte of look-behind; only a legacy mid-line cursor scans
                    # backward and triggers this one-time full reconciliation.
                    if complete_line_offset(s.path, candidate) == candidate:
                        start = candidate
            except OSError:
                continue
            chunks=parse_file(s,start)
            total += self.store.upsert_chunks(chunks)
            if start == 0:
                self.store.reconcile_source(s.source_key, {c.record_id for c in chunks})
            self.store.set_cursor(s.source_key,offset,st.st_ino,st.st_size)
        if (self.config.initial_backfill or refresh_source_schema) and not self._backfill_marker.exists():
            self._backfill_marker.parent.mkdir(parents=True, exist_ok=True)
            self._backfill_marker.write_text(str(time.time()), encoding="utf-8")
        if source_schema_missing:
            self._source_schema_marker.parent.mkdir(parents=True, exist_ok=True)
            self._source_schema_marker.write_text(str(time.time()), encoding="utf-8")
        if zero_record_repair_missing:
            self._zero_record_marker.parent.mkdir(parents=True,exist_ok=True)
            self._zero_record_marker.write_text(str(time.time()),encoding="utf-8")
        if automation_identity_missing:
            self._automation_identity_marker.parent.mkdir(parents=True, exist_ok=True)
            self._automation_identity_marker.write_text(str(time.time()),encoding="utf-8")
        return total

    def reconcile_remote_sources(
        self,
        max_batches: int|None = 16,
        *,
        force: bool = False,
        interruptible: bool = False,
    ) -> dict:
        """Queue only local identities absent from the durable remote source store."""
        required=("meta_value","record_ids_after","enqueue_records","set_meta")
        if any(not hasattr(self.store,name) for name in required):
            return {"complete":True,"checked":0,"queued":0,"skipped":True}
        if force:
            self._remote_source_marker.unlink(missing_ok=True)
            self.store.set_meta(self._remote_source_cursor_key,"")
        if self._remote_source_marker.exists():
            return {"complete":True,"checked":0,"queued":0}
        cursor=self.store.meta_value(self._remote_source_cursor_key) or ""
        checked=queued=batches=0
        while max_batches is None or batches<max_batches:
            if interruptible and (not self.running or self._wake.is_set()):
                return {"complete":False,"checked":checked,"queued":queued,"interrupted":True}
            identities=self.store.record_ids_after(cursor,5000)
            if not identities:
                complete=self.store.pending_count()==0
                if complete:
                    self._remote_source_marker.parent.mkdir(parents=True,exist_ok=True)
                    self._remote_source_marker.write_text(str(time.time()),encoding="utf-8")
                    self.store.set_meta(self._remote_source_cursor_key,"")
                return {"complete":complete,"checked":checked,"queued":queued}
            try:
                missing=self.client.missing_source_identities(identities)
            except Exception as exc:
                log.warning("remote source inventory unavailable: %s",type(exc).__name__)
                return {"complete":False,"checked":checked,"queued":queued,"error":type(exc).__name__}
            queued+=self.store.enqueue_records(missing)
            checked+=len(identities)
            batches+=1
            cursor=identities[-1]
            self.store.set_meta(self._remote_source_cursor_key,cursor)
        return {"complete":False,"checked":checked,"queued":queued}

    def state_status(self) -> dict:
        return {
            "initial_backfill_complete":self._backfill_marker.exists(),
            "source_schema_complete":self._source_schema_marker.exists(),
            "zero_record_repair_complete":self._zero_record_marker.exists(),
            "automation_identity_complete":self._automation_identity_marker.exists(),
            "remote_source_reconciliation":{
                "complete":self._remote_source_marker.exists(),
                "cursor_saved":bool(self.store.meta_value(self._remote_source_cursor_key)),
            },
        }

    def _start_watcher(self) -> None:
        """Wake the bounded scanner on local changes; polling remains the safety net."""
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            return
        daemon = self
        class Handler(FileSystemEventHandler):
            def on_any_event(self, _event):
                daemon._handle_filesystem_event(_event)
        observer = Observer()
        roots = {self.config.home / ".codex", self.config.home / ".pi", self.config.home / ".claude"}
        roots.update(_project_roots(self.config))
        for root in roots:
            if root.exists():
                try:
                    observer.schedule(Handler(), str(root), recursive=True)
                except OSError as exc:
                    log.warning("watcher unavailable for %s: %s", root, type(exc).__name__)
        try:
            observer.start()
        except OSError as exc:
            log.warning("watcher unavailable; continuing with reconciliation polling: %s", type(exc).__name__)
            observer.stop()
            observer.join(timeout=5)
            return
        self._observer = observer

    def _stop_watcher(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
    def flush_once(self):
        rows=self.store.pending(self.config.batch_size); sent=[]
        selected=[]; selected_bytes=0
        max_bytes=max(1, int(self.config.max_batch_bytes))
        for row in rows:
            row_bytes=len(str(row["payload"]).encode("utf-8"))+1
            if selected and selected_bytes+row_bytes>max_bytes:
                break
            selected.append(row); selected_bytes+=row_bytes
        rows=selected
        if not rows:return 0
        try:
            result=self.client.ingest([__import__('json').loads(r['payload']) for r in rows]); sent=[r['record_id'] for r in rows]
            self.store.ack(sent); return int(result.get("accepted",len(sent))) if isinstance(result,dict) else len(sent)
        except Exception as exc:
            for r in rows:self.store.fail(r['record_id'],str(exc),min(3600,30*(2**min(r['attempts'],6))))
            log.warning("ingest failed: %s",exc); return 0
    def drain(self, wait: bool = True) -> int:
        """Flush the existing durable queue without rescanning local sources."""
        total = 0
        while self.store.pending_count():
            sent = self.flush_once()
            if sent:
                total += sent
                continue
            if not wait:
                break
            row = self.store.db.execute("SELECT min(next_at) FROM queue").fetchone()
            next_at = row[0] if row else None
            if next_at is None:
                break
            delay = max(1.0, min(60.0, float(next_at) - time.time()))
            time.sleep(delay)
        return total
    def run(self,once=False):
        if not self.config.enabled:
            return 0
        self.running=True
        def stop(*_):
            self.running=False
            self._wake.set()
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
        if not once:
            self._start_watcher()
        try:
          while self.running:
            if self.native:
                result = self.native.sync()
                if not result.ok:
                    log.warning("native funes sync unavailable: %s", result.error)
                if once:
                    break
                self._wake.wait(max(1,self.config.interval)); self._wake.clear()
                continue
            self.scan_once()
            inventory=self.reconcile_remote_sources(None if once else 16,interruptible=True)
            # A one-shot backfill must drain the durable queue completely when
            # the remote is available; otherwise the first startup would leave
            # most history pending until the next 5-minute pass.  The continuous
            # daemon keeps a bounded batch count so it remains lightweight.  If
            # the remote is offline, flush_once leaves the queue intact and this
            # exits promptly for a later retry.
            if once:
                while self.running and self.store.pending_count():
                    if not self.flush_once():
                        break
                if self.running and not self.store.pending_count():
                    inventory=self.reconcile_remote_sources(1,interruptible=True)
            else:
                for _ in range(8):
                    if not self.flush_once():
                        break
                    # A filesystem event must not wait behind the rest of a
                    # large backfill burst. Finish the current durable request,
                    # then rescan before sending another batch.
                    if not self.running or self._wake.is_set():
                        break
            if once: break
            if not self.running: break
            wait=1 if not inventory.get("complete") and inventory.get("checked") else max(1,self.config.interval)
            self._wake.wait(wait); self._wake.clear()
        finally:
            self._stop_watcher()
