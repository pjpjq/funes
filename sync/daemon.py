from __future__ import annotations
import logging, os, signal, threading, time
from pathlib import Path
from .config import Config
from .discovery import discover_sources
from .parsers import parse_file
from .store import Store
from .client import SyncClient
from .native import NativeFunes
log=logging.getLogger("funes.sync")

class SyncDaemon:
    def __init__(self, config=None, store=None, client=None):
        self.config=config or Config.load(); self.store=store or Store(config=self.config); self.client=client or SyncClient(self.config); self.native=NativeFunes(self.config) if self.config.native_primary else None; self.running=False; self._wake=threading.Event(); self._observer=None
        self._backfill_marker = self.config.state_dir / "initial-backfill.complete"
    def scan_once(self):
        if not self.config.enabled or not self.config.auto_discover:
            return 0
        if not self.config.initial_backfill and not self._backfill_marker.exists():
            # Seed cursors at EOF once so existing history is left untouched,
            # while files created or appended after installation still flow
            # through the normal incremental path.
            sources = discover_sources(self.config)
            present = {s.source_key for s in sources}
            self.store.mark_missing(present)
            for source in sources:
                try:
                    stat = source.path.stat()
                except OSError:
                    continue
                self.store.register_source(source, stat)
                self.store.set_cursor(source.source_key, stat.st_size, stat.st_ino, stat.st_size)
            self._backfill_marker.parent.mkdir(parents=True, exist_ok=True)
            self._backfill_marker.write_text(str(time.time()), encoding="utf-8")
            return 0
        sources=discover_sources(self.config); present={s.source_key for s in sources}
        self.store.mark_missing(present)
        total=0
        for s in sources:
            try: st=s.path.stat()
            except OSError: continue
            old=self.store.db.execute("SELECT size,mtime,inode FROM sources WHERE source_key=?", (s.source_key,)).fetchone()
            self.store.register_source(s,st)
            cur=self.store.cursor(s.source_key); start=0
            # No full reparse for unchanged files; append-only growth resumes at the byte cursor.
            if old and old[0] == st.st_size and old[1] == st.st_mtime and old[2] == st.st_ino:
                continue
            appendable = s.kind in {"codex", "pi", "claude", "codex_session", "pi_session", "claude_session"} or s.kind.endswith("session")
            if appendable and cur and cur.get("inode")==st.st_ino and st.st_size>=cur.get("size",0):
                start=cur.get("offset",0)
            chunks=parse_file(s,start)
            total += self.store.upsert_chunks(chunks)
            if start == 0:
                self.store.reconcile_source(s.source_key, {c.record_id for c in chunks})
            self.store.set_cursor(s.source_key,st.st_size,st.st_ino,st.st_size)
        if self.config.initial_backfill and not self._backfill_marker.exists():
            self._backfill_marker.parent.mkdir(parents=True, exist_ok=True)
            self._backfill_marker.write_text(str(time.time()), encoding="utf-8")
        return total

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
                path = str(getattr(_event, "src_path", ""))
                if any(part in {".git", "target", "logs", ".venv", "node_modules", "__pycache__"} for part in Path(path).parts):
                    return
                if Path(path).suffix.lower() in {".jsonl", ".ndjson", ".json", ".md", ".txt"}:
                    daemon._wake.set()
        observer = Observer()
        roots = {self.config.home / ".codex", self.config.home / ".pi", self.config.home / ".claude"}
        project = Path(os.environ.get("FUNES_PROJECT_ROOT", Path.cwd())).expanduser()
        if project.exists():
            roots.add(project)
        for root in roots:
            if root.exists():
                observer.schedule(Handler(), str(root), recursive=True)
        observer.start()
        self._observer = observer

    def _stop_watcher(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
    def flush_once(self):
        rows=self.store.pending(self.config.batch_size); sent=[]
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
        def stop(*_): self.running=False
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
        if not once:
            self._start_watcher()
        try:
          while self.running:
            self.scan_once()
            if self.native:
                result = self.native.sync()
                if not result.ok:
                    log.warning("native funes sync unavailable: %s", result.error)
                if once:
                    break
                self._wake.wait(max(1,self.config.interval)); self._wake.clear()
                continue
            # A one-shot backfill must drain the durable queue completely when
            # the remote is available; otherwise the first startup would leave
            # most history pending until the next 5-minute pass.  The continuous
            # daemon keeps a bounded batch count so it remains lightweight.  If
            # the remote is offline, flush_once leaves the queue intact and this
            # exits promptly for a later retry.
            if once:
                while self.store.pending_count():
                    if not self.flush_once():
                        break
            else:
                for _ in range(8):
                    if not self.flush_once():
                        break
            if self.store.pending_count() == 0 and self.client.health():
                try:
                    self.client.sync_snapshot()
                except Exception as exc:
                    log.warning("snapshot sync unavailable: %s", type(exc).__name__)
            if once: break
            self._wake.wait(max(1,self.config.interval)); self._wake.clear()
        finally:
            self._stop_watcher()
