from __future__ import annotations
import logging, signal, time
from .config import Config
from .discovery import discover_sources
from .parsers import parse_file
from .store import Store
from .client import SyncClient
log=logging.getLogger("funes.sync")

class SyncDaemon:
    def __init__(self, config=None, store=None, client=None):
        self.config=config or Config.load(); self.store=store or Store(config=self.config); self.client=client or SyncClient(self.config); self.running=False
    def scan_once(self):
        if not self.config.enabled or not self.config.auto_discover:
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
        return total
    def flush_once(self):
        rows=self.store.pending(self.config.batch_size); sent=[]
        if not rows:return 0
        try:
            result=self.client.ingest([__import__('json').loads(r['payload']) for r in rows]); sent=[r['record_id'] for r in rows]
            self.store.ack(sent); return int(result.get("accepted",len(sent))) if isinstance(result,dict) else len(sent)
        except Exception as exc:
            for r in rows:self.store.fail(r['record_id'],str(exc),min(3600,30*(2**min(r['attempts'],6))))
            log.warning("ingest failed: %s",exc); return 0
    def run(self,once=False):
        if not self.config.enabled:
            return 0
        self.running=True
        def stop(*_): self.running=False
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
        while self.running:
            self.scan_once()
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
            time.sleep(max(1,self.config.interval))
