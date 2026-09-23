from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import Config, _restrict_owned_permissions
from .discovery import Source
from .parsers import Chunk


class Store:
    def __init__(self, path: Path|str|None=None, config: Config|None=None):
        self.config=config or Config.load(); self.config.ensure()
        self.path=Path(path or self.config.state_dir/"sync.db"); self.path.parent.mkdir(parents=True,exist_ok=True)
        self._prepare_database_file()
        self._restrict_sqlite_files()
        self.db=sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.db.row_factory=sqlite3.Row; self.db.execute("PRAGMA journal_mode=WAL"); self._schema()
        self._restrict_sqlite_files()
    def _prepare_database_file(self):
        """Create a new POSIX database privately before SQLite opens it."""
        if os.name == "nt":
            return
        try:
            descriptor=os.open(self.path, os.O_CREAT|os.O_EXCL|os.O_RDWR, 0o600)
        except FileExistsError:
            return
        except OSError:
            return
        os.close(descriptor)
    def _restrict_sqlite_files(self):
        for path in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):
            _restrict_owned_permissions(path, 0o600)
    def _schema(self):
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS sources(source_key TEXT PRIMARY KEY,kind TEXT NOT NULL,path TEXT NOT NULL,device_id TEXT,project TEXT,active INTEGER DEFAULT 1,retired INTEGER NOT NULL DEFAULT 0,size INTEGER,mtime REAL,inode INTEGER,updated_at REAL);
        CREATE TABLE IF NOT EXISTS records(record_id TEXT PRIMARY KEY,source_key TEXT NOT NULL,content_hash TEXT NOT NULL,version INTEGER DEFAULT 1,payload TEXT NOT NULL,updated_at REAL);
        CREATE INDEX IF NOT EXISTS records_source ON records(source_key);
        CREATE TABLE IF NOT EXISTS queue(record_id TEXT PRIMARY KEY,attempts INTEGER DEFAULT 0,next_at REAL DEFAULT 0,last_error TEXT,queued_at REAL);
        CREATE INDEX IF NOT EXISTS queue_failed ON queue(last_error) WHERE last_error IS NOT NULL;
        CREATE INDEX IF NOT EXISTS queue_schedule ON queue(next_at,queued_at);
        CREATE INDEX IF NOT EXISTS queue_ready_order ON queue(queued_at,next_at);
        CREATE TABLE IF NOT EXISTS cursors(source_key TEXT PRIMARY KEY,offset INTEGER DEFAULT 0,inode INTEGER,size INTEGER,updated_at REAL);
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL,updated_at REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS sources_active_kind ON sources(active,kind);
        ''')
        source_columns={row[1] for row in self.db.execute("PRAGMA table_info(sources)")}
        if "retired" not in source_columns:
            self.db.execute(
                "ALTER TABLE sources ADD COLUMN retired INTEGER NOT NULL DEFAULT 0"
            )
        self.db.commit()
        retirement_migration="legacy_automation_retirement_v1"
        if not self.db.execute(
            "SELECT 1 FROM meta WHERE key=?",(retirement_migration,)
        ).fetchone():
            now=time.time()
            with self.db:
                self.db.execute('''
                    UPDATE sources SET retired=1,active=0,updated_at=?
                    WHERE kind='codex_memory'
                      AND replace(lower(path),char(92),'/') LIKE '%/.codex/automations/%'
                      AND lower(path) NOT LIKE '%.md'
                      AND lower(path) NOT LIKE '%.toml'
                ''',(now,))
                self.db.execute('''
                    DELETE FROM queue WHERE record_id IN (
                        SELECT r.record_id FROM records r
                        JOIN sources s ON s.source_key=r.source_key
                        WHERE s.retired=1
                    )
                ''')
                self.db.execute(
                    "INSERT INTO meta(key,value,updated_at) VALUES(?,?,?)",
                    (retirement_migration,"completed",now),
                )
    def close(self): self.db.close()
    def register_source(self,s:Source,stat=None):
        now=time.time(); st=stat
        self.db.execute("INSERT INTO sources(source_key,kind,path,device_id,project,active,retired,size,mtime,inode,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(source_key) DO UPDATE SET kind=excluded.kind,path=excluded.path,device_id=excluded.device_id,project=excluded.project,active=1,retired=0,size=excluded.size,mtime=excluded.mtime,inode=excluded.inode,updated_at=excluded.updated_at",(s.source_key,s.kind,str(s.path),s.device_id,s.project,1,0,st.st_size if st else None,st.st_mtime if st else None,st.st_ino if st else None,now)); self.db.commit()
    def mark_missing(self, present:set[str]):
        # Keep records and source rows; only mark source inactive.
        self.db.execute("UPDATE sources SET active=0,updated_at=? WHERE source_key NOT IN (%s)" % (','.join('?'*len(present)) if present else "''"), (time.time(),*present) if present else (time.time(),)); self.db.commit()
    def cursor(self,key):
        r=self.db.execute("SELECT * FROM cursors WHERE source_key=?",(key,)).fetchone(); return dict(r) if r else None
    def set_cursor(self,key,offset,inode,size):
        self.db.execute("INSERT INTO cursors(source_key,offset,inode,size,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(source_key) DO UPDATE SET offset=excluded.offset,inode=excluded.inode,size=excluded.size,updated_at=excluded.updated_at",(key,offset,inode,size,time.time())); self.db.commit()
    def upsert_chunks(self,chunks:list[Chunk]):
        now=time.time(); count=0
        with self.db:
            for c in chunks:
                h=hashlib.sha256(c.raw_text.encode("utf-8")).hexdigest()
                old=self.db.execute("SELECT content_hash,version,payload,source_key FROM records WHERE record_id=?",(c.record_id,)).fetchone()
                value=c.as_dict()
                prior=json.loads(old[2]) if old else {}
                stamp=datetime.fromtimestamp(now, timezone.utc).isoformat()
                value["ingested_at"]=prior.get("ingested_at", stamp)
                value["updated_at"]=prior.get("updated_at", stamp)
                payload=json.dumps(value,ensure_ascii=False,sort_keys=True)
                if old and old[0] == h and old[2] == payload and old[3] == c.source_key:
                    count += 1
                    continue
                value["updated_at"]=stamp
                payload=json.dumps(value,ensure_ascii=False,sort_keys=True)
                version=(old[1]+1 if old and old[0]!=h else (old[1] if old else 1))
                self.db.execute("INSERT INTO records(record_id,source_key,content_hash,version,payload,updated_at) VALUES(?,?,?,?,?,?) ON CONFLICT(record_id) DO UPDATE SET source_key=excluded.source_key,content_hash=excluded.content_hash,version=excluded.version,payload=excluded.payload,updated_at=excluded.updated_at",(c.record_id,c.source_key,h,version,payload,now))
                self.db.execute("INSERT INTO queue(record_id,attempts,next_at,last_error,queued_at) VALUES(?,?,?,?,?) ON CONFLICT(record_id) DO UPDATE SET next_at=MIN(queue.next_at,excluded.next_at),last_error=NULL",(c.record_id,0,0,None,now)); count+=1
        return count
    def reconcile_source(self, source_key: str, current_ids: set[str]):
        """Remove local records deleted from a fully reparsed source.

        Append scans never call this; a full rewrite therefore cannot leave stale
        remote queue entries behind while normal append-only logs remain cheap.
        """
        rows=self.db.execute("SELECT record_id FROM records WHERE source_key=?",(source_key,)).fetchall()
        stale=[r[0] for r in rows if r[0] not in current_ids]
        changed=0
        if stale:
            # Keep the local source-of-truth row and send a soft-missing update;
            # the remote service defaults to the same keep policy.  Physical
            # deletion is intentionally never implied by a local rewrite.
            with self.db:
                for record_id in stale:
                    row=self.db.execute("SELECT payload FROM records WHERE record_id=?",(record_id,)).fetchone()
                    if not row:
                        continue
                    payload=json.loads(row[0])
                    if payload.get("source_missing"):
                        continue
                    payload["source_missing"]=True
                    source_version=str(payload.get("source_version") or payload.get("content_hash") or "")
                    payload["source_version"]=hashlib.sha256((source_version+":source_missing=true").encode()).hexdigest()
                    payload["updated_at"]=str(time.time())
                    self.db.execute("UPDATE records SET payload=?,updated_at=? WHERE record_id=?",(json.dumps(payload,ensure_ascii=False,sort_keys=True),time.time(),record_id))
                    self.db.execute("INSERT INTO queue(record_id,attempts,next_at,last_error,queued_at) VALUES(?,?,?,?,?) ON CONFLICT(record_id) DO UPDATE SET next_at=MIN(queue.next_at,excluded.next_at),last_error=NULL",(record_id,0,0,None,time.time()))
                    changed += 1
        return changed
    def pending(self,limit=50,now=None):
        now=now or time.time(); rows=self.db.execute("SELECT q.*,r.payload FROM queue q JOIN records r ON r.record_id=q.record_id WHERE q.next_at<=? ORDER BY q.queued_at LIMIT ?",(now,limit)).fetchall(); return [dict(r) for r in rows]
    def pending_count(self) -> int:
        """Return queue size without the JSON aggregation used by ``stats``."""
        return int(self.db.execute("SELECT count(*) FROM queue").fetchone()[0])
    def record_ids_after(self, after: str = "", limit: int = 5000) -> list[str]:
        rows=self.db.execute(
            """SELECT r.record_id FROM records r
            LEFT JOIN sources s ON s.source_key=r.source_key
            WHERE r.record_id>? AND COALESCE(s.retired,0)=0
            ORDER BY r.record_id LIMIT ?""",
            (after,max(1,int(limit))),
        ).fetchall()
        return [str(row[0]) for row in rows]
    def enqueue_records(self, record_ids: list[str]) -> int:
        if not record_ids:
            return 0
        before=self.db.total_changes
        now=time.time()
        with self.db:
            self.db.executemany(
                """INSERT OR IGNORE INTO queue(
                record_id,attempts,next_at,last_error,queued_at)
                SELECT r.record_id,0,0,NULL,? FROM records r
                LEFT JOIN sources s ON s.source_key=r.source_key
                WHERE r.record_id=? AND COALESCE(s.retired,0)=0""",
                ((now,record_id) for record_id in dict.fromkeys(record_ids)),
            )
        return self.db.total_changes-before
    def ack(self,ids:list[str]):
        if ids:
            now=time.time()
            with self.db:
                cursor=self.db.executemany("DELETE FROM queue WHERE record_id=?",((i,) for i in ids))
                if cursor.rowcount > 0:
                    stamp=datetime.fromtimestamp(now, timezone.utc).isoformat()
                    self.db.execute("INSERT INTO meta(key,value,updated_at) VALUES('last_successful_sync',?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",(stamp,now))
    def ack_session_records(self, record_ids: list[str]) -> int:
        """Compatibility no-op for legacy identity-only reconciliation callers.

        Presence cannot prove that the pending revision is durably stored.
        Leave the queue and sync checkpoint untouched; only a completed durable
        ingest may call ``ack`` to acknowledge pending updates.
        """
        return 0
    def fail(self,record_id,error,delay=30):
        self.db.execute("UPDATE queue SET attempts=attempts+1,next_at=?,last_error=? WHERE record_id=?",(time.time()+delay,error[:1000],record_id)); self.db.commit()
    @staticmethod
    def _source_category(kind: str) -> str|None:
        if kind in {"codex", "codex_session"}:
            return "codex_sessions"
        if kind in {"pi", "pi_session"}:
            return "pi_sessions"
        if kind in {"claude", "claude_session"}:
            return "claude_sessions"
        if kind in {"codex_memory", "pi_memory", "claude_memory", "agents_md", "persistent"} or kind.endswith("_memory"):
            return "memory_files"
        return None
    def source_counts(self):
        discovered={name:0 for name in ("codex_sessions","pi_sessions","claude_sessions","memory_files")}
        parsed=dict(discovered)
        synced=dict(discovered)
        rows=self.db.execute('''
            SELECT s.kind,count(*) AS discovered,
                   sum(CASE WHEN EXISTS (
                       SELECT 1 FROM records r WHERE r.source_key=s.source_key LIMIT 1
                   ) THEN 1 ELSE 0 END) AS parsed,
                   sum(CASE WHEN EXISTS (
                       SELECT 1 FROM records r WHERE r.source_key=s.source_key LIMIT 1
                   ) AND NOT EXISTS (
                       SELECT 1 FROM records r JOIN queue q ON q.record_id=r.record_id
                       WHERE r.source_key=s.source_key LIMIT 1
                   ) THEN 1 ELSE 0 END) AS synced
            FROM sources s
            WHERE s.active=1
            GROUP BY s.kind
        ''')
        for row in rows:
            category=self._source_category(row["kind"])
            if category:
                discovered[category]+=int(row["discovered"])
                parsed[category]+=int(row["parsed"] or 0)
                synced[category]+=int(row["synced"] or 0)
        return {"discovered":discovered,"parsed":parsed,"synced":synced}
    def meta_value(self,key):
        row=self.db.execute("SELECT value FROM meta WHERE key=?",(key,)).fetchone()
        return row[0] if row else None
    def set_meta(self,key,value):
        now=time.time()
        with self.db:
            self.db.execute(
                "INSERT INTO meta(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                (key,str(value),now),
            )
    def record_counts_by_agent(self):
        rows=self.db.execute('''
            WITH record_counts AS (
                SELECT source_key,count(*) AS n
                FROM records INDEXED BY records_source
                GROUP BY source_key
            )
            SELECT CASE
                       WHEN substr(s.kind,1,5)='codex' THEN 'codex'
                       WHEN substr(s.kind,1,2)='pi' THEN 'pi'
                       WHEN substr(s.kind,1,6)='claude' THEN 'claude_code'
                       WHEN replace(upper(s.path),char(92),'/') LIKE '%/AGENTS.MD' OR upper(s.path)='AGENTS.MD' THEN 'codex'
                       WHEN replace(upper(s.path),char(92),'/') LIKE '%/CLAUDE.MD' OR upper(s.path)='CLAUDE.MD' THEN 'claude_code'
                       WHEN s.kind='persistent' THEN 'shared'
                       ELSE 'unknown'
                   END AS agent,
                   sum(record_counts.n) AS n
            FROM record_counts
            LEFT JOIN sources s ON s.source_key=record_counts.source_key
            GROUP BY agent
        ''')
        return {row["agent"]:int(row["n"]) for row in rows}
    def stats(self):
        by_agent=self.record_counts_by_agent()
        pending=self.pending_count()
        counts=self.source_counts()
        return {"sources":self.db.execute("SELECT count(*) FROM sources").fetchone()[0],"active_sources":self.db.execute("SELECT count(*) FROM sources WHERE active=1").fetchone()[0],"records":self.db.execute("SELECT count(*) FROM records").fetchone()[0],"pending":pending,"pending_uploads":pending,"failed_uploads":self.db.execute("SELECT count(*) FROM queue WHERE last_error IS NOT NULL").fetchone()[0],"last_successful_sync":self.meta_value("last_successful_sync"),"discovered":counts["discovered"],"parsed":counts["parsed"],"synced":counts["synced"],"by_agent":by_agent}
    def search(self,query,limit=20):
        q=f"%{query}%"; rows=self.db.execute("SELECT payload FROM records WHERE payload LIKE ? ORDER BY updated_at DESC LIMIT ?",(q,limit)).fetchall(); return [json.loads(r[0]) for r in rows]
    def get(self,record_id):
        r=self.db.execute("SELECT payload FROM records WHERE record_id=?",(record_id,)).fetchone(); return json.loads(r[0]) if r else None
