from __future__ import annotations

import concurrent.futures
import json
import multiprocessing as mp
import os
import sqlite3
import stat
import time
from datetime import datetime
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from sync.config import Config
from sync.discovery import Source
from sync.parsers import Chunk, _stable_digest, _stable_identity
from sync.store import Store


def test_metadata_only_change_is_queued_without_content_version_bump(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    try:
        base = {
            "record_id": "record",
            "source_key": "source",
            "kind": "agents_md",
            "path": str(tmp_path / "AGENTS.md"),
            "session_id": "",
            "ordinal": 0,
            "role": "system",
            "text": "raw",
            "raw_text": "raw",
            "source_type": "agents_md",
            "device_id": "device",
        }
        store.upsert_chunks([Chunk(**base, source_agent="unknown")])
        store.ack(["record"])
        store.upsert_chunks([Chunk(**base, source_agent="codex")])

        row = store.db.execute(
            "SELECT version,payload FROM records WHERE record_id='record'"
        ).fetchone()
        payload = json.loads(row[1])
        assert row[0] == 1
        assert payload["source_agent"] == "codex"
        assert payload["ingested_at"]
        assert payload["updated_at"]
        assert store.pending_count() == 1
        store.ack(["record"])
        store.upsert_chunks([Chunk(**base, source_agent="codex")])
        assert store.pending_count() == 0
    finally:
        store.close()


def test_record_source_key_stays_consistent_with_updated_payload(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    base = {
        "record_id": "shared-record",
        "kind": "codex_memory",
        "path": "automation.toml",
        "session_id": "",
        "ordinal": 0,
        "role": "system",
        "text": "raw",
        "raw_text": "raw",
    }
    try:
        store.upsert_chunks([Chunk(**base, source_key="source-a")])
        store.ack(["shared-record"])
        store.upsert_chunks([Chunk(**base, source_key="source-b")])

        row = store.db.execute(
            "SELECT source_key,payload FROM records WHERE record_id='shared-record'"
        ).fetchone()
        assert row["source_key"] == "source-b"
        assert json.loads(row["payload"])["source_key"] == "source-b"
    finally:
        store.close()


def test_identical_rescan_repairs_inconsistent_record_source_column(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    chunk = Chunk(
        record_id="record",
        source_key="expected-source",
        kind="codex_memory",
        path="memory.md",
        session_id="",
        ordinal=0,
        role="system",
        text="raw",
        raw_text="raw",
    )
    try:
        store.upsert_chunks([chunk])
        store.ack(["record"])
        store.db.execute(
            "UPDATE records SET source_key='stale-source' WHERE record_id='record'"
        )
        store.db.commit()

        store.upsert_chunks([chunk])

        assert store.db.execute(
            "SELECT source_key FROM records WHERE record_id='record'"
        ).fetchone()[0] == "expected-source"
        assert store.pending_count() == 1
    finally:
        store.close()


def test_memory_identity_does_not_depend_on_mutable_classification():
    old = SimpleNamespace(
        kind="persistent", source_agent="unknown", source_type="persistent"
    )
    new = SimpleNamespace(
        kind="persistent",
        source_agent="claude_code",
        source_type="project_instruction",
    )
    arguments = {"session": "", "message": "memory:persistent:repo:path:heading"}
    assert _stable_identity(old, **arguments) == _stable_identity(new, **arguments)
    assert _stable_identity(new, **arguments) == _stable_digest(
        "native", "unknown", "persistent", "", "memory", arguments["message"]
    )


def test_metadata_revision_changes_source_version():
    base = {
        "record_id": "record",
        "source_key": "source",
        "kind": "agents_md",
        "path": "AGENTS.md",
        "session_id": "",
        "ordinal": 0,
        "role": "system",
        "text": "same raw text",
        "raw_text": "same raw text",
        "source_type": "agents_md",
    }
    old = Chunk(**base, source_agent="unknown").as_dict()
    new = Chunk(**base, source_agent="codex").as_dict()
    assert old["content_hash"] == new["content_hash"]
    assert old["source_version"] != new["source_version"]


def test_filter_metadata_changes_source_version():
    base = {
        "record_id": "record",
        "source_key": "source",
        "kind": "agents_md",
        "path": "AGENTS.md",
        "session_id": "",
        "ordinal": 0,
        "role": "system",
        "text": "same raw text",
        "raw_text": "same raw text",
        "source_agent": "codex",
        "source_type": "agents_md",
    }
    old = Chunk(**base, project="old", repo="repo-a", worktree="tree-a").as_dict()
    new = Chunk(**base, project="new", repo="repo-b", worktree="tree-b").as_dict()
    assert old["content_hash"] == new["content_hash"]
    assert old["source_version"] != new["source_version"]


def test_soft_missing_changes_source_version_and_queues_update(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    chunk = Chunk(
        record_id="record",
        source_key="source",
        kind="persistent",
        path="memory.md",
        session_id="",
        ordinal=0,
        role="system",
        text="raw",
        raw_text="raw",
    )
    try:
        store.upsert_chunks([chunk])
        before = json.loads(
            store.db.execute(
                "SELECT payload FROM records WHERE record_id='record'"
            ).fetchone()[0]
        )
        store.ack(["record"])
        assert store.reconcile_source("source", set()) == 1
        after = json.loads(
            store.db.execute(
                "SELECT payload FROM records WHERE record_id='record'"
            ).fetchone()[0]
        )
        assert after["source_missing"] is True
        assert after["source_version"] != before["source_version"]
        assert store.pending_count() == 1
        store.ack(["record"])
        assert store.reconcile_source("source", set()) == 0
        assert store.pending_count() == 0
    finally:
        store.close()


def test_ack_updates_last_successful_sync_atomically(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    chunk = Chunk(
        record_id="record",
        source_key="source",
        kind="codex",
        path="session.jsonl",
        session_id="session",
        ordinal=0,
        role="user",
        text="raw",
        raw_text="raw",
    )
    try:
        store.upsert_chunks([chunk])
        store.ack(["not-queued"])
        assert store.meta_value("last_successful_sync") is None

        store.db.execute(
            """
            CREATE TRIGGER reject_last_sync BEFORE INSERT ON meta
            WHEN NEW.key='last_successful_sync'
            BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END
            """
        )
        with pytest.raises(sqlite3.IntegrityError, match="synthetic failure"):
            store.ack(["record"])
        assert store.pending_count() == 1
        assert store.meta_value("last_successful_sync") is None

        store.db.execute("DROP TRIGGER reject_last_sync")
        store.ack(["record"])
        stamp = store.meta_value("last_successful_sync")
        assert datetime.fromisoformat(stamp).utcoffset() is not None
        assert store.pending_count() == 0
    finally:
        store.close()

    reopened = Store(config=cfg)
    try:
        assert reopened.meta_value("last_successful_sync") == stamp
    finally:
        reopened.close()


def test_record_inventory_cursor_and_missing_queue_are_restart_safe(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    try:
        for record_id in ("c", "a", "b"):
            store.upsert_chunks(
                [
                    Chunk(
                        record_id=record_id,
                        source_key="source",
                        kind="codex",
                        path="session.jsonl",
                        session_id="session",
                        ordinal=0,
                        role="user",
                        text=record_id,
                        raw_text=record_id,
                    )
                ]
            )
        store.ack(["a", "b", "c"])

        assert store.record_ids_after("", 2) == ["a", "b"]
        assert store.record_ids_after("b", 2) == ["c"]
        assert store.enqueue_records(["b", "missing", "b"]) == 1
        assert store.pending_count() == 1
        assert store.enqueue_records(["b"]) == 0
        store.set_meta("remote_source_cursor", "b")
    finally:
        store.close()

    reopened = Store(config=cfg)
    try:
        assert reopened.meta_value("remote_source_cursor") == "b"
        assert reopened.record_ids_after("b", 2) == ["c"]
        assert reopened.pending_count() == 1
    finally:
        reopened.close()


def test_retired_sources_never_reenter_remote_inventory(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    active = Source("active", "codex", tmp_path / "active.jsonl", "device")
    retired = Source(
        "retired",
        "codex_memory",
        tmp_path / ".codex" / "automations" / "job" / "runs" / "run.jsonl",
        "device",
    )
    try:
        for source in (active, retired):
            store.register_source(source)
            store.upsert_chunks(
                [
                    Chunk(
                        record_id=f"record:{source.source_key}",
                        source_key=source.source_key,
                        kind=source.kind,
                        path=str(source.path),
                        session_id=source.source_key,
                        ordinal=0,
                        role="user",
                        text="raw",
                        raw_text="raw",
                    )
                ]
            )
        store.ack(["record:active", "record:retired"])
        store.db.execute("UPDATE sources SET retired=1 WHERE source_key='retired'")
        store.db.commit()

        assert store.record_ids_after("", 10) == ["record:active"]
        assert store.enqueue_records(["record:retired", "record:active"]) == 1
        assert [row["record_id"] for row in store.pending(10)] == ["record:active"]
    finally:
        store.close()


def test_source_and_upload_counts_are_exact(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    sources = [
        Source("codex:one", "codex", tmp_path / "one.jsonl", "device"),
        Source("codex:two", "codex_session", tmp_path / "two.jsonl", "device"),
        Source("pi:one", "pi", tmp_path / "pi.jsonl", "device"),
        Source("claude:old", "claude", tmp_path / "old.jsonl", "device"),
        Source("memory:one", "persistent", tmp_path / "MEMORY.md", "device"),
        Source("memory:two", "codex_memory", tmp_path / "notes.md", "device"),
    ]
    try:
        for source in sources:
            store.register_source(source)
        store.db.execute("UPDATE sources SET active=0 WHERE source_key='claude:old'")
        for source in (sources[0], sources[2], sources[4]):
            store.upsert_chunks(
                [
                    Chunk(
                        record_id=f"record:{source.source_key}",
                        source_key=source.source_key,
                        kind=source.kind,
                        path=str(source.path),
                        session_id=source.source_key,
                        ordinal=0,
                        role="user",
                        text="raw",
                        raw_text="raw",
                        source_agent=source.source_agent,
                        source_type=source.source_type,
                    )
                ]
            )
        store.ack(["record:codex:one", "record:memory:one"])
        store.fail("record:pi:one", "redacted failure")

        stats = store.stats()

        assert stats["discovered"] == {
            "codex_sessions": 2,
            "pi_sessions": 1,
            "claude_sessions": 0,
            "memory_files": 2,
        }
        assert stats["synced"] == {
            "codex_sessions": 1,
            "pi_sessions": 0,
            "claude_sessions": 0,
            "memory_files": 1,
        }
        assert stats["parsed"] == {
            "codex_sessions": 1,
            "pi_sessions": 1,
            "claude_sessions": 0,
            "memory_files": 1,
        }
        assert stats["pending_uploads"] == stats["pending"] == 1
        assert stats["failed_uploads"] == 1
        assert stats["by_agent"] == {"codex": 1, "pi": 1, "shared": 1}
        assert "redacted failure" not in json.dumps(stats)
    finally:
        store.close()


def test_record_counts_by_agent_use_source_identity_without_parsing_payload(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    sources = [
        Source("codex", "codex", tmp_path / "codex.jsonl", "device"),
        Source("pi", "pi_memory", tmp_path / "memory.md", "device"),
        Source("claude", "persistent", tmp_path / "CLAUDE.md", "device"),
        Source("agents", "agents_md", tmp_path / "AGENTS.md", "device"),
        Source("shared", "persistent", tmp_path / "MEMORY.md", "device"),
        Source("windows-agents", "agents_md", PureWindowsPath(r"C:\repo\AGENTS.md"), "device"),
        Source("windows-claude", "persistent", PureWindowsPath(r"C:\repo\Claude.md"), "device"),
        Source("uppercase-kind", "Codex_session", tmp_path / "uppercase.jsonl", "device"),
    ]
    try:
        for source in sources:
            store.register_source(source)
            store.upsert_chunks(
                [
                    Chunk(
                        record_id=f"record:{source.source_key}",
                        source_key=source.source_key,
                        kind=source.kind,
                        path=str(source.path),
                        session_id=source.source_key,
                        ordinal=0,
                        role="user",
                        text="raw",
                        raw_text="raw",
                        source_agent="legacy_unknown",
                        source_type=source.source_type,
                    )
                ]
            )
        store.db.execute(
            "INSERT INTO records(record_id,source_key,content_hash,version,payload,updated_at) VALUES(?,?,?,?,?,?)",
            ("orphan", "missing", "hash", 1, "not-json", 0),
        )
        store.db.commit()

        assert store.record_counts_by_agent() == {
            "claude_code": 2,
            "codex": 3,
            "pi": 1,
            "shared": 1,
            "unknown": 2,
        }
    finally:
        store.close()


def test_existing_database_gets_meta_migration_without_data_loss(tmp_path):
    state = tmp_path / ".state"
    state.mkdir()
    database = state / "sync.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE sources(source_key TEXT PRIMARY KEY,kind TEXT NOT NULL,path TEXT NOT NULL,device_id TEXT,project TEXT,active INTEGER DEFAULT 1,size INTEGER,mtime REAL,inode INTEGER,updated_at REAL);
        CREATE TABLE records(record_id TEXT PRIMARY KEY,source_key TEXT NOT NULL,content_hash TEXT NOT NULL,version INTEGER DEFAULT 1,payload TEXT NOT NULL,updated_at REAL);
        CREATE TABLE queue(record_id TEXT PRIMARY KEY,attempts INTEGER DEFAULT 0,next_at REAL DEFAULT 0,last_error TEXT,queued_at REAL);
        CREATE TABLE cursors(source_key TEXT PRIMARY KEY,offset INTEGER DEFAULT 0,inode INTEGER,size INTEGER,updated_at REAL);
        INSERT INTO queue(record_id,attempts,next_at,last_error,queued_at) VALUES('pending',0,0,NULL,1);
        INSERT INTO sources(
            source_key,kind,path,device_id,project,active,size,mtime,inode,updated_at
        ) VALUES(
            'legacy-run','codex_memory',
            '/Users/test/.codex/automations/job/runs/run.jsonl',
            'device','',1,42,1,1,1
        );
        INSERT INTO records(
            record_id,source_key,content_hash,version,payload,updated_at
        ) VALUES('legacy-record','legacy-run','hash',1,'{"raw_text":"retained"}',1);
        INSERT INTO queue(record_id,attempts,next_at,last_error,queued_at)
        VALUES('legacy-record',0,0,NULL,1);
        """
    )
    connection.commit()
    connection.close()

    cfg = Config(tmp_path, state, tmp_path / "config.toml")
    store = Store(config=cfg)
    try:
        assert store.pending_count() == 1
        retired = store.db.execute(
            "SELECT active,retired FROM sources WHERE source_key='legacy-run'"
        ).fetchone()
        assert tuple(retired) == (0, 1)
        assert store.get("legacy-record")["raw_text"] == "retained"
        assert "legacy-record" not in store.record_ids_after("", 10)
        assert store.meta_value("legacy_automation_retirement_v1")
        assert store.meta_value("last_successful_sync") is None
        indexes = {
            row[0]
            for row in store.db.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        assert {"records_source", "queue_failed", "queue_schedule", "queue_ready_order", "sources_active_kind"}.issubset(indexes)
    finally:
        store.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not available on Windows")
def test_new_store_restricts_state_directory_and_sqlite_files(tmp_path):
    state = tmp_path / "custom-state"
    state.mkdir(mode=0o755)
    state.chmod(0o755)
    cfg = Config(tmp_path, state, tmp_path / "config.toml")
    database = state / "sync.db"
    store = Store(config=cfg)
    try:
        store.set_meta("permission-test", "written")
        sidecars = [Path(f"{database}-wal"), Path(f"{database}-shm")]
        assert all(path.exists() for path in sidecars)
        assert stat.S_IMODE(state.stat().st_mode) == 0o700
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o600
            for path in [database, *sidecars]
        )
    finally:
        store.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not available on Windows")
def test_existing_store_restricts_sqlite_files_without_data_loss(tmp_path):
    state = tmp_path / "custom-state"
    state.mkdir(mode=0o755)
    database = state / "sync.db"
    legacy = sqlite3.connect(database)
    legacy.execute("PRAGMA journal_mode=WAL")
    legacy.execute("CREATE TABLE retained(value TEXT NOT NULL)")
    legacy.execute("INSERT INTO retained(value) VALUES('private payload')")
    legacy.commit()
    sidecars = [Path(f"{database}-wal"), Path(f"{database}-shm")]
    assert all(path.exists() for path in sidecars)
    state.chmod(0o755)
    for path in [database, *sidecars]:
        path.chmod(0o644)

    cfg = Config(tmp_path, state, tmp_path / "config.toml")
    store = Store(config=cfg)
    try:
        assert store.db.execute("SELECT value FROM retained").fetchone()[0] == "private payload"
        assert stat.S_IMODE(state.stat().st_mode) == 0o700
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o600
            for path in [database, *sidecars]
        )
    finally:
        store.close()
        legacy.close()


def test_pending_query_plan_uses_ready_order_index_without_temp_btree(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    now = time.time()
    try:
        source = Source("source", "codex", tmp_path / "session.jsonl", "device")
        store.register_source(source)
        chunks = [
            Chunk(
                record_id=f"record:{i:02d}",
                source_key="source",
                kind="codex",
                path=str(source.path),
                session_id="session",
                ordinal=i,
                role="user",
                text=f"payload {i}",
                raw_text=f"payload {i}",
            )
            for i in range(10)
        ]
        store.upsert_chunks(chunks)
        store.db.execute(
            "UPDATE queue SET next_at=? WHERE record_id IN ('record:08', 'record:09')",
            (now + 3600,),
        )
        store.db.commit()

        plan = store.db.execute(
            "EXPLAIN QUERY PLAN SELECT q.*,r.payload FROM queue q JOIN records r ON r.record_id=q.record_id WHERE q.next_at<=? ORDER BY q.queued_at LIMIT ?",
            (now, 50),
        ).fetchall()
        details = [row[3] for row in plan]
        assert any("queue_ready_order" in detail for detail in details)
        assert not any("USE TEMP B-TREE FOR ORDER BY" in detail for detail in details)

        pending = store.pending(limit=50, now=now)
        assert len(pending) == 8
        assert [r["record_id"] for r in pending] == [f"record:{i:02d}" for i in range(8)]

        limited = store.pending(limit=3, now=now)
        assert [r["record_id"] for r in limited] == ["record:00", "record:01", "record:02"]
    finally:
        store.close()


def test_ack_session_records_preserves_all_source_kinds_and_checkpoint(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    try:
        sources = [
            Source("s_codex", "codex", tmp_path / "codex.jsonl", "dev"),
            Source("s_codex_s", "codex_session", tmp_path / "codex_s.jsonl", "dev"),
            Source("s_pi", "pi", tmp_path / "pi.jsonl", "dev"),
            Source("s_pi_s", "pi_session", tmp_path / "pi_s.jsonl", "dev"),
            Source("s_claude", "claude", tmp_path / "claude.jsonl", "dev"),
            Source("s_claude_s", "claude_session", tmp_path / "claude_s.jsonl", "dev"),
            Source("s_mem", "codex_memory", tmp_path / "memory.md", "dev"),
            Source("s_agents", "agents_md", tmp_path / "AGENTS.md", "dev"),
            Source("s_persist", "persistent", tmp_path / "persist.json", "dev"),
            Source("s_retired", "codex", tmp_path / "retired.jsonl", "dev"),
        ]
        for s in sources:
            store.register_source(s)
        store.db.execute("UPDATE sources SET retired=1 WHERE source_key='s_retired'")
        store.db.commit()

        chunks = [
            Chunk(
                record_id=f"rec_{s.source_key}",
                source_key=s.source_key,
                kind=s.kind,
                path=str(s.path),
                session_id="sess",
                ordinal=0,
                role="user",
                text="content",
                raw_text="content",
            )
            for s in sources
        ]
        store.upsert_chunks(chunks)
        assert store.pending_count() == 10
        previous_changes = store.db.total_changes

        # Empty list returns 0
        assert store.ack_session_records([]) == 0
        assert store.pending_count() == 10

        # Non-existent ID returns 0
        assert store.ack_session_records(["non_existent_id"]) == 0
        assert store.pending_count() == 10

        # Identity-only ACK cannot confirm any source kind is durably uploaded.
        all_ids = [c.record_id for c in chunks] + ["non_existent_id"]
        deleted = store.ack_session_records(all_ids)

        assert deleted == 0
        assert store.pending_count() == 10

        remaining = [r["record_id"] for r in store.pending(limit=10)]
        assert set(remaining) == {c.record_id for c in chunks}

        sync_meta = store.meta_value("last_successful_sync")
        assert sync_meta is None

        # Repeated compatibility calls remain zero-write no-ops.
        assert store.ack_session_records(all_ids) == 0
        assert store.pending_count() == 10
        assert store.db.total_changes == previous_changes
    finally:
        store.close()
def _concurrent_open_store(db_path_str: str, config_tuple: tuple) -> int:
    cfg = Config(Path(config_tuple[0]), Path(config_tuple[1]), Path(config_tuple[2]))
    store = Store(path=Path(db_path_str), config=cfg)
    try:
        return store.stats()["records"]
    finally:
        store.close()


def _concurrent_write_store(db_path_str: str, config_tuple: tuple, source_key: str) -> bool:
    cfg = Config(Path(config_tuple[0]), Path(config_tuple[1]), Path(config_tuple[2]))
    store = Store(path=Path(db_path_str), config=cfg)
    try:
        source = Source(source_key, "codex", Path(config_tuple[0]) / f"{source_key}.jsonl", "dev")
        store.register_source(source)
        chunks = [
            Chunk(
                f"{source_key}_{i}",
                source_key,
                "codex",
                str(source.path),
                "sess",
                i,
                "user",
                f"msg_{i}",
                f"msg_{i}",
            )
            for i in range(5)
        ]
        store.upsert_chunks(chunks)
        store.ack([f"{source_key}_0", f"{source_key}_1"])
        return True
    finally:
        store.close()


def test_source_record_counts_bootstrap_reentry(tmp_path):
    state = tmp_path / ".state"
    state.mkdir()
    db_file = state / "sync.db"
    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE sources(source_key TEXT PRIMARY KEY,kind TEXT NOT NULL,path TEXT NOT NULL,device_id TEXT,project TEXT,active INTEGER DEFAULT 1,retired INTEGER NOT NULL DEFAULT 0,size INTEGER,mtime REAL,inode INTEGER,updated_at REAL);
        CREATE TABLE records(record_id TEXT PRIMARY KEY,source_key TEXT NOT NULL,content_hash TEXT NOT NULL,version INTEGER DEFAULT 1,payload TEXT NOT NULL,updated_at REAL);
        CREATE TABLE queue(record_id TEXT PRIMARY KEY,attempts INTEGER DEFAULT 0,next_at REAL DEFAULT 0,last_error TEXT,queued_at REAL);
        CREATE TABLE cursors(source_key TEXT PRIMARY KEY,offset INTEGER DEFAULT 0,inode INTEGER,size INTEGER,updated_at REAL);
        CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL,updated_at REAL NOT NULL);

        INSERT INTO sources(source_key, kind, path, device_id, project, active, retired)
        VALUES ('s1', 'codex', '/path/s1.jsonl', 'dev', '', 1, 0),
               ('s2', 'codex', '/path/s2.jsonl', 'dev', '', 1, 0),
               ('s3', 'codex', '/path/s3.jsonl', 'dev', '', 1, 0);

        INSERT INTO records(record_id, source_key, content_hash, version, payload, updated_at)
        VALUES ('s1_1', 's1', 'h', 1, '{}', 1.0),
               ('s1_2', 's1', 'h', 1, '{}', 1.0),
               ('s1_3', 's1', 'h', 1, '{}', 1.0),
               ('s2_1', 's2', 'h', 1, '{}', 1.0),
               ('s2_2', 's2', 'h', 1, '{}', 1.0),
               ('s3_1', 's3', 'h', 1, '{}', 1.0);

        INSERT INTO queue(record_id, attempts, next_at, last_error, queued_at)
        VALUES ('s1_1', 0, 0, NULL, 1.0),
               ('s1_2', 0, 0, NULL, 1.0),
               ('s3_1', 0, 0, NULL, 1.0);
        """
    )
    conn.commit()
    conn.close()

    cfg = Config(tmp_path, state, tmp_path / "config.toml")
    store = Store(config=cfg)
    try:
        assert store.meta_value("stats_cache_v1") == "completed"
        counts = {
            r[0]: (r[1], r[2])
            for r in store.db.execute(
                "SELECT source_key, record_count, pending_count FROM source_record_counts"
            ).fetchall()
        }
        assert counts["s1"] == (3, 2)
        assert counts["s2"] == (2, 0)
        assert counts["s3"] == (1, 1)
        stats = store.stats()
        assert stats["records"] == 6
        assert stats["pending"] == 3
    finally:
        store.close()

    for _ in range(3):
        reopened = Store(config=cfg)
        try:
            assert reopened.stats()["records"] == 6
            assert reopened.stats()["pending"] == 3
        finally:
            reopened.close()

    reopened = Store(config=cfg)
    try:
        reopened.ack(["s1_1"])
        assert reopened.stats()["records"] == 6
        assert reopened.stats()["pending"] == 2
        chunk = Chunk(
            "s3_2",
            "s3",
            "codex",
            "/path/s3.jsonl",
            "sess",
            1,
            "user",
            "text",
            "text",
        )
        reopened.upsert_chunks([chunk])
        assert reopened.stats()["records"] == 7
        assert reopened.stats()["pending"] == 3
    finally:
        reopened.close()


def test_source_record_counts_cross_process_migration_and_writes(tmp_path):
    state = tmp_path / ".state"
    state.mkdir()
    db_file = state / "sync.db"

    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE sources(source_key TEXT PRIMARY KEY,kind TEXT NOT NULL,path TEXT NOT NULL,device_id TEXT,project TEXT,active INTEGER DEFAULT 1,retired INTEGER NOT NULL DEFAULT 0,size INTEGER,mtime REAL,inode INTEGER,updated_at REAL);
        CREATE TABLE records(record_id TEXT PRIMARY KEY,source_key TEXT NOT NULL,content_hash TEXT NOT NULL,version INTEGER DEFAULT 1,payload TEXT NOT NULL,updated_at REAL);
        CREATE TABLE queue(record_id TEXT PRIMARY KEY,attempts INTEGER DEFAULT 0,next_at REAL DEFAULT 0,last_error TEXT,queued_at REAL);
        CREATE TABLE cursors(source_key TEXT PRIMARY KEY,offset INTEGER DEFAULT 0,inode INTEGER,size INTEGER,updated_at REAL);
        CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL,updated_at REAL NOT NULL);
        INSERT INTO records VALUES('r0', 's0', 'h0', 1, '{}', 1.0);
        INSERT INTO queue VALUES('r0', 0, 0, NULL, 1.0);
        """
    )
    conn.commit()
    conn.close()

    cfg_tuple = (str(tmp_path), str(state), str(tmp_path / "config.toml"))

    ctx = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
        futures = []
        for i in range(3):
            futures.append(pool.submit(_concurrent_open_store, str(db_file), cfg_tuple))
        for i in range(3):
            futures.append(pool.submit(_concurrent_write_store, str(db_file), cfg_tuple, f"worker_src_{i}"))
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            assert res is not None

    cfg = Config(Path(cfg_tuple[0]), Path(cfg_tuple[1]), Path(cfg_tuple[2]))
    final_store = Store(path=db_file, config=cfg)
    try:
        stats = final_store.stats()
        assert stats["records"] == 16
        assert stats["pending"] == 10
        actual_records = final_store.db.execute("SELECT count(*) FROM records").fetchone()[0]
        actual_pending = final_store.db.execute("SELECT count(*) FROM queue").fetchone()[0]
        assert stats["records"] == actual_records
        assert stats["pending"] == actual_pending

        rows = final_store.db.execute(
            "SELECT source_key, record_count, pending_count FROM source_record_counts"
        ).fetchall()
        for r in rows:
            sk, rc, pc = r[0], r[1], r[2]
            real_rc = final_store.db.execute(
                "SELECT count(*) FROM records WHERE source_key=?", (sk,)
            ).fetchone()[0]
            real_pc = final_store.db.execute(
                "SELECT count(*) FROM records r JOIN queue q ON q.record_id=r.record_id WHERE r.source_key=?",
                (sk,),
            ).fetchone()[0]
            assert rc == real_rc
            assert pc == real_pc
        assert final_store.meta_value("stats_cache_v1") == "completed"
    finally:
        final_store.close()


def test_source_record_counts_legacy_daemon_writes(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    try:
        source = Source("legacy_src", "codex", tmp_path / "legacy.jsonl", "dev")
        store.register_source(source)

        store.db.execute(
            "INSERT INTO records(record_id, source_key, content_hash, version, payload, updated_at) VALUES ('r1', 'legacy_src', 'h1', 1, '{}', 1.0)"
        )
        store.db.commit()
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='legacy_src'"
        ).fetchone()
        assert (row[0], row[1]) == (1, 0)

        store.db.execute(
            "INSERT INTO queue(record_id, attempts, next_at, last_error, queued_at) VALUES ('r1', 0, 0, NULL, 1.0)"
        )
        store.db.commit()
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='legacy_src'"
        ).fetchone()
        assert (row[0], row[1]) == (1, 1)

        store.db.execute(
            "INSERT INTO records(record_id, source_key, content_hash, version, payload, updated_at) VALUES ('r2', 'legacy_src', 'h2', 1, '{}', 1.0)"
        )
        store.db.execute(
            "INSERT INTO queue(record_id, attempts, next_at, last_error, queued_at) VALUES ('r2', 0, 0, NULL, 1.0)"
        )
        store.db.commit()
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='legacy_src'"
        ).fetchone()
        assert (row[0], row[1]) == (2, 2)

        store.db.execute("DELETE FROM queue WHERE record_id = 'r1'")
        store.db.commit()
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='legacy_src'"
        ).fetchone()
        assert (row[0], row[1]) == (2, 1)

        store.db.execute("DELETE FROM records WHERE record_id = 'r2'")
        store.db.commit()
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='legacy_src'"
        ).fetchone()
        assert (row[0], row[1]) == (1, 0)

        store.db.execute("DELETE FROM queue WHERE record_id = 'r2'")
        store.db.commit()
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='legacy_src'"
        ).fetchone()
        assert (row[0], row[1]) == (1, 0)

        store.db.execute("DELETE FROM records WHERE record_id = 'r1'")
        store.db.commit()
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='legacy_src'"
        ).fetchone()
        assert (row[0], row[1]) == (0, 0)

        stats = store.stats()
        assert stats["records"] == 0
        assert stats["pending"] == 0
    finally:
        store.close()


def test_source_record_counts_record_source_migration(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    try:
        store.register_source(Source("src_alpha", "codex", tmp_path / "alpha.jsonl", "dev"))
        store.register_source(Source("src_beta", "codex", tmp_path / "beta.jsonl", "dev"))
        store.register_source(Source("src_gamma", "codex", tmp_path / "gamma.jsonl", "dev"))

        store.upsert_chunks([
            Chunk("rec_acked", "src_alpha", "codex", str(tmp_path / "alpha.jsonl"), "s", 0, "user", "a", "a"),
            Chunk("rec_pending", "src_alpha", "codex", str(tmp_path / "alpha.jsonl"), "s", 1, "user", "b", "b"),
        ])
        store.ack(["rec_acked"])

        alpha = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_alpha'"
        ).fetchone()
        assert (alpha[0], alpha[1]) == (2, 1)

        store.db.execute("UPDATE records SET source_key='src_beta' WHERE record_id='rec_acked'")
        store.db.commit()
        alpha = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_alpha'"
        ).fetchone()
        beta = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_beta'"
        ).fetchone()
        assert (alpha[0], alpha[1]) == (1, 1)
        assert (beta[0], beta[1]) == (1, 0)

        store.db.execute("UPDATE records SET source_key='src_beta' WHERE record_id='rec_pending'")
        store.db.commit()
        alpha = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_alpha'"
        ).fetchone()
        beta = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_beta'"
        ).fetchone()
        assert (alpha[0], alpha[1]) == (0, 0)
        assert (beta[0], beta[1]) == (2, 1)

        store.upsert_chunks([
            Chunk("rec_pending", "src_gamma", "codex", str(tmp_path / "gamma.jsonl"), "s", 1, "user", "b_updated", "b_updated"),
        ])
        beta = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_beta'"
        ).fetchone()
        gamma = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_gamma'"
        ).fetchone()
        assert (beta[0], beta[1]) == (1, 0)
        assert (gamma[0], gamma[1]) == (1, 1)

        stats = store.stats()
        assert stats["records"] == 2
        assert stats["pending"] == 1
    finally:
        store.close()


def test_source_record_counts_duplicate_queue_ack_idempotence(tmp_path):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=cfg)
    try:
        store.register_source(Source("src_idem", "codex", tmp_path / "idem.jsonl", "dev"))
        store.upsert_chunks([
            Chunk("c1", "src_idem", "codex", str(tmp_path / "idem.jsonl"), "s", 0, "user", "t1", "t1"),
            Chunk("c2", "src_idem", "codex", str(tmp_path / "idem.jsonl"), "s", 1, "user", "t2", "t2"),
        ])
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_idem'"
        ).fetchone()
        assert (row[0], row[1]) == (2, 2)

        store.ack(["c1"])
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_idem'"
        ).fetchone()
        assert (row[0], row[1]) == (2, 1)

        store.ack(["c1"])
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_idem'"
        ).fetchone()
        assert (row[0], row[1]) == (2, 1)

        store.ack(["non_existent_1", "non_existent_2"])
        store.ack([])
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_idem'"
        ).fetchone()
        assert (row[0], row[1]) == (2, 1)

        added = store.enqueue_records(["c1", "c1", "ghost"])
        assert added == 1
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_idem'"
        ).fetchone()
        assert (row[0], row[1]) == (2, 2)

        added = store.enqueue_records(["c1", "c2"])
        assert added == 0
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_idem'"
        ).fetchone()
        assert (row[0], row[1]) == (2, 2)

        store.db.execute("DELETE FROM queue WHERE record_id='c1'")
        store.db.commit()
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_idem'"
        ).fetchone()
        assert (row[0], row[1]) == (2, 1)

        store.db.execute("DELETE FROM queue WHERE record_id='c1'")
        store.db.commit()
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_idem'"
        ).fetchone()
        assert (row[0], row[1]) == (2, 1)

        store.db.execute("DELETE FROM queue WHERE record_id='c2'")
        store.db.commit()
        row = store.db.execute(
            "SELECT record_count, pending_count FROM source_record_counts WHERE source_key='src_idem'"
        ).fetchone()
        assert (row[0], row[1]) == (2, 0)

        store.db.execute(
            "INSERT INTO records(record_id, source_key, content_hash, version, payload, updated_at) VALUES ('ghost', 'src_idem', 'h', 1, '{}', 1.0)"
        )
        store.db.execute("UPDATE source_record_counts SET pending_count=0 WHERE source_key='src_idem'")
        store.db.execute("INSERT INTO queue(record_id, queued_at) VALUES('ghost', 1.0)")
        store.db.execute("UPDATE source_record_counts SET pending_count=0 WHERE source_key='src_idem'")
        store.db.execute("DELETE FROM queue WHERE record_id='ghost'")
        store.db.commit()
        row = store.db.execute(
            "SELECT pending_count FROM source_record_counts WHERE source_key='src_idem'"
        ).fetchone()
        assert row[0] >= 0
    finally:
        store.close()
