from __future__ import annotations

import json
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
