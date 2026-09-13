from __future__ import annotations

import json
import sqlite3
from datetime import datetime
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
        assert "redacted failure" not in json.dumps(stats)
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
        """
    )
    connection.commit()
    connection.close()

    cfg = Config(tmp_path, state, tmp_path / "config.toml")
    store = Store(config=cfg)
    try:
        assert store.pending_count() == 1
        assert store.meta_value("last_successful_sync") is None
        indexes = {
            row[0]
            for row in store.db.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        assert {"records_source", "queue_failed", "queue_schedule", "sources_active_kind"}.issubset(indexes)
    finally:
        store.close()
