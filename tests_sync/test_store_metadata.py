from __future__ import annotations

import json
from types import SimpleNamespace

from sync.config import Config
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
