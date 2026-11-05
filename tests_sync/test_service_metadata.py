from __future__ import annotations

from service.server import Store as ServiceStore
from sync.parsers import Chunk


def test_remote_store_updates_metadata_revision_with_same_raw_text(tmp_path):
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
    store = ServiceStore(str(tmp_path))
    try:
        first = store.ingest([Chunk(**base, source_agent="unknown").as_dict()])
        second = store.ingest([Chunk(**base, source_agent="codex").as_dict()])
        row = store.conn.execute(
            "SELECT source_agent,raw_text FROM memories WHERE source_identity='record'"
        ).fetchone()
        assert first["created"] == 1
        assert second["updated"] == 1
        assert row["source_agent"] == "codex"
        assert row["raw_text"] == "same raw text"
    finally:
        store.close()
