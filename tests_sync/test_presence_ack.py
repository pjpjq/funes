from __future__ import annotations

from dataclasses import replace

import pytest

from sync.config import Config
from sync.daemon import SyncDaemon
from sync.discovery import Source
from sync.parsers import Chunk
from sync.store import Store


@pytest.fixture(params=("codex", "codex_session", "pi", "pi_session", "claude", "claude_session"))
def session_store(tmp_path, request):
    config = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    store = Store(config=config)
    source = Source("session", request.param, tmp_path / "session.jsonl", "device")
    store.register_source(source)
    chunk = Chunk(
        record_id="record",
        source_key=source.source_key,
        kind=source.kind,
        path=str(source.path),
        session_id="session",
        ordinal=0,
        role="user",
        text="original raw",
        raw_text="original raw",
    )
    store.upsert_chunks([chunk])
    store.ack([chunk.record_id])
    try:
        yield config, store, chunk
    finally:
        store.close()


class PresentRemote:
    def __init__(self):
        self.ingested = []

    def missing_source_identities(self, identities):
        return []

    def ingest(self, documents):
        self.ingested.extend(documents)
        return {"durable": True, "accepted": len(documents)}


@pytest.fixture(params=("tombstone", "raw", "attribution", "metadata", "unchanged"))
def pending_revision(session_store, request):
    config, store, chunk = session_store
    original = store.get(chunk.record_id)
    previous_sync = store.meta_value("last_successful_sync")
    assert previous_sync
    if request.param == "tombstone":
        store.reconcile_source(chunk.source_key, set())
    elif request.param == "raw":
        store.upsert_chunks([replace(chunk, text="revised raw", raw_text="revised raw")])
    elif request.param == "attribution":
        store.upsert_chunks([replace(chunk, project="revised-project")])
    elif request.param == "metadata":
        store.upsert_chunks([replace(chunk, metadata={"revision": "new metadata"})])
    else:
        store.enqueue_records([chunk.record_id])
    pending = store.get(chunk.record_id)
    if request.param in {"tombstone", "raw", "attribution"}:
        assert pending["source_version"] != original["source_version"]
    else:
        assert pending["source_version"] == original["source_version"]
    if request.param == "metadata":
        assert pending["metadata"] != original["metadata"]
    assert store.pending_count() == 1
    return config, store, pending, previous_sync


def test_presence_keeps_revision_pending_until_durable_ingest(pending_revision):
    config, store, pending, previous_sync = pending_revision
    remote = PresentRemote()
    daemon = SyncDaemon(config, store, remote)

    result = daemon.reconcile_remote_sources(1)

    assert result == {"complete": False, "checked": 1, "queued": 0, "acknowledged": 0}
    assert store.pending_count() == 1
    assert store.get(pending["record_id"]) == pending
    assert remote.ingested == []
    assert daemon.reconcile_remote_sources(1)["complete"] is False
    assert store.meta_value("last_successful_sync") == previous_sync

    assert daemon.flush_once() == 1
    assert remote.ingested == [pending]
    assert store.pending_count() == 0
    assert daemon.reconcile_remote_sources(1)["complete"] is True


def test_legacy_presence_ack_keeps_revision_and_checkpoint(pending_revision):
    _config, store, pending, previous_sync = pending_revision
    previous_changes = store.db.total_changes

    assert store.ack_session_records([pending["record_id"]]) == 0

    assert store.pending_count() == 1
    assert store.get(pending["record_id"]) == pending
    assert store.meta_value("last_successful_sync") == previous_sync
    assert store.db.total_changes == previous_changes


def test_failed_ingest_keeps_present_revision_pending(pending_revision):
    class FailingRemote(PresentRemote):
        def ingest(self, documents):
            raise RuntimeError("offline test")

    config, store, pending, previous_sync = pending_revision
    daemon = SyncDaemon(config, store, FailingRemote())
    daemon.reconcile_remote_sources(1)

    assert daemon.flush_once() == 0

    assert store.pending_count() == 1
    assert store.get(pending["record_id"]) == pending
    assert store.meta_value("last_successful_sync") == previous_sync
    assert daemon.reconcile_remote_sources(1)["complete"] is False


def test_presence_does_not_requeue_an_acknowledged_record(session_store):
    config, store, _chunk = session_store
    remote = PresentRemote()
    daemon = SyncDaemon(config, store, remote)

    assert daemon.reconcile_remote_sources()["complete"] is True

    assert store.pending_count() == 0
    assert daemon.flush_once() == 0
    assert remote.ingested == []
