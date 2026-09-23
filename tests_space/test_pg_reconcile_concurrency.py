"""Tests for PostgreSQL reconcile concurrency, generation race, batch hydration, and readiness."""
from __future__ import annotations

import json
import os
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest

import space.server as bridge
from space.server import (
    WRITE_LOCK,
    _native_update,
    reconcile_canonical_index,
    source_readiness_state,
    ingest_source_documents,
)


class MockConcurrencyStore:
    """Store test double supporting get, get_many, and generation tracking."""

    def __init__(self, records: dict[str, dict] | None = None):
        self.records = {k: dict(v) for k, v in (records or {}).items()}
        self.get_calls: list[str] = []
        self.get_many_calls: list[list[str]] = []
        self.fail_get_many = False
        self.updates: list[dict] = []
        self.marker = None

    def canonical_index_candidates(self, limit, profile, memory):
        return [dict(r) for r in self.records.values()][:limit]

    def get(self, identity: str) -> dict | None:
        self.get_calls.append(identity)
        item = self.records.get(identity)
        return dict(item) if item else None

    def get_many(self, identities: list[str]) -> list[dict]:
        self.get_many_calls.append(list(identities))
        if self.fail_get_many:
            raise RuntimeError("PostgreSQL batch read connection failed")
        results = []
        for identity in identities:
            item = self.records.get(identity)
            if item is not None:
                results.append(dict(item))
        return results

    def native_optimize_checkpoint(self):
        return {
            "status": None,
            "fingerprint": None,
            "memory": None,
            "revision": 0,
        }

    def update_native_index(self, updates):
        self.updates.extend(updates)

    def set_native_optimize_checkpoint(self, marker):
        self.marker = marker

    def native_index_state_record(self, updates):
        return {
            "_funes_record": "native_index_state",
            "state_version": 1,
            "profile": updates[0]["native_index_profile"],
            "memory": updates[0]["native_index_memory"],
            "revision": 1,
            "eligible": len(updates),
            "indexed": sum(1 for u in updates if u.get("native_index_status") == "indexed"),
            "held": 0,
            "invalid": 0,
            "index_fingerprint": "projected",
        }


class MockSyncer:
    def __init__(self, durable=True):
        self.backend = "postgres"
        self.restoring = False
        self.restore_failed = False
        self.restored = True
        self.durable = durable
        self.uploads = []
        self._progress = {"completed": 0, "total": 100}
        self.probe_ready_called = 0
        self.probe_return = True

    @property
    def progress(self):
        return self._progress

    def upload(self, docs=None):
        self.uploads.append(docs)
        return {"uploaded": self.durable, "durable": self.durable}

    def probe_ready(self) -> bool:
        self.probe_ready_called += 1
        return self.probe_return


def _setup_app_and_profile(monkeypatch, store, syncer):
    app = SimpleNamespace(store=store, syncer=syncer, restore_result=0)
    monkeypatch.setattr(bridge, "source_app", lambda: app)
    monkeypatch.setattr(bridge, "REMOTE", "owner/active-memory")
    monkeypatch.setattr(bridge, "INDEX_REMOTE", "")
    monkeypatch.setenv("FUNES_POSTGRES_DSN", "postgresql://user:pass@host:5432/db")
    monkeypatch.setattr(
        bridge,
        "run",
        lambda *args, **kwargs: (
            0,
            f"ingested sources={len(store.records)} chunks=1 unchanged=0 stale=0 held=0 commit=rev\n",
            "",
        ),
    )
    monkeypatch.setattr(bridge, "request_warm", lambda **_kwargs: None)
    return app


def test_native_update_records_both_generations():
    item = {
        "source_identity": "doc-1",
        "source_version": "v1",
        "content_hash": "hash-1",
        "native_generation": 4,
        "retrieval_generation": 7,
    }
    update = _native_update(item, "indexed", "v1")
    assert update["native_generation"] == 4
    assert update["retrieval_generation"] == 7
    assert isinstance(update["native_generation"], int)
    assert isinstance(update["retrieval_generation"], int)

    # Defaults to 0 when missing or None
    item_default = {
        "source_identity": "doc-2",
        "content_hash": "hash-2",
    }
    update_default = _native_update(item_default, "indexed", "v1")
    assert update_default["native_generation"] == 0
    assert update_default["retrieval_generation"] == 0


def test_reconcile_generation_race_native_generation_mismatch_skipped(monkeypatch):
    row = {
        "source_identity": "race-doc",
        "source_version": "v1",
        "content_hash": "hash-1",
        "raw_text": "Sample raw content",
        "retrieval_text": "Sample retrieval content",
        "source_type": "doc",
        "native_generation": 1,
        "retrieval_generation": 1,
        "native_index_status": "pending",
    }
    store = MockConcurrencyStore({"race-doc": row})
    syncer = MockSyncer()
    app = _setup_app_and_profile(monkeypatch, store, syncer)

    # Simulate generation bump between candidate selection and status validation
    orig_candidates = store.canonical_index_candidates
    def bump_and_return(*args, **kwargs):
        res = orig_candidates(*args, **kwargs)
        # Store is concurrently bumped by reindex control
        store.records["race-doc"]["native_generation"] = 2
        return res
    store.canonical_index_candidates = bump_and_return

    result = reconcile_canonical_index(app)

    # Stale candidate (native_generation=1) should be skipped during status hydration
    assert result["indexed"] == 0
    assert store.updates == []
    assert len(store.get_many_calls) == 1
    assert store.get_calls == []


def test_reconcile_generation_race_retrieval_generation_mismatch_skipped(monkeypatch):
    row = {
        "source_identity": "race-retrieval",
        "source_version": "v1",
        "content_hash": "hash-1",
        "raw_text": "Sample raw content",
        "retrieval_text": "Sample retrieval content",
        "source_type": "doc",
        "native_generation": 1,
        "retrieval_generation": 1,
        "native_index_status": "pending",
    }
    store = MockConcurrencyStore({"race-retrieval": row})
    syncer = MockSyncer()
    app = _setup_app_and_profile(monkeypatch, store, syncer)

    orig_candidates = store.canonical_index_candidates
    def bump_and_return(*args, **kwargs):
        res = orig_candidates(*args, **kwargs)
        store.records["race-retrieval"]["retrieval_generation"] = 2
        return res
    store.canonical_index_candidates = bump_and_return

    result = reconcile_canonical_index(app)

    assert result["indexed"] == 0
    assert store.updates == []
    assert len(store.get_many_calls) == 1
    assert store.get_calls == []


def test_reconcile_generations_match_accepted(monkeypatch):
    row = {
        "source_identity": "matching-doc",
        "source_version": "v1",
        "content_hash": "hash-1",
        "raw_text": "Sample raw content",
        "retrieval_text": "Sample retrieval content",
        "source_type": "doc",
        "native_generation": 3,
        "retrieval_generation": 5,
        "native_index_status": "pending",
    }
    store = MockConcurrencyStore({"matching-doc": row})
    syncer = MockSyncer()
    app = _setup_app_and_profile(monkeypatch, store, syncer)

    result = reconcile_canonical_index(app)

    assert result["indexed"] == 1
    assert len(syncer.uploads) > 0
    assert len(store.get_many_calls) == 1
    assert store.get_calls == []


def test_batch_read_failure_fail_closed_no_get_storm(monkeypatch):
    row = {
        "source_identity": "doc-fail",
        "source_version": "v1",
        "content_hash": "hash-1",
        "raw_text": "Sample raw content",
        "retrieval_text": "Sample retrieval content",
        "source_type": "doc",
        "native_generation": 1,
        "retrieval_generation": 1,
        "native_index_status": "pending",
    }
    store = MockConcurrencyStore({"doc-fail": row})
    store.fail_get_many = True
    syncer = MockSyncer()
    app = _setup_app_and_profile(monkeypatch, store, syncer)

    with pytest.raises(RuntimeError, match="PostgreSQL batch read connection failed"):
        reconcile_canonical_index(app)

    # Must NEVER fall back to point get storm!
    assert store.get_calls == []
    assert len(store.get_many_calls) == 1


def test_batch_read_partial_missing_does_not_fall_back_to_get(monkeypatch):
    row1 = {
        "source_identity": "doc-present",
        "source_version": "v1",
        "content_hash": "hash-1",
        "raw_text": "Sample raw content 1",
        "retrieval_text": "Sample retrieval content 1",
        "source_type": "doc",
        "native_generation": 1,
        "retrieval_generation": 1,
        "native_index_status": "pending",
    }
    row2 = {
        "source_identity": "doc-missing",
        "source_version": "v1",
        "content_hash": "hash-2",
        "raw_text": "Sample raw content 2",
        "retrieval_text": "Sample retrieval content 2",
        "source_type": "doc",
        "native_generation": 1,
        "retrieval_generation": 1,
        "native_index_status": "pending",
    }
    store = MockConcurrencyStore({"doc-present": row1, "doc-missing": row2})
    syncer = MockSyncer()
    app = _setup_app_and_profile(monkeypatch, store, syncer)

    # get_many only returns doc-present
    orig_get_many = store.get_many
    def partial_get_many(ids):
        res = orig_get_many(ids)
        return [r for r in res if r["source_identity"] == "doc-present"]
    store.get_many = partial_get_many

    reconcile_canonical_index(app)

    # doc-present is updated in store, doc-missing skipped without calling get()
    assert [u["source_identity"] for u in store.updates] == ["doc-present"]
    assert store.get_calls == []


def test_readiness_probe_returns_bool_and_does_not_block_under_write_lock(monkeypatch):
    store = MockConcurrencyStore()
    syncer = MockSyncer()
    app = _setup_app_and_profile(monkeypatch, store, syncer)

    finished = threading.Event()
    state_box = []

    def run_probe():
        state_box.append(source_readiness_state())
        finished.set()

    with WRITE_LOCK:
        thread = threading.Thread(target=run_probe)
        thread.start()
        completed = finished.wait(0.5)

    thread.join(timeout=2)
    assert not thread.is_alive()
    assert completed, "source_readiness_state blocked waiting on WRITE_LOCK"

    assert len(state_box) == 1
    state = state_box[0]
    assert state["ready"] is True
    assert isinstance(state["ready"], bool)
    assert "error" not in state
    assert syncer.probe_ready_called == 1


def test_readiness_probe_bool_false_marks_unready(monkeypatch):
    store = MockConcurrencyStore()
    syncer = MockSyncer()
    syncer.probe_return = False
    app = _setup_app_and_profile(monkeypatch, store, syncer)

    state = source_readiness_state()
    assert state["ready"] is False
    assert isinstance(state["ready"], bool)
    assert state["error"] == "postgres_unavailable"
    assert syncer.probe_ready_called == 1


def test_readiness_progress_safe_snapshot_isolation(monkeypatch):
    store = MockConcurrencyStore()
    syncer = MockSyncer()
    syncer.restoring = True
    syncer._progress = {"percent": 42, "status": "restoring"}
    app = _setup_app_and_profile(monkeypatch, store, syncer)

    state = source_readiness_state()
    assert state["restoring"] is True
    assert state["progress"] == {"percent": 42, "status": "restoring"}

    # Concurrently mutating syncer progress does not mutate the returned snapshot
    syncer._progress["percent"] = 99
    syncer._progress["mutated"] = True
    assert state["progress"] == {"percent": 42, "status": "restoring"}


def test_ingest_source_documents_holds_write_lock(monkeypatch):
    store = MockConcurrencyStore()
    syncer = MockSyncer()
    app = _setup_app_and_profile(monkeypatch, store, syncer)

    persisted = threading.Event()
    def fake_persist(app, docs):
        persisted.set()
        return {"durable": True, "items": [{"source_identity": "doc-ingest"}]}
    monkeypatch.setattr(bridge, "persist_source_ingest", fake_persist)

    acquired_during_ingest = []
    # Hold WRITE_LOCK, then attempt ingest in thread; it must not finish persist until lock released
    with WRITE_LOCK:
        thread = threading.Thread(
            target=lambda: ingest_source_documents([{"source_identity": "doc-ingest"}])
        )
        thread.start()
        blocked = not persisted.wait(0.2)
        acquired_during_ingest.append(blocked)

    # After lock released, persist should complete
    assert acquired_during_ingest == [True], "ingest_source_documents bypassed WRITE_LOCK"
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert persisted.is_set()

