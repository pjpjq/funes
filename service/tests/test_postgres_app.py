"""Bounded source-adapter tests: no live service, Hub, or database required."""
from __future__ import annotations

import copy
import json
import os
import sys
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from service import server
from service.postgres_sync import PostgresSync
from space import server as bridge


class AtomicRecords:
    """A transaction double that exposes the actual outer commit boundary."""

    def __init__(self, store):
        self.store = store
        self.depth = 0

    def __enter__(self):
        if self.depth == 0:
            self.before = copy.deepcopy(self.store.records)
        self.depth += 1
        self.store.events.append("begin")
        return self

    def __exit__(self, kind, value, traceback):
        self.depth -= 1
        if self.depth:
            return False
        if kind is not None or self.store.fail_commit:
            self.store.records = self.before
            self.store.events.append("rollback")
            if kind is None:
                raise RuntimeError("postgres://private:password@host raw_text api-key")
        else:
            self.store.events.append("commit")
            self.store.commits += 1
        return False

    def execute(self, statement, parameters=None):
        if statement == "SET LOCAL synchronous_commit=on":
            self.store.events.append("synchronous_commit")
            return SimpleNamespace(close=lambda: None)
        assert statement == "SELECT scope FROM reindex_controls WHERE generation=?"
        row = self.store.records.get(("reindex_control", parameters[0]))
        return SimpleNamespace(fetchone=lambda: row)


class SourceStoreDouble:
    def __init__(self, *args, **kwargs):
        self.events = []
        self.records = {}
        self.commits = 0
        self.fail_commit = False
        self.available = True
        self.lock = threading.RLock()
        self.conn = AtomicRecords(self)

    def verify_schema(self):
        self.events.append("verify_schema")
        if not self.available:
            raise RuntimeError("postgres://private:password@host raw_text api-key")

    def reconnect(self):
        self.events.append("reconnect")
        self.verify_schema()

    def _put(self, record):
        with self.conn:
            kind = record.get("_funes_record", "memory")
            key = (kind, record.get("source_identity") or record.get("generation")
                   or record.get("query") or "state")
            self.records[key] = dict(record)
            self.events.append("persist:" + kind)

    def ingest(self, records):
        results = []
        for record in records:
            status = "updated" if ("memory", record.get("source_identity")) in self.records else "created"
            self._put(record)
            results.append({"source_identity": record["source_identity"], "status": status})
        created = sum(1 for r in results if r["status"] == "created")
        updated = sum(1 for r in results if r["status"] == "updated")
        return {"created": created, "updated": updated, "deduped": 0, "items": results}

    def get(self, identity):
        value = self.records.get(("memory", identity))
        return dict(value) if value else None

    def get_many(self, identities):
        return [self.get(identity) for identity in identities if self.get(identity)]

    def translation_put(self, query, rewritten, **values):
        self._put({"_funes_record": "translation_cache", "query": query,
                   "rewritten": rewritten, **values})

    def record_reindex_control(self, record):
        if ("reindex_control", record["generation"]) in self.records:
            return False
        self._put({**record, "_funes_record": "reindex_control"})
        return True

    def native_index_state_record(self):
        return {"_funes_record": "native_index_state", "state_version": 2,
                "profile": "profile-1", "memory": "hf://memory", "revision": 3,
                "eligible": 1, "indexed": 1, "held": 0, "invalid": 0,
                "index_fingerprint": "index-3"}

    def set_native_index_state(self, record):
        self._put({**record, "_funes_record": "native_index_state"})

    def set_native_optimize_checkpoint(self, record):
        self._put({**record, "_funes_record": "native_optimize_checkpoint"})

    def set_sync(self, **values):
        with self.conn:
            self.records[("sync", "state")] = values
            self.events.append("sync_state")

    def update_native_index(self, updates):
        self.events.append("update_native_index")

    def mark_translations_pending(self, records):
        self.events.append("mark_translations_pending")

    def close(self):
        self.events.append("close")


def test_restore_only_probes_migrated_source_database():
    store = SourceStoreDouble()
    syncer = PostgresSync(store)
    with mock.patch.object(server.SnapshotSync, "restore", side_effect=AssertionError("Hub replay")):
        assert syncer.restore() == 0
    assert store.events == ["verify_schema"]
    assert syncer.restored and not syncer.restore_failed
    assert syncer.progress["rows"] == 0
    assert syncer.progress["phase"] == "complete"


def test_upload_commits_all_record_types_before_durable_ack():
    store = SourceStoreDouble()
    syncer = PostgresSync(store)
    records = [
        {"source_identity": "source-1", "raw_text": "raw\x00source", "source_version": "v1"},
        {"_funes_record": "translation_cache", "query": "query", "rewritten": "rewritten"},
        store.native_index_state_record(),
        {"_funes_record": "native_optimize_checkpoint", "revision": 2,
         "status": "optimized", "fingerprint": "profile-1", "memory": "hf://memory",
         "index_fingerprint": "index-3"},
        {"_funes_record": "reindex_control", "generation": 1, "scope": "all"},
    ]
    original = copy.deepcopy(records)
    result = syncer.upload(records)
    assert result["durable"] is True
    assert store.events[-1] == "commit"
    assert store.commits == 1
    assert len(store.records) == len(records) + 1
    assert records == original
    assert store.records[("memory", "source-1")]["raw_text"] == "raw\x00source"
    assert syncer.upload(records)["durable"] is True
    assert len(store.records) == len(records) + 1
    assert store.commits == 2


def test_commit_failure_rolls_back_and_returns_sanitized_negative_ack():
    store = SourceStoreDouble()
    syncer = PostgresSync(store)
    store.fail_commit = True
    result = syncer.upload([{"source_identity": "source-1", "raw_text": "private source"}])
    assert result == {"uploaded": False, "durable": False, "reason": "postgres_unavailable"}
    assert store.records == {}
    assert store.commits == 0
    assert store.events[-1] == "rollback"
    assert syncer.restore_failed
    public = json.dumps({"result": result, "progress": syncer.progress})
    assert not any(value in public for value in ("password", "api-key", "raw_text", "private"))


def test_sync_without_documents_commits_without_export_scan_or_embedding():
    store = SourceStoreDouble()
    syncer = PostgresSync(store)
    assert syncer.upload()["durable"] is True
    assert store.commits == 1
    assert set(store.records) == {("sync", "state")}
    assert not any(event.startswith("persist:") for event in store.events)


def test_outage_recovery_reuses_app_without_restart():
    store = SourceStoreDouble()
    syncer = PostgresSync(store)
    store.available = False
    assert syncer.restore() == -1
    assert syncer.restore_failed
    store.available = True
    assert syncer.check_ready()
    assert not syncer.restore_failed
    assert syncer.restore_error is None
    assert syncer.upload([{"source_identity": "retry", "raw_text": "retry"}])["durable"]
    assert "reconnect" in store.events


def test_reindex_control_is_committed_before_local_queue_ack():
    store = SourceStoreDouble()
    syncer = PostgresSync(store)
    control = {"_funes_record": "reindex_control", "generation": 3, "scope": "all"}
    store.next_reindex_control = mock.Mock(return_value=control)
    persist = store.record_reindex_control

    def record(value):
        store.events.append("record_control")
        return persist(value)

    store.record_reindex_control = mock.Mock(side_effect=record)
    store.compact_reindex_controls = mock.Mock()
    result = server.queue_reindex(SimpleNamespace(store=store, syncer=syncer), "all")
    assert result["durable"]
    assert store.events.index("commit") < len(store.events) - 1
    assert store.events[-1] == "record_control"
    assert store.records[("reindex_control", 3)] == control


def test_app_selects_pg_without_hub_restore_and_preserves_native_targets(monkeypatch, tmp_path):
    monkeypatch.setenv("FUNES_POSTGRES_DSN", "postgresql://unit-test/not-a-live-database")
    monkeypatch.setenv("FUNES_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FUNES_MEMORY", "hf://native-active")
    monkeypatch.setenv("FUNES_INDEX_MEMORY", "hf://native-build")
    monkeypatch.setenv("FUNES_EMBEDDING_MODEL", "active-model")
    monkeypatch.setenv("FUNES_INDEX_EMBEDDING_MODEL", "build-model")
    monkeypatch.setenv("FUNES_LAZY_RESTORE", "false")
    pg_store = mock.Mock(side_effect=SourceStoreDouble)
    monkeypatch.setitem(sys.modules, "service.postgres", SimpleNamespace(PostgresStore=pg_store))
    with mock.patch.object(server, "Store", side_effect=AssertionError("SQLite fallback")), \
         mock.patch.object(server.SnapshotSync, "restore", side_effect=AssertionError("Hub replay")), \
         mock.patch.object(server.App, "_reconcile_background"), \
         mock.patch.object(server.App, "_reindex_background"):
        app = server.App()
        try:
            pg_store.assert_called_once_with(str(tmp_path))
            assert isinstance(app.syncer, PostgresSync)
            assert app.restore_result == 0
            assert not list(tmp_path.glob("*.sqlite3"))
            assert os.environ["FUNES_MEMORY"] == "hf://native-active"
            assert os.environ["FUNES_INDEX_MEMORY"] == "hf://native-build"
            assert os.environ["FUNES_EMBEDDING_MODEL"] == "active-model"
            assert os.environ["FUNES_INDEX_EMBEDDING_MODEL"] == "build-model"
        finally:
            app.close()


def test_reindex_worker_does_not_scan_after_restore_failure():
    app = server.App.__new__(server.App)
    app.restore_done = threading.Event()
    app.restore_done.set()
    app.reindex_stop = threading.Event()
    app.reindex_wake = mock.Mock()
    app.reindex_wake.wait.side_effect = lambda _: app.reindex_stop.set()
    app.reconcile_interval = 0.01
    app.syncer = SimpleNamespace(restoring=False, restore_failed=True)
    app.store = mock.Mock()
    app._reindex_background()
    app.store.apply_pending_reindex_controls.assert_not_called()
    app.store.compact_reindex_controls.assert_not_called()


def test_space_pg_only_config_fails_closed_and_recovers(monkeypatch):
    monkeypatch.delenv("FUNES_STORAGE_REPO", raising=False)
    monkeypatch.setenv("FUNES_POSTGRES_DSN", "postgresql://unit-test/not-a-live-database")
    for name in ("FUNES_DATA_DIR", "FUNES_LAZY_RESTORE", "FUNES_REQUIRE_DURABLE_ACK",
                 "FUNES_BULK_RESTORE_REBUILD_FTS"):
        monkeypatch.setenv(name, "true" if name != "FUNES_DATA_DIR" else "/unused-test-path")
    monkeypatch.setattr(bridge, "SOURCE_APP", None)
    store = SourceStoreDouble()
    app = SimpleNamespace(store=store, syncer=PostgresSync(store), restore_result=0)
    create = mock.Mock(side_effect=[RuntimeError("postgres://private:password@host"), app])
    monkeypatch.setattr(bridge, "SourceApp", create)
    start = mock.Mock()
    monkeypatch.setattr(bridge, "start_canonical_reconciler", start)
    state = bridge.source_readiness_state()
    assert state == {"configured": True, "ready": False, "restoring": False,
                     "error": "postgres_unavailable"}
    start.assert_not_called()
    assert bridge.source_readiness_state()["ready"] is True
    start.assert_called_once_with(app)
    store.available = False
    state = bridge.source_readiness_state()
    assert state["ready"] is False and state["error"] == "postgres_unavailable"
    store.available = True
    assert bridge.source_readiness_state()["ready"] is True
    assert create.call_count == 2


def test_pg_connection_failure_cannot_fall_back_to_native_ingest(monkeypatch):
    monkeypatch.setenv("FUNES_POSTGRES_DSN", "postgresql://unit-test/not-a-live-database")
    monkeypatch.setattr(bridge, "source_app", lambda: None)
    code, payload, records = bridge.ingest_source_documents([{"raw_text": "private"}])
    assert code == 503
    assert payload == {"ok": False, "durable": False, "error": "postgres_unavailable"}
    assert records == []


@pytest.mark.parametrize("bad_record", [
    {"_funes_record": "unknown"},
    {"_funes_record": "native_index_state", "state_version": 2, "revision": 500},
    {"_funes_record": "native_optimize_checkpoint", "status": "optimized",
     "fingerprint": "wrong-profile", "memory": "hf://memory", "index_fingerprint": "index-3"},
])
def test_invalid_record_or_marker_rolls_back_entire_batch(bad_record):
    store = SourceStoreDouble()
    syncer = PostgresSync(store)
    result = syncer.upload([{"source_identity": "before-bad-marker", "raw_text": "raw"}, bad_record])
    assert result["durable"] is False
    assert store.records == {}
    assert store.commits == 0


def test_conflicting_reindex_control_is_not_acknowledged():
    store = SourceStoreDouble()
    syncer = PostgresSync(store)
    assert syncer.upload_reindex_control({"generation": 1, "scope": "all"})["durable"]
    assert not syncer.upload_reindex_control({"generation": 1, "scope": "retrieval_text"})["durable"]
    assert store.records[("reindex_control", 1)]["scope"] == "all"


def test_regular_ingest_ack_follows_source_and_upload_commits():
    store = SourceStoreDouble()
    app = SimpleNamespace(store=store, syncer=PostgresSync(store))
    documents = [{"source_identity": "regular-ingest", "source_version": "v1", "raw_text": "original\x00text"}]
    with mock.patch.object(server, "prepare_ingest_documents", return_value=documents):
        result = server.ingest_documents(app, documents)
    assert result["durable"]
    assert result["accepted"] == 1
    assert store.commits == 2
    assert store.events[-1] == "commit"
    assert store.get("regular-ingest")["raw_text"] == "original\x00text"


def test_service_ready_checks_database_without_counting_sources():
    store = SourceStoreDouble()
    store.count = mock.Mock(side_effect=AssertionError("readiness full scan"))
    app = SimpleNamespace(store=store, syncer=PostgresSync(store), restore_result=0)
    handler = object.__new__(server.make_handler(app))
    handler.path = "/ready"
    handler._json = lambda status, body: (status, body)
    code, payload = handler.do_GET()
    assert code == 200 and payload["status"] == "ready"
    assert payload["documents"] is None
    store.count.assert_not_called()
    store.available = False
    assert handler.do_GET() == (503, {"status": "not_ready", "error": "postgres_unavailable"})
    store.available = True
    assert handler.do_GET()[0] == 200


@pytest.mark.parametrize("path, method", [
    ("/sources", "sources"),
    ("/sync/status", "sync_status"),
    ("/get?id=source-1", "get"),
])
def test_service_get_disconnect_after_precheck_is_sanitized(path, method):
    store = SourceStoreDouble()
    setattr(store, method, mock.Mock(side_effect=RuntimeError("postgres://private:password@host")))
    app = SimpleNamespace(store=store, syncer=PostgresSync(store), restore_result=0)
    handler = object.__new__(server.make_handler(app))
    handler.path = path
    handler._authorized = lambda: True
    handler._json = lambda status, body: (status, body)
    assert handler.do_GET() == (503, {"error": "postgres_unavailable", "durable": False})
    assert "verify_schema" in store.events


def test_space_source_state_disconnect_after_precheck_is_sanitized(monkeypatch):
    store = SourceStoreDouble()
    store.native_index_checkpoint = mock.Mock(
        side_effect=RuntimeError("postgres://private:password@host")
    )
    app = SimpleNamespace(store=store, syncer=PostgresSync(store), restore_result=0)
    monkeypatch.setattr(bridge, "SOURCE_APP", app)
    assert bridge.source_state() == {
        "configured": True, "ready": False, "restoring": False,
        "error": "postgres_unavailable",
    }
    assert "verify_schema" in store.events


def test_space_sync_cannot_ack_native_when_pg_initialization_fails(monkeypatch):
    monkeypatch.setenv("FUNES_POSTGRES_DSN", "postgresql://unit-test/not-a-live-database")
    monkeypatch.setattr(bridge, "source_app", lambda: None)
    monkeypatch.setattr(bridge, "REMOTE", "hf://native-memory")
    monkeypatch.setattr(bridge, "auth_ok", lambda _: True)
    handler = object.__new__(bridge.Handler)
    handler.path = "/sync"
    handler.body = lambda: {}
    handler.send_json = mock.Mock()
    handler.do_POST()
    handler.send_json.assert_called_once_with(
        503, {"ok": False, "durable": False, "error": "postgres_unavailable"}
    )


def test_space_post_disconnect_is_sanitized(monkeypatch):
    monkeypatch.setenv("FUNES_POSTGRES_DSN", "postgresql://unit-test/not-a-live-database")
    monkeypatch.setattr(bridge, "source_app", mock.Mock(
        side_effect=RuntimeError("postgres://private:password@host")
    ))
    monkeypatch.setattr(bridge, "auth_ok", lambda _: True)
    handler = object.__new__(bridge.Handler)
    handler.path = "/sync"
    handler.body = lambda: {}
    handler.send_json = mock.Mock()
    handler.do_POST()
    handler.send_json.assert_called_once_with(
        503, {"ok": False, "durable": False, "error": "postgres_unavailable"}
    )


def test_ingest_documents_fastpath_avoids_reingesting_rows():
    store = SourceStoreDouble()
    syncer = PostgresSync(store)
    app = SimpleNamespace(store=store, syncer=syncer)
    docs = [
        {"source_identity": "doc-1", "source_version": "v1", "raw_text": "hello"},
        {"source_identity": "doc-2", "source_version": "v1", "raw_text": "world"},
    ]
    with mock.patch.object(server, "prepare_ingest_documents", return_value=docs), \
         mock.patch.object(syncer, "_restore_record", side_effect=AssertionError("re-ingest triggered")):
        result = server.ingest_documents(app, docs)
    assert result["durable"] is True
    assert result["accepted"] == 2
    assert result["sync"]["records"] == 2
    assert result["sync"]["backend"] == "postgres"
    # Rows ingested once, not re-ingested by syncer
    assert store.events.count("persist:memory") == 2
    assert "synchronous_commit" in store.events
    assert "sync_state" in store.events
    assert store.commits == len(docs) + 1


def test_persist_translation_documents_fastpath_avoids_reingesting_rows():
    store = SourceStoreDouble()
    syncer = PostgresSync(store)
    translator = SimpleNamespace(normalize_many=lambda texts: [("norm:" + t, "hash", "v1", "ok") for t in texts])
    app = SimpleNamespace(store=store, syncer=syncer, translator=translator)
    store.ingest([{
        "source_identity": "trans-1", "source_version": "v1", "content_hash": "h1",
        "retrieval_generation": 0, "translation_status": "pending_provider",
        "native_index_status": None, "raw_text": "hello translation",
    }])
    docs = store.get_many(["trans-1"])
    with mock.patch.object(syncer, "_restore_record", side_effect=AssertionError("re-ingest triggered")):
        result = server._persist_translation_documents(app, docs)
    assert result["durable"] is True
    assert result["updated"] == 1
    assert "sync_state" in store.events


def test_ack_committed_persists_sync_state_under_synchronous_commit():
    store = SourceStoreDouble()
    syncer = PostgresSync(store)
    docs = [{"source_identity": "doc-1"}, {"source_identity": "doc-2"}]
    with mock.patch.object(syncer, "_restore_record", side_effect=AssertionError("re-ingest")):
        result = syncer.ack_committed(docs)
    assert result == {"uploaded": True, "durable": True, "backend": "postgres", "records": 2}
    assert "synchronous_commit" in store.events
    assert "sync_state" in store.events
    assert store.commits == 1
    assert syncer.ack_persisted == syncer.ack_committed


def test_ack_committed_fails_closed_when_unavailable():
    store = SourceStoreDouble()
    syncer = PostgresSync(store)
    store.available = False
    result = syncer.ack_committed([{"source_identity": "doc-1"}])
    assert result == {"uploaded": False, "durable": False, "reason": "postgres_unavailable"}
    assert syncer.restore_failed is True

    store.available = True
    syncer.check_ready()
    store.fail_commit = True
    result = syncer.ack_committed([{"source_identity": "doc-1"}])
    assert result == {"uploaded": False, "durable": False, "reason": "postgres_unavailable"}
    assert syncer.restore_failed is True
    assert store.events[-1] == "rollback"


def test_ingest_documents_fails_closed_when_ack_committed_unavailable():
    store = SourceStoreDouble()
    syncer = PostgresSync(store)
    app = SimpleNamespace(store=store, syncer=syncer)
    docs = [{"source_identity": "fail-1", "source_version": "v1", "content_hash": "hash-fail-1", "raw_text": "text"}]
    with mock.patch.object(server, "prepare_ingest_documents", return_value=docs), \
         mock.patch.object(syncer, "ack_committed", return_value={"uploaded": False, "durable": False, "reason": "postgres_unavailable"}):
        result = server.ingest_documents(app, docs)
    assert result["durable"] is False
    assert result["sync"]["reason"] == "postgres_unavailable"
    assert "update_native_index" in store.events


def test_sync_committed_documents_falls_back_to_upload():
    store = SourceStoreDouble()
    docs = [{"source_identity": "doc-fallback"}]

    class FallbackSyncer:
        def __init__(self, store):
            self.store = store
            self.uploaded = []
        def upload(self, canonical):
            self.uploaded.append(canonical)
            return {"uploaded": True, "durable": True, "records": len(canonical)}

    syncer1 = FallbackSyncer(store)
    app1 = SimpleNamespace(store=store, syncer=syncer1)
    res1 = server._sync_committed_documents(app1, docs)
    assert res1["durable"] is True
    assert syncer1.uploaded == [docs]

    different_store = SourceStoreDouble()
    syncer2 = PostgresSync(different_store)
    syncer2.upload = mock.Mock(return_value={"uploaded": True, "durable": True, "records": 1})
    app2 = SimpleNamespace(store=store, syncer=syncer2)
    res2 = server._sync_committed_documents(app2, docs)
    assert res2["durable"] is True
    syncer2.upload.assert_called_once_with(docs)
