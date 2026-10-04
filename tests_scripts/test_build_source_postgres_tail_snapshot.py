from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import stat

import pytest

from scripts.build_source_postgres_tail_snapshot import (
    TailSnapshotError, build_complete_plan, build_tail_snapshot, json_sha256, main,
    prepare_complete_plan,
)
from scripts.plan_source_postgres_tail import build_plan
from service.server import SnapshotSync, Store


NOW = "2026-09-22T12:00:00+00:00"


def doc(identity, text, **extra):
    return {"source_identity": identity, "source_version": "v1", "raw_text": text,
            "retrieval_text": text, "source_agent": "codex", "source_type": "memory",
            "content_type": "decision", "updated_at": NOW, "ingested_at": NOW,
            "metadata": {}, "embedding_generation": 0, **extra}


def make_fixture(tmp_path, monkeypatch, *, initial=None, tail=None, setup=None):
    monkeypatch.setenv("FUNES_STORAGE_KEY", "offline-snapshot-builder-test-only")
    monkeypatch.setenv("FUNES_STORAGE_REPO", "owner/source")
    monkeypatch.setenv("FUNES_BULK_RESTORE_REBUILD_FTS", "false")
    monkeypatch.setattr("service.server.utc_now", lambda: NOW)
    baseline_store = Store(str(tmp_path / "baseline"))
    baseline_store.ingest(initial or [doc("original", "原文\0\ue0000")])
    if setup:
        setup(baseline_store)
    baseline_store.set_sync(last_error="old-error", last_sync=NOW)
    baseline_store.close()
    baseline = tmp_path / "baseline" / "funes.sqlite3"
    baseline_digest = hashlib.sha256(baseline.read_bytes()).hexdigest()
    base = {"version": 1, "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": ["funes-delta-baseline.jsonl.gz.enc"], "controls": []}
    files = [base["snapshot"], *base["deltas"]]
    blobs = {name: hashlib.sha1(name.encode()).hexdigest() for name in files}
    metadata = {"sqlite_sha256": baseline_digest, "source_receipt": {
        "version": 1, "repo": "owner/source", "manifest": base,
        "revision": "a" * 40, "files": files, "blob_ids": blobs}}
    manifest = deepcopy(base)
    cache = tmp_path / "cache"
    (cache / "remote").mkdir(parents=True)
    sync = SnapshotSync(None, rebuild_fts=False)
    downloads = {}
    tail = tail if tail is not None else [
        ("funes-delta-new.jsonl.gz.enc", [doc("new", "new\0raw text", id=98765)])]
    for index, (filename, records) in enumerate(tail):
        plaintext = tmp_path / "payload.jsonl.gz"
        with gzip.open(plaintext, "wt", encoding="utf-8") as stream:
            for item in records:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        encrypted = cache / "remote" / filename
        sync._encrypt_file(plaintext, encrypted)
        plaintext.unlink()
        downloads[filename] = {"size": encrypted.stat().st_size,
                               "sha256": hashlib.sha256(encrypted.read_bytes()).hexdigest()}
        group = "controls" if filename.startswith("funes-reindex-") else "deltas"
        manifest[group].append(filename)
        blobs[filename] = hashlib.sha1(str(index).encode()).hexdigest()
    active = [manifest["snapshot"], *manifest["deltas"], *manifest["controls"]]
    plan = build_plan(metadata, {"head": "b" * 40, "manifest": manifest,
                                "repo_files": active, "blob_ids": blobs}, "owner/source")
    plan["downloads"] = downloads
    return {"baseline": baseline, "metadata": metadata, "plan": plan, "cache": cache,
            "output_dir": tmp_path / "built", "sync": sync}


def read_tables(path):
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as conn:
        return {table: conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                for table in ("memories", "translation_cache", "reindex_controls", "sync_state")}


def test_builds_closed_snapshot_without_mutating_inputs_or_claiming_boundary(tmp_path, monkeypatch):
    args = make_fixture(tmp_path, monkeypatch)
    baseline_before = args["baseline"].read_bytes()
    encrypted = args["cache"] / "remote" / args["plan"]["tail_files"][0]
    encrypted_before = encrypted.read_bytes()
    events = []
    report = build_tail_snapshot(**args, progress=events.append, batch_size=1)
    target = args["output_dir"] / "funes.sqlite3"
    assert args["baseline"].read_bytes() == baseline_before
    assert encrypted.read_bytes() == encrypted_before
    assert report["counts"] == {"memories": 2, "translation_cache": 0,
                                "reindex_controls": 0, "sync_state": 1}
    assert report["records_by_type"] == {"memory": 1}
    assert report["sqlite_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert report["final_write_boundary"] is False
    assert report["fts_ready"] is report["fts_rebuilt"] is False
    assert report["status"] == "built_not_imported"
    assert json.loads((args["output_dir"] / "snapshot-report.json").read_text()) == report
    assert stat.S_IMODE(args["output_dir"].stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert sorted(path.name for path in args["output_dir"].iterdir()) == [
        "funes.sqlite3", "snapshot-report.json"]
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT id,source_identity,raw_text FROM memories ORDER BY id").fetchall() == [
            (1, "original", "原文\0\ue0000"), (2, "new", "new\0raw text")]
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert events[-1]["phase"] == "hashing_completed_snapshot"


def test_replay_matches_production_for_existing_rows_tombstones_cache_and_controls(tmp_path, monkeypatch):
    initial = [doc("changed", "old raw"), doc("tombstone", "source raw"),
               doc("already-control-applied", "stable raw")]

    def setup(store):
        store.record_reindex_control({"generation": 4, "scope": "all", "created_at": NOW})
        store.drain_reindex_controls(1)

    records = [
        doc("changed", "new raw\0\ue0000", id=1, source_version="v2",
            native_generation=0, retrieval_generation=0),
        doc("tombstone", "source raw", id=2, source_missing=True,
            retrieval_generation=4, native_generation=4, embedding_generation=4),
        doc("new-session", "session raw", source_type="codex_session", id=1234),
        {"_funes_record": "translation_cache", "query": "问题\0", "rewritten": "query\0rewrite",
         "created_at": "old-cache-timestamp", "translation_status": "ok"},
        {"_funes_record": "native_index_state", "state_version": 2, "revision": 100,
         "profile": "profile", "memory": "native-memory", "eligible": 999, "indexed": 999},
        {"_funes_record": "native_optimize_checkpoint", "revision": 100,
         "fingerprint": "profile", "memory": "native-memory", "status": "optimized",
         "optimized_at": NOW, "index_layout_version": 2},
        {"_funes_record": "native_optimize_checkpoint", "revision": 99,
         "fingerprint": "profile", "memory": "native-memory", "status": "pending",
         "optimized_at": NOW, "index_layout_version": 2},
    ]
    control = {"_funes_record": "reindex_control", "generation": 6,
               "scope": "retrieval_text", "created_at": NOW}
    tail = [("deltas/ab/funes-delta-test.jsonl.gz.enc", records),
            ("funes-reindex-six.jsonl.gz.enc", [control])]
    args = make_fixture(tmp_path, monkeypatch, initial=initial, tail=tail, setup=setup)

    # Independent oracle: the production SnapshotSync sequence, including its
    # single outer bulk scope and replay=True cursor rewind.
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    import shutil
    shutil.copyfile(args["baseline"], reference_dir / "funes.sqlite3")
    reference = Store(str(reference_dir))
    reference.begin_bulk_restore()
    for _, items in tail:
        reference.restore_documents(items, 2, apply_controls=False)
    reference.compact_reindex_controls(replay=True)
    reference.drain_reindex_controls(2)
    reference.finish_bulk_restore(rebuild_fts=False)
    reference.set_sync(last_error=None)
    reference.close()

    # Intentional offline-restore correction: production translation_put
    # regenerates created_at, while the builder retains the serialized clock.
    with sqlite3.connect(reference_dir / "funes.sqlite3") as conn:
        conn.execute("UPDATE translation_cache SET created_at=?", ("old-cache-timestamp",))
    conn.close()  # immutable readback below must not bypass an uncheckpointed WAL

    report = build_tail_snapshot(**args, batch_size=2)
    target = args["output_dir"] / "funes.sqlite3"
    assert read_tables(target) == read_tables(reference_dir / "funes.sqlite3")
    assert report["native_index_revision"] == 100
    assert report["native_optimize_revision"] == 100
    assert report["records_by_type"] == {"memory": 3, "translation_cache": 1,
                                        "native_index_state": 1, "native_optimize_checkpoint": 2,
                                        "reindex_control": 1}
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT source_missing FROM memories WHERE id=2").fetchone() == (1,)
        assert conn.execute("SELECT raw_text,embedding_generation FROM memories WHERE id=1").fetchone() == (
            "new raw\0\ue0000", 4)
        assert conn.execute("SELECT native_eligible_count FROM sync_state").fetchone() == (4,)
        assert conn.execute("SELECT created_at FROM translation_cache").fetchone() == ("old-cache-timestamp",)
        assert conn.execute("SELECT count(*) FROM reindex_controls WHERE applied_at IS NULL").fetchone() == (0,)


def test_only_one_native_rebuild_and_no_startup_identifier_or_fts_rebuild(tmp_path, monkeypatch):
    def setup(store):
        with store.conn:
            store.conn.execute("UPDATE memories SET search_identifiers='must not rebuild'")
            store.conn.execute("UPDATE sync_state SET fts_ready=1,fts_schema_version=0")
    args = make_fixture(tmp_path, monkeypatch, tail=[], setup=setup)
    original = Store._rebuild_native_checkpoint_state_locked
    calls = []

    def counted(self, *params, **kwargs):
        calls.append(1)
        return original(self, *params, **kwargs)

    def reject_startup(*_):
        raise AssertionError("must not run Store startup migrations")

    monkeypatch.setattr(Store, "_init_schema", reject_startup)
    monkeypatch.setattr(Store, "_rebuild_native_checkpoint_state_locked", counted)
    report = build_tail_snapshot(**args)
    assert calls == [1]
    with sqlite3.connect(args["output_dir"] / "funes.sqlite3") as conn:
        assert conn.execute("SELECT search_identifiers FROM memories").fetchone() == ("must not rebuild",)
        assert conn.execute("SELECT fts_ready,fts_schema_version FROM sync_state").fetchone() == (0, 0)
    assert report["startup_schema_migrations_skipped"] is True


def test_plaintext_and_bulk_settings_are_bounded_and_restored(tmp_path, monkeypatch):
    args = make_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("FUNES_BULK_RESTORE_REBUILD_FTS", "true")
    monkeypatch.setenv("FUNES_BULK_RESTORE_CACHE_SIZE", "-9999999")
    decrypt = args["sync"]._decrypt_file
    checked = []

    def check(source, target):
        import os
        assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
        assert target.is_relative_to(args["output_dir"])
        assert os.environ["FUNES_BULK_RESTORE_REBUILD_FTS"] == "false"
        assert os.environ["FUNES_BULK_RESTORE_CACHE_SIZE"] == "-16384"
        assert os.environ["FUNES_BULK_RESTORE_MMAP_SIZE"] == "0"
        checked.append(1)
        decrypt(source, target)

    monkeypatch.setattr(args["sync"], "_decrypt_file", check)
    build_tail_snapshot(**args)
    import os
    assert checked == [1]
    assert os.environ["FUNES_BULK_RESTORE_REBUILD_FTS"] == "true"
    assert os.environ["FUNES_BULK_RESTORE_CACHE_SIZE"] == "-9999999"


@pytest.mark.parametrize("change,code", [
    (lambda a: a["plan"].update(revision="main"), "invalid_source_identity"),
    (lambda a: a["plan"].update(repo="../source"), "invalid_repo_id"),
    (lambda a: a["plan"].update(final_write_boundary=True), "not_boundary_evidence"),
    (lambda a: a["plan"].update(tail_files=[]), "receipt_mismatch"),
    (lambda a: a["plan"].update(covered_files=99), "receipt_mismatch"),
    (lambda a: a["plan"]["manifest"].update(snapshot="funes-snapshot-replaced.jsonl.gz.enc"), "blob_identity"),
    (lambda a: a["plan"].update(downloads={}), "incomplete"),
])
def test_unpinned_or_modified_plan_is_rejected_before_copy(tmp_path, monkeypatch, change, code):
    args = make_fixture(tmp_path, monkeypatch)
    change(args)
    with pytest.raises(TailSnapshotError, match=code):
        build_tail_snapshot(**args)
    assert not args["output_dir"].exists()


@pytest.mark.parametrize("id", [0, -1, True, "1", 1 << 63])
def test_invalid_row_ids_are_rejected_and_owned_copy_is_cleaned(tmp_path, monkeypatch, id):
    args = make_fixture(tmp_path, monkeypatch, tail=[
        ("funes-delta-invalid.jsonl.gz.enc", [doc("new", "raw", id=id)])])
    before = args["baseline"].read_bytes()
    with pytest.raises(TailSnapshotError, match="invalid_memory_id"):
        build_tail_snapshot(**args)
    assert not args["output_dir"].exists()
    assert args["baseline"].read_bytes() == before


@pytest.mark.parametrize("record,code", [
    ([], "non_object"),
    ({"_funes_record": "unexpected"}, "unsupported"),
    (doc("", "raw"), "invalid_memory_identity"),
    (doc("new\0identity", "raw"), "invalid_memory_identity"),
    (doc("new", 7), "invalid_raw_text"),
    (doc("new", "raw", embedding_generation=-1), "invalid_memory_generation"),
    ({"_funes_record": "reindex_control", "generation": 0, "scope": "all"}, "invalid_reindex"),
])
def test_malformed_records_fail_closed(tmp_path, monkeypatch, record, code):
    args = make_fixture(tmp_path, monkeypatch, tail=[("funes-delta-invalid.jsonl.gz.enc", [record])])
    with pytest.raises(TailSnapshotError, match=code):
        build_tail_snapshot(**args)
    assert not args["output_dir"].exists()


def test_encrypted_digest_mismatch_fails_after_copy_without_touching_baseline(tmp_path, monkeypatch):
    args = make_fixture(tmp_path, monkeypatch)
    name = args["plan"]["tail_files"][0]
    encrypted = args["cache"] / "remote" / name
    encrypted.write_bytes(encrypted.read_bytes() + b"corruption")
    with pytest.raises(TailSnapshotError, match="encrypted_download_digest_mismatch"):
        build_tail_snapshot(**args)
    assert not args["output_dir"].exists()
    assert hashlib.sha256(args["baseline"].read_bytes()).hexdigest() == args["metadata"]["sqlite_sha256"]


def test_changed_hash_cannot_bypass_authentication_and_partial_replay_is_discarded(tmp_path, monkeypatch):
    args = make_fixture(tmp_path, monkeypatch, tail=[
        ("funes-delta-one.jsonl.gz.enc", [doc("one", "committed only to disposable copy")]),
        ("funes-delta-two.jsonl.gz.enc", [doc("two", "must authenticate first")])])
    name = args["plan"]["tail_files"][-1]
    encrypted = args["cache"] / "remote" / name
    data = bytearray(encrypted.read_bytes())
    data[-1] ^= 1
    encrypted.write_bytes(data)
    args["plan"]["downloads"][name]["sha256"] = hashlib.sha256(data).hexdigest()
    from cryptography.exceptions import InvalidTag
    with pytest.raises(InvalidTag):
        build_tail_snapshot(**args)
    assert not args["output_dir"].exists()
    assert read_tables(args["baseline"])["memories"][0][1] == "original"


def test_wrong_baseline_hash_and_live_wal_are_rejected(tmp_path, monkeypatch):
    args = make_fixture(tmp_path, monkeypatch)
    args["plan"]["baseline_sqlite_sha256"] = args["metadata"]["sqlite_sha256"] = "0" * 64
    with pytest.raises(TailSnapshotError, match="baseline_digest_mismatch"):
        build_tail_snapshot(**args)
    assert not args["output_dir"].exists()
    wal = Path(str(args["baseline"]) + "-wal")
    wal.write_bytes(b"live WAL")
    with pytest.raises(TailSnapshotError, match="baseline_has_live_wal"):
        build_tail_snapshot(**args)
    assert wal.read_bytes() == b"live WAL"


def test_cache_symlink_and_path_traversal_are_rejected(tmp_path, monkeypatch):
    args = make_fixture(tmp_path, monkeypatch)
    name = args["plan"]["tail_files"][0]
    encrypted = args["cache"] / "remote" / name
    outside = tmp_path / "outside.enc"
    encrypted.rename(outside)
    encrypted.symlink_to(outside)
    with pytest.raises(TailSnapshotError, match="unsafe_tail_path"):
        build_tail_snapshot(**args)
    assert outside.exists() and not args["output_dir"].exists()
    args["plan"]["manifest"]["deltas"][-1] = "../funes-delta-evil.jsonl.gz.enc"
    with pytest.raises(TailSnapshotError):
        build_tail_snapshot(**args)


def test_preexisting_output_is_never_deleted_or_overwritten(tmp_path, monkeypatch):
    args = make_fixture(tmp_path, monkeypatch)
    args["output_dir"].mkdir()
    marker = args["output_dir"] / "owned-by-someone-else"
    marker.write_text("preserve")
    with pytest.raises(TailSnapshotError, match="output_already_exists"):
        build_tail_snapshot(**args)
    assert marker.read_text() == "preserve"


def test_changed_encrypted_file_during_decrypt_is_rejected(tmp_path, monkeypatch):
    args = make_fixture(tmp_path, monkeypatch)
    decrypt = args["sync"]._decrypt_file

    def changed(source, target):
        decrypt(source, target)
        with source.open("ab") as stream:
            stream.write(b"changed after authentication")

    monkeypatch.setattr(args["sync"], "_decrypt_file", changed)
    with pytest.raises(TailSnapshotError, match="encrypted_download_changed"):
        build_tail_snapshot(**args)
    assert not args["output_dir"].exists()
    assert hashlib.sha256(args["baseline"].read_bytes()).hexdigest() == args["metadata"]["sqlite_sha256"]


def test_output_schema_remains_identical_to_baseline_for_pg_tail_application(tmp_path, monkeypatch):
    args = make_fixture(tmp_path, monkeypatch)
    build_tail_snapshot(**args)
    from scripts.migrate_source_postgres import Snapshot
    with Snapshot(args["baseline"]) as baseline, Snapshot(args["output_dir"] / "funes.sqlite3") as desired:
        assert desired.schema == baseline.schema
        assert desired.counts["memories"] == baseline.counts["memories"] + 1


def test_schema_upgrade_is_explicit_and_does_not_mutate_baseline(tmp_path, monkeypatch):
    def setup(store):
        store.set_sync(native_checkpoint_state_version=1)
    args = make_fixture(tmp_path, monkeypatch, setup=setup)
    before = args["baseline"].read_bytes()
    with pytest.raises(TailSnapshotError, match="baseline_native_schema_requires_explicit_upgrade"):
        build_tail_snapshot(**args)
    assert args["baseline"].read_bytes() == before
    assert not args["output_dir"].exists()


def boundary_fixture(tmp_path, plan, **updates):
    value = {"version": 1, "kind": "funes_source_write_boundary", "status": "verified",
             "repo": plan["repo"], "revision": plan["revision"],
             "baseline_sqlite_sha256": plan["baseline_sqlite_sha256"],
             "plan_sha256": json_sha256(plan), "source_writes_fenced": True,
             "background_writers_stopped": True, "in_flight_source_writes": 0,
             "pinned_after_fence": True, "verified_at": "2026-01-01T00:00:00+00:00", **updates}
    path = tmp_path / "boundary.json"
    path.write_text(json.dumps(value))
    return {"boundary_evidence": path, "boundary_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_only_hash_pinned_verified_external_boundary_can_mark_final(tmp_path, monkeypatch):
    args = make_fixture(tmp_path, monkeypatch)
    boundary = boundary_fixture(tmp_path, args["plan"])
    report = build_tail_snapshot(**args, **boundary)
    assert report["final_write_boundary"] is True
    assert report["boundary_evidence"]["kind"] == "external_operator_attestation"
    assert report["boundary_evidence"]["live_writer_state_checked_by_builder"] is False
    assert report["status"] == "built_not_imported"


@pytest.mark.parametrize("updates", [
    {"revision": "c" * 40}, {"plan_sha256": "0" * 64}, {"status": "unverified"},
    {"source_writes_fenced": False}, {"background_writers_stopped": False},
    {"in_flight_source_writes": 1}, {"pinned_after_fence": False},
])
def test_unverified_boundary_cannot_claim_final(tmp_path, monkeypatch, updates):
    args = make_fixture(tmp_path, monkeypatch)
    boundary = boundary_fixture(tmp_path, args["plan"], **updates)
    with pytest.raises(TailSnapshotError, match="unverified_or_unbound_write_boundary"):
        build_tail_snapshot(**args, **boundary)
    assert not args["output_dir"].exists()


def test_boundary_hash_is_mandatory_and_checks_original_bytes(tmp_path, monkeypatch):
    args = make_fixture(tmp_path, monkeypatch)
    boundary = boundary_fixture(tmp_path, args["plan"])
    with pytest.raises(TailSnapshotError, match="boundary_path_and_hash_required"):
        build_tail_snapshot(**args, boundary_evidence=boundary["boundary_evidence"])
    boundary["boundary_sha256"] = "0" * 64
    with pytest.raises(TailSnapshotError, match="boundary_digest_mismatch"):
        build_tail_snapshot(**args, **boundary)


def test_cli_suppresses_untrusted_exceptions_and_removes_owned_output(tmp_path, monkeypatch, capsys):
    args = make_fixture(tmp_path, monkeypatch)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(args["plan"]))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(args["metadata"]))

    def broken(*_):
        raise ValueError("sensitive raw payload must never be logged")

    monkeypatch.setattr(SnapshotSync, "_decrypt_file", broken)
    assert main(["--baseline", str(args["baseline"]), "--metadata", str(metadata),
                 "--plan", str(plan), "--cache-dir", str(args["cache"]),
                 "--output-dir", str(args["output_dir"])]) == 2
    output = capsys.readouterr()
    assert "sensitive" not in output.err + output.out
    assert "tail_snapshot_build_failed" in output.err
    assert not args["output_dir"].exists()


def test_cli_success_returns_file_identity_and_nonfinal_boundary(tmp_path, monkeypatch, capsys):
    args = make_fixture(tmp_path, monkeypatch)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(args["plan"]))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(args["metadata"]))
    assert main(["--baseline", str(args["baseline"]), "--metadata", str(metadata),
                 "--plan", str(plan), "--cache-dir", str(args["cache"]),
                 "--output-dir", str(args["output_dir"])]) == 0
    output = capsys.readouterr()
    assert not output.err
    final = json.loads(output.out.splitlines()[-1])
    assert final["phase"] == "completed"
    assert final["counts"]["memories"] == 2
    assert final["final_write_boundary"] is False
    assert len(final["sqlite_sha256"]) == 64


def make_complete_fixture(tmp_path, monkeypatch, *, initial=None, files=None, setup=None):
    args = make_fixture(tmp_path, monkeypatch, initial=initial, tail=[], setup=setup)
    files = files if files is not None else [
        ("funes-snapshot-compacted.jsonl.gz.enc", [doc("original", "authoritative raw")])]
    manifest = {"version": 1, "snapshot": files[0][0], "deltas": [], "controls": []}
    downloads, blobs = {}, {}
    for filename, records in files:
        plaintext = tmp_path / "complete-payload.jsonl.gz"
        with gzip.open(plaintext, "wt", encoding="utf-8") as stream:
            for item in records:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        encrypted = args["cache"] / "remote" / filename
        encrypted.parent.mkdir(parents=True, exist_ok=True)
        args["sync"]._encrypt_file(plaintext, encrypted)
        plaintext.unlink()
        downloads[filename] = {"size": encrypted.stat().st_size,
                               "sha256": hashlib.sha256(encrypted.read_bytes()).hexdigest()}
        blobs[filename] = hashlib.sha1(encrypted.read_bytes()).hexdigest()
        if filename != manifest["snapshot"]:
            manifest["controls" if filename.startswith("funes-reindex-") else "deltas"].append(filename)
    current = {"head": "c" * 40, "manifest": manifest,
               "repo_files": [name for name, _ in files], "blob_ids": blobs}
    args["plan"] = build_complete_plan(args["metadata"], current, "owner/source", args["sync"])
    args["plan"]["downloads"] = downloads
    args["mode"] = "complete"
    return args


def native_state(revision, **extra):
    return {"_funes_record": "native_index_state", "state_version": 2,
            "revision": revision, "profile": "profile", "memory": "native-memory",
            "eligible": 3, "indexed": 1, "held": 0, "invalid": 0, **extra}


def optimize(revision, **extra):
    return {"_funes_record": "native_optimize_checkpoint", "revision": revision,
            "fingerprint": "profile", "memory": "native-memory", "status": "optimized",
            "optimized_at": NOW, "index_layout_version": 2, **extra}


def test_complete_rebuild_preserves_ids_schema_and_source_timestamps_not_stale_baseline(tmp_path, monkeypatch):
    initial = [doc("removed", "stale raw"), doc("alpha", "stale alpha", native_generation=90),
               doc("beta", "stale beta", source_missing=True)]

    def setup(store):
        store.translation_put("removed-cache", "stale cache")
        store.record_reindex_control({"generation": 90, "scope": "all", "created_at": NOW})
        store.set_native_index_state(native_state(900))
        store.set_native_optimize_checkpoint(optimize(900))
        with store.conn:
            store.conn.execute("UPDATE sqlite_sequence SET seq=77 WHERE name='memories'")

    old = "2024-01-02T03:04:05+00:00"
    source = [doc("beta", "new beta", id=1, ingested_at=old),
              doc("new", "新\0原文\ue0000", id=2, ingested_at=old),
              doc("alpha", "new alpha", id=3, ingested_at=old, native_generation=1),
              {"_funes_record": "translation_cache", "query": "new-cache", "rewritten": "cache",
               "created_at": old}, native_state(5), optimize(5)]
    files = [("funes-snapshot-compacted.jsonl.gz.enc", source)]
    args = make_complete_fixture(tmp_path, monkeypatch, initial=initial, files=files, setup=setup)
    before = args["baseline"].read_bytes()
    events = []
    report = build_tail_snapshot(**args, batch_size=2, progress=events.append)
    target = args["output_dir"] / "funes.sqlite3"
    assert args["baseline"].read_bytes() == before
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT id,source_identity,raw_text,ingested_at FROM memories ORDER BY id").fetchall() == [
            (2, "alpha", "new alpha", old), (3, "beta", "new beta", old), (79, "new", "新\0原文\ue0000", old)]
        assert conn.execute("SELECT native_generation FROM memories WHERE source_identity='alpha'").fetchone() == (1,)
        assert conn.execute("SELECT source_missing FROM memories WHERE source_identity='beta'").fetchone() == (0,)
        assert conn.execute("SELECT query,created_at FROM translation_cache").fetchall() == [("new-cache", old)]
        assert conn.execute("SELECT count(*) FROM reindex_controls").fetchone() == (0,)
        assert conn.execute("SELECT native_index_revision,native_optimize_revision,last_sync FROM sync_state").fetchone() == (5, 5, None)
        assert conn.execute("SELECT seq FROM sqlite_sequence WHERE name='memories'").fetchone() == (79,)
    from scripts.migrate_source_postgres import Snapshot
    with Snapshot(args["baseline"]) as baseline, Snapshot(target) as desired:
        assert desired.schema == baseline.schema
    assert report["source_membership"] == "complete_manifest_union_only"
    assert report["identity_mapping"] == {
        "baseline_identities_restored": 2, "new_identities": 1,
        "source_ingestion_timestamps": 3, "baseline_ingestion_timestamps": 0,
        "source_cache_creation_timestamps": 1, "baseline_max_id": 3,
        "new_id_floor": 77, "baseline_identities_absent": 1}
    assert report["mode"] == "complete" and "tail_files" not in report
    assert "copying_baseline" not in {item["phase"] for item in events}
    assert report["final_write_boundary"] is False


def test_complete_merge_and_native_state_match_fresh_production_restore(tmp_path, monkeypatch):
    docs = [doc("beta", "beta", native_index_profile="profile", native_index_memory="native-memory",
                native_index_status="indexed"), doc("alpha", "alpha"), doc("new", "new")]
    files = [("funes-snapshot-compacted.jsonl.gz.enc", [*docs, native_state(80), optimize(80)]),
             ("deltas/a/funes-delta-final.jsonl.gz.enc", [
                 doc("alpha", "changed alpha", source_version="v2", source_missing=True),
                 native_state(81), optimize(81), native_state(79), optimize(79)])]
    args = make_complete_fixture(tmp_path, monkeypatch, initial=[doc("alpha", "baseline")], files=files)
    reference = Store(str(tmp_path / "production"))
    reference.begin_bulk_restore()
    for _, records in files:
        reference.restore_documents(records, 2, apply_controls=False)
    reference.compact_reindex_controls(replay=True)
    reference.drain_reindex_controls(2)
    reference.finish_bulk_restore(rebuild_fts=False)
    reference.set_sync(last_error=None)
    reference.close()
    report = build_tail_snapshot(**args, batch_size=2)
    actual = read_tables(args["output_dir"] / "funes.sqlite3")
    expected = read_tables(tmp_path / "production" / "funes.sqlite3")
    assert sorted(row[1:] for row in actual["memories"]) == sorted(row[1:] for row in expected["memories"])
    for table in ("translation_cache", "reindex_controls", "sync_state"):
        assert actual[table] == expected[table]
    assert report["native_index_revision"] == report["native_optimize_revision"] == 81


def test_complete_replays_all_controls_and_drops_absent_baseline_controls(tmp_path, monkeypatch):
    def setup(store):
        store.record_reindex_control({"generation": 99, "scope": "all", "created_at": NOW})
    files = [("funes-snapshot-compacted.jsonl.gz.enc", [doc("original", "raw")]),
             ("funes-reindex-final.jsonl.gz.enc", [
                 {"_funes_record": "reindex_control", "generation": 2, "scope": "all", "created_at": NOW}])]
    args = make_complete_fixture(tmp_path, monkeypatch, files=files, setup=setup)
    report = build_tail_snapshot(**args)
    with sqlite3.connect(args["output_dir"] / "funes.sqlite3") as conn:
        assert conn.execute("SELECT generation,scope,applied_at IS NOT NULL FROM reindex_controls").fetchall() == [(2, "all", 1)]
        assert conn.execute("SELECT native_generation,retrieval_generation FROM memories").fetchone() == (2, 2)
    assert report["records_by_type"] == {"memory": 1, "reindex_control": 1}


def test_complete_duplicate_identity_keeps_baseline_id_and_first_source_timestamp(tmp_path, monkeypatch):
    first = "2020-01-01T00:00:00+00:00"
    files = [("funes-snapshot-compacted.jsonl.gz.enc", [
        doc("original", "first", ingested_at=first),
        doc("original", "second", source_version="v2", ingested_at=NOW),
        doc("new", "new raw"), doc("new", "new raw")])]
    args = make_complete_fixture(tmp_path, monkeypatch, files=files)
    report = build_tail_snapshot(**args, batch_size=4)
    with sqlite3.connect(args["output_dir"] / "funes.sqlite3") as conn:
        assert conn.execute("SELECT id,raw_text,ingested_at FROM memories WHERE source_identity='original'").fetchone() == (1, "second", first)
        assert conn.execute("SELECT count(*) FROM memories").fetchone() == (2,)
    assert report["identity_mapping"]["baseline_identities_restored"] == 1
    assert report["identity_mapping"]["new_identities"] == 1


def test_complete_legacy_missing_ingestion_timestamp_falls_back_to_baseline(tmp_path, monkeypatch):
    value = doc("original", "raw")
    del value["ingested_at"]
    args = make_complete_fixture(tmp_path, monkeypatch, files=[("funes-snapshot-compacted.jsonl.gz.enc", [value])])
    monkeypatch.setattr("service.server.utc_now", lambda: "future-restore-time")
    report = build_tail_snapshot(**args)
    with sqlite3.connect(args["output_dir"] / "funes.sqlite3") as conn:
        assert conn.execute("SELECT ingested_at FROM memories").fetchone() == (NOW,)
    assert report["identity_mapping"]["baseline_ingestion_timestamps"] == 1


def test_complete_plan_requires_explicit_mode_and_does_not_weaken_suffix_guard(tmp_path, monkeypatch):
    args = make_complete_fixture(tmp_path, monkeypatch)
    args["mode"] = "suffix"
    with pytest.raises(TailSnapshotError, match="explicit_plan_mode_mismatch"):
        build_tail_snapshot(**args)
    from scripts.plan_source_postgres_tail import TailPlanError
    with pytest.raises(TailPlanError, match="baseline_snapshot_replaced"):
        build_plan(args["metadata"], {"head": args["plan"]["revision"], "manifest": args["plan"]["manifest"],
                                     "repo_files": args["plan"]["restore_files"], "blob_ids": args["plan"]["blob_ids"]},
                   args["plan"]["repo"])
    assert not args["output_dir"].exists()


@pytest.mark.parametrize("change,code", [
    (lambda p: p.update(restore_files=[]), "receipt_mismatch"),
    (lambda p: p.update(tail_files=[]), "suffix_claim"),
    (lambda p: p.update(downloads={}), "incomplete"),
    (lambda p: p.update(mode="suffix"), "mode_mismatch"),
    (lambda p: p.update(blob_ids={}), "blob_identity"),
])
def test_incomplete_or_mislabelled_complete_plan_fails_closed(tmp_path, monkeypatch, change, code):
    args = make_complete_fixture(tmp_path, monkeypatch)
    change(args["plan"])
    with pytest.raises(TailSnapshotError, match=code):
        build_tail_snapshot(**args)
    assert not args["output_dir"].exists()


@pytest.mark.parametrize("mode", ["suffix", "complete"])
@pytest.mark.parametrize("record,code", [
    (native_state(100, state_version=999), "state_version"),
    (native_state(100, eligible=-1), "checkpoint_fields"),
    (native_state(100, profile=[]), "checkpoint_fields"),
    (native_state(100, index_fingerprint="not-a-matching-fingerprint"), "checkpoint_fingerprint"),
    (optimize(100, optimized_at=None), "optimize_checkpoint"),
    (optimize(100, fingerprint=""), "optimize_checkpoint"),
    (optimize(100, index_layout_version=-1), "optimize_checkpoint"),
])
def test_invalid_checkpoints_fail_instead_of_silently_disappearing(tmp_path, monkeypatch, mode, record, code):
    if mode == "complete":
        args = make_complete_fixture(tmp_path, monkeypatch, files=[("funes-snapshot-compacted.jsonl.gz.enc", [record])])
    else:
        args = make_fixture(tmp_path, monkeypatch, tail=[("funes-delta-bad.jsonl.gz.enc", [record])])
    with pytest.raises(TailSnapshotError, match=code):
        build_tail_snapshot(**args)
    assert not args["output_dir"].exists()


def test_complete_wrong_baseline_hash_and_bad_authenticated_blob_clean_owned_output(tmp_path, monkeypatch):
    args = make_complete_fixture(tmp_path, monkeypatch)
    actual_hash = args["plan"]["baseline_sqlite_sha256"]
    args["plan"]["baseline_sqlite_sha256"] = args["metadata"]["sqlite_sha256"] = "0" * 64
    with pytest.raises(TailSnapshotError, match="baseline_digest_mismatch"):
        build_tail_snapshot(**args)
    assert not args["output_dir"].exists()
    args["plan"]["baseline_sqlite_sha256"] = args["metadata"]["sqlite_sha256"] = actual_hash
    name = args["plan"]["restore_files"][0]
    encrypted = args["cache"] / "remote" / name
    data = bytearray(encrypted.read_bytes())
    data[-1] ^= 1
    encrypted.write_bytes(data)
    args["plan"]["downloads"][name]["sha256"] = hashlib.sha256(data).hexdigest()
    from cryptography.exceptions import InvalidTag
    with pytest.raises(InvalidTag):
        build_tail_snapshot(**args)
    assert not args["output_dir"].exists()


def test_prepare_complete_plan_never_resolves_head_and_pins_every_read(tmp_path, monkeypatch):
    from types import SimpleNamespace
    args = make_complete_fixture(tmp_path, monkeypatch)
    calls = []
    revision = args["plan"]["revision"]
    sync = args["sync"]
    sync.token = "test-only"

    class API:
        def repo_info(self, **_):
            raise AssertionError("must not resolve mutable HEAD")

        def list_repo_tree(self, repo, **kwargs):
            assert repo == "owner/source" and kwargs["revision"] == revision
            calls.append("tree")
            return [SimpleNamespace(path=name, blob_id=blob)
                    for name, blob in {**args["plan"]["blob_ids"], sync.manifest_filename: "d" * 40}.items()]

    def manifest(files, pinned):
        assert pinned == revision and sync.manifest_filename in files
        calls.append("manifest")
        return args["plan"]["manifest"]

    def download(**kwargs):
        assert kwargs["revision"] == revision and kwargs["repo_type"] == "dataset"
        calls.append("blob")
        return args["cache"] / "remote" / kwargs["filename"]

    monkeypatch.setattr(sync, "_download_restore_manifest", manifest)
    plan = prepare_complete_plan(args["metadata"], args["cache"], revision, sync, API(), download)
    assert calls == ["tree", "manifest", "blob"]
    assert plan["downloads"] == args["plan"]["downloads"]
    assert plan["mode"] == "complete" and plan["revision"] == revision
    assert plan["final_write_boundary"] is False
    with pytest.raises(TailSnapshotError, match="explicit_full_revision_required"):
        prepare_complete_plan(args["metadata"], args["cache"], "main", sync, API(), download)


def test_complete_cli_and_boundary_evidence_are_bound_to_entire_plan(tmp_path, monkeypatch, capsys):
    args = make_complete_fixture(tmp_path, monkeypatch)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(args["plan"]))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(args["metadata"]))
    boundary = boundary_fixture(tmp_path, args["plan"])
    assert main(["--mode", "complete", "--baseline", str(args["baseline"]), "--metadata", str(metadata),
                 "--plan", str(plan), "--cache-dir", str(args["cache"]), "--output-dir", str(args["output_dir"]),
                 "--boundary-evidence", str(boundary["boundary_evidence"]),
                 "--boundary-sha256", boundary["boundary_sha256"]]) == 0
    final = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert final["counts"]["memories"] == 1 and final["final_write_boundary"] is True


def test_complete_skips_startup_scans_and_rebuilds_native_state_only_once(tmp_path, monkeypatch):
    args = make_complete_fixture(tmp_path, monkeypatch)
    original = Store._rebuild_native_checkpoint_state_locked
    calls = []

    def counted(self, *params, **kwargs):
        calls.append(1)
        return original(self, *params, **kwargs)

    def reject_startup(*_):
        raise AssertionError("must not run Store startup migrations")

    monkeypatch.setattr(Store, "_init_schema", reject_startup)
    monkeypatch.setattr(Store, "_rebuild_native_checkpoint_state_locked", counted)
    report = build_tail_snapshot(**args)
    assert calls == [1]
    assert report["full_identifiers_rebuilt"] is True
    assert report["second_full_identifier_scan"] is False


def test_record_progress_does_not_materialize_or_print_source(tmp_path, monkeypatch):
    from collections import Counter
    from scripts.build_source_postgres_tail_snapshot import checked_records
    records = (doc("identity", "private raw") for _ in range(10000))
    events = []
    assert sum(1 for _ in checked_records(records, Counter(), events.append)) == 10000
    assert events == [{"phase": "replaying_records", "records_seen": 10000}]


@pytest.mark.parametrize("batch_size", [1, 2, 50])
def test_suffix_preserves_new_source_ingestion_timestamp_without_changing_baseline_ids_or_ingestion(tmp_path, monkeypatch, batch_size):
    original = "2023-02-03T04:05:06+00:00"
    later = "2024-02-03T04:05:06+00:00"

    def setup(store):
        with store.conn:
            store.conn.execute("UPDATE sqlite_sequence SET seq=40 WHERE name='memories'")

    records = [doc("original", "updated original", id=999, source_version="v2", ingested_at=original),
               doc("new", "new raw", id=1, ingested_at=original),
               doc("new", "latest raw", id=999, source_version="v2", ingested_at=later)]
    args = make_fixture(tmp_path, monkeypatch, tail=[("funes-delta-final.jsonl.gz.enc", records)], setup=setup)
    before = args["baseline"].read_bytes()
    report = build_tail_snapshot(**args, batch_size=batch_size)
    with sqlite3.connect(args["output_dir"] / "funes.sqlite3") as conn:
        assert conn.execute("SELECT id,source_identity,raw_text,ingested_at FROM memories ORDER BY id").fetchall() == [
            (1, "original", "updated original", NOW), (41, "new", "latest raw", original)]
        assert conn.execute("SELECT seq FROM sqlite_sequence WHERE name='memories'").fetchone() == (41,)
    assert args["baseline"].read_bytes() == before
    assert report["timestamp_preservation"]["source_ingestion_timestamps"] == 1
    assert report["production_restore_deviations"] == ["preserve_serialized_source_timestamps"]
    assert report["id_policy"] == "baseline_ids_unchanged;new_ids_above_sqlite_sequence;remote_numeric_ids_ignored"


def test_suffix_preserves_serialized_cache_timestamps_in_order_and_leaves_untouched_cache(tmp_path, monkeypatch):
    first = "2023-01-02T03:04:05+00:00"
    second = "2024-01-02T03:04:05+00:00"

    def setup(store):
        store.translation_put("untouched", "old value")
        store.translation_put("changed", "old changed")

    records = [{"_funes_record": "translation_cache", "query": "changed", "rewritten": "first",
                "created_at": first},
               {"_funes_record": "translation_cache", "query": "new", "rewritten": "new value",
                "created_at": first},
               {"_funes_record": "translation_cache", "query": "changed", "rewritten": "latest",
                "created_at": second}]
    args = make_fixture(tmp_path, monkeypatch, tail=[("funes-delta-cache.jsonl.gz.enc", records)], setup=setup)
    report = build_tail_snapshot(**args)
    with sqlite3.connect(args["output_dir"] / "funes.sqlite3") as conn:
        assert conn.execute("SELECT query,rewritten,created_at FROM translation_cache ORDER BY query").fetchall() == [
            ("changed", "latest", second), ("new", "new value", first), ("untouched", "old value", NOW)]
    assert report["timestamp_preservation"]["source_cache_creation_timestamps"] == 3


def test_suffix_missing_source_timestamps_keep_production_fallback(tmp_path, monkeypatch):
    memory = doc("new", "raw")
    memory.pop("ingested_at")
    records = [memory, {"_funes_record": "translation_cache", "query": "new", "rewritten": "raw"}]
    args = make_fixture(tmp_path, monkeypatch, tail=[("funes-delta-legacy.jsonl.gz.enc", records)])
    report = build_tail_snapshot(**args)
    with sqlite3.connect(args["output_dir"] / "funes.sqlite3") as conn:
        assert conn.execute("SELECT ingested_at FROM memories WHERE source_identity='new'").fetchone() == (NOW,)
        assert conn.execute("SELECT created_at FROM translation_cache").fetchone() == (NOW,)
    assert report["timestamp_preservation"] == {
        "source_ingestion_timestamps": 0, "source_cache_creation_timestamps": 0}


@pytest.mark.parametrize("record,code", [
    (doc("new", "raw", ingested_at=123), "invalid_source_ingestion_timestamp"),
    ({"_funes_record": "translation_cache", "query": "new", "rewritten": "raw", "created_at": []},
     "invalid_source_cache_timestamp"),
])
def test_suffix_invalid_source_timestamps_fail_closed(tmp_path, monkeypatch, record, code):
    args = make_fixture(tmp_path, monkeypatch, tail=[("funes-delta-invalid-clock.jsonl.gz.enc", [record])])
    before = args["baseline"].read_bytes()
    with pytest.raises(TailSnapshotError, match=code):
        build_tail_snapshot(**args)
    assert not args["output_dir"].exists()
    assert args["baseline"].read_bytes() == before
