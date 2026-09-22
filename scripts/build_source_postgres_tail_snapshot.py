#!/usr/bin/env python3
"""Build an offline source snapshot from a frozen baseline and pinned Hub files.

During replay only a newly created, mode-0700 output directory is writable. The baseline,
encrypted cache, Hub, PostgreSQL and native Lance index are never modified.
The result is NOT a final write boundary by default. An optional hash-pinned
external boundary attestation must explicitly bind this exact plan/revision.

Replay uses Store's production merge/control semantics, not INSERT-only SQL.
Opening the copy deliberately skips startup schema migrations: those can scan
all raw text, rebuild identifiers/FTS, or reclassify old rows. Incompatible
schemas fail closed instead. The one outer bulk restore still recomputes native
pending/counts and secondary indexes, but never rebuilds FTS or performs a second
full identifier scan. Complete replay computes identifiers once per input record
through normal production ingest.

Default suffix mode requires unchanged covered blobs and manifest prefixes.
It preserves baseline IDs/ingestion times and serialized timestamps on newly
created source rows and replayed cache entries; remote numeric IDs are not used.
Separate complete mode replays the ENTIRE authoritative manifest into an empty
schema clone, retaining only baseline identity-to-ID mappings, not stale rows or
checkpoint state. Its small restore adapter also preserves serialized ingestion
and cache timestamps (production ingest otherwise regenerates these timestamps).
--prepare-complete-plan downloads encrypted files at an explicitly pinned SHA;
this preparation is the only operation allowed to write the encrypted cache.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.migrate_source_postgres import TABLES, file_signature, quoted  # noqa: E402
from scripts.plan_source_postgres_tail import build_plan, write_json  # noqa: E402
from scripts.validate_source_postgres_tail import RECORD_TYPES  # noqa: E402
from service.server import (  # noqa: E402
    FIELDS, FTS_SCHEMA_VERSION, NATIVE_CHECKPOINT_STATE_VERSION, REINDEX_SCOPES, SnapshotSync, Store,
    utc_now,
)

CHUNK = 4 * 1024 * 1024
SQLITE_MAX_INT = (1 << 63) - 1


class TailSnapshotError(RuntimeError):
    """Fixed operator-facing codes; never include paths, payloads or secrets."""


def require(condition, code):
    if not condition:
        raise TailSnapshotError(code)


def json_sha256(value: dict) -> str:
    # Same canonical plan digest as validate_source_postgres_tail.py.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def is_sha(value, lengths=(64,)) -> bool:
    return isinstance(value, str) and len(value) in lengths and re.fullmatch(
        r"[0-9a-f]+", value
    ) is not None


def sha256_file(path: Path, progress=None, phase="hashing") -> str:
    digest = hashlib.sha256()
    read = last = 0
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
            read += len(block)
            if progress and read - last >= 256 * 1024 * 1024:
                progress({"phase": phase, "bytes": read})
                last = read
    return digest.hexdigest()


def build_complete_plan(metadata: dict, current: dict, repo: str,
                        sync: SnapshotSync) -> dict:
    """An explicit replacement plan; deliberately never call the suffix planner."""
    try:
        receipt = metadata["source_receipt"]
        baseline = receipt["manifest"]
        covered = [baseline["snapshot"], *baseline["deltas"], *baseline["controls"]]
        require(receipt.get("version") == 1 and receipt.get("repo") == repo
                and receipt.get("files") == covered, "baseline_receipt_is_not_complete")
        sync._validate_restore_manifest(baseline, set(covered))
        require(all(is_sha(receipt["blob_ids"].get(name), (40, 64)) for name in covered),
                "invalid_baseline_blob_identity")
        manifest = sync._validate_restore_manifest(current["manifest"], set(current["repo_files"]))
        files = [manifest["snapshot"], *manifest["deltas"], *manifest["controls"]]
        require(is_sha(current["head"], (40,)) and is_sha(receipt["revision"], (40,))
                and is_sha(metadata["sqlite_sha256"]), "invalid_source_identity")
        require(all(is_sha(current["blob_ids"].get(name), (40, 64)) for name in files),
                "invalid_blob_identity")
        return {
            "version": 1, "mode": "complete", "created_at": utc_now(), "repo": repo,
            "baseline_sqlite_sha256": metadata["sqlite_sha256"],
            "baseline_revision": receipt["revision"], "revision": current["head"],
            "restore_files": files, "blob_ids": {name: current["blob_ids"][name] for name in files},
            "manifest": manifest, "final_write_boundary": False,
        }
    except TailSnapshotError:
        raise
    except Exception:
        raise TailSnapshotError("invalid_complete_manifest_receipt") from None


def replay_files(plan: dict, mode: str) -> list[str]:
    return plan["restore_files" if mode == "complete" else "tail_files"]


def validate_plan(metadata: dict, plan: dict, sync: SnapshotSync, mode="suffix") -> None:
    require(isinstance(metadata, dict) and isinstance(plan, dict), "invalid_plan")
    require(mode in {"suffix", "complete"} and plan.get("mode", "suffix") == mode,
            "explicit_plan_mode_mismatch")
    repo = plan.get("repo")
    require(isinstance(repo, str) and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}", repo
    ), "invalid_repo_id")
    require(not sync.repo or sync.repo == repo, "configured_repo_mismatch")
    require(is_sha(plan.get("revision"), (40,))
            and is_sha(plan.get("baseline_revision"), (40,))
            and is_sha(plan.get("baseline_sqlite_sha256")), "invalid_source_identity")
    require(plan.get("final_write_boundary") is False, "plan_is_not_boundary_evidence")
    try:
        receipt = metadata["source_receipt"]
        manifest = plan["manifest"]
        active = [manifest["snapshot"], *manifest["deltas"], *manifest["controls"]]
        blobs = (plan["blob_ids"] if mode == "complete"
                 else {**receipt["blob_ids"], **plan["blob_ids"]})
        require(all(is_sha(blobs.get(name), (40, 64)) for name in active),
                "invalid_blob_identity")
        current = {"head": plan["revision"], "manifest": manifest,
                   "repo_files": active, "blob_ids": blobs}
        if mode == "complete":
            require("tail_files" not in plan and "covered_files" not in plan,
                    "complete_plan_contains_suffix_claim")
            rebuilt = build_complete_plan(metadata, current, repo, sync)
        else:
            rebuilt = build_plan(metadata, current, repo)
        require(all(plan.get(key) == value for key, value in rebuilt.items()
                    if key != "created_at"), "tail_plan_receipt_mismatch")
    except TailSnapshotError:
        raise
    except Exception:
        raise TailSnapshotError("tail_plan_receipt_mismatch") from None
    downloads = plan.get("downloads")
    require(isinstance(downloads, dict) and set(downloads) == set(replay_files(plan, mode)),
            "incomplete_tail_download_plan")
    for receipt in downloads.values():
        require(isinstance(receipt, dict) and type(receipt.get("size")) is int
                and receipt["size"] > 0 and is_sha(receipt.get("sha256")),
                "invalid_download_receipt")


def prepare_complete_plan(metadata: dict, cache: Path, revision: str,
                          sync: SnapshotSync, api, download, progress=None) -> dict:
    """Read one immutable revision; no HEAD fallback and no Hub write methods."""
    require(is_sha(revision, (40,)), "explicit_full_revision_required")
    require(bool(sync.repo) and bool(sync.token), "HF_TOKEN_and_FUNES_STORAGE_REPO_required")
    require(not cache.is_symlink(), "unsafe_tail_cache")
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache = cache.resolve(strict=True)
    require(not (cache / "remote").is_symlink(), "unsafe_tail_cache")
    sync.store = SimpleNamespace(data_dir=cache)
    files, blobs = sync._repo_tree_files(api, revision)
    require(sync.manifest_filename in files, "explicit_restore_manifest_required")
    manifest = sync._download_restore_manifest(files, revision)
    plan = build_complete_plan(metadata, {"head": revision, "manifest": manifest,
                                         "repo_files": files, "blob_ids": blobs}, sync.repo, sync)
    plan["downloads"] = {}
    for index, name in enumerate(plan["restore_files"]):
        downloaded = Path(download(repo_id=sync.repo, repo_type="dataset", filename=name,
                                   revision=revision, token=sync.token, local_dir=str(cache / "remote")))
        expected = checked_encrypted_path(cache, name)
        require(downloaded.resolve(strict=True) == expected.resolve(strict=True),
                "unexpected_download_path")
        before = file_signature(expected)
        digest = sha256_file(expected, progress, "hashing_encrypted_download")
        require(before == file_signature(expected), "encrypted_download_changed")
        plan["downloads"][name] = {"size": before[2], "sha256": digest}
        if progress:
            progress({"phase": "downloaded_complete_manifest", "files": index + 1,
                      "total_files": len(plan["restore_files"]), "revision": revision})
    validate_plan(metadata, plan, sync, "complete")
    return plan


def checked_encrypted_path(cache: Path, filename: str) -> Path:
    require(SnapshotSync._safe_repo_path(filename) and filename.endswith(".enc"),
            "unsafe_tail_path")
    root = cache / "remote"
    require(root.is_dir() and not root.is_symlink(), "unsafe_tail_cache")
    candidate = root
    for component in Path(filename).parts:
        candidate /= component
        require(not candidate.is_symlink(), "unsafe_tail_path")
    require(candidate.is_file() and candidate.resolve().is_relative_to(root.resolve()),
            "unsafe_or_missing_tail_file")
    require(stat.S_ISREG(candidate.stat().st_mode), "unsafe_tail_path")
    return candidate


def assert_frozen(path: Path, signature: tuple) -> None:
    require(not path.is_symlink() and file_signature(path) == signature,
            "baseline_changed")
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(path) + suffix)
        require(not sidecar.exists() or sidecar.stat().st_size == 0,
                "baseline_has_live_wal_or_journal")


def validate_schema(conn: sqlite3.Connection) -> None:
    # A read-only compatibility gate, NOT a schema migration. Keep this set
    # aligned with the methods invoked below; absent columns must never be
    # silently invented in an authoritative baseline.
    required = {
        "memories": {"id", "source_identity", "source_version", "raw_text",
                     "retrieval_text", "search_identifiers", "metadata_json",
                     "source_metadata_clock_json", "native_index_pending", *FIELDS},
        "translation_cache": {"query", "rewritten", "created_at", "translation_hash",
                              "translation_version", "translation_status"},
        "reindex_controls": {"generation", "scope", "created_at", "row_cursor", "applied_at"},
        "sync_state": {"id", "last_sync", "last_error", "snapshot_path", "restored_at",
                       "native_optimize_provider", "native_optimize_model",
                       "native_optimize_dimensions", "native_optimize_schema_version",
                       "native_optimize_layout_version", "native_optimize_memory",
                       "native_optimize_fingerprint", "native_optimize_index_fingerprint",
                       "native_optimize_status", "native_optimized_at", "native_optimize_revision",
                       "native_checkpoint_profile", "native_checkpoint_memory",
                       "native_index_revision", "native_eligible_count", "native_indexed_count",
                       "native_held_count", "native_invalid_count", "native_checkpoint_state_version",
                       "fts_schema_version", "fts_ready"},
    }
    for table, names in required.items():
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({quoted(table)})")}
        require(names <= columns, "baseline_schema_requires_explicit_upgrade")
    state = conn.execute("SELECT id,native_checkpoint_state_version FROM sync_state").fetchall()
    require(len(state) == 1 and tuple(state[0]) == (1, NATIVE_CHECKPOINT_STATE_VERSION),
            "baseline_native_schema_requires_explicit_upgrade")
    fts = [row[1] for row in conn.execute("PRAGMA table_info(memories_fts)")]
    require(fts == ["raw_text", "retrieval_text", "search_identifiers"],
            "baseline_fts_schema_requires_explicit_upgrade")


class ExistingSnapshotStore(Store):
    """Production restore methods with no implicit startup upgrades/scans."""

    def __init__(self, data_dir: str):
        try:
            super().__init__(data_dir)
        except BaseException:
            if hasattr(self, "conn"):
                self.conn.close()
            raise

    def _init_schema(self) -> None:
        validate_schema(self.conn)
        self.conn.execute("PRAGMA temp_store=FILE")
        self.conn.execute("PRAGMA cache_size=-16384")
        self.conn.execute("PRAGMA mmap_size=0")


def cache_timestamp_records(store: Store, documents, stats: dict):
    """Preserve each consumed cache record's clock without buffering the stream."""
    for item in documents:
        if item.get("_funes_record") == "translation_cache":
            stamp = item.get("created_at")
            require(stamp is None or isinstance(stamp, str), "invalid_source_cache_timestamp")
            yield item
            # Store has consumed this item and applied translation_put.
            if stamp is not None:
                with store.conn:
                    changed = store.conn.execute("UPDATE translation_cache SET created_at=? WHERE query=?",
                                                 (stamp, item["query"]))
                    require(changed.rowcount == 1, "cache_timestamp_preservation_failed")
                stats["source_cache_creation_timestamps"] += 1
        else:
            yield item


class SuffixSnapshotStore(ExistingSnapshotStore):
    """Append using stable local IDs; preserve only newly supplied source clocks."""

    def __init__(self, data_dir: str):
        self.timestamp_stats = {"source_ingestion_timestamps": 0,
                                "source_cache_creation_timestamps": 0}
        super().__init__(data_dir)

    def ingest(self, docs: list[dict]) -> dict:
        require(self._bulk_restore_depth > 0, "suffix_restore_requires_bulk_scope")
        result = super().ingest(docs)
        stamps = []
        for doc, item in zip(docs, result["items"], strict=True):
            # Never replace the baseline row's original ingestion clock, nor
            # the first source clock when another delta updates that identity.
            if item["status"] != "created":
                continue
            stamp = doc.get("ingested_at", (doc.get("metadata") or {}).get("ingested_at"))
            require(stamp is None or isinstance(stamp, str), "invalid_source_ingestion_timestamp")
            if stamp is not None:
                stamps.append((stamp, item["id"], item["source_identity"]))
        if stamps:
            with self.conn:
                changed = self.conn.executemany(
                    "UPDATE memories SET ingested_at=? WHERE id=? AND source_identity=?", stamps)
                require(changed.rowcount == len(stamps), "source_timestamp_preservation_failed")
            self.timestamp_stats["source_ingestion_timestamps"] += len(stamps)
        return result

    def restore_documents(self, documents, batch_size=500, *, apply_controls=True) -> int:
        return super().restore_documents(cache_timestamp_records(self, documents, self.timestamp_stats),
                                         batch_size, apply_controls=apply_controls)


def source_schema(conn: sqlite3.Connection) -> dict:
    return {table: [tuple(row) for row in conn.execute(f"PRAGMA table_info({quoted(table)})")]
            for table in TABLES}


def create_empty_schema_clone(baseline: Path, target: Path) -> tuple[int, int]:
    """Keep four source-table schemas byte-compatible, not any baseline rows."""
    with closing(sqlite3.connect(baseline.as_uri() + "?mode=ro&immutable=1", uri=True)) as source:
        source.execute("PRAGMA query_only=ON")
        source.execute("PRAGMA mmap_size=0")
        source.execute("PRAGMA cache_size=-16384")
        original_schema = source_schema(source)
        maximum = source.execute("SELECT COALESCE(max(id),0) FROM memories").fetchone()[0]
        sequence = source.execute("SELECT seq FROM sqlite_sequence WHERE name='memories'").fetchone()
        floor = max(maximum, int(sequence[0] or 0) if sequence else 0)
        require(0 <= floor < SQLITE_MAX_INT, "baseline_ids_exhausted")
        with target.open("xb"):
            os.chmod(target, 0o600)
        with closing(sqlite3.connect(target)) as destination, destination:
            for table in (*TABLES, "memories_fts"):
                row = source.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                                     (table,)).fetchone()
                require(row is not None and isinstance(row[0], str), "missing_source_table_schema")
                destination.execute(row[0])
            destination.execute(
                """INSERT INTO sync_state(id,native_checkpoint_state_version,fts_schema_version,fts_ready)
                VALUES(1,?,?,0)""", (NATIVE_CHECKPOINT_STATE_VERSION, FTS_SCHEMA_VERSION))
            destination.execute("INSERT INTO sqlite_sequence(name,seq) VALUES('memories',?)", (floor,))
            require(source_schema(destination) == original_schema, "cloned_source_schema_mismatch")
            validate_schema(destination)
    return maximum, floor


class CompleteSnapshotStore(ExistingSnapshotStore):
    """Production merges on a fresh DB with bounded ID/timestamp preservation.

    No baseline row payload, generation, translation cache, controls or native
    state is carried forward. Only IDs and missing legacy ingestion timestamps
    can come from the baseline. New IDs allocate above its lifetime high-water
    mark, including previously deleted IDs tracked by sqlite_sequence.
    """

    def __init__(self, data_dir: str, baseline: Path, id_floor: int):
        self.id_floor = id_floor
        self.identity_stats = {"baseline_identities_restored": 0, "new_identities": 0,
                               "source_ingestion_timestamps": 0, "baseline_ingestion_timestamps": 0,
                               "source_cache_creation_timestamps": 0}
        self.baseline_conn = sqlite3.connect(baseline.as_uri() + "?mode=ro&immutable=1", uri=True)
        self.baseline_conn.row_factory = sqlite3.Row
        try:
            self.baseline_conn.execute("PRAGMA query_only=ON")
            self.baseline_conn.execute("PRAGMA cache_size=-16384")
            self.baseline_conn.execute("PRAGMA mmap_size=0")
            super().__init__(data_dir)
            with self.conn:
                self._create_generation_indexes_locked()
        except BaseException:
            if hasattr(self, "conn"):
                self.conn.close()
            self.baseline_conn.close()
            raise

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.baseline_conn.close()

    def ingest(self, docs: list[dict]) -> dict:
        require(self._bulk_restore_depth > 0, "complete_restore_requires_bulk_scope")
        # The production method allocates temporary IDs. The outer bulk scope
        # keeps ID-dependent FTS/native triggers disabled until remapping ends.
        seq = self.conn.execute("SELECT seq FROM sqlite_sequence WHERE name='memories'").fetchone()[0]
        require(seq <= SQLITE_MAX_INT - len(docs), "baseline_ids_exhausted")
        result = super().ingest(docs)
        created = [(doc, item) for doc, item in zip(docs, result["items"], strict=True)
                   if item["status"] == "created"]
        if not created:
            return result
        identities = [item["source_identity"] for _, item in created]
        marks = ",".join("?" for _ in identities)
        previous = {row["source_identity"]: row for row in self.baseline_conn.execute(
            f"SELECT id,source_identity,ingested_at FROM memories WHERE source_identity IN ({marks})",
            identities)}
        mapped = {}
        with self.conn:
            for doc, item in created:
                identity = item["source_identity"]
                old = previous.get(identity)
                target_id = old["id"] if old else item["id"]
                stamp = doc.get("ingested_at", (doc.get("metadata") or {}).get("ingested_at"))
                require(stamp is None or isinstance(stamp, str), "invalid_source_ingestion_timestamp")
                if stamp is not None:
                    self.identity_stats["source_ingestion_timestamps"] += 1
                elif old is not None:
                    stamp = old["ingested_at"]
                    self.identity_stats["baseline_ingestion_timestamps"] += 1
                changed = self.conn.execute(
                    "UPDATE memories SET id=?,ingested_at=COALESCE(?,ingested_at) WHERE id=? AND source_identity=?",
                    (target_id, stamp, item["id"], identity))
                require(changed.rowcount == 1, "complete_identity_remap_failed")
                mapped[identity] = target_id
                self.identity_stats["baseline_identities_restored" if old else "new_identities"] += 1
            maximum = self.conn.execute("SELECT COALESCE(max(id),0) FROM memories").fetchone()[0]
            self.conn.execute("UPDATE sqlite_sequence SET seq=? WHERE name='memories'",
                              (max(maximum, self.id_floor),))
        for item in result["items"]:
            if item["source_identity"] in mapped:
                item["id"] = mapped[item["source_identity"]]
        return result

    def restore_documents(self, documents, batch_size=500, *, apply_controls=True) -> int:
        return super().restore_documents(cache_timestamp_records(self, documents, self.identity_stats),
                                         batch_size, apply_controls=apply_controls)


@contextmanager
def bounded_restore_environment():
    values = {"FUNES_BULK_RESTORE_REBUILD_FTS": "false",
              "FUNES_BULK_RESTORE_CACHE_SIZE": "-16384",
              "FUNES_BULK_RESTORE_MMAP_SIZE": "0",
              "FUNES_BULK_RESTORE_SYNCHRONOUS": "FULL"}
    old = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def checked_records(records, counts: Counter, progress=None):
    for item in records:
        require(isinstance(item, dict), "non_object_tail_record")
        kind = item.get("_funes_record", "memory")
        require(isinstance(kind, str) and kind in RECORD_TYPES, "unsupported_tail_record_type")
        if kind == "memory":
            require(isinstance(item.get("source_identity"), str)
                    and bool(item["source_identity"])
                    and not any(ord(char) < 32 for char in item["source_identity"]),
                    "invalid_memory_identity")
            if "id" in item:
                require(type(item["id"]) is int and 0 < item["id"] <= SQLITE_MAX_INT,
                        "invalid_memory_id")
            raw = item.get("raw_text", item.get("text"))
            require(isinstance(raw, str) and bool(raw), "invalid_raw_text")
            require(item.get("retrieval_text") is None
                    or isinstance(item["retrieval_text"], str), "invalid_retrieval_text")
            metadata = item.get("metadata", item.get("metadata_json", {}))
            require(isinstance(metadata, dict), "invalid_memory_metadata")
            for field in ("retrieval_generation", "native_generation", "embedding_generation"):
                value = item.get(field, metadata.get(field))
                require(value is None or (type(value) is int and 0 <= value <= SQLITE_MAX_INT),
                        "invalid_memory_generation")
        elif kind == "reindex_control":
            require(type(item.get("generation")) is int
                    and 0 < item["generation"] <= SQLITE_MAX_INT
                    and isinstance(item.get("scope"), str)
                    and item["scope"] in REINDEX_SCOPES, "invalid_reindex_control")
        elif kind == "translation_cache":
            require(all(isinstance(item.get(key), str) and item[key]
                        for key in ("query", "rewritten")), "invalid_translation_cache")
        else:
            require(type(item.get("revision")) is int
                    and 0 <= item["revision"] <= SQLITE_MAX_INT, "invalid_native_revision")
            if kind == "native_index_state":
                require(type(item.get("state_version")) is int
                        and item["state_version"] == NATIVE_CHECKPOINT_STATE_VERSION,
                        "unsupported_native_checkpoint_state_version")
                require(all(type(item.get(name, 0)) is int
                            and 0 <= item.get(name, 0) <= SQLITE_MAX_INT
                            for name in ("eligible", "indexed", "held", "invalid"))
                        and all(isinstance(item.get(name, ""), str) for name in ("profile", "memory")),
                        "invalid_native_checkpoint_fields")
                require(item.get("index_fingerprint") in (None, Store._native_state_fingerprint(item)),
                        "invalid_native_checkpoint_fingerprint")
            else:
                require(all(isinstance(item.get(name), str) and item[name]
                            for name in ("optimized_at", "fingerprint", "status"))
                        and type(item.get("index_layout_version", 0)) is int
                        and 0 <= item.get("index_layout_version", 0) <= SQLITE_MAX_INT
                        and all(item.get(name) is None or (type(item[name]) is int
                                                          and 0 <= item[name] <= SQLITE_MAX_INT)
                                for name in ("dimensions", "schema_version"))
                        and all(item.get(name) is None or isinstance(item[name], str)
                                for name in ("provider", "model", "memory", "index_fingerprint")),
                        "invalid_native_optimize_checkpoint")
        counts[kind] += 1
        if progress and sum(counts.values()) % 10000 == 0:
            progress({"phase": "replaying_records", "records_seen": sum(counts.values())})
        yield item


def verify_boundary(path: Path | None, expected_sha256: str | None, plan: dict) -> dict:
    """Verify binding/integrity of an externally audited fence, not live writers.

    The coordinator owns the truth of these assertions and must hash-pin the
    already verified evidence. A bare --final flag or plan boolean is not proof.
    """
    require((path is None) == (expected_sha256 is None), "boundary_path_and_hash_required")
    if path is None:
        return {"verified": False, "kind": "none"}
    require(is_sha(expected_sha256), "invalid_boundary_digest")
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected_sha256, "boundary_digest_mismatch")
    value = json.loads(payload)
    require(isinstance(value, dict) and value.get("version") == 1
            and value.get("kind") == "funes_source_write_boundary"
            and value.get("status") == "verified"
            and value.get("repo") == plan["repo"]
            and value.get("revision") == plan["revision"]
            and value.get("plan_sha256") == json_sha256(plan)
            and value.get("baseline_sqlite_sha256") == plan["baseline_sqlite_sha256"]
            and value.get("source_writes_fenced") is True
            and value.get("background_writers_stopped") is True
            and type(value.get("in_flight_source_writes")) is int
            and value["in_flight_source_writes"] == 0
            and value.get("pinned_after_fence") is True, "unverified_or_unbound_write_boundary")
    try:
        verified_at = datetime.fromisoformat(value["verified_at"].replace("Z", "+00:00"))
        require(verified_at.tzinfo is not None and verified_at <= datetime.now(timezone.utc),
                "invalid_boundary_timestamp")
    except (KeyError, TypeError, ValueError, AttributeError):
        raise TailSnapshotError("invalid_boundary_timestamp") from None
    return {"verified": True, "kind": "external_operator_attestation",
            "sha256": expected_sha256, "verified_at": value["verified_at"],
            "live_writer_state_checked_by_builder": False}


def build_tail_snapshot(baseline: Path, metadata: dict, plan: dict, cache: Path,
                        output_dir: Path, sync: SnapshotSync, *, batch_size: int = 50,
                        mode: str = "suffix",
                        boundary_evidence: Path | None = None,
                        boundary_sha256: str | None = None, progress=None) -> dict:
    """Copy/hash first, authenticate/replay serially, checkpoint/close/hash last."""
    validate_plan(metadata, plan, sync, mode)
    files = replay_files(plan, mode)
    require(type(batch_size) is int and 1 <= batch_size <= 500, "invalid_batch_size")
    boundary = verify_boundary(boundary_evidence, boundary_sha256, plan)
    require(not baseline.is_symlink() and baseline.is_file(), "unsafe_or_missing_baseline")
    baseline = baseline.resolve(strict=True)
    signature = file_signature(baseline)
    assert_frozen(baseline, signature)
    cache = cache.resolve(strict=True)
    for filename in files:
        checked_encrypted_path(cache, filename)
    require(not output_dir.is_symlink() and not output_dir.exists(), "output_already_exists")
    output_dir = output_dir.parent.resolve(strict=True) / output_dir.name
    with closing(sqlite3.connect(baseline.as_uri() + "?mode=ro&immutable=1", uri=True)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA cache_size=-16384")
        conn.execute("PRAGMA mmap_size=0")
        validate_schema(conn)
        baseline_schema = source_schema(conn)
        baseline_counts = {table: conn.execute(f"SELECT count(*) FROM {quoted(table)}").fetchone()[0]
                           for table in TABLES}
    assert_frozen(baseline, signature)
    output_dir.mkdir(mode=0o700)  # exclusive; an existing directory is never owned
    owned = output_dir.stat()
    store = None
    try:
        target = output_dir / "funes.sqlite3"
        if mode == "complete":
            if progress:
                progress({"phase": "hashing_baseline", "total_bytes": signature[2]})
            baseline_digest = sha256_file(baseline, progress, "hashing_baseline")
        else:
            if progress:
                progress({"phase": "copying_baseline", "total_bytes": signature[2]})
            digest = hashlib.sha256()
            copied = last = 0
            with baseline.open("rb") as source, target.open("xb") as destination:
                os.chmod(target, 0o600)
                while block := source.read(CHUNK):
                    destination.write(block)
                    digest.update(block)
                    copied += len(block)
                    if progress and copied - last >= 256 * 1024 * 1024:
                        progress({"phase": "copying_baseline", "bytes": copied})
                        last = copied
                destination.flush()
                os.fsync(destination.fileno())
            baseline_digest = digest.hexdigest()
        assert_frozen(baseline, signature)
        require(baseline_digest == plan["baseline_sqlite_sha256"], "baseline_digest_mismatch")
        identity_stats = None
        if mode == "complete":
            maximum, floor = create_empty_schema_clone(baseline, target)
            assert_frozen(baseline, signature)
        records = Counter()
        restored = 0
        with bounded_restore_environment():
            store = (CompleteSnapshotStore(str(output_dir), baseline, floor)
                     if mode == "complete" else SuffixSnapshotStore(str(output_dir)))
            store.begin_bulk_restore()
            with tempfile.TemporaryDirectory(prefix="plaintext-", dir=output_dir) as directory:
                for index, filename in enumerate(files):
                    encrypted = checked_encrypted_path(cache, filename)
                    before = file_signature(encrypted)
                    receipt = plan["downloads"][filename]
                    if progress:
                        progress({"phase": "authenticating_file", "file_index": index + 1,
                                  "total_files": len(files), "encrypted_bytes": before[2]})
                    require(before[2] == receipt["size"]
                            and sha256_file(encrypted) == receipt["sha256"],
                            "encrypted_download_digest_mismatch")
                    plaintext = Path(directory) / Path(filename).name.removesuffix(".enc")
                    try:
                        # Authentication completes before the first record is read.
                        sync._decrypt_file(encrypted, plaintext)
                        require(file_signature(encrypted) == before, "encrypted_download_changed")
                        restored += store.restore_documents(
                            checked_records(sync._iter_file(plaintext), records, progress),
                            batch_size, apply_controls=False,
                        )
                    finally:
                        plaintext.unlink(missing_ok=True)
                    require(file_signature(encrypted) == before, "encrypted_download_changed")
                    if progress and ((index + 1) % 25 == 0 or index + 1 == len(files)):
                        progress({"phase": "replaying", "files": index + 1,
                                  "total_files": len(files), "records": sum(records.values())})
            if progress:
                progress({"phase": "replaying_controls"})
            compacted = store.compact_reindex_controls(replay=True)
            controls = store.drain_reindex_controls(batch_size)
            if progress:
                progress({"phase": "rebuilding_native_state_and_secondary_indexes", "rebuild_fts": False})
            store.finish_bulk_restore(rebuild_fts=False)
            store.set_sync(last_error=None)
            counts = {table: store.conn.execute(f"SELECT count(*) FROM {quoted(table)}").fetchone()[0]
                      for table in TABLES}
            require(source_schema(store.conn) == baseline_schema, "output_source_schema_mismatch")
            if mode == "complete":
                identity_stats = dict(store.identity_stats)
                require(counts["memories"] == (identity_stats["baseline_identities_restored"]
                                               + identity_stats["new_identities"]),
                        "complete_authoritative_membership_mismatch")
                identity_stats.update(
                    baseline_max_id=maximum, new_id_floor=floor,
                    baseline_identities_absent=baseline_counts["memories"] - identity_stats["baseline_identities_restored"])
            else:
                timestamp_stats = dict(store.timestamp_stats)
            state = store.conn.execute(
                "SELECT native_index_revision,native_optimize_revision,fts_ready FROM sync_state WHERE id=1"
            ).fetchone()
            require(state["fts_ready"] == 0, "derived_fts_unexpectedly_ready")
            require(store.conn.execute("SELECT count(*) FROM reindex_controls WHERE applied_at IS NULL")
                    .fetchone()[0] == 0, "pending_reindex_controls")
            if progress:
                progress({"phase": "checkpointing_and_checking_sqlite"})
            require(tuple(store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()) == (0, 0, 0),
                    "sqlite_checkpoint_incomplete")
            require(store.conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete",
                    "sqlite_journal_not_closed")
            require([tuple(row) for row in store.conn.execute("PRAGMA quick_check")] == [("ok",)],
                    "sqlite_quick_check_failed")
            store.close()
            store = None
        require(not any(Path(str(target) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")),
                "output_has_sqlite_sidecars")
        assert_frozen(baseline, signature)
        if progress:
            progress({"phase": "hashing_completed_snapshot"})
        output_digest = sha256_file(target, progress, "hashing_completed_snapshot")
        assert_frozen(baseline, signature)
        active = [plan["manifest"]["snapshot"], *plan["manifest"]["deltas"], *plan["manifest"]["controls"]]
        blobs = {**metadata["source_receipt"]["blob_ids"], **plan["blob_ids"]}
        report = {
            "version": 1, "mode": mode, "built_at": utc_now(), "status": "built_not_imported",
            "revision": plan["revision"], "plan_sha256": json_sha256(plan),
            "baseline_sqlite_sha256": plan["baseline_sqlite_sha256"],
            "baseline_counts": baseline_counts, "sqlite_sha256": output_digest,
            "sqlite_size": target.stat().st_size, "counts": counts,
            "replayed_files": len(files), "records_by_type": dict(records),
            "restored_memory_changes": restored, "control_compaction": compacted,
            "control_replay": controls, "native_index_revision": state["native_index_revision"],
            "native_optimize_revision": state["native_optimize_revision"],
            "authenticated_decryption": True, "baseline_modified": False,
            "postgres_modified": False, "hub_modified": False, "native_lance_modified": False,
            "startup_schema_migrations_skipped": True, "fts_rebuilt": False, "fts_ready": False,
            "full_identifiers_rebuilt": mode == "complete", "second_full_identifier_scan": False,
            "identifier_policy": "production_ingest_per_record_only", "boundary_evidence": boundary,
            "final_write_boundary": boundary["verified"],
            "source_receipt": {"version": 1, "repo": plan["repo"], "revision": plan["revision"],
                               "manifest": plan["manifest"], "files": active,
                               "blob_ids": {name: blobs[name] for name in active}},
        }
        if mode == "suffix":
            report.update(
                tail_files=len(files), timestamp_preservation=timestamp_stats,
                timestamp_policy="existing_ingested_at_unchanged;first_source_ingested_at_else_restore_time;source_cache_created_at_else_restore_time",
                id_policy="baseline_ids_unchanged;new_ids_above_sqlite_sequence;remote_numeric_ids_ignored",
                production_restore_deviations=["preserve_serialized_source_timestamps"],
            )
        else:
            report.update(
                restore_files=len(files), identity_mapping=identity_stats,
                source_membership="complete_manifest_union_only",
                schema_cloned_without_baseline_rows=True,
                timestamp_policy="first_source_ingested_at_else_baseline_else_restore_time;source_cache_created_at_else_restore_time",
                production_restore_deviations=["preserve_baseline_identity_ids", "preserve_serialized_source_timestamps"],
            )
        write_json(output_dir / "snapshot-report.json", report)
        return report
    except BaseException:
        if store is not None:
            store.close()
        # No expensive derived rebuild on a failed replay: discard this owned
        # copy. Never delete baseline, downloads, or a pre-existing output.
        current = output_dir.lstat() if output_dir.exists() else None
        if (current and stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino)):
            shutil.rmtree(output_dir)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--mode", choices=("suffix", "complete"), default="suffix")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, help="must not exist; parent must exist")
    parser.add_argument("--prepare-complete-plan", action="store_true",
                        help="download entire manifest at --revision; --plan must not exist; does not build SQLite")
    parser.add_argument("--revision", help="full immutable Hub SHA, required only with --prepare-complete-plan")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--boundary-evidence", type=Path)
    parser.add_argument("--boundary-sha256", help="SHA-256 of externally verified boundary evidence bytes")
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if args.prepare_complete_plan:
            require(args.mode == "complete" and args.baseline is None and args.output_dir is None
                    and args.boundary_evidence is None and args.boundary_sha256 is None,
                    "invalid_complete_preparation_arguments")
            require(not args.plan.exists() and not args.plan.is_symlink(), "plan_already_exists")
            from huggingface_hub import HfApi, hf_hub_download
            sync = SnapshotSync(None, rebuild_fts=False)
            plan = prepare_complete_plan(
                json.loads(args.metadata.read_text()), args.cache_dir, args.revision, sync,
                HfApi(token=sync.token), hf_hub_download,
                progress=lambda item: print(json.dumps(item), flush=True))
            write_json(args.plan, plan)
            print(json.dumps({"phase": "complete_plan_prepared", "revision": plan["revision"],
                              "plan_sha256": json_sha256(plan), "files": len(plan["restore_files"]),
                              "final_write_boundary": False}), flush=True)
            return 0
        require(args.baseline is not None and args.output_dir is not None and args.revision is None,
                "baseline_and_output_required_revision_is_in_plan")
        report = build_tail_snapshot(
            args.baseline, json.loads(args.metadata.read_text()), json.loads(args.plan.read_text()),
            args.cache_dir, args.output_dir, SnapshotSync(None, rebuild_fts=False),
            batch_size=args.batch_size, mode=args.mode, boundary_evidence=args.boundary_evidence,
            boundary_sha256=args.boundary_sha256,
            progress=lambda item: print(json.dumps(item), flush=True),
        )
        print(json.dumps({"phase": "completed", "status": report["status"],
                          "sqlite_sha256": report["sqlite_sha256"], "counts": report["counts"],
                          "final_write_boundary": report["final_write_boundary"]}), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error_class": type(error).__name__,
                          "error": str(error) if isinstance(error, TailSnapshotError)
                          else "tail_snapshot_build_failed"}), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
