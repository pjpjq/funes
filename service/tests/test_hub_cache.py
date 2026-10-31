"""Tests for Hugging Face Hub cache reuse via HF Bucket mount."""
from __future__ import annotations
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

from service.hub_cache import (
    DEFAULT_INTERVAL,
    DEFAULT_MAX_BYTES,
    DEFAULT_PREFIX,
    MAX_MANIFEST_SIZE,
    HubCache,
    validate_hash,
    validate_prefix,
    validate_rel_path,
    validate_repo_name,
)


class MockHfApi:
    """Mock HfApi for tracking batch_bucket_files calls and optionally mirroring to mount."""

    def __init__(self, mount_dir: Path | None = None) -> None:
        self.calls: list[dict] = []
        self.mount_dir = mount_dir
        self.manifest_failure: Exception | None = None
        self.blob_failure: Exception | None = None
        self.in_upload = threading.Event()
        self.release_upload = threading.Event()
        self.block_on_upload = False

    def batch_bucket_files(
        self,
        bucket_id: str,
        add: list[tuple[str, str]] | None = None,
        delete: list[str] | None = None,
    ) -> None:
        add_items = list(add or [])
        delete_items = list(delete or [])
        self.calls.append(
            {
                "bucket_id": bucket_id,
                "add": add_items,
                "delete": delete_items,
            }
        )

        if self.block_on_upload:
            self.in_upload.set()
            self.release_upload.wait(timeout=5.0)

        is_manifest = any("manifest.json" in dest for _, dest in add_items)
        if is_manifest and self.manifest_failure is not None:
            raise self.manifest_failure
        if not is_manifest and self.blob_failure is not None:
            raise self.blob_failure

        if self.mount_dir is not None:
            for local_path, dest_path in add_items:
                target = self.mount_dir / dest_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_path, target)


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------


def test_repo_name_validation() -> None:
    assert validate_repo_name("models--meta-llama--Llama-2-7b")
    assert validate_repo_name("models--sentence-transformers--all-MiniLM-L6-v2")
    assert validate_repo_name("datasets--glue")
    assert validate_repo_name("models--bert-base-uncased.1_2")

    assert not validate_repo_name("models--")
    assert not validate_repo_name("models--foo/bar")
    assert not validate_repo_name("models--foo\bar")
    assert not validate_repo_name("models--foo..bar")
    assert not validate_repo_name("../models--foo")
    assert not validate_repo_name("spaces--my-space")
    assert not validate_repo_name("")


def test_hash_validation() -> None:
    assert validate_hash("a" * 40)
    assert validate_hash("0123456789abcdef0123456789abcdef01234567")
    assert validate_hash("b" * 64)

    assert not validate_hash("a" * 39)
    assert not validate_hash("a" * 41)
    assert not validate_hash("a" * 63)
    assert not validate_hash("a" * 65)
    assert not validate_hash("g" * 40)
    assert not validate_hash("../" + "a" * 37)
    assert not validate_hash("")


def test_rel_path_validation() -> None:
    assert validate_rel_path("config.json")
    assert validate_rel_path(".gitattributes")
    assert validate_rel_path("sub/folder/model.safetensors")
    assert validate_rel_path("a/b/c/d.bin")

    assert not validate_rel_path("")
    assert not validate_rel_path("/absolute/path")
    assert not validate_rel_path(r"\absolute\path")
    assert not validate_rel_path("../escape")
    assert not validate_rel_path("sub/../escape")
    assert not validate_rel_path("sub/./escape")
    assert not validate_rel_path(".")


def test_prefix_validation() -> None:
    assert validate_prefix("cache-v1")
    assert validate_prefix("custom_prefix/sub")
    assert validate_prefix("cache.2026-09-24")

    assert not validate_prefix("")
    assert not validate_prefix("/cache-v1")
    assert not validate_prefix("cache-v1/")
    assert not validate_prefix("../escape")
    assert not validate_prefix("cache/../escape")
    assert not validate_prefix("invalid@prefix")


# ---------------------------------------------------------------------------
# from_env Configuration Tests
# ---------------------------------------------------------------------------


def test_from_env_variations(tmp_path: Path) -> None:
    # 1. Missing bucket or mount -> None
    assert HubCache.from_env({}) is None
    assert HubCache.from_env({"FUNES_HUB_CACHE_BUCKET": "my-bucket"}) is None
    assert HubCache.from_env({"FUNES_HUB_CACHE_MOUNT": str(tmp_path)}) is None

    # 2. Defaults applied correctly
    env = {
        "FUNES_HUB_CACHE_BUCKET": "owner/bucket",
        "FUNES_HUB_CACHE_MOUNT": str(tmp_path / "mount"),
        "HF_HUB_CACHE": str(tmp_path / "cache"),
    }
    cache = HubCache.from_env(env)
    assert cache is not None
    assert cache.bucket_id == "owner/bucket"
    assert cache.mount_dir == (tmp_path / "mount").resolve()
    assert cache.cache_dir == (tmp_path / "cache").resolve()
    assert cache.interval == DEFAULT_INTERVAL
    assert cache.max_bytes == DEFAULT_MAX_BYTES
    assert cache.prefix == DEFAULT_PREFIX

    # 3. Custom settings and malformed fallback
    env_custom = {
        "FUNES_HUB_CACHE_BUCKET": "owner/bucket",
        "FUNES_HUB_CACHE_MOUNT": str(tmp_path / "mount"),
        "FUNES_HUB_CACHE_INTERVAL": "invalid_int",
        "FUNES_HUB_CACHE_MAX_BYTES": "invalid_bytes",
        "FUNES_HUB_CACHE_PREFIX": "my-prefix/v2",
        "HF_HOME": str(tmp_path / "hf_home"),
    }
    cache_custom = HubCache.from_env(env_custom)
    assert cache_custom is not None
    assert cache_custom.interval == DEFAULT_INTERVAL
    assert cache_custom.max_bytes == DEFAULT_MAX_BYTES
    assert cache_custom.prefix == "my-prefix/v2"
    assert cache_custom.cache_dir == (tmp_path / "hf_home" / "hub").resolve()


# ---------------------------------------------------------------------------
# Blocker 1: Lexical Snapshot Resolution without Remote Stat
# ---------------------------------------------------------------------------


def test_lexical_snapshot_resolution_no_remote_stat(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()
    repo = "models--test--model"
    blob_hash = "a" * 40
    rev = "b" * 40

    blobs_dir = cache_dir / repo / "blobs"
    blobs_dir.mkdir(parents=True)
    local_blob_symlink = blobs_dir / blob_hash
    # Target points to a remote mount path that does NOT exist locally
    nonexistent_remote_target = mount_dir / "cache-v1" / repo / "blobs" / blob_hash
    os.symlink(str(nonexistent_remote_target), str(local_blob_symlink))

    snap_dir = cache_dir / repo / "snapshots" / rev
    snap_dir.mkdir(parents=True)
    snap_file = snap_dir / "config.json"
    os.symlink("../../blobs/" + blob_hash, str(snap_file))

    api = MockHfApi(mount_dir)
    cache = HubCache("test-bucket", mount_dir, cache_dir, api=api)
    cache._published_blobs = {repo: {blob_hash: 100}}

    res = cache.checkpoint()
    assert res["status"] == "ok"
    assert cache._published_snapshots.get(repo, {}).get(rev, {}).get("config.json") == blob_hash
    # Blob is a symlink, so it was NOT treated as candidate for upload
    assert res["uploaded_files"] == 0


# ---------------------------------------------------------------------------
# Blocker 2: Staged Blob Retry on Manifest Upload Failure
# ---------------------------------------------------------------------------


def test_staged_blob_retry_on_manifest_failure(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()
    repo = "models--test--model"
    blob1_hash = "a" * 40
    blob2_hash = "b" * 40
    rev = "c" * 40

    blobs_dir = cache_dir / repo / "blobs"
    blobs_dir.mkdir(parents=True)
    (blobs_dir / blob1_hash).write_bytes(b"blob1_content")
    (blobs_dir / blob2_hash).write_bytes(b"blob2_content")

    snap_dir = cache_dir / repo / "snapshots" / rev
    snap_dir.mkdir(parents=True)
    os.symlink("../../blobs/" + blob1_hash, str(snap_dir / "config.json"))

    api = MockHfApi(mount_dir)
    api.manifest_failure = RuntimeError("Simulated manifest network glitch")
    cache = HubCache("test-bucket", mount_dir, cache_dir, api=api)

    # 1. First checkpoint: blobs upload succeeds, manifest upload fails
    res1 = cache.checkpoint()
    assert res1["status"] == "error"
    assert res1["error_class"] == "RuntimeError"
    assert res1["uploaded_files"] == 0
    # Staged blobs are preserved, not committed to published
    assert blob1_hash in cache._staged_blobs[repo]
    assert blob2_hash in cache._staged_blobs[repo]
    assert repo not in cache._published_blobs

    # 2. Second checkpoint: manifest upload succeeds
    api.manifest_failure = None
    res2 = cache.checkpoint()
    assert res2["status"] == "ok"
    assert res2["uploaded_files"] == 2
    # Staged blobs are now committed to published blobs
    assert not cache._staged_blobs
    assert blob1_hash in cache._published_blobs[repo]
    assert blob2_hash in cache._published_blobs[repo]
    assert cache._published_snapshots[repo][rev]["config.json"] == blob1_hash

    # Verify API calls: blob upload was done ONLY in first run, not repeated!
    blob_adds = [c for c in api.calls if not any("manifest.json" in d for _, d in c["add"])]
    manifest_adds = [c for c in api.calls if any("manifest.json" in d for _, d in c["add"])]
    assert len(blob_adds) == 1
    assert len(manifest_adds) == 2


# ---------------------------------------------------------------------------
# Blocker 3: Non-blocking status() During Network I/O
# ---------------------------------------------------------------------------


def test_status_non_blocking_during_network_io(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()
    repo = "models--test--model"
    blob_hash = "a" * 40

    blobs_dir = cache_dir / repo / "blobs"
    blobs_dir.mkdir(parents=True)
    (blobs_dir / blob_hash).write_bytes(b"data")

    api = MockHfApi(mount_dir)
    api.block_on_upload = True
    cache = HubCache("test-bucket", mount_dir, cache_dir, api=api)

    t = threading.Thread(target=cache.checkpoint, daemon=True)
    t.start()

    assert api.in_upload.wait(timeout=2.0)

    # status() called while batch_bucket_files is blocked in another thread
    t0 = time.monotonic()
    st = cache.status()
    duration = time.monotonic() - t0

    assert duration < 0.05, f"status() blocked for {duration:.4f}s"
    assert st["enabled"] is True

    api.release_upload.set()
    t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Blocker 4: Preserve Existing Local Files and Retain Metadata in restore()
# ---------------------------------------------------------------------------


def test_restore_preserves_existing_local_metadata(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()
    repo = "models--test--model"
    blob_hash = "a" * 40
    rev = "b" * 40

    # Local cache already has a real file and snapshot
    blobs_dir = cache_dir / repo / "blobs"
    blobs_dir.mkdir(parents=True)
    local_blob = blobs_dir / blob_hash
    local_blob.write_bytes(b"existing_local_bytes")

    snap_dir = cache_dir / repo / "snapshots" / rev
    snap_dir.mkdir(parents=True)
    snap_file = snap_dir / "config.json"
    os.symlink("../../blobs/" + blob_hash, str(snap_file))

    # Mount contains manifest
    manifest_dir = mount_dir / "cache-v1"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "version": 1,
        "blobs": {repo: {blob_hash: {"size": 20}}},
        "snapshots": {repo: {rev: {"config.json": blob_hash}}},
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    api = MockHfApi(mount_dir)
    cache = HubCache("test-bucket", mount_dir, cache_dir, api=api)

    res_restore = cache.restore()
    assert res_restore["status"] == "ok"
    assert res_restore["restored_links"] == 0
    # Content untouched
    assert local_blob.read_bytes() == b"existing_local_bytes"
    # Metadata retained in published blobs and snapshots
    assert blob_hash in cache._published_blobs[repo]
    assert cache._published_snapshots[repo][rev]["config.json"] == blob_hash

    # Checkpoint does NOT re-upload existing blob
    res_chk = cache.checkpoint()
    assert res_chk["status"] == "ok"
    assert res_chk["uploaded_files"] == 0
    assert len(api.calls) == 0


# ---------------------------------------------------------------------------
# Blocker 5: Path Traversal, Symlink Escapes, and Oversized Manifest
# ---------------------------------------------------------------------------


def test_security_traversal_intermediate_symlinks(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()
    outside_dir = (tmp_path / "outside").resolve()
    outside_dir.mkdir()

    repo = "models--test--model"
    blob_hash = "a" * 40
    rev = "b" * 40

    snap_rev_dir = cache_dir / repo / "snapshots" / rev
    snap_rev_dir.mkdir(parents=True)
    # Intermediate directory is a symlink pointing outside
    os.symlink(str(outside_dir), str(snap_rev_dir / "bad_link"))

    manifest_dir = mount_dir / "cache-v1"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "version": 1,
        "blobs": {repo: {blob_hash: {"size": 10}}},
        "snapshots": {
            repo: {
                rev: {
                    "bad_link/evil.txt": blob_hash,
                    "../../escape.txt": blob_hash,
                }
            }
        },
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cache = HubCache("test-bucket", mount_dir, cache_dir)
    res = cache.restore()
    assert res["status"] == "ok"
    assert not (outside_dir / "evil.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_oversized_manifest_rejection(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()

    manifest_dir = mount_dir / "cache-v1"
    manifest_dir.mkdir(parents=True)
    large_manifest = manifest_dir / "manifest.json"
    with open(large_manifest, "wb") as f:
        f.truncate(MAX_MANIFEST_SIZE + 1024)

    cache = HubCache("test-bucket", mount_dir, cache_dir)
    res = cache.restore()
    assert res["status"] == "error"
    assert res["error_class"] == "ManifestOversizedError"
    assert cache.status()["error_class"] == "ManifestOversizedError"


def test_malformed_manifest_rejection(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()
    manifest_dir = mount_dir / "cache-v1"
    manifest_dir.mkdir(parents=True)

    # 1. Invalid JSON
    (manifest_dir / "manifest.json").write_text("{bad-json", encoding="utf-8")
    cache = HubCache("test-bucket", mount_dir, cache_dir)
    res = cache.restore()
    assert res["status"] == "error"
    assert res["error_class"] == "JSONDecodeError"

    # 2. Invalid version
    (manifest_dir / "manifest.json").write_text('{"version": 2}', encoding="utf-8")
    res2 = cache.restore()
    assert res2["status"] == "error"
    assert res2["error_class"] == "ManifestValidationError"


# ---------------------------------------------------------------------------
# Roundtrip: Cold Checkpoint -> Wipe Local Cache -> Restore Zero Copy
# ---------------------------------------------------------------------------


def test_roundtrip_cold_checkpoint_and_restore_zero_copy(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()
    repo = "models--test--model"
    blob_hash = "a" * 40
    rev = "b" * 40

    # 1. Seed local cache with real blob and snapshot
    blobs_dir = cache_dir / repo / "blobs"
    blobs_dir.mkdir(parents=True)
    (blobs_dir / blob_hash).write_bytes(b"initial_weights_data")

    snap_dir = cache_dir / repo / "snapshots" / rev
    snap_dir.mkdir(parents=True)
    os.symlink("../../blobs/" + blob_hash, str(snap_dir / "weights.bin"))

    api = MockHfApi(mount_dir)
    cache1 = HubCache("test-bucket", mount_dir, cache_dir, api=api)

    # 2. Checkpoint uploads to bucket
    chk_res1 = cache1.checkpoint()
    assert chk_res1["status"] == "ok"
    assert chk_res1["uploaded_files"] == 1
    assert chk_res1["uploaded_bytes"] == len(b"initial_weights_data")

    # 3. Wipe local cache completely (simulating fresh container)
    shutil.rmtree(cache_dir)
    assert not cache_dir.exists()

    # 4. Restore in fresh container
    cache2 = HubCache("test-bucket", mount_dir, cache_dir, api=api)
    restore_res = cache2.restore()
    assert restore_res["status"] == "ok"
    assert restore_res["restored_links"] == 2

    # Zero blob copying: restored blob is a symlink pointing to mount
    restored_blob = cache_dir / repo / "blobs" / blob_hash
    assert os.path.islink(restored_blob)
    assert os.readlink(restored_blob) == str(mount_dir / "cache-v1" / repo / "blobs" / blob_hash)

    # Snapshot symlink points to ../../blobs/<hash>
    restored_snap = cache_dir / repo / "snapshots" / rev / "weights.bin"
    assert os.path.islink(restored_snap)
    assert os.readlink(restored_snap) == "../../blobs/" + blob_hash

    # Reading through symlink chain yields original contents
    assert restored_snap.read_bytes() == b"initial_weights_data"

    # 5. Subsequent checkpoint does NOT re-upload anything
    chk_res2 = cache2.checkpoint()
    assert chk_res2["status"] == "ok"
    assert chk_res2["uploaded_files"] == 0


# ---------------------------------------------------------------------------
# Batching and Boundary Tests
# ---------------------------------------------------------------------------


def test_batching_upload_boundary(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()
    repo = "models--test--model"

    blobs_dir = cache_dir / repo / "blobs"
    blobs_dir.mkdir(parents=True)

    # Create 20 distinct blobs (> DEFAULT_BATCH_SIZE of 16)
    for i in range(20):
        h = f"{i:040x}"
        (blobs_dir / h).write_bytes(f"content_{i}".encode("utf-8"))

    api = MockHfApi(mount_dir)
    cache = HubCache("test-bucket", mount_dir, cache_dir, api=api)

    res = cache.checkpoint()
    assert res["status"] == "ok"
    assert res["uploaded_files"] == 20

    # Check batching: 2 blob batches (16 + 4) + 1 manifest batch
    blob_batches = [c for c in api.calls if not any("manifest.json" in d for _, d in c["add"])]
    assert len(blob_batches) == 2
    assert len(blob_batches[0]["add"]) == 16
    assert len(blob_batches[1]["add"]) == 4


def test_max_bytes_boundary(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()
    repo = "models--test--model"

    blobs_dir = cache_dir / repo / "blobs"
    blobs_dir.mkdir(parents=True)

    # Blob 1: 500 bytes, Blob 2: 700 bytes
    h1 = "1" * 40
    h2 = "2" * 40
    (blobs_dir / h1).write_bytes(b"x" * 500)
    (blobs_dir / h2).write_bytes(b"y" * 700)

    api = MockHfApi(mount_dir)
    # Set max_bytes to 600 -> only blob 1 fits
    cache = HubCache("test-bucket", mount_dir, cache_dir, max_bytes=600, api=api)

    res = cache.checkpoint()
    assert res["status"] == "ok"
    assert res["uploaded_files"] == 1
    assert h1 in cache._published_blobs[repo]
    assert h2 not in cache._published_blobs[repo]


# ---------------------------------------------------------------------------
# Worker Thread Lifecycle Tests
# ---------------------------------------------------------------------------


def test_lifecycle_start_stop(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()

    cache = HubCache("test-bucket", mount_dir, cache_dir, interval=100)
    assert cache.status()["state"] == "idle"

    cache.start()
    assert cache.status()["state"] == "running"

    # Idempotent start
    cache.start()
    assert cache.status()["state"] == "running"

    cache.stop(timeout=1.0)
    assert cache.status()["state"] == "stopped"

    # Idempotent stop
    cache.stop(timeout=1.0)
    assert cache.status()["state"] == "stopped"


def test_restore_rejects_negative_size_and_non_str(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()
    repo = "models--test--model"
    blob_hash = "a" * 40

    manifest_dir = mount_dir / "cache-v1"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "version": 1,
        "blobs": {
            repo: {
                blob_hash: {"size": -100},
                "invalid_hash": {"size": 20},
                ("b" * 40): {"size": "not_an_int"},
                ("c" * 40): {"size": True},
            }
        },
        "snapshots": {},
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cache = HubCache("test-bucket", mount_dir, cache_dir)
    res = cache.restore()
    assert res["status"] == "ok"
    assert res["restored_links"] == 0
    assert not (cache_dir / repo / "blobs" / blob_hash).exists()


def test_restore_rejects_parent_symlinks(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()
    outside_dir = (tmp_path / "outside").resolve()
    outside_dir.mkdir()

    repo = "models--test--model"
    blob_hash = "a" * 40

    # Local repo dir is a symlink pointing outside
    cache_dir.mkdir(parents=True)
    os.symlink(str(outside_dir), str(cache_dir / repo))

    manifest_dir = mount_dir / "cache-v1"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "version": 1,
        "blobs": {repo: {blob_hash: {"size": 100}}},
        "snapshots": {},
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cache = HubCache("test-bucket", mount_dir, cache_dir)
    res = cache.restore()
    assert res["status"] == "ok"
    assert res["restored_links"] == 0
    # Outside dir must remain untouched
    assert not (outside_dir / "blobs").exists()

def test_checkpoint_nested_snapshot_and_dir_symlink_ignored(tmp_path: Path) -> None:
    cache_dir = (tmp_path / "cache").resolve()
    mount_dir = (tmp_path / "mount").resolve()
    repo = "models--test--model"
    blob_hash = "a" * 40
    rev = "b" * 40

    blobs_dir = cache_dir / repo / "blobs"
    blobs_dir.mkdir(parents=True)
    (blobs_dir / blob_hash).write_bytes(b"blob_bytes")

    snap_rev_dir = cache_dir / repo / "snapshots" / rev
    sub_dir = snap_rev_dir / "nested" / "sub"
    sub_dir.mkdir(parents=True)
    # Nested snapshot file
    os.symlink(os.path.relpath(blobs_dir / blob_hash, sub_dir), str(sub_dir / "model.safetensors"))

    # Dir symlink pointing outside or somewhere else
    outside_dir = (tmp_path / "outside").resolve()
    outside_dir.mkdir()
    os.symlink(str(outside_dir), str(snap_rev_dir / "symlink_dir"))

    api = MockHfApi(mount_dir)
    cache = HubCache("test-bucket", mount_dir, cache_dir, api=api)
    res = cache.checkpoint()
    assert res["status"] == "ok"
    assert res["uploaded_files"] == 1
    # Check that nested snapshot was captured
    assert cache._published_snapshots[repo][rev]["nested/sub/model.safetensors"] == blob_hash
    # Dir symlink is not included
    assert "symlink_dir" not in cache._published_snapshots[repo][rev]
