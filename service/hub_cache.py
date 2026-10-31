"""Minimal Python module for cross-container HF Hub cache reuse via HF Bucket mount."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping

DEFAULT_INTERVAL = 300
DEFAULT_MAX_BYTES = 32 * 1024 * 1024 * 1024  # 32 GiB
DEFAULT_PREFIX = "cache-v1"
DEFAULT_BATCH_SIZE = 16
MAX_MANIFEST_SIZE = 50 * 1024 * 1024  # 50 MiB

ENV_BUCKET = "FUNES_HUB_CACHE_BUCKET"
ENV_MOUNT = "FUNES_HUB_CACHE_MOUNT"
ENV_INTERVAL = "FUNES_HUB_CACHE_INTERVAL"
ENV_MAX_BYTES = "FUNES_HUB_CACHE_MAX_BYTES"
ENV_PREFIX = "FUNES_HUB_CACHE_PREFIX"

HASH_REGEX = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
REPO_REGEX = re.compile(r"^(models|datasets)--[a-zA-Z0-9_.-]+$")
PREFIX_REGEX = re.compile(r"^[a-zA-Z0-9_.-]+(/[a-zA-Z0-9_.-]+)*$")


def validate_prefix(prefix: str) -> bool:
    if not prefix or "\x00" in prefix or ".." in prefix:
        return False
    if prefix.startswith("/") or prefix.endswith("/"):
        return False
    norm = os.path.normpath(prefix)
    if norm != prefix or norm.startswith("."):
        return False
    return bool(PREFIX_REGEX.match(prefix))


def validate_repo_name(name: str) -> bool:
    if not name or "\x00" in name or "/" in name or "\\" in name:
        return False
    return bool(REPO_REGEX.match(name)) and ".." not in name


def validate_hash(value: str) -> bool:
    if not value or "\x00" in value:
        return False
    return bool(HASH_REGEX.match(value))


def validate_rel_path(rel_path: Any) -> bool:
    if not isinstance(rel_path, str):
        return False
    if not rel_path or chr(0) in rel_path or '..' in rel_path:
        return False
    if any(ord(c) < 32 for c in rel_path):
        return False
    if rel_path.startswith('/') or rel_path.startswith('\\'):
        return False
    parts = rel_path.replace('\\', '/').split('/')
    if any(p in ('', '.', '..') for p in parts):
        return False
    norm = os.path.normpath(rel_path)
    if os.path.isabs(norm) or norm.startswith('..') or norm == '.':
        return False
    parts = norm.replace('\\', '/').split('/')
    if any(p in ('', '.', '..') for p in parts):
        return False
    return True

class HubCache:
    """Manages syncing local Hugging Face Hub cache with a mounted HF Bucket."""

    def __init__(
        self,
        bucket_id: str,
        mount_dir: str | Path,
        cache_dir: str | Path,
        interval: int = DEFAULT_INTERVAL,
        max_bytes: int = DEFAULT_MAX_BYTES,
        prefix: str = DEFAULT_PREFIX,
        api: Any | None = None,
    ) -> None:
        self.bucket_id = str(bucket_id).strip()
        if not self.bucket_id:
            raise ValueError("bucket_id must not be empty")

        clean_prefix = str(prefix).strip("/") or DEFAULT_PREFIX
        if not validate_prefix(clean_prefix):
            raise ValueError(f"Invalid cache prefix: {prefix!r}")
        self.prefix = clean_prefix

        self.mount_dir = Path(mount_dir).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.interval = max(1, int(interval))
        self.max_bytes = max(0, int(max_bytes))
        self._api = api

        self._published_blobs: dict[str, dict[str, int]] = {}
        self._staged_blobs: dict[str, dict[str, int]] = {}
        self._published_snapshots: dict[str, dict[str, dict[str, str]]] = {}
        self._uploaded_bytes: int = 0
        self._uploaded_count: int = 0
        self._restored_links: int = 0
        self._restore_seconds: float = 0.0
        self._last_checkpoint_at: float | None = None

        self._state: str = "idle"
        self._last_error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._checkpoint_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._lock = self._state_lock

    @property
    def api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi()
        return self._api

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> HubCache | None:
        env = os.environ if environ is None else environ
        bucket = env.get(ENV_BUCKET, "").strip()
        mount = env.get(ENV_MOUNT, "").strip()
        if not bucket or not mount:
            return None

        interval_raw = env.get(ENV_INTERVAL, "").strip()
        try:
            interval = int(interval_raw) if interval_raw else DEFAULT_INTERVAL
        except ValueError:
            interval = DEFAULT_INTERVAL

        max_bytes_raw = env.get(ENV_MAX_BYTES, "").strip()
        try:
            max_bytes = int(max_bytes_raw) if max_bytes_raw else DEFAULT_MAX_BYTES
        except ValueError:
            max_bytes = DEFAULT_MAX_BYTES

        prefix = env.get(ENV_PREFIX, DEFAULT_PREFIX).strip() or DEFAULT_PREFIX

        if env.get("HF_HUB_CACHE"):
            cache_dir = env["HF_HUB_CACHE"]
        elif env.get("HF_HOME"):
            cache_dir = os.path.join(env["HF_HOME"], "hub")
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

        return cls(
            bucket_id=bucket,
            mount_dir=mount,
            cache_dir=cache_dir,
            interval=interval,
            max_bytes=max_bytes,
            prefix=prefix,
        )

    def restore(self) -> dict:
        """Restore local cache symlinks from mount manifest without reading remote blobs."""
        t0 = time.monotonic()
        try:
            with self._checkpoint_lock:
                return self._restore_unlocked()
        finally:
            with self._state_lock:
                self._restore_seconds = round(time.monotonic() - t0, 4)

    def _restore_unlocked(self) -> dict:
            manifest_path = self.mount_dir / self.prefix / "manifest.json"
            if not os.path.lexists(manifest_path):
                return {"restored_links": 0, "status": "no_manifest"}

            try:
                size = os.path.getsize(manifest_path)
                if size > MAX_MANIFEST_SIZE:
                    with self._state_lock:
                        self._last_error = "ManifestOversizedError"
                    return {
                        "restored_links": 0,
                        "status": "error",
                        "error_class": "ManifestOversizedError",
                    }
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception as e:
                err = e.__class__.__name__
                with self._state_lock:
                    self._last_error = err
                return {
                    "restored_links": 0,
                    "status": "error",
                    "error_class": err,
                }

            if not isinstance(manifest, dict) or manifest.get("version") != 1:
                err = "ManifestValidationError"
                with self._state_lock:
                    self._last_error = err
                return {
                    "restored_links": 0,
                    "status": "error",
                    "error_class": err,
                }

            blobs_section = manifest.get("blobs", {})
            snapshots_section = manifest.get("snapshots", {})
            if not isinstance(blobs_section, dict) or not isinstance(snapshots_section, dict):
                err = "ManifestValidationError"
                with self._state_lock:
                    self._last_error = err
                return {
                    "restored_links": 0,
                    "status": "error",
                    "error_class": err,
                }

            restored_count = 0
            published_blobs_to_add: dict[str, dict[str, int]] = {}
            published_snapshots_to_add: dict[str, dict[str, dict[str, str]]] = {}

            # 1. Restore blob symlinks -> mounted blob files
            # Performance constraint: do NOT stat or read mounted blob files!
            for repo, repo_blobs in blobs_section.items():
                if not validate_repo_name(repo) or not isinstance(repo_blobs, dict):
                    continue
                repo_dir = self.cache_dir / repo
                if repo_dir.is_symlink():
                    continue
                local_blobs_dir = repo_dir / "blobs"
                if local_blobs_dir.is_symlink():
                    continue
                local_blobs_dir.mkdir(parents=True, exist_ok=True)
                if repo_dir.is_symlink() or local_blobs_dir.is_symlink():
                    continue

                for blob_hash, meta in repo_blobs.items():
                    if not validate_hash(blob_hash):
                        continue
                    if not isinstance(meta, dict):
                        continue
                    size = meta.get("size")
                    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                        continue
                    size_int = size
                    # Always retain metadata even if local blob already exists
                    published_blobs_to_add.setdefault(repo, {})[blob_hash] = size_int

                    local_blob_path = local_blobs_dir / blob_hash
                    if os.path.lexists(local_blob_path):
                        # Local real blob or symlink already exists; do not overwrite
                        continue

                    mounted_blob_path = (
                        self.mount_dir / self.prefix / repo / "blobs" / blob_hash
                    )
                    os.symlink(str(mounted_blob_path), str(local_blob_path))
                    restored_count += 1

            # 2. Restore snapshot symlinks -> local blob symlinks
            for repo, repo_snaps in snapshots_section.items():
                if not validate_repo_name(repo) or not isinstance(repo_snaps, dict):
                    continue
                repo_dir = self.cache_dir / repo
                if repo_dir.is_symlink():
                    continue
                snapshots_dir = repo_dir / "snapshots"
                if snapshots_dir.is_symlink():
                    continue
                for rev, files in repo_snaps.items():
                    if not validate_hash(rev) or not isinstance(files, dict):
                        continue
                    rev_dir = snapshots_dir / rev
                    if rev_dir.is_symlink():
                        continue
                    for rel_path, blob_hash in files.items():
                        if not validate_rel_path(rel_path) or not validate_hash(blob_hash):
                            continue

                        local_snapshot_file = rev_dir / rel_path
                        # Strict lexical boundary check against path traversal
                        norm_full = os.path.abspath(os.path.normpath(str(local_snapshot_file)))
                        norm_rev = os.path.abspath(str(rev_dir))
                        if not (norm_full == norm_rev or norm_full.startswith(norm_rev + os.sep)):
                            continue

                        # Check intermediate snapshot directories are not symlinks
                        curr = rev_dir
                        escaped_symlink = False
                        for part in Path(rel_path).parent.parts:
                            curr = curr / part
                            if os.path.islink(curr):
                                escaped_symlink = True
                                break
                        if escaped_symlink:
                            continue

                        # Always retain snapshot metadata even if local file already exists
                        published_snapshots_to_add.setdefault(repo, {}).setdefault(
                            rev, {}
                        )[rel_path] = blob_hash

                        if os.path.lexists(local_snapshot_file):
                            continue

                        local_blob_path = repo_dir / "blobs" / blob_hash
                        local_snapshot_file.parent.mkdir(parents=True, exist_ok=True)
                        rel_target = os.path.relpath(
                            local_blob_path, local_snapshot_file.parent
                        )
                        os.symlink(rel_target, str(local_snapshot_file))
                        restored_count += 1

            with self._state_lock:
                for repo, hashes in published_blobs_to_add.items():
                    self._published_blobs.setdefault(repo, {}).update(hashes)
                for repo, snaps in published_snapshots_to_add.items():
                    for rev, files in snaps.items():
                        self._published_snapshots.setdefault(repo, {}).setdefault(rev, {}).update(files)

                self._restored_links += restored_count
                self._uploaded_bytes = sum(
                    sum(hashes.values()) for hashes in self._published_blobs.values()
                )
                self._uploaded_count = sum(
                    len(hashes) for hashes in self._published_blobs.values()
                )

            return {"restored_links": restored_count, "status": "ok"}

    def _load_manifest_metadata(self) -> None:
        manifest_path = self.mount_dir / self.prefix / "manifest.json"
        if not os.path.lexists(manifest_path):
            return
        try:
            if os.path.getsize(manifest_path) > MAX_MANIFEST_SIZE:
                return
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if not isinstance(manifest, dict) or manifest.get("version") != 1:
                return
            blobs_sec = manifest.get("blobs", {})
            snapshots_sec = manifest.get("snapshots", {})
            if not isinstance(blobs_sec, dict) or not isinstance(snapshots_sec, dict):
                return

            loaded_blobs: dict[str, dict[str, int]] = {}
            for repo, hashes in blobs_sec.items():
                if not validate_repo_name(repo) or not isinstance(hashes, dict):
                    continue
                for h, m in hashes.items():
                    if not validate_hash(h) or not isinstance(m, dict):
                        continue
                    sz = m.get("size")
                    if isinstance(sz, bool) or not isinstance(sz, int) or sz < 0:
                        continue
                    loaded_blobs.setdefault(repo, {})[h] = sz

            loaded_snapshots: dict[str, dict[str, dict[str, str]]] = {}
            for repo, snaps in snapshots_sec.items():
                if not validate_repo_name(repo) or not isinstance(snaps, dict):
                    continue
                for rev, files in snaps.items():
                    if not validate_hash(rev) or not isinstance(files, dict):
                        continue
                    for rel, bh in files.items():
                        if validate_rel_path(rel) and validate_hash(bh):
                            loaded_snapshots.setdefault(repo, {}).setdefault(rev, {})[rel] = bh

            with self._state_lock:
                for repo, hashes in loaded_blobs.items():
                    self._published_blobs.setdefault(repo, {}).update(hashes)
                for repo, snaps in loaded_snapshots.items():
                    for rev, files in snaps.items():
                        self._published_snapshots.setdefault(repo, {}).setdefault(rev, {}).update(files)
                self._uploaded_bytes = sum(
                    sum(h.values()) for h in self._published_blobs.values()
                )
                self._uploaded_count = sum(
                    len(h) for h in self._published_blobs.values()
                )
        except Exception:
            pass

    def checkpoint(self) -> dict:
        """Scan local cache, upload new blobs to bucket in batches <= 16, then publish manifest."""
        with self._checkpoint_lock:
            return self._checkpoint_unlocked()

    def _checkpoint_unlocked(self) -> dict:
        if not self.cache_dir.exists():
            return {"uploaded_files": 0, "uploaded_bytes": 0, "status": "ok", "error_class": None}

        with self._state_lock:
            needs_seed = not self._published_blobs
        if needs_seed:
            self._load_manifest_metadata()

        with self._state_lock:
            published_map = {
                repo: dict(hashes) for repo, hashes in self._published_blobs.items()
            }
            staged_map = {
                repo: dict(hashes) for repo, hashes in self._staged_blobs.items()
            }
            current_total_bytes = self._uploaded_bytes + sum(
                sum(hashes.values()) for hashes in self._staged_blobs.values()
            )

        candidates: list[tuple[str, str, str, int]] = []

        # Scan repos in local cache
        for entry in self.cache_dir.iterdir():
            if entry.is_symlink() or not entry.is_dir() or not validate_repo_name(entry.name):
                continue
            repo = entry.name
            blobs_dir = entry / "blobs"
            if blobs_dir.is_symlink() or not blobs_dir.is_dir():
                continue

            for blob_entry in blobs_dir.iterdir():
                # Only local regular files (not symlinks, not sockets)
                if blob_entry.is_symlink() or not blob_entry.is_file():
                    continue
                blob_hash = blob_entry.name
                if not validate_hash(blob_hash):
                    continue
                if blob_hash in published_map.get(repo, {}) or blob_hash in staged_map.get(repo, {}):
                    continue

                try:
                    blob_size = blob_entry.stat().st_size
                except OSError:
                    continue

                if current_total_bytes + blob_size > self.max_bytes:
                    continue

                candidates.append((repo, blob_hash, str(blob_entry), blob_size))
                current_total_bytes += blob_size

        upload_failed = False

        # Batch uploads bounded to at most DEFAULT_BATCH_SIZE (16)
        for i in range(0, len(candidates), DEFAULT_BATCH_SIZE):
            if self._stop_event.is_set():
                break
            batch = candidates[i : i + DEFAULT_BATCH_SIZE]
            add_payload = [
                (local_path, f"{self.prefix}/{repo}/blobs/{b_hash}")
                for repo, b_hash, local_path, _ in batch
            ]
            try:
                # Network I/O strictly outside _state_lock
                self.api.batch_bucket_files(self.bucket_id, add=add_payload)
                with self._state_lock:
                    for repo, b_hash, _, b_size in batch:
                        self._staged_blobs.setdefault(repo, {})[b_hash] = b_size
            except Exception as e:
                with self._state_lock:
                    self._last_error = e.__class__.__name__
                upload_failed = True
                break

        # Scan snapshots purely lexically without traversing or stating remote files
        with self._state_lock:
            all_known_blobs = {
                repo: set(self._published_blobs.get(repo, {}).keys())
                | set(self._staged_blobs.get(repo, {}).keys())
                for repo in set(self._published_blobs.keys()) | set(self._staged_blobs.keys())
            }

        current_snapshots: dict[str, dict[str, dict[str, str]]] = {}
        for entry in self.cache_dir.iterdir():
            if entry.is_symlink() or not entry.is_dir() or not validate_repo_name(entry.name):
                continue
            repo = entry.name
            snapshots_dir = entry / "snapshots"
            if snapshots_dir.is_symlink() or not snapshots_dir.is_dir():
                continue

            blobs_dir_abs = os.path.abspath(str(entry / "blobs"))

            for rev_entry in snapshots_dir.iterdir():
                if rev_entry.is_symlink() or not rev_entry.is_dir() or not validate_hash(rev_entry.name):
                    continue
                rev = rev_entry.name

                # Scan snapshots using os.scandir with is_dir(follow_symlinks=False)
                # to strictly avoid stat() on remote mount symlinks.
                def _scan_snapshots(curr_dir: str, rel_prefix: str) -> None:
                    try:
                        with os.scandir(curr_dir) as it:
                            for item in it:
                                iname = item.name
                                if not iname or "\x00" in iname or iname in (".", ".."):
                                    continue
                                try:
                                    is_sym = item.is_symlink()
                                except OSError:
                                    continue

                                if is_sym:
                                    # Snapshot file: strictly readlink without remote stat
                                    try:
                                        target = os.readlink(item.path)
                                    except OSError:
                                        continue
                                    norm_target = os.path.abspath(
                                        os.path.normpath(os.path.join(curr_dir, target))
                                    )
                                    if os.path.dirname(norm_target) != blobs_dir_abs:
                                        continue
                                    target_hash = os.path.basename(norm_target)
                                    if not validate_hash(target_hash):
                                        continue
                                    if target_hash not in all_known_blobs.get(repo, set()):
                                        continue
                                    rel_path = f"{rel_prefix}/{iname}" if rel_prefix else iname
                                    if not validate_rel_path(rel_path):
                                        continue
                                    current_snapshots.setdefault(repo, {}).setdefault(rev, {})[
                                        rel_path
                                    ] = target_hash
                                else:
                                    try:
                                        if item.is_dir(follow_symlinks=False):
                                            sub_rel = f"{rel_prefix}/{iname}" if rel_prefix else iname
                                            _scan_snapshots(item.path, sub_rel)
                                    except OSError:
                                        continue
                    except OSError:
                        pass

                _scan_snapshots(str(rev_entry), "")

        manifest_committed = False
        committed_files = 0
        committed_bytes = 0

        # Publish manifest only if all batch uploads succeeded and changes exist
        if not upload_failed:
            with self._state_lock:
                has_staged = bool(self._staged_blobs)
                snapshots_changed = (current_snapshots != self._published_snapshots)
                should_publish = has_staged or snapshots_changed

                if should_publish:
                    manifest_blobs: dict[str, dict[str, dict[str, int]]] = {}
                    for repo, hashes in self._published_blobs.items():
                        manifest_blobs.setdefault(repo, {}).update(
                            {h: {"size": sz} for h, sz in hashes.items()}
                        )
                    for repo, hashes in self._staged_blobs.items():
                        manifest_blobs.setdefault(repo, {}).update(
                            {h: {"size": sz} for h, sz in hashes.items()}
                        )
                    staged_copy = {
                        repo: dict(hashes) for repo, hashes in self._staged_blobs.items()
                    }

            if should_publish:
                manifest_data = {
                    "version": 1,
                    "prefix": self.prefix,
                    "updated_at": time.time(),
                    "blobs": manifest_blobs,
                    "snapshots": current_snapshots,
                }
                manifest_tmp_dir = self.cache_dir / ".hub_cache_tmp"
                manifest_tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_fd, tmp_manifest = tempfile.mkstemp(
                    dir=manifest_tmp_dir, prefix="manifest_", suffix=".json"
                )
                try:
                    with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                        json.dump(manifest_data, f)
                    manifest_obj = f"{self.prefix}/manifest.json"
                    # Network I/O outside _state_lock
                    self.api.batch_bucket_files(
                        self.bucket_id, add=[(tmp_manifest, manifest_obj)]
                    )
                    manifest_committed = True
                except Exception as e:
                    with self._state_lock:
                        self._last_error = e.__class__.__name__
                    upload_failed = True
                finally:
                    try:
                        os.unlink(tmp_manifest)
                    except OSError:
                        pass

                if manifest_committed:
                    with self._state_lock:
                        self._last_checkpoint_at = time.time()
                        # Commit state ONLY after manifest publish succeeds
                        for repo, hashes in staged_copy.items():
                            self._published_blobs.setdefault(repo, {}).update(hashes)
                        self._staged_blobs.clear()
                        self._published_snapshots = current_snapshots
                        self._uploaded_bytes = sum(
                            sum(h.values()) for h in self._published_blobs.values()
                        )
                        self._uploaded_count = sum(
                            len(h) for h in self._published_blobs.values()
                        )
                        committed_files = sum(len(h) for h in staged_copy.values())
                        committed_bytes = sum(sum(h.values()) for h in staged_copy.values())

        status_code = "error" if upload_failed else "ok"
        with self._state_lock:
            if not upload_failed and self._last_checkpoint_at is None:
                self._last_checkpoint_at = time.time()
            err = self._last_error if upload_failed else None

        return {
            "uploaded_files": committed_files,
            "uploaded_bytes": committed_bytes,
            "status": status_code,
            "error_class": err,
        }

    def start(self) -> None:
        """Start background thread performing periodic checkpoints."""
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._state = "running"
            self._thread = threading.Thread(
                target=self._run_loop, name="HubCacheWorker", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop background thread within bounded timeout."""
        self._stop_event.set()
        with self._state_lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._state_lock:
            self._state = "stopped"

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=self.interval):
                break
            try:
                self.checkpoint()
            except Exception as e:
                with self._state_lock:
                    self._last_error = e.__class__.__name__

    def status(self) -> dict:
        """Report status without exposing paths, tokens, or raw contents."""
        with self._state_lock:
            return {
                "enabled": True,
                "state": self._state,
                "restored_links": self._restored_links,
                "restored_files": self._restored_links,
                "uploaded_files": self._uploaded_count,
                "uploaded_bytes": self._uploaded_bytes,
                "bytes": self._uploaded_bytes,
                "restore_seconds": self._restore_seconds,
                "last_checkpoint_at": self._last_checkpoint_at,
                "error_class": self._last_error,
            }
