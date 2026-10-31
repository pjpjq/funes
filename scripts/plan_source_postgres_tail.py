#!/usr/bin/env python3
"""Pin/download only the immutable Hub suffix after an attested source snapshot.

This preparation tool never opens SQLite/PostgreSQL or modifies a Hub repository.
It deliberately refuses a replaced baseline, changed covered blobs, or reordered
manifests. A later final cutover must capture another head after writers stop.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from service.server import SnapshotSync, utc_now  # noqa: E402


class TailPlanError(RuntimeError):
    """Payload/credential-free operator error."""


def build_plan(metadata: dict, current: dict, repo: str) -> dict:
    receipt = metadata.get("source_receipt", {})
    baseline = receipt.get("manifest")
    manifest = current.get("manifest")
    if receipt.get("version") != 1 or receipt.get("repo") != repo:
        raise TailPlanError("receipt_repo_or_version_mismatch")
    if not isinstance(baseline, dict) or not isinstance(manifest, dict):
        raise TailPlanError("explicit_restore_manifests_required")
    baseline_files = [baseline["snapshot"], *baseline["deltas"], *baseline["controls"]]
    if receipt.get("files") != baseline_files:
        raise TailPlanError("baseline_receipt_is_not_complete")
    validator = SnapshotSync(None)
    validator._validate_restore_manifest(baseline, set(baseline_files))
    validator._validate_restore_manifest(manifest, set(current["repo_files"]))
    if baseline["snapshot"] != manifest["snapshot"]:
        raise TailPlanError("baseline_snapshot_replaced")
    for group in ("deltas", "controls"):
        if manifest[group][:len(baseline[group])] != baseline[group]:
            raise TailPlanError("covered_manifest_prefix_changed")
    old_blobs, new_blobs = receipt.get("blob_ids", {}), current.get("blob_ids", {})
    for name in baseline_files:
        if not old_blobs.get(name) or old_blobs[name] != new_blobs.get(name):
            raise TailPlanError("covered_blob_missing_or_changed")
    tail = [*manifest["deltas"][len(baseline["deltas"]):],
            *manifest["controls"][len(baseline["controls"]):]]
    if any(not new_blobs.get(name) for name in tail):
        raise TailPlanError("tail_blob_identity_missing")
    if not metadata.get("sqlite_sha256") or not receipt.get("revision") or not current.get("head"):
        raise TailPlanError("source_identity_missing")
    return {
        "version": 1, "created_at": utc_now(), "repo": repo,
        "baseline_sqlite_sha256": metadata["sqlite_sha256"],
        "baseline_revision": receipt["revision"], "revision": current["head"],
        "covered_files": len(baseline_files), "tail_files": tail,
        "blob_ids": {name: new_blobs[name] for name in tail},
        "manifest": manifest, "final_write_boundary": False,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        from huggingface_hub import HfApi, hf_hub_download

        metadata = json.loads(Path(args.metadata).read_text())
        cache = Path(args.cache_dir).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        sync = SnapshotSync(SimpleNamespace(data_dir=cache))
        if not sync.repo or not sync.token:
            raise TailPlanError("HF_TOKEN_and_FUNES_STORAGE_REPO_required")
        current = sync._remote_restore_state(HfApi(token=sync.token))
        plan = build_plan(metadata, current, sync.repo)
        write_json(Path(args.output), plan)
        print(json.dumps({"phase": "pinned", "revision": plan["revision"],
                          "covered_files": plan["covered_files"],
                          "tail_files": len(plan["tail_files"])}), flush=True)
        if args.download:
            def download(name):
                path = Path(hf_hub_download(repo_id=sync.repo, repo_type="dataset", filename=name,
                                           revision=plan["revision"], token=sync.token,
                                           local_dir=str(cache / "remote")))
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                return name, {"size": path.stat().st_size, "sha256": digest.hexdigest()}

            with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 16))) as pool:
                plan["downloads"] = dict(pool.map(download, plan["tail_files"]))
            write_json(Path(args.output), plan)
            print(json.dumps({"phase": "downloaded", "files": len(plan["downloads"]),
                              "bytes": sum(x["size"] for x in plan["downloads"].values()),
                              "final_write_boundary": False}), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error_class": type(error).__name__,
                          "error": str(error) if isinstance(error, TailPlanError) else "tail_preparation_failed"}),
              file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
