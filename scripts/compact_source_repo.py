#!/usr/bin/env python3
"""Restart-safe compaction operator for funes source repository on Hugging Face.

Consolidates all historical deltas and base snapshot into a single encrypted
full snapshot, commits the snapshot and updated manifest atomically, and
deletes legacy root deltas in bounded batches (<=2000 per commit).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from service.server import Store, SnapshotSync, ENCRYPTED_MAGIC


DEFAULT_REPO = "bolikoto/funes-memory-source"
DEFAULT_BATCH_SIZE = 1000
MAX_BATCH_SIZE = 2000
DEFAULT_SNAPSHOT_FILE = "funes-snapshot.jsonl.gz"
DEFAULT_MANIFEST_FILE = "funes-restore-manifest-v1.json"
DEFAULT_DELTA_PREFIX = "funes-delta-"
DEFAULT_CONTROL_PREFIX = "funes-reindex-"
ENCRYPTED_SUFFIXES = (".jsonl.enc", ".jsonl.gz.enc")


def sanitize_text(text: str, secrets: list[str]) -> str:
    """Strip secret values from output text."""
    result = text
    for secret in secrets:
        if secret and len(secret) >= 4:
            result = result.replace(secret, "[REDACTED]")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restart-safe compaction operator for funes source storage repositories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        default=os.getenv("FUNES_STORAGE_REPO") or DEFAULT_REPO,
        help="Hugging Face repository ID (dataset repo)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--storage-key",
        default=None,
        help="Storage encryption key (defaults to FUNES_STORAGE_KEY / FUNES_API_TOKEN / FUNES_AUTH_TOKEN)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Maximum number of legacy deltas to delete per commit (capped at {MAX_BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect repository and plan compaction without making commits",
    )
    parser.add_argument(
        "--force-recompact",
        action="store_true",
        help="Force snapshot consolidation even if active manifest already has no deltas",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Custom local staging directory for temporary database and artifacts",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts for transient Hub network operations",
    )
    return parser.parse_args(argv)


def get_credentials(
    args: argparse.Namespace,
) -> tuple[str, str, str]:
    """Extract and validate credentials without exposing values."""
    repo = args.repo.strip() if args.repo else ""
    if not repo:
        raise ValueError("Repository ID is required (via --repo or FUNES_STORAGE_REPO)")

    token = args.token or os.getenv("HF_TOKEN", "")
    token = token.strip() if token else ""

    storage_key = (
        args.storage_key
        or os.getenv("FUNES_STORAGE_KEY")
        or os.getenv("FUNES_API_TOKEN")
        or os.getenv("FUNES_AUTH_TOKEN", "")
    )
    storage_key = storage_key.strip() if storage_key else ""

    return repo, token, storage_key


def inspect_source_repo(
    api: Any,
    repo: str,
    token: str,
    *,
    manifest_filename: str = DEFAULT_MANIFEST_FILE,
    snapshot_filename: str = DEFAULT_SNAPSHOT_FILE,
    delta_prefix: str = DEFAULT_DELTA_PREFIX,
    control_prefix: str = DEFAULT_CONTROL_PREFIX,
) -> dict[str, Any]:
    """Inspect remote repository state and partition active vs legacy objects."""
    info = api.repo_info(repo_id=repo, repo_type="dataset", token=token or None)
    head = getattr(info, "sha", "")
    if not isinstance(head, str) or not head:
        raise RuntimeError("Hub repository head is unavailable")

    repo_files: set[str] = set()
    blob_ids: dict[str, str] = {}
    for item in api.list_repo_tree(
        repo,
        repo_type="dataset",
        recursive=True,
        revision=head,
        token=token or None,
    ):
        name = getattr(item, "path", "")
        if isinstance(name, str):
            repo_files.add(name)
            blob_id = getattr(item, "blob_id", "")
            if isinstance(blob_id, str) and blob_id:
                blob_ids[name] = blob_id

    manifest: dict[str, Any] | None = None
    if manifest_filename in repo_files:
        from huggingface_hub import hf_hub_download
        import json

        with tempfile.TemporaryDirectory(prefix="funes-manifest-check-") as tmp_dir:
            downloaded = hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=manifest_filename,
                revision=head,
                token=token or None,
                local_dir=tmp_dir,
            )
            raw = Path(downloaded).read_text(encoding="utf-8")
            parsed = json.loads(raw)
            # Basic validation
            if (
                isinstance(parsed, dict)
                and parsed.get("version") == 1
                and isinstance(parsed.get("snapshot"), str)
                and isinstance(parsed.get("deltas"), list)
                and isinstance(parsed.get("controls"), list)
            ):
                manifest = parsed

    target_snapshot_name = snapshot_filename + ".enc"

    # Identify root deltas
    root_deltas = sorted(
        name
        for name in repo_files
        if "/" not in name
        and name.startswith(delta_prefix)
        and name.endswith(ENCRYPTED_SUFFIXES)
    )

    # Active objects from manifest
    active_snapshot = manifest.get("snapshot") if manifest else None
    active_deltas = manifest.get("deltas", []) if manifest else []
    active_controls = manifest.get("controls", []) if manifest else []

    protected: set[str] = {
        manifest_filename,
        ".gitattributes",
    }
    if active_snapshot:
        protected.add(active_snapshot)
    for d in active_deltas:
        protected.add(d)
    for c in active_controls:
        protected.add(c)

    # Unreferenced root deltas that are safe to prune
    unreferenced_root_deltas = sorted(
        d for d in root_deltas if d not in protected
    )

    # Repository is considered already compacted if:
    # 1. Manifest exists and points to an existing snapshot
    # 2. Manifest has no active deltas pending consolidation
    is_already_compacted = (
        manifest is not None
        and active_snapshot in repo_files
        and len(active_deltas) == 0
    )

    return {
        "head": head,
        "repo_files": repo_files,
        "blob_ids": blob_ids,
        "manifest": manifest,
        "target_snapshot_name": target_snapshot_name,
        "active_snapshot": active_snapshot,
        "active_deltas": active_deltas,
        "active_controls": active_controls,
        "protected": protected,
        "root_deltas": root_deltas,
        "unreferenced_root_deltas": unreferenced_root_deltas,
        "is_already_compacted": is_already_compacted,
    }


def plan_compaction(
    inventory: dict[str, Any],
    batch_size: int = DEFAULT_BATCH_SIZE,
    force_recompact: bool = False,
) -> dict[str, Any]:
    """Generate safe compaction and deletion plan."""
    bounded_batch_size = max(1, min(batch_size, MAX_BATCH_SIZE))
    is_compacted = inventory["is_already_compacted"] and not force_recompact

    unref_deltas = inventory["unreferenced_root_deltas"]
    unref_count = len(unref_deltas)
    prune_batches = math.ceil(unref_count / bounded_batch_size) if unref_count > 0 else 0

    return {
        "head": inventory["head"],
        "has_manifest": inventory["manifest"] is not None,
        "active_snapshot": inventory["active_snapshot"],
        "target_snapshot": inventory["target_snapshot_name"],
        "active_deltas_count": len(inventory["active_deltas"]),
        "active_controls_count": len(inventory["active_controls"]),
        "total_repo_files": len(inventory["repo_files"]),
        "root_deltas_count": len(inventory["root_deltas"]),
        "unreferenced_deltas_count": unref_count,
        "batch_size": bounded_batch_size,
        "prune_batches": prune_batches,
        "is_already_compacted": is_compacted,
        "action_compaction": "skip" if is_compacted else "consolidate",
        "action_prune": unref_count > 0,
    }


def execute_compaction(
    syncer: SnapshotSync,
    api: Any,
    temp_dir: str | Path,
    base_state: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Restore entire dataset into isolated temporary SQLite, materialize full snapshot, and commit atomically."""
    if dry_run:
        return {
            "dry_run": True,
            "commit_oid": None,
            "retained_suffix": False,
            "restored_records": 0,
        }

    temp_path = Path(temp_dir)
    restored_count = syncer.restore()
    if syncer.restore_failed:
        raise RuntimeError(f"Source store restore failed: {syncer.restore_error}")

    plaintext = temp_path / syncer.filename
    syncer.store.snapshot(plaintext)
    if not plaintext.is_file() or plaintext.stat().st_size == 0:
        raise RuntimeError("Materialized snapshot is missing or empty")

    encrypted = plaintext.with_name(plaintext.name + ".enc")
    syncer._encrypt_file(plaintext, encrypted)
    if not encrypted.is_file() or encrypted.stat().st_size == 0:
        raise RuntimeError("Encrypted snapshot is missing or empty")

    # Verify encrypted magic header
    with encrypted.open("rb") as f:
        header = f.read(len(ENCRYPTED_MAGIC))
        if header != ENCRYPTED_MAGIC:
            raise RuntimeError("Invalid encryption magic on consolidated snapshot")

    target = syncer.filename + ".enc"
    commit_message = "funes consolidated source snapshot"

    commit_result, retained_suffix = syncer._commit_compact_snapshot(
        api,
        encrypted,
        target,
        commit_message,
        base_state,
    )

    commit_oid = syncer._commit_oid(commit_result)

    # Immediately remove plaintext to minimize exposure and memory/disk footprint
    plaintext.unlink(missing_ok=True)
    encrypted.unlink(missing_ok=True)

    return {
        "dry_run": False,
        "commit_oid": commit_oid,
        "retained_suffix": retained_suffix,
        "restored_records": restored_count,
    }


def execute_prune_deltas(
    api: Any,
    repo: str,
    candidates: list[str],
    protected: set[str],
    parent_commit: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
    retries: int = 3,
) -> list[dict[str, Any]]:
    """Delete legacy unreferenced root deltas in bounded batches (<=2000 per commit)."""
    bounded_batch_size = max(1, min(batch_size, MAX_BATCH_SIZE))

    # Strict safety invariant check
    for item in candidates:
        if item in protected:
            raise ValueError(f"Safety violation: attempted to delete protected artifact '{item}'")
        if "/" in item:
            raise ValueError(f"Safety violation: attempted to delete non-root artifact '{item}'")

    if not candidates:
        return []

    batches = [
        candidates[i : i + bounded_batch_size]
        for i in range(0, len(candidates), bounded_batch_size)
    ]

    if dry_run:
        return [
            {
                "batch_index": idx,
                "total_batches": len(batches),
                "file_count": len(b),
                "commit_oid": None,
            }
            for idx, b in enumerate(batches, start=1)
        ]

    from huggingface_hub import CommitOperationDelete

    current_head = parent_commit
    results: list[dict[str, Any]] = []

    for idx, batch in enumerate(batches, start=1):
        operations = [CommitOperationDelete(path_in_repo=p) for p in batch]
        commit_msg = (
            f"funes prune legacy source deltas (batch {idx}/{len(batches)}, {len(batch)} files)"
        )
        last_error = None
        new_commit_oid = None

        for attempt in range(retries):
            try:
                res = api.create_commit(
                    repo_id=repo,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=commit_msg,
                    parent_commit=current_head,
                )
                new_commit_oid = getattr(res, "oid", None)
                if not new_commit_oid and hasattr(api, "repo_info"):
                    info = api.repo_info(repo_id=repo, repo_type="dataset")
                    new_commit_oid = getattr(info, "sha", None)
                break
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"Failed to delete batch {idx}/{len(batches)} ({len(batch)} files): {exc}"
                    ) from last_error

        current_head = new_commit_oid or current_head
        results.append(
            {
                "batch_index": idx,
                "total_batches": len(batches),
                "file_count": len(batch),
                "commit_oid": current_head,
            }
        )

    return results


def compact_source_repo(
    *,
    repo: str = DEFAULT_REPO,
    token: str = "",
    storage_key: str = "",
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
    force_recompact: bool = False,
    temp_dir: str | None = None,
    retries: int = 3,
    api: Any = None,
) -> dict[str, Any]:
    """Execute complete restart-safe source repo compaction workflow."""
    bounded_batch_size = max(1, min(batch_size, MAX_BATCH_SIZE))
    secrets_to_redact = [token, storage_key]

    if not token and not dry_run:
        raise RuntimeError("HF_TOKEN is required for compaction operations")
    if not storage_key and not dry_run:
        raise RuntimeError("FUNES_STORAGE_KEY is required for compaction operations")

    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token or None)

    # 1. Inspect repository state
    inventory = inspect_source_repo(
        api=api,
        repo=repo,
        token=token,
    )
    plan = plan_compaction(
        inventory=inventory,
        batch_size=bounded_batch_size,
        force_recompact=force_recompact,
    )

    current_head = inventory["head"]
    compaction_result: dict[str, Any] = {"action": plan["action_compaction"]}

    if dry_run:
        # Under dry-run, simulate plan and return
        return {
            "dry_run": True,
            "repo": repo,
            "plan": plan,
            "compaction": compaction_result,
            "prune_batches": [
                {
                    "batch_index": idx,
                    "total_batches": plan["prune_batches"],
                    "file_count": min(
                        bounded_batch_size,
                        plan["unreferenced_deltas_count"] - (idx - 1) * bounded_batch_size,
                    ),
                    "commit_oid": None,
                }
                for idx in range(1, plan["prune_batches"] + 1)
            ],
        }

    # 2. Execute snapshot consolidation if needed
    if plan["action_compaction"] == "consolidate":
        with tempfile.TemporaryDirectory(
            prefix="funes-compact-run-", dir=temp_dir
        ) as staging:
            store = Store(staging)
            syncer = SnapshotSync(store)
            syncer.repo = repo
            syncer.token = token
            syncer.storage_key = storage_key

            base_state = {
                "head": inventory["head"],
                "manifest": inventory["manifest"],
                "repo_files": inventory["repo_files"],
                "blob_ids": inventory["blob_ids"],
            }

            try:
                comp_res = execute_compaction(
                    syncer=syncer,
                    api=api,
                    temp_dir=staging,
                    base_state=base_state,
                    dry_run=False,
                )
                compaction_result.update(comp_res)
                if comp_res.get("commit_oid"):
                    current_head = comp_res["commit_oid"]
            finally:
                store.close()

    # 3. Refresh remote inventory to get accurate updated manifest and head
    post_inventory = inspect_source_repo(
        api=api,
        repo=repo,
        token=token,
    )

    # 4. Prune unreferenced legacy root deltas in bounded batches
    prune_candidates = post_inventory["unreferenced_root_deltas"]
    protected_set = post_inventory["protected"]

    prune_results = execute_prune_deltas(
        api=api,
        repo=repo,
        candidates=prune_candidates,
        protected=protected_set,
        parent_commit=post_inventory["head"],
        batch_size=bounded_batch_size,
        dry_run=False,
        retries=retries,
    )

    final_head = prune_results[-1]["commit_oid"] if prune_results else post_inventory["head"]

    return {
        "dry_run": False,
        "repo": repo,
        "plan": plan,
        "compaction": compaction_result,
        "pruned_deltas_count": len(prune_candidates),
        "prune_batches": prune_results,
        "initial_head": inventory["head"],
        "final_head": final_head,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        repo, token, storage_key = get_credentials(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    secrets = [token, storage_key]

    try:
        result = compact_source_repo(
            repo=repo,
            token=token,
            storage_key=storage_key,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            force_recompact=args.force_recompact,
            temp_dir=args.temp_dir,
            retries=args.retries,
        )

        plan = result["plan"]
        print(f"Repository: {result['repo']}")
        print(f"Initial head: {plan['head'][:8] if plan['head'] else 'none'}")
        print(f"Files in repository: {plan['total_repo_files']}")
        print(f"Root deltas: {plan['root_deltas_count']}")
        print(f"Active deltas in manifest: {plan['active_deltas_count']}")
        print(f"Compaction status: {plan['action_compaction']}")
        print(
            f"Legacy deltas to prune: {plan['unreferenced_deltas_count']} "
            f"({plan['prune_batches']} batches, max {plan['batch_size']}/batch)"
        )

        if result.get("dry_run"):
            print("\n[DRY RUN] Plan verified successfully. No remote changes made.")
            return 0

        print(f"\nFinal head: {result.get('final_head', '')[:8]}")
        print(f"Pruned legacy deltas: {result.get('pruned_deltas_count', 0)}")
        print("Compaction complete and verified.")
        return 0

    except Exception as exc:
        err_msg = sanitize_text(str(exc), secrets)
        print(f"Compaction failed: {err_msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
