#!/usr/bin/env python3
"""Offline authentication/JSON validation of a pinned source migration tail.

Reads encrypted downloads only. Never writes to SQLite, PostgreSQL, or the Hub;
never claims a final write boundary or a successfully applied migration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.plan_source_postgres_tail import write_json  # noqa: E402
from service.server import SnapshotSync, utc_now  # noqa: E402

RECORD_TYPES = frozenset({
    "memory", "translation_cache", "reindex_control",
    "native_optimize_checkpoint", "native_index_state",
})


class TailValidationError(RuntimeError):
    """Only fixed error codes may appear in operator output."""


def validate_tail(plan: dict, cache: Path, sync: SnapshotSync, progress=None) -> dict:
    """Verify every encrypted hash/tag and record without applying any record."""
    files = plan.get("tail_files")
    downloads = plan.get("downloads")
    if (plan.get("version") != 1 or not plan.get("revision")
            or not plan.get("baseline_sqlite_sha256")
            or not isinstance(files, list) or not all(isinstance(x, str) for x in files)
            or len(set(files)) != len(files) or not isinstance(downloads, dict)
            or set(downloads) != set(files)):
        raise TailValidationError("incomplete_tail_download_plan")
    root = (cache / "remote").resolve()
    counts = Counter()
    byte_counts = Counter()
    total_bytes = 0
    with tempfile.TemporaryDirectory(prefix="funes-tail-validation-") as temporary:
        # TemporaryDirectory creates a mode-0700 parent; plaintext never leaves it.
        for index, filename in enumerate(files):
            relative = Path(filename)
            encrypted = (root / relative).resolve()
            if (relative.is_absolute() or ".." in relative.parts
                    or not filename.endswith(".enc")
                    or not encrypted.is_relative_to(root) or not encrypted.is_file()):
                raise TailValidationError("unsafe_or_missing_tail_file")
            expected = downloads[filename]
            if not isinstance(expected, dict):
                raise TailValidationError("invalid_download_receipt")
            before = encrypted.stat()
            digest = hashlib.sha256()
            with encrypted.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            if (before.st_size != expected.get("size")
                    or digest.hexdigest() != expected.get("sha256")):
                raise TailValidationError("encrypted_download_digest_mismatch")
            plaintext = Path(temporary) / relative.name.removesuffix(".enc")
            try:
                # Uses the same authenticated streaming decryptor as production.
                sync._decrypt_file(encrypted, plaintext)
                for record in sync._iter_file(plaintext):
                    if not isinstance(record, dict):
                        raise TailValidationError("non_object_tail_record")
                    kind = record.get("_funes_record", "memory")
                    if not isinstance(kind, str) or kind not in RECORD_TYPES:
                        raise TailValidationError("unsupported_tail_record_type")
                    counts[kind] += 1
                    if kind == "memory":
                        for field in ("raw_text", "retrieval_text"):
                            value = record.get(field)
                            if value is not None and not isinstance(value, str):
                                raise TailValidationError("non_text_memory_field")
                            if isinstance(value, str):
                                byte_counts[field] += len(value.encode("utf-8"))
            finally:
                plaintext.unlink(missing_ok=True)
            after = encrypted.stat()
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, key) != getattr(after, key) for key in fields):
                raise TailValidationError("encrypted_download_changed_during_validation")
            total_bytes += before.st_size
            if progress and ((index + 1) % 50 == 0 or index + 1 == len(files)):
                progress({"phase": "validating", "files": index + 1,
                          "total_files": len(files), "records": sum(counts.values())})
    return {
        "version": 1, "validated_at": utc_now(), "status": "validated_not_applied",
        "revision": plan["revision"],
        "baseline_sqlite_sha256": plan["baseline_sqlite_sha256"],
        "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "files": len(files), "encrypted_bytes": total_bytes,
        "records": sum(counts.values()), "records_by_type": dict(counts),
        "text_bytes": dict(byte_counts), "authenticated_decryption": True,
        "database_modified": False, "final_write_boundary": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        plan = json.loads(Path(args.plan).read_text())
        cache = Path(args.cache_dir).resolve()
        sync = SnapshotSync(SimpleNamespace(data_dir=cache))
        report = validate_tail(plan, cache, sync,
                               lambda item: print(json.dumps(item), flush=True))
        write_json(Path(args.output), report)
        print(json.dumps(report), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error_class": type(error).__name__,
                          "error": str(error) if isinstance(error, TailValidationError)
                          else "tail_validation_failed"}), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
