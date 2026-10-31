import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.validate_source_postgres_tail import TailValidationError, main, validate_tail
from service.server import SnapshotSync


def fixture(tmp_path, monkeypatch, records=None):
    monkeypatch.setenv("FUNES_STORAGE_KEY", "offline-tail-test-only")
    sync = SnapshotSync(SimpleNamespace(data_dir=tmp_path))
    filename = "funes-delta-test.jsonl.gz.enc"
    plaintext = tmp_path / "test.jsonl.gz"
    records = records if records is not None else [
        {"raw_text": "中文原文\u0000\ue0000", "retrieval_text": "English raw\u0000text"},
        {"_funes_record": "native_index_state", "revision": 7},
    ]
    with gzip.open(plaintext, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    encrypted = tmp_path / "remote" / filename
    sync._encrypt_file(plaintext, encrypted)
    plan = {"version": 1, "revision": "pinned-sha", "baseline_sqlite_sha256": "baseline-sha",
            "tail_files": [filename], "downloads": {filename: {
                "size": encrypted.stat().st_size,
                "sha256": hashlib.sha256(encrypted.read_bytes()).hexdigest()}}}
    return plan, sync, encrypted


def test_validates_raw_unicode_nul_and_records_without_modifying_inputs(tmp_path, monkeypatch):
    plan, sync, encrypted = fixture(tmp_path, monkeypatch)
    before = encrypted.read_bytes()
    progress = []
    report = validate_tail(plan, tmp_path, sync, progress.append)
    assert report["records_by_type"] == {"memory": 1, "native_index_state": 1}
    assert report["text_bytes"]["raw_text"] == len("中文原文\u0000\ue0000".encode())
    assert report["status"] == "validated_not_applied"
    assert report["database_modified"] is report["final_write_boundary"] is False
    assert encrypted.read_bytes() == before
    assert progress[0]["files"] == 1


def test_corrupted_encrypted_download_rejected_before_decryption(tmp_path, monkeypatch):
    plan, sync, encrypted = fixture(tmp_path, monkeypatch)
    encrypted.write_bytes(encrypted.read_bytes() + b"bad")
    with pytest.raises(TailValidationError, match="digest_mismatch"):
        validate_tail(plan, tmp_path, sync)


def test_modified_download_and_receipt_still_requires_valid_authentication_tag(tmp_path, monkeypatch):
    plan, sync, encrypted = fixture(tmp_path, monkeypatch)
    value = bytearray(encrypted.read_bytes())
    value[-1] ^= 1
    encrypted.write_bytes(value)
    plan["downloads"][encrypted.name]["sha256"] = hashlib.sha256(value).hexdigest()
    from cryptography.exceptions import InvalidTag
    with pytest.raises(InvalidTag):
        validate_tail(plan, tmp_path, sync)


@pytest.mark.parametrize("records,code", [
    (["raw content is not an object"], "non_object"),
    ([{"_funes_record": "unexpected"}], "unsupported"),
    ([{"_funes_record": []}], "unsupported"),
    ([{"raw_text": 7}], "non_text"),
])
def test_malformed_records_fail_closed(tmp_path, monkeypatch, records, code):
    plan, sync, _ = fixture(tmp_path, monkeypatch, records)
    with pytest.raises(TailValidationError, match=code):
        validate_tail(plan, tmp_path, sync)


def test_symlink_escape_rejected(tmp_path, monkeypatch):
    plan, sync, encrypted = fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside.enc"
    encrypted.rename(outside)
    encrypted.symlink_to(outside)
    with pytest.raises(TailValidationError, match="unsafe"):
        validate_tail(plan, tmp_path, sync)


def test_incomplete_download_receipt_rejected(tmp_path, monkeypatch):
    plan, sync, _ = fixture(tmp_path, monkeypatch)
    plan["downloads"] = {}
    with pytest.raises(TailValidationError, match="incomplete"):
        validate_tail(plan, tmp_path, sync)


def test_cli_suppresses_untrusted_exception_text(tmp_path, monkeypatch, capsys):
    plan, _, _ = fixture(tmp_path, monkeypatch)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan))
    def broken(*_):
        raise ValueError("sensitive untrusted record")
    monkeypatch.setattr(SnapshotSync, "_decrypt_file", broken)
    assert main(["--plan", str(plan_file), "--cache-dir", str(tmp_path),
                 "--output", str(tmp_path / "report.json")]) == 2
    output = capsys.readouterr()
    assert "sensitive" not in output.err
    assert "tail_validation_failed" in output.err
    assert not (tmp_path / "report.json").exists()
