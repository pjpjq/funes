from copy import deepcopy

import pytest

from scripts.plan_source_postgres_tail import TailPlanError, build_plan


def fixture():
    base = {"version": 1, "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": ["funes-delta-old.jsonl.gz.enc"], "controls": []}
    files = [base["snapshot"], *base["deltas"]]
    blobs = {name: f"blob-{i}" for i, name in enumerate(files)}
    metadata = {"sqlite_sha256": "sqlite-hash", "source_receipt": {
        "version": 1, "repo": "owner/source", "manifest": base,
        "files": files, "blob_ids": blobs, "revision": "baseline-sha"}}
    manifest = deepcopy(base)
    manifest["deltas"].append("funes-delta-new.jsonl.gz.enc")
    manifest["controls"].append("funes-reindex-new.jsonl.gz.enc")
    active = [manifest["snapshot"], *manifest["deltas"], *manifest["controls"]]
    current = {"head": "current-sha", "manifest": manifest, "repo_files": active,
               "blob_ids": {**blobs, active[-2]: "new-blob", active[-1]: "control-blob"}}
    return metadata, current


def test_tail_only_is_pinned_and_never_claims_final_write_boundary():
    metadata, current = fixture()
    plan = build_plan(metadata, current, "owner/source")
    assert plan["tail_files"] == current["manifest"]["deltas"][1:] + current["manifest"]["controls"]
    assert plan["covered_files"] == 2
    assert plan["revision"] == "current-sha"
    assert plan["final_write_boundary"] is False


@pytest.mark.parametrize("mutate", [
    lambda m, c: m["source_receipt"]["files"].pop(),
    lambda m, c: c["blob_ids"].update({c["manifest"]["snapshot"]: "changed"}),
    lambda m, c: c["manifest"]["deltas"].reverse(),
    lambda m, c: c["blob_ids"].pop(c["manifest"]["deltas"][-1]),
    lambda m, c: m["source_receipt"].update(repo="wrong/source"),
    lambda m, c: c.update(head=""),
])
def test_changed_or_incomplete_baselines_fail_closed(mutate):
    metadata, current = fixture()
    mutate(metadata, current)
    with pytest.raises(TailPlanError):
        build_plan(metadata, current, "owner/source")
