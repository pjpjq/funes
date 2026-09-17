import argparse
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import compact_source_repo as compactor
from service.server import Store, SnapshotSync, ENCRYPTED_MAGIC


class MockHfHub:
    """Simulates Hugging Face Hub dataset repository for tests."""

    def __init__(self, files: dict[str, bytes] | None = None, head: str = "head-0"):
        self.files = dict(files or {})
        self.head = head
        self.commits: list[dict] = []

    def repo_info(self, repo_id: str, repo_type: str = "dataset", token: str | None = None):
        return mock.Mock(sha=self.head)

    def list_repo_tree(
        self,
        repo_id: str,
        repo_type: str = "dataset",
        recursive: bool = True,
        revision: str | None = None,
        token: str | None = None,
    ):
        for path in sorted(self.files.keys()):
            yield mock.Mock(path=path, blob_id=f"blob-{path}")

    def create_commit(
        self,
        repo_id: str,
        repo_type: str = "dataset",
        operations=None,
        commit_message: str = "",
        parent_commit: str | None = None,
        token: str | None = None,
        **kwargs,
    ):
        new_head = f"head-{len(self.commits) + 1}"
        for op in (operations or []):
            if hasattr(op, "path_or_fileobj"):
                content = op.path_or_fileobj
                if isinstance(content, (str, Path)):
                    self.files[op.path_in_repo] = Path(content).read_bytes()
                elif isinstance(content, bytes):
                    self.files[op.path_in_repo] = content
            else:
                self.files.pop(op.path_in_repo, None)
        self.head = new_head
        self.commits.append(
            {
                "oid": new_head,
                "message": commit_message,
                "operations": operations,
                "parent_commit": parent_commit,
            }
        )
        return mock.Mock(oid=new_head)


def make_fake_download(hub: MockHfHub):
    def fake_download(repo_id, filename, revision=None, token=None, local_dir=None, **kwargs):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        if filename not in hub.files:
            raise FileNotFoundError(f"File {filename} not in mock hub")
        p.write_bytes(hub.files[filename])
        return str(p)

    return fake_download


class TestCompactSourceRepo(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = compactor.parse_args([])
        self.assertEqual(args.repo, compactor.DEFAULT_REPO)
        self.assertIsNone(args.token)
        self.assertIsNone(args.storage_key)
        self.assertEqual(args.batch_size, compactor.DEFAULT_BATCH_SIZE)
        self.assertFalse(args.dry_run)
        self.assertFalse(args.force_recompact)

    def test_parse_args_overrides(self):
        args = compactor.parse_args([
            "--repo", "custom/repo",
            "--token", "hf_custom",
            "--storage-key", "key_custom",
            "--batch-size", "1500",
            "--dry-run",
            "--force-recompact",
        ])
        self.assertEqual(args.repo, "custom/repo")
        self.assertEqual(args.token, "hf_custom")
        self.assertEqual(args.storage_key, "key_custom")
        self.assertEqual(args.batch_size, 1500)
        self.assertTrue(args.dry_run)
        self.assertTrue(args.force_recompact)

    def test_batch_size_capped_at_max(self):
        # Even if batch-size is specified as 5000, plan and execution cap at 2000
        inventory = {
            "head": "head-0",
            "repo_files": set(),
            "blob_ids": {},
            "manifest": None,
            "target_snapshot_name": "funes-snapshot.jsonl.gz.enc",
            "active_snapshot": None,
            "active_deltas": [],
            "active_controls": [],
            "protected": set(),
            "root_deltas": [],
            "unreferenced_root_deltas": [f"funes-delta-{i}.jsonl.gz.enc" for i in range(5000)],
            "is_already_compacted": True,
        }
        plan = compactor.plan_compaction(inventory, batch_size=5000)
        self.assertEqual(plan["batch_size"], compactor.MAX_BATCH_SIZE)
        self.assertEqual(plan["batch_size"], 2000)
        self.assertEqual(plan["prune_batches"], 3)  # 2000 + 2000 + 1000 = 3 batches

    def test_inspect_source_repo_partitions_active_and_legacy(self):
        manifest_data = {
            "version": 1,
            "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": ["funes-delta-active1.jsonl.gz.enc", "funes-delta-active2.jsonl.gz.enc"],
            "controls": [],
        }
        hub = MockHfHub(
            files={
                ".gitattributes": b"",
                "funes-restore-manifest-v1.json": json.dumps(manifest_data).encode("utf-8"),
                "funes-snapshot.jsonl.gz.enc": b"content",
                "funes-delta-active1.jsonl.gz.enc": b"content",
                "funes-delta-active2.jsonl.gz.enc": b"content",
                "funes-delta-legacy1.jsonl.gz.enc": b"content",
                "funes-delta-legacy2.jsonl.gz.enc": b"content",
            }
        )

        with mock.patch("huggingface_hub.hf_hub_download", side_effect=make_fake_download(hub)):
            inv = compactor.inspect_source_repo(
                api=hub,
                repo="test/repo",
                token="test-token",
            )

        self.assertEqual(inv["head"], "head-0")
        self.assertEqual(inv["active_snapshot"], "funes-snapshot.jsonl.gz.enc")
        self.assertEqual(inv["active_deltas"], ["funes-delta-active1.jsonl.gz.enc", "funes-delta-active2.jsonl.gz.enc"])
        self.assertEqual(inv["unreferenced_root_deltas"], ["funes-delta-legacy1.jsonl.gz.enc", "funes-delta-legacy2.jsonl.gz.enc"])
        self.assertIn("funes-delta-active1.jsonl.gz.enc", inv["protected"])
        self.assertIn("funes-snapshot.jsonl.gz.enc", inv["protected"])
        self.assertFalse(inv["is_already_compacted"])

    def test_plan_compaction_with_active_deltas(self):
        inventory = {
            "head": "head-0",
            "repo_files": {"funes-snapshot.jsonl.gz.enc", "funes-delta-1.jsonl.gz.enc"},
            "blob_ids": {},
            "manifest": {
                "version": 1,
                "snapshot": "funes-snapshot.jsonl.gz.enc",
                "deltas": ["funes-delta-1.jsonl.gz.enc"],
                "controls": [],
            },
            "target_snapshot_name": "funes-snapshot.jsonl.gz.enc",
            "active_snapshot": "funes-snapshot.jsonl.gz.enc",
            "active_deltas": ["funes-delta-1.jsonl.gz.enc"],
            "active_controls": [],
            "protected": {"funes-snapshot.jsonl.gz.enc", "funes-delta-1.jsonl.gz.enc"},
            "root_deltas": ["funes-delta-1.jsonl.gz.enc"],
            "unreferenced_root_deltas": [],
            "is_already_compacted": False,
        }
        plan = compactor.plan_compaction(inventory, batch_size=1000)
        self.assertEqual(plan["action_compaction"], "consolidate")
        self.assertEqual(plan["active_deltas_count"], 1)
        self.assertFalse(plan["action_prune"])

    def test_plan_compaction_resumes_when_already_compacted(self):
        inventory = {
            "head": "head-0",
            "repo_files": {"funes-snapshot.jsonl.gz.enc", "funes-delta-old.jsonl.gz.enc"},
            "blob_ids": {},
            "manifest": {
                "version": 1,
                "snapshot": "funes-snapshot.jsonl.gz.enc",
                "deltas": [],
                "controls": [],
            },
            "target_snapshot_name": "funes-snapshot.jsonl.gz.enc",
            "active_snapshot": "funes-snapshot.jsonl.gz.enc",
            "active_deltas": [],
            "active_controls": [],
            "protected": {"funes-snapshot.jsonl.gz.enc"},
            "root_deltas": ["funes-delta-old.jsonl.gz.enc"],
            "unreferenced_root_deltas": ["funes-delta-old.jsonl.gz.enc"],
            "is_already_compacted": True,
        }
        plan = compactor.plan_compaction(inventory, batch_size=1000)
        self.assertEqual(plan["action_compaction"], "skip")
        self.assertTrue(plan["action_prune"])
        self.assertEqual(plan["unreferenced_deltas_count"], 1)
        self.assertEqual(plan["prune_batches"], 1)

    def test_dry_run_executes_no_commits(self):
        manifest_data = {
            "version": 1,
            "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": ["funes-delta-1.jsonl.gz.enc"],
            "controls": [],
        }
        hub = MockHfHub(
            files={
                ".gitattributes": b"",
                "funes-restore-manifest-v1.json": json.dumps(manifest_data).encode("utf-8"),
                "funes-snapshot.jsonl.gz.enc": b"content",
                "funes-delta-1.jsonl.gz.enc": b"content",
                "funes-delta-old.jsonl.gz.enc": b"content",
            }
        )

        with mock.patch("huggingface_hub.hf_hub_download", side_effect=make_fake_download(hub)):
            result = compactor.compact_source_repo(
                repo="test/repo",
                token="test-token",
                storage_key="test-key",
                dry_run=True,
                api=hub,
            )

        self.assertTrue(result["dry_run"])
        self.assertEqual(len(hub.commits), 0)
        self.assertEqual(result["plan"]["action_compaction"], "consolidate")
        self.assertEqual(result["plan"]["unreferenced_deltas_count"], 1)

    def test_safety_invariant_protected_files_never_deleted(self):
        api = mock.Mock()
        protected = {"funes-snapshot.jsonl.gz.enc", "funes-restore-manifest-v1.json"}
        candidates = ["funes-snapshot.jsonl.gz.enc"]

        with self.assertRaises(ValueError) as ctx:
            compactor.execute_prune_deltas(
                api=api,
                repo="test/repo",
                candidates=candidates,
                protected=protected,
                parent_commit="head-0",
            )
        self.assertIn("Safety violation: attempted to delete protected artifact", str(ctx.exception))
        api.create_commit.assert_not_called()

    def test_safety_invariant_non_root_files_never_deleted(self):
        api = mock.Mock()
        protected = set()
        candidates = ["sharded/ab/funes-delta-test.jsonl.gz.enc"]

        with self.assertRaises(ValueError) as ctx:
            compactor.execute_prune_deltas(
                api=api,
                repo="test/repo",
                candidates=candidates,
                protected=protected,
                parent_commit="head-0",
            )
        self.assertIn("Safety violation: attempted to delete non-root artifact", str(ctx.exception))
        api.create_commit.assert_not_called()

    def test_execute_compaction_materializes_and_encrypts_full_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(td)
            store.ingest([{"source_identity": "test-doc-1", "raw_text": "sample text"}])
            syncer = SnapshotSync(store)
            syncer.repo = "owner/repo"
            syncer.token = "test-token"
            syncer.storage_key = "test-key"

            api = mock.Mock()
            api.create_commit.return_value = mock.Mock(oid="head-compacted")

            base_state = {
                "head": "head-base",
                "manifest": {"version": 1, "snapshot": "funes-snapshot.jsonl.gz.enc", "deltas": [], "controls": []},
                "repo_files": {"funes-snapshot.jsonl.gz.enc"},
                "blob_ids": {},
            }

            syncer.restore = mock.Mock(return_value=1)
            syncer.restore_failed = False

            res = compactor.execute_compaction(
                syncer=syncer,
                api=api,
                temp_dir=td,
                base_state=base_state,
                dry_run=False,
            )

            self.assertEqual(res["commit_oid"], "head-compacted")
            self.assertEqual(res["restored_records"], 1)
            api.create_commit.assert_called_once()
            call_kwargs = api.create_commit.call_args.kwargs
            ops = call_kwargs["operations"]
            self.assertEqual(len(ops), 2)
            self.assertEqual(ops[0].path_in_repo, "funes-snapshot.jsonl.gz.enc")
            self.assertEqual(ops[1].path_in_repo, "funes-restore-manifest-v1.json")

            # Verify local files cleaned up
            self.assertFalse((Path(td) / "funes-snapshot.jsonl.gz").exists())
            self.assertFalse((Path(td) / "funes-snapshot.jsonl.gz.enc").exists())
            store.close()

    def test_end_to_end_compaction_and_batch_pruning(self):
        manifest_data = {
            "version": 1,
            "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": ["funes-delta-1.jsonl.gz.enc", "funes-delta-2.jsonl.gz.enc"],
            "controls": [],
        }
        hub = MockHfHub(
            files={
                ".gitattributes": b"",
                "funes-restore-manifest-v1.json": json.dumps(manifest_data).encode("utf-8"),
                "funes-snapshot.jsonl.gz.enc": b"content-base",
                "funes-delta-1.jsonl.gz.enc": b"content-d1",
                "funes-delta-2.jsonl.gz.enc": b"content-d2",
                "funes-delta-legacy1.jsonl.gz.enc": b"content-l1",
                "funes-delta-legacy2.jsonl.gz.enc": b"content-l2",
                "funes-delta-legacy3.jsonl.gz.enc": b"content-l3",
            }
        )

        with mock.patch("huggingface_hub.hf_hub_download", side_effect=make_fake_download(hub)), \
             mock.patch.object(SnapshotSync, "restore", return_value=10):
            res = compactor.compact_source_repo(
                repo="test/repo",
                token="test-token",
                storage_key="test-key",
                batch_size=2,
                dry_run=False,
                api=hub,
            )

        self.assertFalse(res["dry_run"])
        # Commit 1: snapshot + manifest (deltas = [])
        # Commit 2: delete legacy deltas batch 1 (2 files)
        # Commit 3: delete legacy deltas batch 2 (2 files)
        # Commit 4: delete legacy deltas batch 3 (1 file)
        # Total commits: 4
        self.assertEqual(len(hub.commits), 4)

        # Invariant: remaining files in hub should only be .gitattributes, manifest, and snapshot
        expected_remaining = {".gitattributes", "funes-restore-manifest-v1.json", "funes-snapshot.jsonl.gz.enc"}
        self.assertEqual(set(hub.files.keys()), expected_remaining)

        # Verify updated manifest has empty deltas
        updated_manifest = json.loads(hub.files["funes-restore-manifest-v1.json"].decode("utf-8"))
        self.assertEqual(updated_manifest["deltas"], [])
        self.assertEqual(updated_manifest["snapshot"], "funes-snapshot.jsonl.gz.enc")

    def test_resume_after_interrupted_pruning(self):
        # Simulate repository after snapshot consolidation has already occurred:
        # manifest has deltas = [], but 3 unreferenced legacy deltas still remain in repo
        manifest_data = {
            "version": 1,
            "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": [],
            "controls": [],
        }
        hub = MockHfHub(
            files={
                ".gitattributes": b"",
                "funes-restore-manifest-v1.json": json.dumps(manifest_data).encode("utf-8"),
                "funes-snapshot.jsonl.gz.enc": b"consolidated-snapshot-bytes",
                "funes-delta-rem1.jsonl.gz.enc": b"content",
                "funes-delta-rem2.jsonl.gz.enc": b"content",
                "funes-delta-rem3.jsonl.gz.enc": b"content",
            }
        )

        with mock.patch("huggingface_hub.hf_hub_download", side_effect=make_fake_download(hub)), \
             mock.patch.object(compactor, "execute_compaction") as mock_compaction:
            res = compactor.compact_source_repo(
                repo="test/repo",
                token="test-token",
                storage_key="test-key",
                batch_size=2,
                dry_run=False,
                api=hub,
            )

        # Compaction execution should have been skipped!
        mock_compaction.assert_not_called()

        # Only prune commits should have been made (batch 1 of 2, batch 2 of 1)
        self.assertEqual(len(hub.commits), 2)
        expected_remaining = {".gitattributes", "funes-restore-manifest-v1.json", "funes-snapshot.jsonl.gz.enc"}
        self.assertEqual(set(hub.files.keys()), expected_remaining)
        self.assertEqual(res["pruned_deltas_count"], 3)

    def test_sanitize_text_redacts_tokens_and_storage_keys(self):
        secret_token = "hf_SecretToken123456789"
        secret_key = "MySuperSecretEncryptionKeyPassphrase"
        log_message = f"Error connecting with {secret_token} using {secret_key} on server"

        sanitized = compactor.sanitize_text(log_message, [secret_token, secret_key])
        self.assertNotIn(secret_token, sanitized)
        self.assertNotIn(secret_key, sanitized)
        self.assertIn("[REDACTED]", sanitized)

    def test_cli_main_help(self):
        with self.assertRaises(SystemExit) as ctx, mock.patch("sys.stdout", new=io.StringIO()):
            compactor.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_cli_main_missing_credentials_fails_closed(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("sys.stderr", new=io.StringIO()):
            code = compactor.main([])
            self.assertEqual(code, 2)



    def test_sharded_deltas_are_preserved_and_never_pruned(self):
        manifest_data = {
            "version": 1,
            "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": ["funes-delta-old1.jsonl.gz.enc", "deltas/4a/funes-delta-4a123.jsonl.gz.enc"],
            "controls": [],
        }
        hub = MockHfHub(
            files={
                ".gitattributes": b"",
                "funes-restore-manifest-v1.json": json.dumps(manifest_data).encode("utf-8"),
                "funes-snapshot.jsonl.gz.enc": b"content-base",
                "funes-delta-old1.jsonl.gz.enc": b"content-old1",
                "deltas/4a/funes-delta-4a123.jsonl.gz.enc": b"content-sharded",
                "funes-delta-unref.jsonl.gz.enc": b"content-unref",
            }
        )

        with mock.patch("huggingface_hub.hf_hub_download", side_effect=make_fake_download(hub)):
            inv = compactor.inspect_source_repo(
                api=hub,
                repo="test/repo",
                token="test-token",
            )

        self.assertIn("deltas/4a/funes-delta-4a123.jsonl.gz.enc", inv["protected"])
        self.assertNotIn("deltas/4a/funes-delta-4a123.jsonl.gz.enc", inv["root_deltas"])
        self.assertEqual(inv["unreferenced_root_deltas"], ["funes-delta-unref.jsonl.gz.enc"])


    def test_prune_refreshes_current_head_on_retry_conflict(self):
        api = mock.Mock()
        # First commit call fails with stale parent conflict; second succeeds
        api.create_commit.side_effect = [
            RuntimeError("A commit has happened since. Please refresh."),
            mock.Mock(oid="head-refreshed-commit"),
        ]
        # repo_info returns refreshed head
        api.repo_info.return_value = mock.Mock(sha="head-externally-advanced")

        candidates = ["funes-delta-old1.jsonl.gz.enc"]
        protected = {"funes-snapshot.jsonl.gz.enc"}

        results = compactor.execute_prune_deltas(
            api=api,
            repo="owner/repo",
            candidates=candidates,
            protected=protected,
            parent_commit="head-stale",
            token="test-token",
            retries=3,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["commit_oid"], "head-refreshed-commit")
        self.assertEqual(api.create_commit.call_count, 2)

        # Verify first call used stale parent, second call used refreshed parent
        first_call = api.create_commit.call_args_list[0].kwargs
        second_call = api.create_commit.call_args_list[1].kwargs
        self.assertEqual(first_call["parent_commit"], "head-stale")
        self.assertEqual(first_call["token"], "test-token")
        self.assertEqual(second_call["parent_commit"], "head-externally-advanced")
        self.assertEqual(second_call["token"], "test-token")

        # Verify repo_info was called with explicit token
        api.repo_info.assert_called_with(repo_id="owner/repo", repo_type="dataset", token="test-token")

    def test_inspect_source_repo_full_manifest_validation_fails_closed_on_missing_file(self):
        manifest_data = {
            "version": 1,
            "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": ["funes-delta-nonexistent.jsonl.gz.enc"],
            "controls": [],
        }
        hub = MockHfHub(
            files={
                ".gitattributes": b"",
                "funes-restore-manifest-v1.json": json.dumps(manifest_data).encode("utf-8"),
                "funes-snapshot.jsonl.gz.enc": b"content-base",
            }
        )

        with mock.patch("huggingface_hub.hf_hub_download", side_effect=make_fake_download(hub)):
            with self.assertRaises(FileNotFoundError):
                compactor.inspect_source_repo(
                    api=hub,
                    repo="test/repo",
                    token="test-token",
                )

    def test_inspect_source_repo_full_manifest_validation_fails_closed_on_invalid_schema(self):
        # Version 2 is unsupported
        manifest_data = {
            "version": 2,
            "snapshot": "funes-snapshot.jsonl.gz.enc",
            "deltas": [],
            "controls": [],
        }
        hub = MockHfHub(
            files={
                ".gitattributes": b"",
                "funes-restore-manifest-v1.json": json.dumps(manifest_data).encode("utf-8"),
                "funes-snapshot.jsonl.gz.enc": b"content-base",
            }
        )

        with mock.patch("huggingface_hub.hf_hub_download", side_effect=make_fake_download(hub)):
            with self.assertRaises(ValueError):
                compactor.inspect_source_repo(
                    api=hub,
                    repo="test/repo",
                    token="test-token",
                )

    def test_inspect_source_repo_honors_env_prefixes(self):
        custom_manifest = {
            "version": 1,
            "snapshot": "custom-snapshot-01.jsonl.gz.enc",
            "deltas": ["custom-delta-01.jsonl.gz.enc"],
            "controls": ["custom-reindex-01.jsonl.gz.enc"],
        }
        hub = MockHfHub(
            files={
                ".gitattributes": b"",
                "custom-manifest.json": json.dumps(custom_manifest).encode("utf-8"),
                "custom-snapshot-01.jsonl.gz.enc": b"content",
                "custom-delta-01.jsonl.gz.enc": b"content",
                "custom-reindex-01.jsonl.gz.enc": b"content",
                "custom-delta-legacy.jsonl.gz.enc": b"content",
            }
        )

        with mock.patch("huggingface_hub.hf_hub_download", side_effect=make_fake_download(hub)):
            inv = compactor.inspect_source_repo(
                api=hub,
                repo="test/repo",
                token="test-token",
                manifest_filename="custom-manifest.json",
                snapshot_prefix="custom-snapshot-",
                delta_prefix="custom-delta-",
                control_prefix="custom-reindex-",
            )

        self.assertEqual(inv["active_snapshot"], "custom-snapshot-01.jsonl.gz.enc")
        self.assertEqual(inv["active_deltas"], ["custom-delta-01.jsonl.gz.enc"])
        self.assertEqual(inv["active_controls"], ["custom-reindex-01.jsonl.gz.enc"])
        self.assertEqual(inv["unreferenced_root_deltas"], ["custom-delta-legacy.jsonl.gz.enc"])

if __name__ == "__main__":
    unittest.main()
