import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).parents[1] / "funes_sync.py"
spec = importlib.util.spec_from_file_location("funes_sync", MODULE)
mod = importlib.util.module_from_spec(spec)
sys.modules["funes_sync"] = mod
spec.loader.exec_module(mod)


class SyncLayerTests(unittest.TestCase):
    def test_stable_dedup_does_not_rewrite_unchanged_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "home"
            memory = home / ".codex" / "memories" / "note.md"
            memory.parent.mkdir(parents=True)
            memory.write_text("决策：保留原始来源\n", encoding="utf-8")
            store = mod.SyncStore(root / ".funes")
            calls = []

            def runner(cmd, **kwargs):
                calls.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, "", "")

            engine = mod.SyncEngine(store, mod.Discoverer(home=home, repos=[]), runner)
            with patch.dict(os.environ, {"FUNES_MEMORY": "", "FUNES_CONFIG": str(root / "missing.toml")}, clear=False):
                first = engine.sync_once()
            generated = next(store.sources.glob("*.jsonl"))
            before = generated.read_bytes()
            first_mtime = generated.stat().st_mtime_ns
            with patch.dict(os.environ, {"FUNES_MEMORY": "", "FUNES_CONFIG": str(root / "missing.toml")}, clear=False):
                second = engine.sync_once()
            self.assertEqual(first["changed"], 1)
            self.assertEqual(second["changed"], 0)
            self.assertEqual(before, generated.read_bytes())
            self.assertEqual(first_mtime, generated.stat().st_mtime_ns)
            self.assertEqual(len(calls), 1)

    def test_offline_push_keeps_pending_queue_until_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "home"
            source = home / ".codex" / "AGENTS.md"
            source.parent.mkdir(parents=True)
            source.write_text("offline queue", encoding="utf-8")
            store = mod.SyncStore(root / ".funes")
            calls = []
            push_results = [1, 0]

            def runner(cmd, **kwargs):
                calls.append(cmd)
                if cmd[1] == "push":
                    return subprocess.CompletedProcess(cmd, push_results.pop(0), "", "offline" if push_results else "")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            engine = mod.SyncEngine(store, mod.Discoverer(home=home, repos=[]), runner)
            with patch.dict(os.environ, {"FUNES_MEMORY": "org/memory", "FUNES_BIN": "funes"}, clear=False):
                failed = engine.sync_once()
                self.assertFalse(failed["pushed"])
                self.assertEqual(failed["pending"], 1)
                state = store.load()
                self.assertEqual(len(state["pending"]), 1)
                retried = engine.sync_once()
            self.assertTrue(retried["pushed"])
            self.assertEqual(retried["pending"], 0)
            self.assertEqual(len(store.load()["pending"]), 0)
            self.assertEqual([c[1] for c in calls], ["index", "push", "index", "push"])

    def test_memory_conversion_preserves_raw_unicode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_path = root / "MEMORY.md"
            raw = "# 记忆\n\n不要丢失中文与 emoji 🚀。\n"
            source_path.write_text(raw, encoding="utf-8")
            source = mod.Source(source_path, "memory")
            destination = root / "out.jsonl"
            mod.convert_source(source, destination)
            records = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[1]["message"]["content"], raw)
            self.assertEqual(records[0]["uuid"].split("-")[0], records[1]["uuid"].split("-")[0])


if __name__ == "__main__":
    unittest.main()
