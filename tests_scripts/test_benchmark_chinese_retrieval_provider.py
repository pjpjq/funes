import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import benchmark_chinese_retrieval_provider as benchmark


REQUIRED_CANONICAL_FIELDS = {
    "source_identity",
    "source_version",
    "retrieval_text",
    "content_hash",
    "updated_at",
    "metadata",
}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ProviderBenchmarkTest(unittest.TestCase):
    def test_canonical_sources_keep_after_index_shadow_only(self) -> None:
        memories = benchmark._memories()[:3]
        shadows = {memory.raw: f"English shadow {index}" for index, memory in enumerate(memories)}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before_path = root / "before" / "canonical.jsonl"
            after_path = root / "after" / "canonical.jsonl"

            benchmark._write_canonical_source(before_path, memories, None)
            benchmark._write_canonical_source(after_path, memories, shadows)

            before = _read_jsonl(before_path)
            after = _read_jsonl(after_path)
            after_jsonl = after_path.read_text(encoding="utf-8")
            self.assertTrue(all(REQUIRED_CANONICAL_FIELDS <= row.keys() for row in before + after))
            self.assertTrue(all("raw_text" not in row for row in before + after))
            self.assertEqual([row["retrieval_text"] for row in before], [memory.raw for memory in memories])
            self.assertEqual(
                [row["retrieval_text"] for row in after],
                [shadows[memory.raw] for memory in memories],
            )
            self.assertTrue(all(memory.raw not in after_jsonl for memory in memories))
            self.assertTrue(
                all(
                    row["content_hash"]
                    == hashlib.sha256(str(row["retrieval_text"]).encode("utf-8")).hexdigest()
                    for row in before + after
                )
            )

    def test_ingest_docs_uses_explicit_home_and_memory(self) -> None:
        seen: dict[str, object] = {}

        def fake_run(command, **kwargs):
            seen["command"] = command
            seen["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            benchmark.subprocess,
            "run",
            side_effect=fake_run,
        ):
            root = Path(temp)
            binary = root / "funes"
            home = root / "home"
            memory = root / "memory"
            source = root / "canonical.jsonl"

            benchmark._ingest_docs(binary, home, memory, source, 30, "provider-secret")

        self.assertEqual(
            seen["command"],
            [str(binary), "ingest-docs", str(source), "--memory", str(memory)],
        )
        kwargs = seen["kwargs"]
        self.assertEqual(kwargs["env"]["FUNES_HOME"], str(home))
        self.assertEqual(kwargs["timeout"], 30)

    def test_run_arm_serves_the_explicit_memory(self) -> None:
        calls: dict[str, object] = {}

        class FakeWorker:
            def __init__(self, binary, memory, home, *, timeout):
                calls["worker"] = (binary, memory, home, timeout)

            def recall(self, query, **kwargs):
                calls["recall"] = (query, kwargs)
                return "  → get bench-expected --from 0 --to 3\n"

            def close(self):
                calls["closed"] = True

        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            benchmark,
            "_ingest_docs",
            return_value=0.25,
        ), mock.patch.object(benchmark, "NativeMcpWorker", FakeWorker):
            root = Path(temp)
            query = benchmark.Query("中文查询", "English query", "expected")
            memory = root / "memory"
            summary, rows = benchmark._run_arm(
                name="after",
                binary=root / "funes",
                home=root / "home",
                memory=memory,
                source=root / "canonical.jsonl",
                queries=[query],
                query_shadows={query.text: query.shadow},
                candidates=12,
                index_timeout=30,
                recall_timeout=10.0,
                token="",
            )

        self.assertEqual(calls["worker"][1], str(memory))
        self.assertEqual(
            calls["recall"],
            ("English query", {"k": 5, "candidates": 12, "half_life": 0, "neighbors": 0}),
        )
        self.assertTrue(calls["closed"])
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(summary["recall@5"], 1.0)


if __name__ == "__main__":
    unittest.main()
