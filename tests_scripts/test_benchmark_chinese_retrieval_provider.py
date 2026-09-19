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
    def test_provider_uses_role_specific_prompts_and_query_validation(self) -> None:
        class Response:
            def __init__(self, content: str):
                self.content = content

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": self.content}}]}
                ).encode("utf-8")

        normalizer = benchmark.ProviderNormalizer(
            endpoint="https://provider.example/v1/chat/completions",
            model="test-model",
            token="test-token",
            timeout=1.0,
            retries=1,
        )
        raw_query = "Funes MCP 2.0 怎么检索？"
        with mock.patch.object(
            benchmark.urllib.request,
            "urlopen",
            side_effect=[
                Response("English CPA document"),
                Response("Funes MCP 2.0 retrieval"),
                Response("unrelated answer 1024"),
            ],
        ) as request:
            document = normalizer.normalize_document("中文 CPA 文档")
            query = normalizer.normalize_query(raw_query)
            rejected = normalizer.normalize_query("Funes MCP previous_response_id 怎么检索？")

        document_payload = json.loads(request.call_args_list[0].args[0].data)
        query_payload = json.loads(request.call_args_list[1].args[0].data)
        self.assertEqual(document_payload["messages"][0]["content"], benchmark.RETRIEVAL_PROMPT)
        self.assertNotIn("max_tokens", document_payload)
        self.assertEqual(query_payload["messages"][0]["content"], benchmark.QUERY_RETRIEVAL_PROMPT)
        self.assertEqual(query_payload["max_tokens"], 128)
        self.assertEqual(document.validation_status, "accepted")
        self.assertEqual(query.effective_output, "Funes MCP 2.0 retrieval")
        self.assertEqual(rejected.provider_output, "unrelated answer 1024")
        self.assertEqual(
            rejected.effective_output,
            "Funes MCP previous_response_id 怎么检索？",
        )
        self.assertEqual(rejected.validation_status, "fallback_invalid_query")
        self.assertEqual(normalizer.stats.calls, 3)

    def test_checkpoint_separates_roles_and_rejects_single_prompt_format(self) -> None:
        shared_input = "Funes 中文检索"
        document_outputs = {
            shared_input: benchmark.ProviderOutput(
                provider_output="Funes document retrieval",
                effective_output="Funes document retrieval",
                validation_status="accepted",
            )
        }
        query_outputs = {
            shared_input: benchmark.ProviderOutput(
                provider_output="unrelated answer 1024",
                effective_output=shared_input,
                validation_status="fallback_invalid_query",
            )
        }
        provider = {
            "endpoint": "https://provider.example/v1/chat/completions",
            "model": "test-model",
            **benchmark._prompt_metadata(),
        }
        checkpoint = {
            "benchmark_kind": benchmark.BENCHMARK_KIND,
            "provider": provider,
            "provider_outputs": benchmark._provider_outputs(
                [shared_input], document_outputs, "document"
            )
            + benchmark._provider_outputs([shared_input], query_outputs, "query"),
        }

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.json"

            def load(value):
                path.write_text(json.dumps(value), encoding="utf-8")
                return benchmark._load_provider_checkpoint(
                    path,
                    endpoint=provider["endpoint"],
                    model=provider["model"],
                    document_inputs=[shared_input],
                    query_inputs=[shared_input],
                )

            loaded = load(checkpoint)
            self.assertIsNotNone(loaded)
            loaded_documents, loaded_queries, _metadata = loaded
            self.assertEqual(
                loaded_documents[shared_input].effective_output,
                "Funes document retrieval",
            )
            self.assertEqual(loaded_queries[shared_input].provider_output, "unrelated answer 1024")
            self.assertEqual(loaded_queries[shared_input].effective_output, shared_input)

            stale_query_prompt = json.loads(json.dumps(checkpoint))
            stale_query_prompt["provider"]["query_prompt_sha256"] = "stale"
            self.assertIsNone(load(stale_query_prompt))

            single_prompt = json.loads(json.dumps(checkpoint))
            single_prompt["provider"] = {
                "endpoint": provider["endpoint"],
                "model": provider["model"],
                "prompt_version": benchmark.PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(
                    benchmark.RETRIEVAL_PROMPT.encode("utf-8")
                ).hexdigest(),
            }
            self.assertIsNone(load(single_prompt))

            tampered_hash = json.loads(json.dumps(checkpoint))
            tampered_hash["provider_outputs"][0]["input_sha256"] = "tampered"
            self.assertIsNone(load(tampered_hash))

            duplicate_row = json.loads(json.dumps(checkpoint))
            duplicate_row["provider_outputs"].append(
                duplicate_row["provider_outputs"][0]
            )
            self.assertIsNone(load(duplicate_row))

            unexpected_row = json.loads(json.dumps(checkpoint))
            extra = dict(unexpected_row["provider_outputs"][0])
            extra["input"] = "unexpected"
            extra["input_sha256"] = hashlib.sha256(b"unexpected").hexdigest()
            unexpected_row["provider_outputs"].append(extra)
            self.assertIsNone(load(unexpected_row))

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
