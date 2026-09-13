import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from scripts import benchmark_voyage_retrieval as benchmark


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class VoyageRetrievalBenchmarkTest(unittest.TestCase):
    @staticmethod
    def _voyage_http_payload(**overrides):
        payload = {
            "query": "raw query must not escape",
            "results": [{"raw_text": "raw memory must not escape"}],
            "retrieval_backend": "voyage_lance_bm25_rrf",
            "embedding_profile": {
                "provider": "voyage",
                "model": "voyage-4-lite",
                "dimensions": 1024,
                "schema_version": 2,
                "fingerprint": "voyage-profile-fingerprint",
            },
        }
        payload.update(overrides)
        return payload

    def test_default_cli_is_an_offline_self_check(self) -> None:
        output = io.StringIO()
        with mock.patch.object(
            benchmark.urllib.request,
            "urlopen",
            side_effect=AssertionError("default mode must not use the network"),
        ), contextlib.redirect_stdout(output):
            status = benchmark.main([])

        result = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "offline_self_check")
        self.assertEqual(result["memories"], 60)
        self.assertEqual(result["queries"], 20)

    def test_matrix_contains_only_the_requested_compatible_arms(self) -> None:
        self.assertEqual(
            [arm.name for arm in benchmark.MATRIX],
            [
                "voyage-4-lite",
                "voyage-4",
                "voyage-4-doc__voyage-4-lite-query",
                "voyage-code-4",
                "voyage-4-lite__rerank-3-lite",
            ],
        )
        mixed = benchmark.MATRIX[2]
        self.assertEqual((mixed.document_model, mixed.query_model), ("voyage-4", "voyage-4-lite"))
        self.assertTrue(benchmark._models_share_space(mixed.document_model, mixed.query_model))
        self.assertFalse(benchmark._models_share_space("voyage-code-4", "voyage-4-lite"))

    def test_embedding_requests_use_roles_and_persistent_cache(self) -> None:
        secret = "voyage-secret-must-not-leak"
        requests = []

        def urlopen(request, timeout):
            requests.append((request, timeout))
            payload = json.loads(request.data)
            data = [
                {"index": index, "embedding": [float(index + 1), 0.5]}
                for index, _text in enumerate(payload["input"])
            ]
            return Response(
                {
                    "data": list(reversed(data)),
                    "model": payload["model"],
                    "usage": {"total_tokens": 4},
                }
            )

        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            benchmark.urllib.request,
            "urlopen",
            side_effect=urlopen,
        ):
            cache_path = Path(temp) / "embeddings.json"
            cache = benchmark.EmbeddingCache(cache_path)
            client = benchmark.VoyageClient(
                api_key=secret,
                cache=cache,
                timeout=2.0,
                attempts=1,
            )
            documents = client.embed(
                ["中文文档", "English document"],
                model="voyage-4-lite",
                input_type="document",
            )
            cached = client.embed(
                ["中文文档", "English document"],
                model="voyage-4-lite",
                input_type="document",
            )
            query = client.embed(
                ["混合 query"],
                model="voyage-4-lite",
                input_type="query",
            )

            cache_text = cache_path.read_text(encoding="utf-8")

        self.assertEqual(documents, [[1.0, 0.5], [2.0, 0.5]])
        self.assertEqual(cached, documents)
        self.assertEqual(query, [[1.0, 0.5]])
        self.assertEqual(len(requests), 2)
        payloads = [json.loads(request.data) for request, _timeout in requests]
        self.assertEqual([payload["input_type"] for payload in payloads], ["document", "query"])
        self.assertTrue(all(request.full_url == benchmark.EMBEDDINGS_URL for request, _ in requests))
        self.assertTrue(
            all(request.get_header("Authorization") == "Bearer " + secret for request, _ in requests)
        )
        self.assertNotIn(secret, cache_text)
        self.assertNotIn("中文文档", cache_text)
        self.assertNotIn("English document", cache_text)
        self.assertEqual(client.stats.embedding_api_calls, 2)
        self.assertEqual(client.stats.embedding_cache_hits, 2)
        self.assertEqual(client.stats.embedding_cache_misses, 3)

    def test_embedding_response_model_must_match_the_requested_profile(self) -> None:
        def urlopen(_request, timeout):
            self.assertEqual(timeout, 2.0)
            return Response(
                {
                    "data": [{"index": 0, "embedding": [1.0, 0.5]}],
                    "model": "different-model",
                }
            )

        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            benchmark.urllib.request,
            "urlopen",
            side_effect=urlopen,
        ):
            client = benchmark.VoyageClient(
                api_key="secret",
                cache=benchmark.EmbeddingCache(Path(temp) / "cache.json"),
                timeout=2.0,
                attempts=1,
            )
            with self.assertRaisesRegex(benchmark.BenchmarkError, "model does not match"):
                client.embed(
                    ["query"],
                    model="voyage-4-lite",
                    input_type="query",
                )

    def test_rerank_returns_original_candidate_indices(self) -> None:
        seen = {}

        def urlopen(request, timeout):
            seen["request"] = request
            seen["timeout"] = timeout
            return Response(
                {
                    "data": [
                        {"index": 2, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.8},
                        {"index": 1, "relevance_score": 0.1},
                    ],
                    "model": "rerank-3-lite",
                }
            )

        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            benchmark.urllib.request,
            "urlopen",
            side_effect=urlopen,
        ):
            client = benchmark.VoyageClient(
                api_key="secret",
                cache=benchmark.EmbeddingCache(Path(temp) / "cache.json"),
                timeout=3.0,
                attempts=1,
            )
            ranked = client.rerank(
                "query",
                ["zero", "one", "two"],
                model="rerank-3-lite",
            )

        payload = json.loads(seen["request"].data)
        self.assertEqual(ranked, [2, 0, 1])
        self.assertEqual(seen["request"].full_url, benchmark.RERANK_URL)
        self.assertEqual(payload["model"], "rerank-3-lite")
        self.assertEqual(payload["top_k"], 3)
        self.assertFalse(payload["return_documents"])

    def test_metrics_include_recall_mrr_and_linear_percentiles(self) -> None:
        self.assertEqual(
            benchmark._quality_metrics([1, 3, 5, None]),
            {"recall@1": 0.25, "recall@3": 0.5, "recall@5": 0.75, "mrr": 0.3833},
        )
        self.assertEqual(
            benchmark._latency_metrics([10.0, 20.0, 30.0, 40.0]),
            {"p50": 25.0, "p95": 38.5, "max": 40.0},
        )

    def test_live_matrix_warms_each_arm_in_an_isolated_cache_namespace(self) -> None:
        embedding_requests = []

        def urlopen(request, timeout):
            payload = json.loads(request.data)
            embedding_requests.append((request, timeout, payload))
            return Response(
                {
                    "data": [
                        {"index": index, "embedding": [float(index + 1), 1.0]}
                        for index, _text in enumerate(payload["input"])
                    ],
                    "model": payload["model"],
                }
            )

        arms = (
            benchmark.BenchmarkArm("same-model-a", "voyage-4-lite", "voyage-4-lite"),
            benchmark.BenchmarkArm("same-model-b", "voyage-4-lite", "voyage-4-lite"),
        )
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            benchmark,
            "MATRIX",
            arms,
        ), mock.patch.object(benchmark.urllib.request, "urlopen", side_effect=urlopen):
            result = benchmark.run_voyage_benchmark(
                api_key="secret",
                cache_path=Path(temp) / "embeddings.json",
                timeout=2.0,
                attempts=1,
                batch_size=128,
                candidate_k=20,
            )

            cache_files = sorted(Path(temp).glob("*.json"))

        self.assertEqual(len(embedding_requests), 4)
        self.assertEqual(len(cache_files), 2)
        self.assertEqual(result["method"]["cache_protocol"], "isolated_per_arm_then_fully_warm")
        self.assertEqual([row["cache_namespace"] for row in result["arms"]], [arm.name for arm in arms])
        for arm_result in result["arms"]:
            self.assertEqual(arm_result["actual_backend"], "Voyage native REST API")
            self.assertEqual(
                arm_result["actual_profile"]["document_model"],
                arm_result["document_model"],
            )
            self.assertEqual(arm_result["warmup"]["api_and_cache"]["embedding_api_calls"], 2)
            self.assertEqual(arm_result["warmup"]["api_network_latency_ms"]["embedding"]["requests"], 2)
            self.assertEqual(arm_result["api_and_cache"]["embedding_api_calls"], 0)
            self.assertEqual(arm_result["api_network_latency_ms"]["embedding"]["requests"], 0)
            self.assertIsNone(arm_result["api_network_latency_ms"]["embedding"]["latency_ms"])
            self.assertEqual(arm_result["measured_queries"], 20)
            self.assertIn("p95", arm_result["end_to_end_latency_ms"])

    def test_http_latency_warms_then_measures_50_without_raw_output(self) -> None:
        requests = []

        def urlopen(request, timeout):
            requests.append((request, timeout))
            return Response(self._voyage_http_payload())

        clock = iter(index / 1000 for index in range(500))
        with mock.patch.object(
            benchmark.urllib.request,
            "urlopen",
            side_effect=urlopen,
        ), mock.patch.object(benchmark.time, "monotonic", side_effect=lambda: next(clock)):
            result = benchmark.run_http_latency(
                remote_url="https://funes.example/base/",
                token="funes-secret-must-not-leak",
                requests=50,
                timeout=5.0,
            )

        rendered = json.dumps(result, ensure_ascii=False)
        self.assertEqual(len(requests), 51)
        self.assertEqual(result["requests"], 50)
        self.assertEqual(result["warmup_requests"], 1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["end_to_end_latency_ms"], result["latency_ms"])
        self.assertEqual(result["voyage_validation"]["verified_voyage_requests"], 50)
        self.assertEqual(result["voyage_validation"]["non_voyage_requests"], 0)
        self.assertEqual(
            result["voyage_validation"]["observed_retrieval_backends"],
            [{"retrieval_backend": "voyage_lance_bm25_rrf", "observations": 51}],
        )
        self.assertEqual(
            result["voyage_validation"]["observed_embedding_profiles"][0]["embedding_profile"]["fingerprint"],
            "voyage-profile-fingerprint",
        )
        self.assertEqual(sum(row["requests"] for row in result["by_category"].values()), 50)
        self.assertEqual(
            set(result["by_category"]),
            {"chinese", "english", "mixed", "semantic_paraphrase", "exact_identifier", "code_error"},
        )
        self.assertTrue(all(request.full_url == "https://funes.example/base/recall" for request, _ in requests))
        self.assertNotIn("funes-secret-must-not-leak", rendered)
        self.assertNotIn("raw query", rendered)
        self.assertNotIn("raw memory", rendered)

    def test_http_latency_does_not_count_degraded_or_mismatched_responses_as_voyage(self) -> None:
        cases = (
            (
                "degraded",
                self._voyage_http_payload(retrieval_degraded="voyage_unavailable"),
                "degraded",
            ),
            (
                "backend mismatch",
                self._voyage_http_payload(retrieval_backend="legacy_sidecar"),
                "backend_mismatch",
            ),
            (
                "provider mismatch",
                self._voyage_http_payload(
                    embedding_profile={
                        "provider": "local",
                        "model": "BAAI/bge-small-en-v1.5",
                        "dimensions": 384,
                        "schema_version": 1,
                        "fingerprint": "local-profile-fingerprint",
                    }
                ),
                "profile_mismatch",
            ),
            (
                "model mismatch",
                self._voyage_http_payload(
                    embedding_profile={
                        "provider": "voyage",
                        "model": "voyage-4",
                        "dimensions": 1024,
                        "schema_version": 2,
                        "fingerprint": "different-model-profile",
                    }
                ),
                "profile_mismatch",
            ),
            (
                "dimension mismatch",
                self._voyage_http_payload(
                    embedding_profile={
                        "provider": "voyage",
                        "model": "voyage-4-lite",
                        "dimensions": 384,
                        "schema_version": 2,
                        "fingerprint": "different-dimension-profile",
                    }
                ),
                "profile_mismatch",
            ),
        )
        for label, adversarial_payload, expected_failure in cases:
            with self.subTest(label=label):
                responses = iter(
                    [Response(self._voyage_http_payload()), Response(adversarial_payload)]
                )
                with mock.patch.object(
                    benchmark.urllib.request,
                    "urlopen",
                    side_effect=lambda *_args, **_kwargs: next(responses),
                ):
                    result = benchmark.run_http_latency(
                        remote_url="https://funes.example",
                        token="secret",
                        requests=1,
                        timeout=5.0,
                    )

                validation = result["voyage_validation"]
                self.assertEqual(result["status"], "failed_validation")
                self.assertEqual(validation["verified_voyage_requests"], 0)
                self.assertEqual(validation["non_voyage_requests"], 1)
                self.assertEqual(validation["failure_counts"][expected_failure], 1)

    def test_http_error_does_not_expose_key_or_response_body(self) -> None:
        secret = "voyage-secret-must-not-leak"
        error = urllib.error.HTTPError(
            benchmark.EMBEDDINGS_URL,
            401,
            "body echoed " + secret,
            {},
            None,
        )
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            benchmark.urllib.request,
            "urlopen",
            side_effect=error,
        ):
            client = benchmark.VoyageClient(
                api_key=secret,
                cache=benchmark.EmbeddingCache(Path(temp) / "cache.json"),
                timeout=1.0,
                attempts=1,
            )
            with self.assertRaises(benchmark.BenchmarkError) as caught:
                client.embed(["sensitive input"], model="voyage-4-lite", input_type="document")

        error.close()
        message = str(caught.exception)
        self.assertIn("HTTP 401", message)
        self.assertNotIn(secret, message)
        self.assertNotIn("sensitive input", message)


if __name__ == "__main__":
    unittest.main()
