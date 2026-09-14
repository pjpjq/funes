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
    def __init__(self, payload, *, status=200):
        self.payload = payload
        self.status = status
        self.read_calls = 0
        self.read_sizes = []
        self.close_calls = 0
        self.exhausted = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size=None):
        self.read_calls += 1
        self.read_sizes.append(size)
        if isinstance(self.payload, bytes):
            encoded = self.payload
        else:
            encoded = json.dumps(self.payload).encode("utf-8")
        if size is None or len(encoded) <= size:
            self.exhausted = True
            return encoded
        return encoded[:size]

    def isclosed(self):
        return self.exhausted or self.close_calls > 0

    def close(self):
        self.close_calls += 1


class FakePersistentConnection:
    def __init__(self, protocol, host, port, timeout, actions):
        self.protocol = protocol
        self.host = host
        self.port = port
        self.timeout = timeout
        self.actions = actions
        self.sock = None
        self.connect_calls = 0
        self.close_calls = 0
        self.requests = []
        self.tunnels = []

    def set_tunnel(self, host, port=None, headers=None):
        self.tunnels.append((host, port, dict(headers or {})))

    def connect(self):
        self.connect_calls += 1
        self.sock = object()

    def request(self, method, target, body=None, headers=None):
        self.requests.append((method, target, body, dict(headers or {})))

    def getresponse(self):
        if not self.actions:
            raise AssertionError("fake HTTP response queue exhausted")
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action

    def close(self):
        self.close_calls += 1
        self.sock = None


@contextlib.contextmanager
def fake_funes_http(actions, *, proxies=None, bypass=False):
    actions = list(actions)
    connections = []

    def factory(protocol):
        def create(host, port=None, timeout=None):
            connection = FakePersistentConnection(
                protocol,
                host,
                port,
                timeout,
                actions,
            )
            connections.append(connection)
            return connection

        return create

    with mock.patch.object(
        benchmark.http.client,
        "HTTPConnection",
        side_effect=factory("http"),
    ), mock.patch.object(
        benchmark.http.client,
        "HTTPSConnection",
        side_effect=factory("https"),
    ), mock.patch.object(
        benchmark.urllib.request,
        "getproxies",
        return_value={} if proxies is None else proxies,
    ), mock.patch.object(
        benchmark.urllib.request,
        "proxy_bypass",
        return_value=bypass,
    ):
        yield connections

    if actions:
        raise AssertionError(f"{len(actions)} fake HTTP responses were not consumed")


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
        responses = [Response(self._voyage_http_payload()) for _ in range(51)]
        clock = iter(index / 1000 for index in range(500))
        with fake_funes_http(responses) as connections, mock.patch.object(
            benchmark.time,
            "monotonic",
            side_effect=lambda: next(clock),
        ):
            result = benchmark.run_http_latency(
                remote_url="https://funes.example/base/",
                token="funes-secret-must-not-leak",
                requests=50,
                timeout=5.0,
            )

        rendered = json.dumps(result, ensure_ascii=False)
        self.assertEqual(len(connections), 1)
        connection = connections[0]
        self.assertEqual(connection.connect_calls, 1)
        self.assertEqual(len(connection.requests), 51)
        self.assertTrue(all(response.read_calls == 1 for response in responses))
        self.assertEqual(result["requests"], 50)
        self.assertEqual(result["warmup_requests"], 1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["end_to_end_latency_ms"], result["latency_ms"])
        self.assertIn("unmeasured warmup", result["latency_scope"])
        self.assertEqual(result["transport"], "direct_https_keep_alive")
        self.assertFalse(result["proxy_used"])
        self.assertEqual(
            result["connection_reuse"]["mode"],
            "single_persistent_connection_with_bounded_reconnect",
        )
        self.assertEqual(result["connection_reuse"]["transport_connections_opened"], 1)
        self.assertEqual(result["connection_reuse"]["request_attempts"], 51)
        self.assertTrue(
            result["connection_reuse"][
                "single_transport_connection_for_warmup_and_measurements"
            ]
        )
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
        self.assertTrue(
            all(
                method == "POST"
                and target == "/base/recall"
                and headers["Authorization"] == "Bearer funes-secret-must-not-leak"
                and headers["Connection"] == "keep-alive"
                for method, target, _body, headers in connection.requests
            )
        )
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
                responses = [
                    Response(self._voyage_http_payload()),
                    Response(adversarial_payload),
                ]
                with fake_funes_http(responses):
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

    def test_http_latency_reconnects_once_after_disconnect(self) -> None:
        secret = "funes-secret-must-not-leak"
        responses = [
            Response(self._voyage_http_payload()),
            benchmark.http.client.RemoteDisconnected("raw disconnect detail " + secret),
            Response(self._voyage_http_payload()),
        ]
        with fake_funes_http(responses) as connections:
            result = benchmark.run_http_latency(
                remote_url="https://funes.example",
                token=secret,
                requests=1,
                timeout=5.0,
            )

        rendered = json.dumps(result, ensure_ascii=False)
        self.assertEqual(len(connections), 2)
        self.assertEqual([connection.connect_calls for connection in connections], [1, 1])
        self.assertEqual(sum(len(connection.requests) for connection in connections), 3)
        self.assertEqual(result["connection_reuse"]["disconnect_retries"], 1)
        self.assertEqual(result["connection_reuse"]["transport_connections_opened"], 2)
        self.assertEqual(result["connection_reuse"]["transport_reconnections"], 1)
        self.assertFalse(
            result["connection_reuse"][
                "single_transport_connection_for_warmup_and_measurements"
            ]
        )
        self.assertNotIn(secret, rendered)
        self.assertNotIn("raw disconnect detail", rendered)

    def test_http_latency_stops_after_one_disconnect_retry(self) -> None:
        secret = "funes-secret-must-not-leak"
        responses = [
            Response(self._voyage_http_payload()),
            benchmark.http.client.RemoteDisconnected("first raw failure " + secret),
            benchmark.http.client.RemoteDisconnected("second raw failure " + secret),
        ]
        with fake_funes_http(responses) as connections, self.assertRaises(
            benchmark.BenchmarkError
        ) as caught:
            benchmark.run_http_latency(
                remote_url="https://funes.example",
                token=secret,
                requests=1,
                timeout=5.0,
            )

        message = str(caught.exception)
        self.assertEqual(len(connections), 2)
        self.assertEqual(sum(len(connection.requests) for connection in connections), 3)
        self.assertIn("transport failure (RemoteDisconnected)", message)
        self.assertNotIn(secret, message)
        self.assertNotIn("raw failure", message)

    def test_https_proxy_uses_connect_without_target_authorization(self) -> None:
        secret = "funes-secret-must-not-leak"
        responses = [
            Response(self._voyage_http_payload()),
            Response(self._voyage_http_payload()),
        ]
        with fake_funes_http(
            responses,
            proxies={"https": "http://127.0.0.1:6324"},
        ) as connections:
            result = benchmark.run_http_latency(
                remote_url="https://funes.example/base",
                token=secret,
                requests=1,
                timeout=5.0,
            )

        self.assertEqual(len(connections), 1)
        connection = connections[0]
        self.assertEqual(
            (connection.protocol, connection.host, connection.port),
            ("https", "127.0.0.1", 6324),
        )
        self.assertEqual(connection.tunnels, [("funes.example", 443, {})])
        self.assertNotIn("Authorization", connection.tunnels[0][2])
        self.assertTrue(
            all(
                target == "/base/recall"
                and headers["Authorization"] == "Bearer " + secret
                for _method, target, _body, headers in connection.requests
            )
        )
        self.assertEqual(result["transport"], "https_over_http_connect_keep_alive")
        self.assertTrue(result["proxy_used"])
        self.assertNotIn(secret, json.dumps(result))
        self.assertNotIn("127.0.0.1", json.dumps(result))

    def test_no_proxy_bypasses_configured_proxy(self) -> None:
        responses = [
            Response(self._voyage_http_payload()),
            Response(self._voyage_http_payload()),
        ]
        with fake_funes_http(
            responses,
            proxies={"https": "http://127.0.0.1:6324"},
            bypass=True,
        ) as connections:
            result = benchmark.run_http_latency(
                remote_url="https://funes.example",
                token="secret",
                requests=1,
                timeout=5.0,
            )

        self.assertEqual((connections[0].host, connections[0].port), ("funes.example", 443))
        self.assertEqual(connections[0].tunnels, [])
        self.assertFalse(result["proxy_used"])
        self.assertEqual(result["transport"], "direct_https_keep_alive")

    def test_http_url_proxy_and_response_errors_are_sanitized(self) -> None:
        invalid_urls = (
            "https://user:password@funes.example",
            "https://funes.example?token=raw-secret",
            "https://funes.example#raw-secret",
        )
        for remote_url in invalid_urls:
            with self.subTest(remote_url=remote_url), self.assertRaises(
                benchmark.BenchmarkError
            ) as caught:
                benchmark.run_http_latency(
                    remote_url=remote_url,
                    token="funes-secret",
                    requests=1,
                    timeout=5.0,
                )
            self.assertNotIn("password", str(caught.exception))
            self.assertNotIn("raw-secret", str(caught.exception))

        with mock.patch.object(
            benchmark.urllib.request,
            "getproxies",
            return_value={"https": "http://proxy-user:proxy-secret@127.0.0.1:6324"},
        ), mock.patch.object(
            benchmark.urllib.request,
            "proxy_bypass",
            return_value=False,
        ), self.assertRaises(benchmark.BenchmarkError) as caught:
            benchmark.run_http_latency(
                remote_url="https://funes.example",
                token="funes-secret",
                requests=1,
                timeout=5.0,
            )
        self.assertNotIn("proxy-user", str(caught.exception))
        self.assertNotIn("proxy-secret", str(caught.exception))

        error_response = Response(b"raw response funes-secret", status=302)
        with fake_funes_http([error_response]), self.assertRaises(
            benchmark.BenchmarkError
        ) as caught:
            benchmark.run_http_latency(
                remote_url="https://funes.example",
                token="funes-secret",
                requests=1,
                timeout=5.0,
            )
        self.assertEqual(error_response.read_calls, 1)
        self.assertEqual(
            error_response.read_sizes,
            [benchmark.MAX_HTTP_ERROR_BODY_BYTES + 1],
        )
        self.assertEqual(str(caught.exception), "Funes recall HTTP 302")
        self.assertNotIn("raw response", str(caught.exception))

    def test_large_http_error_body_is_bounded_and_discards_connection(self) -> None:
        secret = "funes-secret-must-not-leak"
        marker = b"raw-large-error-body-must-not-leak"
        error_response = Response(
            marker + b"x" * (2 * 1024 * 1024),
            status=503,
        )
        with fake_funes_http([error_response]) as connections:
            client = benchmark._PersistentFunesHttpClient(
                endpoint=benchmark._validate_remote_url("https://funes.example"),
                token=secret,
                timeout=5.0,
            )
            with self.assertRaises(benchmark.BenchmarkError) as caught:
                client.post_recall("raw query must not leak")

            self.assertIsNone(client._connection)

        message = str(caught.exception)
        self.assertEqual(message, "Funes recall HTTP 503")
        self.assertEqual(error_response.read_calls, 1)
        self.assertEqual(
            error_response.read_sizes,
            [benchmark.MAX_HTTP_ERROR_BODY_BYTES + 1],
        )
        self.assertEqual(connections[0].close_calls, 1)
        self.assertNotIn(marker.decode(), message)
        self.assertNotIn(secret, message)
        self.assertNotIn(marker.decode(), repr(client.__dict__))

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
