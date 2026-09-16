import gzip
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import sync.client as client_module
from sync.client import SyncClient
from sync.config import Config


def cfg(tmp_path):
    return Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")


def test_keychain_lookup_is_darwin_only_and_keeps_value_out_of_argv(monkeypatch):
    monkeypatch.setattr(client_module.sys, "platform", "darwin")
    monkeypatch.setenv("USER", "tester")
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="keychain-value\n")

    monkeypatch.setattr(client_module.subprocess, "run", run)
    assert client_module._keychain_token("funes-api-token") == "keychain-value"

    args, kwargs = calls[0]
    assert args == [
        "/usr/bin/security",
        "find-generic-password",
        "-a",
        "tester",
        "-s",
        "funes-api-token",
        "-w",
    ]
    assert "keychain-value" not in args
    assert kwargs["stdin"] is client_module.subprocess.DEVNULL

    monkeypatch.setattr(client_module.sys, "platform", "linux")
    assert client_module._keychain_token("funes-api-token") == ""


def test_sync_methods_use_keychain_credentials_when_environment_is_empty(tmp_path, monkeypatch):
    for name in ("FUNES_API_TOKEN", "FUNES_HF_TOKEN", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    credentials = {
        "funes-api-token": "api-keychain-value",
        "funes-hf-token": "hf-keychain-value",
    }
    monkeypatch.setattr(client_module, "_keychain_token", credentials.__getitem__)
    requests = []

    class Response:
        def __init__(self, payload, status=200):
            self.payload = payload
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def urlopen(req, **kwargs):
        requests.append(req)
        if req.full_url.endswith("/ingest"):
            return Response({"durable": True, "accepted": 1})
        if req.full_url.endswith("/sync"):
            return Response({"durable": True})
        if req.full_url.endswith("/reindex"):
            return Response(
                {"durable": True, "queued": True, "scope": "all"}, status=202
            )
        return Response({})

    monkeypatch.setattr(client_module, "open_no_redirect", urlopen)
    client = SyncClient(cfg(tmp_path))

    assert client.health()
    assert client.ingest([{"raw_text": "one"}])["accepted"] == 1
    assert client.sync_snapshot()["durable"] is True
    assert client.reindex("all")["queued"] is True

    assert len(requests) == 4
    for req in requests:
        headers = {key.lower(): value for key, value in req.header_items()}
        assert headers["authorization"] == "Bearer hf-keychain-value"
        assert headers["x-funes-authorization"] == "Bearer api-keychain-value"
    assert json.loads(requests[-1].data) == {"scope": "all"}


def test_source_inventory_check_is_bounded_and_validated(tmp_path, monkeypatch):
    monkeypatch.setenv("FUNES_API_TOKEN", "api-environment-value")
    monkeypatch.setattr(client_module, "_keychain_token", lambda _service: "")
    requests=[]

    class Response:
        status=200

        def __enter__(self):
            return self

        def __exit__(self,*_):
            return False

        def read(self,_limit):
            return json.dumps(
                {"ok":True,"present":["present"],"missing":["missing"]}
            ).encode()

    def urlopen(req,**_kwargs):
        requests.append(req)
        return Response()

    monkeypatch.setattr(client_module,"open_no_redirect",urlopen)
    client=SyncClient(cfg(tmp_path))

    assert client.missing_source_identities([]) == []
    assert client.missing_source_identities(["present","missing","present"]) == ["missing"]
    assert json.loads(requests[0].data) == {
        "source_identities":["present","missing"]
    }
    assert requests[0].full_url.endswith("/sources/check")
    with pytest.raises(ValueError,match="1-5000"):
        client.missing_source_identities([str(value) for value in range(5001)])


def test_source_inventory_check_rejects_inconsistent_response(tmp_path, monkeypatch):
    monkeypatch.setenv("FUNES_API_TOKEN", "api-environment-value")
    monkeypatch.setattr(client_module,"_keychain_token",lambda _service: "")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self,*_):
            return False

        def read(self,_limit):
            return b'{"ok":true,"present":["one"],"missing":[]}'

    monkeypatch.setattr(client_module,"open_no_redirect",lambda *_args,**_kwargs: Response())

    with pytest.raises(RuntimeError,match="inconsistent inventory"):
        SyncClient(cfg(tmp_path)).missing_source_identities(["one","two"])


def test_ingest_gzips_large_payload_and_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("FUNES_API_TOKEN", "api-environment-value")
    monkeypatch.setattr(client_module, "_keychain_token", lambda _service: "")
    requests = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"durable":true,"accepted":1}'

    def urlopen(req, **_kwargs):
        requests.append(req)
        return Response()

    monkeypatch.setattr(client_module, "open_no_redirect", urlopen)
    records = [{"source_identity": "large", "raw_text": "repeated memory text " * 2048}]
    client = SyncClient(cfg(tmp_path))

    assert client.ingest(records)["accepted"] == 1
    compressed = requests[-1]
    compressed_headers = {key.lower(): value for key, value in compressed.header_items()}
    original = json.dumps({"device_id": client.config.device_id, "documents": records}, ensure_ascii=False).encode()
    assert compressed_headers["content-encoding"] == "gzip"
    assert gzip.decompress(compressed.data) == original
    assert len(compressed.data) < len(original) // 10

    monkeypatch.setenv("FUNES_HTTP_GZIP", "false")
    assert client.ingest(records)["accepted"] == 1
    disabled_headers = {key.lower(): value for key, value in requests[-1].header_items()}
    assert "content-encoding" not in disabled_headers
    assert requests[-1].data == original


@pytest.mark.parametrize("status", [200, 201])
def test_reindex_rejects_non_202_success_status(tmp_path, monkeypatch, status):
    monkeypatch.setenv("FUNES_API_TOKEN", "api-environment-value")
    monkeypatch.delenv("FUNES_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(client_module, "_keychain_token", lambda _service: "")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"durable":true,"queued":true,"scope":"all"}'

    response = Response()
    response.status = status
    monkeypatch.setattr(client_module, "open_no_redirect", lambda *_args, **_kwargs: response)
    with pytest.raises(RuntimeError, match=f"HTTP {status}, expected 202"):
        SyncClient(cfg(tmp_path)).reindex("all")


def test_environment_credentials_take_precedence_over_keychain(tmp_path, monkeypatch):
    monkeypatch.setenv("FUNES_API_TOKEN", "api-environment-value")
    monkeypatch.setenv("FUNES_HF_TOKEN", "hf-environment-value")

    def unexpected_lookup(_service):
        pytest.fail("Keychain lookup should not run when environment credentials exist")

    monkeypatch.setattr(client_module, "_keychain_token", unexpected_lookup)

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def urlopen(req, **kwargs):
        headers = {key.lower(): value for key, value in req.header_items()}
        assert headers["authorization"] == "Bearer hf-environment-value"
        assert headers["x-funes-authorization"] == "Bearer api-environment-value"
        return Response()

    monkeypatch.setattr(client_module, "open_no_redirect", urlopen)
    assert SyncClient(cfg(tmp_path)).health()


class _IngestResponse:
    def __init__(self, status, payload, headers=None):
        self.status = status
        self.payload = payload
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _async_client_environment(monkeypatch, timeout="30"):
    monkeypatch.setenv("FUNES_API_TOKEN", "api-environment-value")
    monkeypatch.delenv("FUNES_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("FUNES_REMOTE_TIMEOUT", timeout)
    monkeypatch.setattr(client_module, "_keychain_token", lambda _service: "")


def test_ingest_requests_async_and_polls_until_durable_with_one_deadline(tmp_path, monkeypatch):
    _async_client_environment(monkeypatch)
    clock = [100.0]
    sleeps = []
    requests = []
    timeouts = []
    operation_id = "operation-1"
    responses = [
        _IngestResponse(
            202,
            {
                "operation_id": operation_id,
                "status_url": f"/ingest/operations/{operation_id}",
                "status": "running",
                "durable": False,
            },
            {"Retry-After": "3"},
        ),
        _IngestResponse(
            202,
            {"operation_id": operation_id, "status": "running", "durable": False},
            {"Retry-After": "4"},
        ),
        _IngestResponse(
            200,
            {
                "operation_id": operation_id,
                "status": "succeeded",
                "durable": True,
                "accepted": 1,
            },
        ),
    ]

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    def urlopen(req, timeout):
        requests.append(req)
        timeouts.append(timeout)
        return responses.pop(0)

    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(client_module.time, "sleep", sleep)
    monkeypatch.setattr(client_module, "open_no_redirect", urlopen)

    result = SyncClient(cfg(tmp_path)).ingest([{"raw_text": "queued until durable"}])

    assert result["durable"] is True
    assert result["accepted"] == 1
    assert [req.get_method() for req in requests] == ["POST", "GET", "GET"]
    assert requests[1].full_url == f"http://127.0.0.1:7860/ingest/operations/{operation_id}"
    post_headers = {name.lower(): value for name, value in requests[0].header_items()}
    assert post_headers["prefer"] == "respond-async"
    assert "prefer" not in {
        name.lower(): value for name, value in requests[1].header_items()
    }
    assert sleeps == [3.0, 4.0]
    assert timeouts == [30.0, 27.0, 23.0]


def test_ingest_reposts_same_payload_after_async_operation_failure(tmp_path, monkeypatch):
    _async_client_environment(monkeypatch)
    clock = [0.0]
    requests = []
    operation_id = "operation-retry"
    responses = [
        _IngestResponse(
            202,
            {
                "operation_id": operation_id,
                "status_url": f"/ingest/operations/{operation_id}",
                "durable": False,
            },
            {"Retry-After": "1"},
        ),
        _IngestResponse(
            503,
            {
                "operation_id": operation_id,
                "status": "failed",
                "durable": False,
                "error": "ingest_failed",
            },
            {"Retry-After": "2"},
        ),
        _IngestResponse(
            202,
            {
                "operation_id": operation_id,
                "status_url": f"/ingest/operations/{operation_id}",
                "durable": False,
            },
            {"Retry-After": "1"},
        ),
        _IngestResponse(
            200,
            {"operation_id": operation_id, "durable": True, "accepted": 1},
        ),
    ]

    def urlopen(req, timeout):
        requests.append((req, timeout))
        return responses.pop(0)

    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        client_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr(client_module, "open_no_redirect", urlopen)

    result = SyncClient(cfg(tmp_path)).ingest([{"raw_text": "retry safely"}])

    assert result["durable"] is True
    assert [req.get_method() for req, _timeout in requests] == ["POST", "GET", "POST", "GET"]
    assert requests[0][0].data == requests[2][0].data
    assert all(0 < timeout <= 30 for _req, timeout in requests)


def test_ingest_timeout_never_treats_running_operation_as_durable(tmp_path, monkeypatch):
    _async_client_environment(monkeypatch, timeout="3")
    clock = [0.0]
    requests = []
    operation_id = "operation-timeout"
    responses = [
        _IngestResponse(
            202,
            {
                "operation_id": operation_id,
                "status_url": f"/ingest/operations/{operation_id}",
                "durable": False,
            },
            {"Retry-After": "1"},
        ),
        _IngestResponse(
            202,
            {"operation_id": operation_id, "status": "running", "durable": False},
            {"Retry-After": "10"},
        ),
    ]

    def urlopen(req, timeout):
        requests.append((req, timeout))
        return responses.pop(0)

    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        client_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr(client_module, "open_no_redirect", urlopen)

    with pytest.raises(RuntimeError, match="timed out before durable confirmation"):
        SyncClient(cfg(tmp_path)).ingest([{"raw_text": "must remain locally queued"}])

    assert [req.get_method() for req, _timeout in requests] == ["POST", "GET"]
    assert [timeout for _req, timeout in requests] == [3.0, 2.0]
    assert clock[0] == 3.0


def test_ingest_does_not_retry_permanent_http_error(tmp_path, monkeypatch):
    _async_client_environment(monkeypatch)
    sleeps = []

    def urlopen(req, timeout):
        raise client_module.error.HTTPError(req.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(client_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(client_module, "open_no_redirect", urlopen)
    with pytest.raises(client_module.error.HTTPError) as exc_info:
        SyncClient(cfg(tmp_path)).ingest([{"raw_text": "not acknowledged"}])
    assert exc_info.value.code == 401
    assert sleeps == []


def test_ingest_reposts_same_payload_when_operation_status_is_missing(tmp_path, monkeypatch):
    _async_client_environment(monkeypatch)
    clock = [0.0]
    requests = []
    operation_id = "operation-missing"
    responses = [
        _IngestResponse(
            202,
            {
                "operation_id": operation_id,
                "status_url": f"/ingest/operations/{operation_id}",
                "durable": False,
            },
            {"Retry-After": "1"},
        ),
        client_module.error.HTTPError(
            f"http://127.0.0.1:7860/ingest/operations/{operation_id}",
            404,
            "not found",
            {"Retry-After": "1"},
            None,
        ),
        _IngestResponse(
            202,
            {
                "operation_id": operation_id,
                "status_url": f"/ingest/operations/{operation_id}",
                "durable": False,
            },
            {"Retry-After": "1"},
        ),
        _IngestResponse(
            200,
            {"operation_id": operation_id, "durable": True, "accepted": 1},
        ),
    ]

    def urlopen(req, timeout):
        requests.append((req, timeout))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        client_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr(client_module, "open_no_redirect", urlopen)

    result = SyncClient(cfg(tmp_path)).ingest([{"raw_text": "same request body"}])

    assert result["durable"] is True
    assert [req.get_method() for req, _timeout in requests] == ["POST", "GET", "POST", "GET"]
    assert requests[0][0].data == requests[2][0].data


def test_ingest_rejects_cross_origin_status_url_without_forwarding_auth(tmp_path, monkeypatch):
    _async_client_environment(monkeypatch)
    requests = []

    def urlopen(req, timeout):
        requests.append(req)
        return _IngestResponse(
            202,
            {
                "operation_id": "unsafe-operation",
                "status_url": "https://attacker.invalid/collect",
                "durable": False,
            },
        )

    monkeypatch.setattr(client_module, "open_no_redirect", urlopen)
    with pytest.raises(RuntimeError, match="status URL changed origin"):
        SyncClient(cfg(tmp_path)).ingest([{"raw_text": "credential stays local"}])

    assert len(requests) == 1
    assert requests[0].full_url == "http://127.0.0.1:7860/ingest"
    assert requests[0].get_method() == "POST"


def test_ingest_trickle_response_cannot_outlive_total_deadline(tmp_path, monkeypatch):
    _async_client_environment(monkeypatch, timeout="0.25")

    class TrickleHandler(BaseHTTPRequestHandler):
        def log_message(self, _fmt, *_args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            payload = b'{"durable":true,"accepted":1}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            for byte in payload:
                try:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                time.sleep(0.05)

    server = ThreadingHTTPServer(("127.0.0.1", 0), TrickleHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    config = cfg(tmp_path)
    config.remote_url = f"http://127.0.0.1:{server.server_address[1]}"
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="timed out before durable confirmation"):
            SyncClient(config).ingest([{"raw_text": "must remain queued"}])
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    assert elapsed < 0.8


def test_ingest_repeated_429_contention_eventually_succeeds(tmp_path, monkeypatch):
    _async_client_environment(monkeypatch)
    clock = [0.0]
    sleeps = []
    requests = []
    operation_id = "op-contention"
    responses = [
        # 6 consecutive 429 responses with Retry-After: 2
        # (exceeds the legacy consecutive_failures limit of 4)
        client_module.error.HTTPError(
            "http://127.0.0.1:7860/ingest",
            429,
            "Too Many Requests",
            {"Retry-After": "2"},
            None,
        ),
        client_module.error.HTTPError(
            "http://127.0.0.1:7860/ingest",
            429,
            "Too Many Requests",
            {"Retry-After": "2"},
            None,
        ),
        _IngestResponse(
            429,
            {"ok": False, "error": "ingest_busy", "retry_after": 2},
            {"Retry-After": "2"},
        ),
        _IngestResponse(
            429,
            {"ok": False, "error": "ingest_busy", "retry_after": 2},
            {"Retry-After": "2"},
        ),
        client_module.error.HTTPError(
            "http://127.0.0.1:7860/ingest",
            429,
            "Too Many Requests",
            {"Retry-After": "2"},
            None,
        ),
        _IngestResponse(
            429,
            {"ok": False, "error": "ingest_busy", "retry_after": 2},
            {"Retry-After": "2"},
        ),
        # 7th request finally succeeds
        _IngestResponse(
            200,
            {"operation_id": operation_id, "durable": True, "accepted": 1},
        ),
    ]

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    def urlopen(req, timeout):
        requests.append((req, timeout))
        res = responses.pop(0)
        if isinstance(res, Exception):
            raise res
        return res

    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(client_module.time, "sleep", sleep)
    monkeypatch.setattr(client_module, "open_no_redirect", urlopen)

    result = SyncClient(cfg(tmp_path)).ingest([{"raw_text": "drains after contention"}])

    assert result["durable"] is True
    assert result["accepted"] == 1
    assert len(requests) == 7
    assert sleeps == [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    assert clock[0] == 12.0


def test_ingest_permanent_4xx_error_fails_immediately_without_retry(tmp_path, monkeypatch):
    _async_client_environment(monkeypatch)
    sleeps = []

    # 1. urllib HTTPError for 400 Bad Request
    def urlopen_400(req, timeout):
        raise client_module.error.HTTPError(req.full_url, 400, "bad request", {}, None)

    monkeypatch.setattr(client_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(client_module, "open_no_redirect", urlopen_400)
    with pytest.raises(client_module.error.HTTPError) as exc_info:
        SyncClient(cfg(tmp_path)).ingest([{"raw_text": "bad request payload"}])
    assert exc_info.value.code == 400
    assert sleeps == []

    # 2. Response object with status 403 Forbidden
    def urlopen_403(req, timeout):
        return _IngestResponse(403, {"error": "forbidden"})

    monkeypatch.setattr(client_module, "open_no_redirect", urlopen_403)
    with pytest.raises(RuntimeError, match="remote ingest returned HTTP 403"):
        SyncClient(cfg(tmp_path)).ingest([{"raw_text": "forbidden payload"}])
    assert sleeps == []


def test_ingest_repeated_429_exhausts_transient_retries_without_unbounded_loop(tmp_path, monkeypatch):
    _async_client_environment(monkeypatch)
    monkeypatch.setenv("FUNES_REMOTE_TRANSIENT_RETRIES", "5")
    clock = [0.0]
    sleeps = []
    requests = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    def urlopen(req, timeout):
        requests.append((req, timeout))
        raise client_module.error.HTTPError(
            req.full_url,
            429,
            "Too Many Requests",
            {"Retry-After": "2"},
            None,
        )

    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(client_module.time, "sleep", sleep)
    monkeypatch.setattr(client_module, "open_no_redirect", urlopen)

    with pytest.raises(RuntimeError, match="remote ingest failed after retries"):
        SyncClient(cfg(tmp_path)).ingest([{"raw_text": "must fail after max retries"}])

    assert len(requests) == 5
    assert len(sleeps) == 4
    assert clock[0] == 8.0
