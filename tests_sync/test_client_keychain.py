import gzip
import json
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
