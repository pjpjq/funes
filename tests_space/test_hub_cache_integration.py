"""Persistent derived cache is optional, observable, and never a readiness gate."""
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from types import SimpleNamespace

import space.server as bridge


def test_cache_restore_precedes_sources_and_native_warm(monkeypatch, tmp_path):
    calls = []
    cache = SimpleNamespace(
        restore=lambda: calls.append("restore"),
        start=lambda: calls.append("cache_start"),
        stop=lambda: None,
        status=lambda: {"enabled": True, "restored_links": 4},
    )
    monkeypatch.setattr(bridge.HubCache, "from_env", lambda: cache)
    monkeypatch.setattr(bridge, "HUB_CACHE", None)
    monkeypatch.setattr(bridge, "HOME", tmp_path)
    monkeypatch.setattr(bridge.atexit, "register", lambda fn: None)
    monkeypatch.setattr(bridge, "source_app", lambda: calls.append("sources"))
    monkeypatch.setattr(bridge, "request_warm", lambda: calls.append("warm"))
    monkeypatch.setattr(bridge, "ThreadingHTTPServer", lambda *a: SimpleNamespace(
        serve_forever=lambda: calls.append("http")))
    bridge.serve(port=0)
    assert calls == ["restore", "cache_start", "sources", "warm", "http"]
    assert bridge.hub_cache_status()["restored_links"] == 4


def test_invalid_optional_cache_cannot_abort_startup_or_expose_error(monkeypatch):
    def invalid_config():
        raise ValueError("must-not-leak-secret")
    monkeypatch.setattr(bridge.HubCache, "from_env", invalid_config)
    monkeypatch.setattr(bridge, "HUB_CACHE", None)
    monkeypatch.setattr(bridge, "HUB_CACHE_ERROR", "")
    bridge.start_hub_cache()
    assert bridge.hub_cache_status() == {"enabled": False, "error_class": "ValueError"}


def test_native_children_inherit_local_cache_not_bucket_mount(monkeypatch, tmp_path):
    hf_home = tmp_path / "local-hf"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_CACHE", str(hf_home / "hub"))
    monkeypatch.setenv("FUNES_HUB_CACHE_MOUNT", str(tmp_path / "bucket"))
    env = bridge.native_environment()
    assert env["HF_HOME"] == str(hf_home)
    assert env["HF_HUB_CACHE"] == str(hf_home / "hub")
