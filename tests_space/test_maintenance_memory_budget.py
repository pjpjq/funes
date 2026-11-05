import os
from types import SimpleNamespace

import pytest

import space.server as bridge


@pytest.mark.parametrize("operation", ["optimize-index", "ingest-docs", "compact"])
def test_maintenance_budget_is_child_scoped(monkeypatch, operation):
    monkeypatch.setenv("FUNES_MAINTENANCE_MEMORY_BUDGET", "inherited")
    monkeypatch.setenv("LANCE_FTS_NUM_SHARDS", "7")
    monkeypatch.setenv("LANCE_FTS_PARTITION_SIZE", "2048")
    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    captured = []

    def popen(*args, **kwargs):
        captured.append(kwargs["env"].copy())
        return SimpleNamespace(pid=123, returncode=0, communicate=lambda **kwargs: ("ok", ""))

    monkeypatch.setattr(bridge.subprocess, "Popen", popen)
    assert bridge.run(operation, "fixture/memory") == (0, "ok", "")
    env = captured[0]
    assert env["HF_TOKEN"] == "fixture-token"
    if operation == "optimize-index":
        assert env["FUNES_MAINTENANCE_MEMORY_BUDGET"] == "1"
        assert env["LANCE_FTS_NUM_SHARDS"] == "1"
        assert env["LANCE_FTS_PARTITION_SIZE"] == "512"
    else:
        assert "FUNES_MAINTENANCE_MEMORY_BUDGET" not in env
        assert env["LANCE_FTS_NUM_SHARDS"] == "7"
        assert env["LANCE_FTS_PARTITION_SIZE"] == "2048"
    assert os.environ["FUNES_MAINTENANCE_MEMORY_BUDGET"] == "inherited"
    assert os.environ["LANCE_FTS_NUM_SHARDS"] == "7"
    assert os.environ["LANCE_FTS_PARTITION_SIZE"] == "2048"


def test_query_worker_does_not_inherit_maintenance_cache_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("FUNES_MAINTENANCE_MEMORY_BUDGET", "1")
    worker = bridge.NativeMcpWorker("fake-funes", "fixture/memory", tmp_path)
    assert "FUNES_MAINTENANCE_MEMORY_BUDGET" not in worker._environment()
