import json
import os
import stat
import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest

from service import runtime_diagnostics as diag


@pytest.fixture(autouse=True)
def isolated_os(monkeypatch):
    monkeypatch.setattr(diag, "os", SimpleNamespace(**vars(os)))


@pytest.fixture
def roots(tmp_path, monkeypatch):
    diag.stop()
    cgroup, proc = tmp_path / "cgroup", tmp_path / "proc"
    cgroup.mkdir()
    proc.mkdir()
    monkeypatch.setattr(diag, "CGROUP", cgroup)
    monkeypatch.setattr(diag, "PROC", proc)
    monkeypatch.setattr(diag, "_LATEST", {})
    monkeypatch.setattr(diag, "_SAMPLED_AT", None)
    monkeypatch.setattr(diag, "_EVENTS", deque(maxlen=16))
    monkeypatch.setattr(diag, "_THREAD", None)
    monkeypatch.setattr(diag, "_STOP", threading.Event())
    yield cgroup, proc
    diag.stop()


def test_collect_allowlists_memory_process_and_io(roots, monkeypatch):
    cgroup, proc = roots
    monkeypatch.setattr(diag.os, "getpid", lambda: 123)
    for name, value in {"memory.current": "8192", "memory.max": "16000000000",
                        "memory.peak": "16384", "cgroup.procs": "123\n124\n",
                        "memory.events": "oom 2\noom_kill 1\nPRIVATE_TOKEN 123\n",
                        "memory.events.local": "oom_kill 1\n",
                        "memory.stat": "anon 4096\nfile 2048\nPRIVATE_TEXT 7\n",
                        "cpu.stat": "usage_usec 100\nnr_throttled 2\nsecret 1\n"}.items():
        (cgroup / name).write_text(value)
    for pid, name in ((123, "python3"), (124, "PRIVATE_TOKEN")):
        directory = proc / str(pid)
        directory.mkdir()
        (directory / "status").write_text(f"Name:\t{name}\nPPid:\t1\nVmRSS:\t12 kB\nVmHWM:\t15 kB\nRssAnon:\t8 kB\nRssFile:\t4 kB\nThreads:\t2\nPRIVATE: raw text\n")
        (directory / "io").write_text("read_bytes: 100\nwrite_bytes: 200\nsecret: 12\n")
        (directory / "stat").write_text(f"{pid} (private name) S " + "0 " * 10 + "17 23\n")
    result = diag.collect()
    assert result["memory"]["current_bytes"] == 8192
    assert result["memory"]["max_bytes"] == 16000000000
    assert result["memory"]["events"]["oom_kill"] == 1
    assert result["memory"]["stat"] == {"anon": 4096, "file": 2048}
    assert result["process_count"] == 2 and not result["processes_truncated"]
    assert result["processes"][0]["rss_bytes"] == 12 * 1024
    assert result["processes"][0]["user_ticks"] == 17
    assert result["processes"][0]["system_ticks"] == 23
    assert result["processes"][1]["role"] == "other"
    assert result["processes"][0]["io"] == {"read_bytes:": 100, "write_bytes:": 200}
    assert "PRIVATE" not in json.dumps(result) and "secret" not in json.dumps(result)


def test_missing_or_oversized_files_are_safe(roots):
    cgroup, _ = roots
    result = diag.collect()
    assert result["memory"]["available"] is False
    assert result["process_count"] is None and result["processes_truncated"]
    (cgroup / "memory.current").write_text("1" * 10000)
    (cgroup / "memory.max").write_text("max")
    (cgroup / "memory.events").write_text("oom_kill PRIVATE_TOKEN\noom -1\n")
    assert diag._read(cgroup / "memory.current", 64) is None
    result = diag.collect()
    assert result["memory"]["current_bytes"] is None
    assert result["memory"]["unlimited"] is True
    assert result["memory"]["events"] == {}


def test_process_count_is_bounded(roots, monkeypatch):
    cgroup, _ = roots
    (cgroup / "cgroup.procs").write_text("\n".join(map(str, range(1, 100))))
    monkeypatch.setattr(diag.os, "getpid", lambda: 101)
    calls = []
    monkeypatch.setattr(diag, "_process", lambda pid: calls.append(pid) or {"pid": pid})
    result = diag.collect()
    assert len(calls) == diag.MAX_PROCESSES and calls[0] == 101
    assert result["process_count"] == 99 and result["processes_truncated"]


def test_events_are_bounded_and_do_not_leak_arguments(roots, capsys):
    for _ in range(20):
        diag.event("native_exit", pid=12, operation="PRIVATE_RAW_TEXT", exit_code=-9)
    diag.event("PRIVATE_EVENT")
    result = diag.status()
    assert len(result["events"]) == 16
    assert result["events"][-1]["operation"] == "other"
    assert result["events"][-1]["exit_code"] == -9
    assert "PRIVATE" not in json.dumps(result) + capsys.readouterr().err
    result["events"].clear()
    assert len(diag.status()["events"]) == 16


def test_logging_and_sampling_failures_are_nonfatal(roots, monkeypatch):
    monkeypatch.setattr(diag.os, "open", lambda *args: (_ for _ in ()).throw(OSError("PRIVATE_TOKEN")))
    diag.event("ingest_restart")
    monkeypatch.setattr(diag, "collect", lambda: (_ for _ in ()).throw(ValueError("PRIVATE_TOKEN")))
    diag._sample()
    result = diag.status()
    assert result["snapshot"]["error"] == "sampling_failed"
    assert "PRIVATE" not in json.dumps(result)


def test_log_pipe_is_nonblocking_bounded_and_closed(monkeypatch):
    calls = []
    monkeypatch.setattr(diag.os, "open", lambda name, flags: calls.append((name, flags)) or 123)
    monkeypatch.setattr(diag.os, "fstat", lambda fd: SimpleNamespace(st_mode=stat.S_IFIFO))
    monkeypatch.setattr(diag.os, "fpathconf", lambda fd, name: 4096)
    monkeypatch.setattr(diag.os, "write", lambda fd, data: calls.append((fd, data)))
    monkeypatch.setattr(diag.os, "close", lambda fd: calls.append(("closed", fd)))
    assert os.fstat is not diag.os.fstat and os.open is not diag.os.open
    diag._emit({"event": "resource_sample", "processes": [{"pid": 1}] * 1000})
    assert calls[0][0] == "/proc/self/fd/2" and calls[0][1] & os.O_NONBLOCK
    assert len(calls[1][1]) <= 4096 and b'"processes_truncated":true' in calls[1][1]
    assert calls[-1] == ("closed", 123)
    monkeypatch.setattr(diag.os, "write", lambda *args: (_ for _ in ()).throw(BlockingIOError()))
    diag._emit({"event": "native_exit"})
    assert calls[-1] == ("closed", 123)


def test_start_is_singleton_and_stop_reaps_thread(roots, monkeypatch):
    ready = threading.Event()
    monkeypatch.setattr(diag, "collect", lambda: ready.set() or {"pid": 1})
    monkeypatch.setattr(diag, "_emit", lambda _: None)
    diag.start()
    assert ready.wait(1)
    thread = diag._THREAD
    diag.start()
    assert diag._THREAD is thread and diag.status()["running"]
    deadline = time.monotonic() + 1
    while diag.status()["sample_age_seconds"] is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert diag.status()["sample_age_seconds"] is not None
    diag.stop()
    assert not thread.is_alive() and not diag.status()["running"]
