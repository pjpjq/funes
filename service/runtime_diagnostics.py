"""Bounded resource evidence; never read argv, environment, or memory contents."""
import copy
import json
import os
import stat
import threading
import time
from collections import deque
from pathlib import Path


CGROUP = Path("/sys/fs/cgroup")
PROC = Path("/proc")
INTERVAL = 30
MAX_PROCESSES = 64
_LOCK = threading.Lock()
_STOP = threading.Event()
_THREAD = None
_LATEST = {}
_SAMPLED_AT = None
_EVENTS = deque(maxlen=16)
_KINDS = frozenset(("service_start", "service_exit", "ingest_restart",
                    "ingest_exec_failed", "ingest_restart_thread_failed",
                    "native_start", "native_exit", "native_timeout"))
_OPERATIONS = frozenset(("optimize-index", "ingest-docs", "index", "push", "compact"))


def _read(path, limit=8192):
    try:
        with path.open("r", encoding="ascii", errors="replace") as stream:
            data = stream.read(limit + 1)
        return data if len(data) <= limit else None
    except (OSError, ValueError):
        return None


def _integer(value):
    if isinstance(value, str) and value.isascii() and value.isdecimal() and len(value) <= 20:
        return int(value)
    return None


def _counters(path, allowed):
    result = {}
    for line in (_read(path) or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in allowed:
            value = _integer(parts[1])
            if value is not None:
                result[parts[0]] = value
    return result


def _memory():
    result = {}
    for name in ("current", "max", "peak"):
        raw = _read(CGROUP / ("memory." + name), 64)
        result[name + "_bytes"] = _integer(raw.strip()) if raw is not None else None
        if name == "max":
            result["unlimited"] = raw is not None and raw.strip() == "max"
    result["available"] = result["current_bytes"] is not None
    for name in ("events", "events.local"):
        result[name.replace(".", "_")] = _counters(
            CGROUP / ("memory." + name), {"low", "high", "max", "oom", "oom_kill", "oom_group_kill"})
    result["stat"] = _counters(CGROUP / "memory.stat", {
        "anon", "file", "kernel", "kernel_stack", "pagetables", "sock", "shmem",
        "file_mapped", "file_dirty", "file_writeback", "slab", "inactive_file", "active_file"})
    return result


def _process(pid):
    text = _read(PROC / str(pid) / "status")
    if text is None:
        return None
    values = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
    name = values.get("Name", "").strip()
    role = "native" if name == "funes" else "postgres" if name == "postgres" else "python" if name in ("python", "python3", "python3.11", "python3.12") else "other"
    result = {"pid": pid, "role": role}
    for name in ("PPid", "Threads"):
        result[name.lower()] = _integer(values.get(name, "").strip())
    for name, key in (("VmRSS", "rss_bytes"), ("VmHWM", "peak_rss_bytes"),
                      ("RssAnon", "anon_bytes"), ("RssFile", "file_bytes"), ("RssShmem", "shmem_bytes")):
        parts = values.get(name, "").split()
        value = _integer(parts[0]) if len(parts) == 2 and parts[1] == "kB" else None
        result[key] = value * 1024 if value is not None else None
    stat = _read(PROC / str(pid) / "stat")
    if stat is not None and ") " in stat:
        fields = stat.rsplit(") ", 1)[1].split()
        if len(fields) >= 13:
            result["user_ticks"] = _integer(fields[11])
            result["system_ticks"] = _integer(fields[12])
    result["io"] = _counters(PROC / str(pid) / "io", {"read_bytes:", "write_bytes:", "rchar:", "wchar:"})
    return result


def collect():
    raw = _read(CGROUP / "cgroup.procs")
    pids = sorted({pid for line in (raw or "").splitlines()
                   if (pid := _integer(line.strip())) is not None and pid > 0})
    selected = [os.getpid(), *(pid for pid in pids if pid != os.getpid())][:MAX_PROCESSES]
    return {"at": time.time(), "pid": os.getpid(), "memory": _memory(),
            "cpu": _counters(CGROUP / "cpu.stat", {"usage_usec", "user_usec", "system_usec", "nr_throttled", "throttled_usec"}),
            "process_count": len(pids) if raw is not None else None,
            "processes_truncated": raw is None or len(pids) > MAX_PROCESSES,
            "processes": [row for pid in selected if (row := _process(pid)) is not None]}


def _emit(value):
    descriptor = None
    try:
        # A separate open description must not change stderr flags or block business work.
        descriptor = os.open("/proc/self/fd/2", os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
            return
        limit = min(4096, os.fpathconf(descriptor, "PC_PIPE_BUF"))
        line = ("funes_runtime " + json.dumps(value, separators=(",", ":")) + "\n").encode()
        if len(line) > limit and "processes" in value:
            summary = {**value, "processes": [], "processes_truncated": True}
            line = ("funes_runtime " + json.dumps(summary, separators=(",", ":")) + "\n").encode()
        if len(line) <= limit:
            os.write(descriptor, line)
    except Exception:
        pass
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def event(kind, *, pid=None, operation=None, exit_code=None):
    """Diagnostic failures must not change the caller's operation or exception."""
    try:
        if kind not in _KINDS:
            return
        value = {"event": kind, "at": time.time(), "service_pid": os.getpid(), "memory": _memory()}
        if type(pid) is int and pid > 0:
            value["pid"] = pid
        if operation is not None:
            value["operation"] = operation if isinstance(operation, str) and operation in _OPERATIONS else "other"
        if type(exit_code) is int and -65536 <= exit_code <= 65536:
            value["exit_code"] = exit_code
        with _LOCK:
            _EVENTS.append(value)
        _emit(value)
    except Exception:
        pass


def _sample():
    global _LATEST, _SAMPLED_AT
    try:
        value = collect()
    except Exception:
        value = {"at": time.time(), "pid": os.getpid(), "error": "sampling_failed"}
    with _LOCK:
        _LATEST = value
        _SAMPLED_AT = time.monotonic()
    _emit({"event": "resource_sample", **value})


def _loop():
    while not _STOP.is_set():
        _sample()
        if _STOP.wait(INTERVAL):
            break


def start():
    global _THREAD
    try:
        with _LOCK:
            if _THREAD is not None and _THREAD.is_alive():
                return
            _STOP.clear()
            _THREAD = threading.Thread(target=_loop, name="funes-runtime-diagnostics", daemon=True)
            _THREAD.start()
        event("service_start")
    except Exception:
        pass


def stop():
    try:
        _STOP.set()
        thread = _THREAD
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
        event("service_exit")
    except Exception:
        pass


def status():
    try:
        with _LOCK:
            return {"running": _THREAD is not None and _THREAD.is_alive(),
                    "sample_age_seconds": None if _SAMPLED_AT is None else round(max(0, time.monotonic() - _SAMPLED_AT), 3),
                    "snapshot": copy.deepcopy(_LATEST), "events": copy.deepcopy(list(_EVENTS))}
    except Exception:
        return {"running": False, "error": "diagnostic_unavailable"}
