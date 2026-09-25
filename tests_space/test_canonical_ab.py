"""Unit tests for CanonicalABController in space/canonical_ab.py."""

import json
import threading
import time
import pytest

from space.canonical_ab import (
    CanonicalABController,
    FIXED_PROFILES,
    CYCLES_PER_PROFILE,
    MAX_CYCLES,
    DEFAULT_DEADLINE_SECONDS,
)


def test_fixed_profiles_specs():
    assert len(FIXED_PROFILES) == 3
    assert CYCLES_PER_PROFILE == 4
    assert MAX_CYCLES == 12
    assert DEFAULT_DEADLINE_SECONDS == 1200.0

    p0 = FIXED_PROFILES[0]
    assert p0["name"] == "baseline_c2_r64"
    assert p0["request_rows"] == 64
    assert p0["max_chars"] == 96000
    assert p0["batch_size"] == 128
    assert p0["concurrency"] == 2

    p1 = FIXED_PROFILES[1]
    assert p1["name"] == "c4_r64"
    assert p1["request_rows"] == 64
    assert p1["max_chars"] == 96000
    assert p1["batch_size"] == 128
    assert p1["concurrency"] == 4

    p2 = FIXED_PROFILES[2]
    assert p2["name"] == "c4_r128"
    assert p2["request_rows"] == 128
    assert p2["max_chars"] == 192000
    assert p2["batch_size"] == 128
    assert p2["concurrency"] == 4


def test_normal_progression_completion():
    controller = CanonicalABController()
    assert controller.phase == "idle"
    assert controller.begin_cycle() is None

    controller.start()
    assert controller.phase == "running"
    assert controller.run_id == "run_1"

    expected_names = (
        ["baseline_c2_r64"] * 4
        + ["c4_r64"] * 4
        + ["c4_r128"] * 4
    )

    for i, expected_name in enumerate(expected_names):
        p = controller.begin_cycle()
        assert p is not None
        assert p["name"] == expected_name
        assert p["cycle_index"] == i + 1
        assert p["run_id"] == "run_1"
        assert len(p["cycle_token"]) > 0

        ok = controller.finish_cycle(
            result={"attempted": p["request_rows"], "indexed": p["request_rows"], "held": 0, "durable": True},
            duration_ms=100 + i * 10,
            metrics={"http_429": 0, "http_5xx": 0, "transport_errors": 0, "tokens": 500},
            profile=p,
        )
        assert ok is True

    # 13th cycle must return None
    assert controller.begin_cycle() is None

    st = controller.status()
    assert st["phase"] == "completed"
    assert st["completed_cycles"] == 12
    assert st["total_cycles"] == 12
    assert st["error_reason"] is None
    assert st["active_profile"] is None
    assert len(st["records"]) == 12

    # Check records ordering
    for i, rec in enumerate(st["records"]):
        assert rec["profile"] == expected_names[i]
        assert rec["cycle_index"] == i + 1
        assert rec["durable"] is True
        assert rec["attempted"] > 0
        assert rec["duration_ms"] > 0
        assert rec["metrics"]["tokens"] == 500


def test_held_alone_does_not_abort():
    controller = CanonicalABController()
    controller.start()

    p = controller.begin_cycle()
    assert p is not None

    # Cycle has held > 0, but durable=True, attempted > 0, no 429/5xx/transport_errors
    ok = controller.finish_cycle(
        result={"attempted": 64, "indexed": 60, "held": 4, "durable": True},
        duration_ms=120,
        metrics={"http_429": 0, "http_5xx": 0, "transport_errors": 0},
        profile=p,
    )
    assert ok is True

    st = controller.status()
    assert st["phase"] == "running"
    assert st["error_reason"] is None
    assert len(st["records"]) == 1
    assert st["records"][0]["held"] == 4
    assert st["records"][0]["indexed"] == 60

    # Next cycle continues normally
    p2 = controller.begin_cycle()
    assert p2 is not None


def test_start_conflict_rejected():
    controller = CanonicalABController()
    controller.start()
    assert controller.phase == "running"

    with pytest.raises(RuntimeError, match="canonical_ab_already_running"):
        controller.start()

    assert controller.phase == "running"


def test_abort_active_boundary():
    controller = CanonicalABController()
    controller.start()

    p1 = controller.begin_cycle()
    assert p1 is not None

    # Abort while cycle 1 is active
    st_abort = controller.abort()
    assert st_abort["phase"] == "aborted"
    assert controller.phase == "aborted"

    # Subsequent begin_cycle must return None
    assert controller.begin_cycle() is None

    # In-flight cycle 1 finishes and is accepted
    ok = controller.finish_cycle(
        result={"attempted": 64, "indexed": 64, "held": 0, "durable": True},
        duration_ms=150,
        metrics={"http_429": 0},
        profile=p1,
    )
    assert ok is True

    # Phase remains aborted
    st = controller.status()
    assert st["phase"] == "aborted"
    assert len(st["records"]) == 1
    assert st["records"][0]["profile"] == "baseline_c2_r64"

    # Re-submitting the same token is rejected (already consumed)
    assert controller.finish_cycle(
        result={"attempted": 64, "indexed": 64, "held": 0, "durable": True},
        duration_ms=150,
        metrics={},
        profile=p1,
    ) is False


def test_baseline_cannot_be_attributed_after_start():
    controller = CanonicalABController()
    controller.start()

    # Reconcile finishes from baseline without a profile token
    res = {"attempted": 64, "indexed": 64, "held": 0, "durable": True}
    assert controller.finish_cycle(result=res, duration_ms=100, metrics={}, profile=None) is False
    assert len(controller.status()["records"]) == 0

    # Baseline with bogus profile token
    bogus = {"run_id": "other_run", "cycle_token": "fake_token", "name": "baseline_c2_r64"}
    assert controller.finish_cycle(result=res, duration_ms=100, metrics={}, profile=bogus) is False
    assert len(controller.status()["records"]) == 0

    # Legitimate cycle from begin_cycle is attributed
    p = controller.begin_cycle()
    assert p is not None
    assert controller.finish_cycle(result=res, duration_ms=100, metrics={}, profile=p) is True
    assert len(controller.status()["records"]) == 1


def test_first_non_durable_abort():
    controller = CanonicalABController()
    controller.start()

    p = controller.begin_cycle()
    assert p is not None

    ok = controller.finish_cycle(
        result={"attempted": 64, "indexed": 0, "held": 0, "durable": False},
        duration_ms=80,
        metrics={},
        profile=p,
    )
    assert ok is True

    st = controller.status()
    assert st["phase"] == "error"
    assert st["error_reason"] == "not_durable"
    assert len(st["records"]) == 1
    assert st["records"][0]["durable"] is False

    # After error, begin_cycle returns None
    assert controller.begin_cycle() is None


def test_attempted_zero_abort():
    controller = CanonicalABController()
    controller.start()

    p = controller.begin_cycle()
    assert p is not None

    ok = controller.finish_cycle(
        result={"attempted": 0, "indexed": 0, "held": 0, "durable": True},
        duration_ms=50,
        metrics={},
        profile=p,
    )
    assert ok is True

    st = controller.status()
    assert st["phase"] == "error"
    assert st["error_reason"] == "attempted_zero"
    assert controller.begin_cycle() is None


def test_http_429_abort():
    controller = CanonicalABController()
    controller.start()

    p = controller.begin_cycle()
    assert p is not None

    ok = controller.finish_cycle(
        result={"attempted": 64, "indexed": 50, "held": 0, "durable": True},
        duration_ms=200,
        metrics={"http_429": 1, "http_5xx": 0, "transport_errors": 0},
        profile=p,
    )
    assert ok is True

    st = controller.status()
    assert st["phase"] == "error"
    assert st["error_reason"] == "http_429"
    assert controller.begin_cycle() is None


def test_http_5xx_abort():
    controller = CanonicalABController()
    controller.start()

    p = controller.begin_cycle()
    assert p is not None

    ok = controller.finish_cycle(
        result={"attempted": 64, "indexed": 50, "held": 0, "durable": True},
        duration_ms=200,
        metrics={"http_429": 0, "http_5xx": 2, "transport_errors": 0},
        profile=p,
    )
    assert ok is True

    st = controller.status()
    assert st["phase"] == "error"
    assert st["error_reason"] == "http_5xx"
    assert controller.begin_cycle() is None


def test_transport_errors_abort():
    controller = CanonicalABController()
    controller.start()

    p = controller.begin_cycle()
    assert p is not None

    ok = controller.finish_cycle(
        result={"attempted": 64, "indexed": 50, "held": 0, "durable": True},
        duration_ms=200,
        metrics={"http_429": 0, "http_5xx": 0, "transport_errors": 1},
        profile=p,
    )
    assert ok is True

    st = controller.status()
    assert st["phase"] == "error"
    assert st["error_reason"] == "transport_errors"
    assert controller.begin_cycle() is None


def test_explicit_error_abort_no_raw_text_leak():
    controller = CanonicalABController()
    controller.start()

    p = controller.begin_cycle()
    assert p is not None

    ok = controller.finish_cycle(
        result={"attempted": 64, "indexed": 64, "held": 0, "durable": True},
        duration_ms=100,
        metrics={},
        error="FATAL: password authentication failed for user 'admin' postgresql://...",
        profile=p,
    )
    assert ok is True

    st = controller.status()
    assert st["phase"] == "error"
    assert st["error_reason"] == "error_reported"
    # Ensure sensitive credentials or raw error text are NOT leaked anywhere in status
    st_json = json.dumps(st)
    assert "password" not in st_json
    assert "FATAL" not in st_json
    assert "postgresql" not in st_json


def test_timeout():
    current_time = 1000.0

    def mock_clock():
        return current_time

    controller = CanonicalABController(clock=mock_clock, deadline_seconds=1200.0)
    controller.start()
    assert controller.phase == "running"

    p1 = controller.begin_cycle()
    assert p1 is not None

    # Advance clock past 20 minutes (1200 seconds)
    current_time += 1205.0

    assert controller.begin_cycle() is None
    st = controller.status()
    assert st["phase"] in {"aborted", "error"}
    assert st["error_reason"] == "timeout"


def test_locking_concurrency():
    controller = CanonicalABController()
    controller.start()

    dispatched = []
    completed = []
    lock = threading.Lock()

    def worker():
        for _ in range(3):
            p = controller.begin_cycle()
            if p is None:
                break
            with lock:
                dispatched.append(p)
            time.sleep(0.005)
            ok = controller.finish_cycle(
                result={"attempted": p["request_rows"], "indexed": p["request_rows"], "held": 0, "durable": True},
                duration_ms=10,
                metrics={"http_429": 0},
                profile=p,
            )
            if ok:
                with lock:
                    completed.append(p)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    st = controller.status()
    assert len(dispatched) <= 12
    assert len(completed) <= 12
    assert len(st["records"]) == len(completed)
    assert st["completed_cycles"] == len(completed)
    if len(completed) == 12:
        assert st["phase"] == "completed"


def test_status_json_serializable_and_bounded():
    controller = CanonicalABController()
    assert json.dumps(controller.status())

    controller.start()
    p = controller.begin_cycle()
    controller.finish_cycle(
        result={"attempted": 64, "indexed": 60, "held": 4, "durable": True},
        duration_ms=123,
        metrics={
            "http_429": 0,
            "tokens": 450,
            "unauthorized_secret_metric": "ignore_me",
        },
        profile=p,
    )

    st = controller.status()
    dumped = json.dumps(st)
    loaded = json.loads(dumped)
    assert loaded["run_id"] == "run_1"
    assert loaded["records"][0]["metrics"].get("tokens") == 450
    assert "unauthorized_secret_metric" not in loaded["records"][0]["metrics"]


def test_restart_after_completed_or_aborted():
    controller = CanonicalABController()
    controller.start()
    assert controller.run_id == "run_1"
    controller.abort()
    assert controller.phase == "aborted"

    # Now can start run_2
    controller.start()
    assert controller.run_id == "run_2"
    assert controller.phase == "running"
    assert controller.status()["completed_cycles"] == 0
    assert len(controller.status()["records"]) == 0
