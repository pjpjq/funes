"""Pure in-memory bounded canonical backfill A/B controller.

Fixed profiles:
- baseline_c2_r64: request_rows=64, max_chars=96000, batch_size=128, concurrency=2
- c4_r64: request_rows=64, max_chars=96000, batch_size=128, concurrency=4
- c4_r128: request_rows=128, max_chars=192000, batch_size=128, concurrency=4

Fixed progression: 4 cycles per profile, 12 cycles max.
Fixed deadline: 20 minutes (1200 seconds).
Pure in-memory, thread-safe with single lock.
"""

from __future__ import annotations

import copy
import math
import threading
import time
import uuid
from typing import Any, Callable, Mapping

FIXED_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "name": "baseline_c2_r64",
        "request_rows": 64,
        "max_chars": 96000,
        "batch_size": 128,
        "concurrency": 2,
    },
    {
        "name": "c4_r64",
        "request_rows": 64,
        "max_chars": 96000,
        "batch_size": 128,
        "concurrency": 4,
    },
    {
        "name": "c4_r128",
        "request_rows": 128,
        "max_chars": 192000,
        "batch_size": 128,
        "concurrency": 4,
    },
)

CYCLES_PER_PROFILE: int = 4
MAX_CYCLES: int = len(FIXED_PROFILES) * CYCLES_PER_PROFILE  # 12
DEFAULT_DEADLINE_SECONDS: float = 20.0 * 60.0  # 1200 seconds (20 minutes)

ALLOWLISTED_METRICS: frozenset[str] = frozenset({
    "http_429",
    "http_5xx",
    "transport_errors",
    "requests",
    "retries",
    "tokens",
    "duration_ms",
    "batch_size",
    "concurrency",
    "request_rows",
    "max_chars",
    "latency_ms",
    "request_duration_ms",
    "pacer_wait_ms",
    "backoff_ms",
})

ALLOWLISTED_PHASES = frozenset({
    "secret_scan", "remote_open", "revision_lookup", "vector_reuse", "embedding",
    "lance_write_commit", "lance_append", "lance_delete", "captured_files", "write_ops",
    "hf_commit_chunk", "hf_commit_wait", "recall_open", "recall_models", "recall_embed",
    "recall_vector", "recall_fts", "recall_rerank", "recall_neighbors",
})


def _sanitize_metrics(raw_metrics: Any) -> dict[str, int | float]:
    if not isinstance(raw_metrics, Mapping):
        return {}
    clean: dict[str, int | float] = {}
    for k, v in raw_metrics.items():
        if not isinstance(k, str) or k not in ALLOWLISTED_METRICS:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1_000_000_000:
            clean[k] = int(v) if isinstance(v, int) or v.is_integer() else round(float(v), 4)
    return clean


def _sanitize_phase_ms(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    return {
        key: round(float(value), 2)
        for key, value in raw.items()
        if key in ALLOWLISTED_PHASES
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value < 86_400_000
    }


def _count(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1_000_000_000
    ):
        return 0
    return int(value)


class CanonicalABController:
    """Thread-safe bounded in-memory A/B controller for canonical backfill."""

    def __init__(
        self,
        clock: Callable[[], float] | None = None,
        deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    ) -> None:
        self._lock = threading.RLock()
        self._clock = clock if clock is not None else time.monotonic
        self._deadline_seconds = float(deadline_seconds)

        self._run_counter: int = 0
        self._run_id: str | None = None
        self._phase: str = "idle"  # idle | running | completed | aborted | error
        self._error_reason: str | None = None
        self._started_at: float = 0.0
        self._dispatched_cycles: int = 0
        self._completed_cycles: int = 0
        self._records: list[dict[str, Any]] = []
        self._in_flight: dict[str, dict[str, Any]] = {}

    def _check_deadline_locked(self) -> None:
        if self._phase == "running":
            now = self._clock()
            if (now - self._started_at) >= self._deadline_seconds:
                self._phase = "aborted"
                self._error_reason = "timeout"

    def start(self) -> dict[str, Any]:
        """Start a new A/B run. Rejects concurrent start if currently running."""
        with self._lock:
            self._check_deadline_locked()
            if self._phase == "running" or self._in_flight:
                raise RuntimeError("canonical_ab_already_running")

            self._run_counter += 1
            self._run_id = f"run_{self._run_counter}"
            self._phase = "running"
            self._error_reason = None
            self._started_at = self._clock()
            self._dispatched_cycles = 0
            self._completed_cycles = 0
            self._records = []
            self._in_flight = {}
            return self._status_locked()

    def abort(self, reason: str = "aborted") -> dict[str, Any]:
        """Abort current run. Does not cancel active write, but stops next overrides."""
        with self._lock:
            self._check_deadline_locked()
            if self._phase == "running":
                self._phase = "aborted"
                self._error_reason = "user_aborted" if reason == "aborted" else "aborted"
            return self._status_locked()

    def begin_cycle(self) -> dict[str, Any] | None:
        """Reserve next profile cycle with immutable token. Returns None when inactive."""
        with self._lock:
            self._check_deadline_locked()
            if self._phase != "running":
                return None
            # The existing reconciler owns execution. Never dispatch another
            # override while an aborted/timed-out native cycle is draining.
            if self._in_flight:
                return None
            if self._dispatched_cycles >= MAX_CYCLES:
                return None

            profile_idx = self._dispatched_cycles // CYCLES_PER_PROFILE
            base_profile = FIXED_PROFILES[profile_idx]
            cycle_token = f"{self._run_id}:{self._dispatched_cycles + 1}:{uuid.uuid4().hex[:8]}"
            cycle_item = {
                "run_id": self._run_id,
                "cycle_token": cycle_token,
                "cycle_index": self._dispatched_cycles + 1,
                "name": base_profile["name"],
                "request_rows": base_profile["request_rows"],
                "max_chars": base_profile["max_chars"],
                "batch_size": base_profile["batch_size"],
                "concurrency": base_profile["concurrency"],
            }
            self._in_flight[cycle_token] = dict(cycle_item)
            self._dispatched_cycles += 1
            return dict(cycle_item)

    def finish_cycle(
        self,
        result: dict[str, Any] | None = None,
        duration_ms: int = 0,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
        profile: dict[str, Any] | None = None,
        phase_ms: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Record completed cycle. Thread-safe, bounded, safe error triggers."""
        with self._lock:
            self._check_deadline_locked()

            # Handle flexible argument positions: finish_cycle(profile, result, duration_ms, metrics, error)
            if isinstance(result, dict) and "cycle_token" in result and isinstance(duration_ms, dict):
                actual_profile = result
                actual_result = duration_ms
                actual_duration_ms = metrics if isinstance(metrics, (int, float)) else 0
                actual_metrics = error if isinstance(error, dict) else kwargs.get("metrics")
                actual_error = kwargs.get("error") if error is actual_metrics else error
            else:
                actual_profile = profile or kwargs.get("profile")
                actual_result = result
                actual_duration_ms = duration_ms
                actual_metrics = metrics
                actual_error = error

            # If profile token was embedded in result
            if actual_profile is None and isinstance(actual_result, dict) and "cycle_token" in actual_result:
                actual_profile = actual_result

            # Validate token against active in-flight cycles for this run
            if not isinstance(actual_profile, dict):
                return False

            token = actual_profile.get("cycle_token")
            run_id = actual_profile.get("run_id")
            if not token or run_id != self._run_id:
                return False

            if token not in self._in_flight:
                return False

            in_flight_info = self._in_flight.pop(token)
            profile_name = str(in_flight_info.get("name", "unknown"))
            cycle_index = int(in_flight_info.get("cycle_index", len(self._records) + 1))

            clean_metrics = _sanitize_metrics(actual_metrics)
            safe_duration_ms = (
                min(86_400_000, max(0, int(actual_duration_ms)))
                if not isinstance(actual_duration_ms, bool)
                and isinstance(actual_duration_ms, (int, float))
                and math.isfinite(actual_duration_ms)
                else 0
            )

            # Error triggers:
            # - not durable
            # - attempted <= 0
            # - held alone does NOT abort unless other triggers
            # - metrics: http_429 > 0 / http_5xx > 0 / transport_errors > 0
            # - error string passed
            has_error = False
            error_reason: str | None = None

            if actual_error is not None and str(actual_error).strip() != "":
                has_error = True
                error_reason = "error_reported"
            elif not isinstance(actual_result, dict):
                has_error = True
                error_reason = "invalid_result"
            else:
                attempted = _count(actual_result.get("attempted", 0))
                durable = actual_result.get("durable", False)

                if not bool(durable):
                    has_error = True
                    error_reason = "not_durable"
                elif attempted <= 0:
                    has_error = True
                    error_reason = "attempted_zero"
                elif not _count(actual_result.get("indexed", 0)) and not _count(actual_result.get("held", 0)):
                    # A durably persisted retry checkpoint is not successful
                    # indexing, even when the source-store ACK succeeded.
                    has_error = True
                    error_reason = "no_progress"
                elif clean_metrics.get("http_429", 0) > 0:
                    has_error = True
                    error_reason = "http_429"
                elif clean_metrics.get("http_5xx", 0) > 0:
                    has_error = True
                    error_reason = "http_5xx"
                elif clean_metrics.get("transport_errors", 0) > 0:
                    has_error = True
                    error_reason = "transport_errors"

            if isinstance(actual_result, dict):
                rec_attempted = _count(actual_result.get("attempted", 0))
                rec_indexed = _count(actual_result.get("indexed", 0))
                rec_held = _count(actual_result.get("held", 0))
                rec_durable = bool(actual_result.get("durable", False))
            else:
                rec_attempted = 0
                rec_indexed = 0
                rec_held = 0
                rec_durable = False

            record = {
                "profile": profile_name,
                "cycle_index": cycle_index,
                "attempted": rec_attempted,
                "indexed": rec_indexed,
                "held": rec_held,
                "durable": rec_durable,
                "duration_ms": safe_duration_ms,
                "metrics": clean_metrics,
                "phase_ms": _sanitize_phase_ms(phase_ms),
            }

            if len(self._records) < MAX_CYCLES:
                self._records.append(record)

            if has_error:
                if self._phase == "running":
                    self._phase = "error"
                    self._error_reason = error_reason
                    self._in_flight.clear()
            else:
                self._completed_cycles += 1
                if self._phase == "running" and self._completed_cycles >= MAX_CYCLES:
                    self._phase = "completed"

            return True

    def _status_locked(self) -> dict[str, Any]:
        self._check_deadline_locked()
        elapsed = 0.0
        if self._run_id is not None:
            elapsed = max(0.0, float(self._clock() - self._started_at))

        active_profile_name: str | None = None
        if self._phase == "running" and self._in_flight:
            active_profile_name = next(iter(self._in_flight.values()))["name"]
        elif self._phase == "running" and self._dispatched_cycles < MAX_CYCLES:
            profile_idx = self._dispatched_cycles // CYCLES_PER_PROFILE
            active_profile_name = FIXED_PROFILES[profile_idx]["name"]

        return {
            "run_id": self._run_id,
            "phase": self._phase,
            "error_reason": self._error_reason,
            "total_cycles": MAX_CYCLES,
            "completed_cycles": len(self._records),
            "active_profile": active_profile_name,
            "current_cycle_index": self._dispatched_cycles,
            "in_flight_cycles": len(self._in_flight),
            "elapsed_seconds": round(elapsed, 3),
            "deadline_seconds": self._deadline_seconds,
            "records": copy.deepcopy(self._records),
        }

    def status(self) -> dict[str, Any]:
        """Return JSON-safe bounded status dict."""
        with self._lock:
            return self._status_locked()

    @property
    def run_id(self) -> str | None:
        with self._lock:
            return self._run_id

    @property
    def phase(self) -> str:
        with self._lock:
            self._check_deadline_locked()
            return self._phase

    @property
    def is_active(self) -> bool:
        with self._lock:
            self._check_deadline_locked()
            return self._phase == "running"
