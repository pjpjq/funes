from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

from .config import Config


@dataclass
class NativeResult:
    ok: bool
    indexed: tuple[str, ...] = ()
    error: str = ""


class NativeFunes:
    """Small adapter to the upstream Funes index/push pipeline.

    This is intentionally the only production publication path: the native
    binary owns Lance, embedding, BM25, reranking, recency, CAS commits and its
    TruffleHog gate. The Python HTTP client remains available for migration and
    metadata-only deployments.
    """

    def __init__(self, config: Config, runner=None):
        self.config = config
        self.binary = config.native_bin or os.environ.get("FUNES_BIN") or shutil.which("funes")
        self.runner = runner or subprocess.run

    def _call(self, args: list[str], timeout: int) -> tuple[int, str]:
        if not self.binary:
            return 127, "funes binary not found"
        try:
            result = self.runner(
                [self.binary, *args],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
            return int(result.returncode), ""
        except (OSError, subprocess.SubprocessError) as exc:
            return 1, type(exc).__name__

    def sync(self, harnesses: tuple[str, ...] = ("codex", "pi", "claude")) -> NativeResult:
        if not self.config.native_memory:
            return NativeResult(False, error="FUNES_MEMORY is not configured")
        indexed = []
        for harness in harnesses:
            code, error = self._call(["index", "--harness", harness, "--yes"], timeout=900)
            if code != 0:
                return NativeResult(False, tuple(indexed), error or f"index:{harness}:{code}")
            indexed.append(harness)
        code, error = self._call(
            ["push", self.config.native_memory, "--yes", "--force-reindex"], timeout=1800
        )
        if code != 0:
            return NativeResult(False, tuple(indexed), error or f"push:{code}")
        return NativeResult(True, tuple(indexed))
