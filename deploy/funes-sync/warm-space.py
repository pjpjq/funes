#!/usr/bin/env python3
"""Best-effort native worker refresh after a durable local push."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def keychain(service: str) -> str:
    if sys.platform != "darwin":
        return ""
    try:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-a",
                os.environ.get("USER", ""),
                "-s",
                service,
                "-w",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or "").strip() if result.returncode == 0 else ""


def zshrc_env(name: str) -> str:
    expressions = {
        "FUNES_API_TOKEN": '"${FUNES_API_TOKEN:-}"',
        "FUNES_HF_TOKEN": '"${FUNES_HF_TOKEN:-}"',
        "HF_TOKEN": '"${HF_TOKEN:-}"',
    }
    expression = expressions.get(name)
    if (
        sys.platform != "darwin"
        or expression is None
        or not Path.home().joinpath(".zshrc").is_file()
    ):
        return ""
    try:
        result = subprocess.run(
            [
                "/bin/zsh",
                "-lc",
                f'source "$HOME/.zshrc" >/dev/null 2>&1; printf %s {expression}',
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or "").strip() if result.returncode == 0 else ""


def warm_interval() -> float:
    try:
        value = float(os.environ.get("FUNES_NATIVE_WARM_MIN_INTERVAL", "300"))
    except ValueError:
        value = 300.0
    return min(max(value, 0.0), 3600.0)


def stamp_path(base: str) -> Path:
    state_dir = Path(
        os.environ.get("FUNES_SYNC_STATE_DIR", "~/.local/share/funes-sync")
    ).expanduser()
    remote_hash = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    return state_dir / f"native-warm-{remote_hash}.last-success"


def recently_warmed(stamp: Path, interval: float) -> bool:
    if interval <= 0:
        return False
    try:
        age = time.time() - stamp.stat().st_mtime
    except OSError:
        return False
    return 0 <= age < interval


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def main() -> None:
    base = os.environ.get("FUNES_REMOTE_URL", "").rstrip("/")
    if not base:
        return
    stamp = stamp_path(base)
    if recently_warmed(stamp, warm_interval()):
        return

    api_token = (
        os.environ.get("FUNES_API_TOKEN")
        or keychain("funes-api-token")
        or zshrc_env("FUNES_API_TOKEN")
    )
    hub_token = (
        os.environ.get("HF_TOKEN")
        or keychain("funes-hf-token")
        or zshrc_env("FUNES_HF_TOKEN")
        or zshrc_env("HF_TOKEN")
    )
    if not api_token:
        return
    headers = {
        "Content-Type": "application/json",
        "X-Funes-Authorization": "Bearer " + api_token,
    }
    if hub_token:
        headers["Authorization"] = "Bearer " + hub_token
    request = urllib.request.Request(
        base + "/warm", data=b"{}", headers=headers, method="POST"
    )
    try:
        opener = urllib.request.build_opener(NoRedirectHandler())
        with opener.open(request, timeout=10) as response:
            response.read()
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()
    except (OSError, ValueError):
        # The dataset push is already durable; a later reconciliation or restart
        # can warm the Space if this best-effort notification is unavailable.
        pass


if __name__ == "__main__":
    main()
