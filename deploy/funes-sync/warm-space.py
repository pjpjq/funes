#!/usr/bin/env python3
"""Best-effort native worker refresh after a durable local push."""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request


def keychain(service: str) -> str:
    if sys.platform != "darwin":
        return ""
    try:
        result = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-a", os.environ.get("USER", ""), "-s", service, "-w"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return ""
    return (result.stdout or "").strip() if result.returncode == 0 else ""


base = os.environ.get("FUNES_REMOTE_URL", "").rstrip("/")
api_token = os.environ.get("FUNES_API_TOKEN") or keychain("funes-api-token")
hub_token = os.environ.get("HF_TOKEN") or keychain("funes-hf-token")
if not base or not api_token:
    raise SystemExit(0)
headers = {"Content-Type": "application/json", "X-Funes-Authorization": "Bearer " + api_token}
if hub_token:
    headers["Authorization"] = "Bearer " + hub_token
request = urllib.request.Request(base + "/warm", data=b"{}", headers=headers, method="POST")
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()
except (OSError, ValueError):
    # The dataset push is already durable; a later reconciliation or restart
    # can warm the Space if this best-effort notification is unavailable.
    pass
