from __future__ import annotations
import gzip
import json, os, subprocess, sys, time
from urllib import request, error
from .config import Config
from .http import open_no_redirect


def _keychain_token(service: str) -> str:
    """Read a token from the macOS Keychain without putting it in argv."""
    if sys.platform != "darwin":
        return ""
    account = os.environ.get("USER") or str(os.getuid())
    try:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-a",
                account,
                "-s",
                service,
                "-w",
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or "").rstrip("\r\n")


class SyncClient:
    def __init__(self, config: Config|None=None): self.config=config or Config.load()

    def _auth_headers(self, *, json_content: bool = False) -> dict[str, str]:
        headers = {"User-Agent": "funes-sync/1"}
        if json_content:
            headers["Content-Type"] = "application/json"
        token = os.environ.get("FUNES_API_TOKEN") or _keychain_token("funes-api-token")
        if not token:
            raise RuntimeError("FUNES_API_TOKEN is not configured")
        hub_token = os.environ.get("FUNES_HF_TOKEN") or os.environ.get("HF_TOKEN") or _keychain_token("funes-hf-token")
        if hub_token:
            headers["Authorization"] = "Bearer " + hub_token
            headers["X-Funes-Authorization"] = "Bearer " + token
        else:
            headers["Authorization"] = "Bearer " + token
        return headers

    def ingest(self, records:list[dict]):
        if not records: return {"accepted":0}
        url=self.config.remote_url.rstrip("/")+"/ingest"
        # The native Space bridge and the migration compatibility service both
        # accept this canonical envelope; durable=true is mandatory in either
        # implementation.
        body=json.dumps({"device_id":self.config.device_id,"documents":records}, ensure_ascii=False).encode()
        headers=self._auth_headers(json_content=True)
        gzip_enabled = os.environ.get("FUNES_HTTP_GZIP", "true").strip().lower() not in {"0", "false", "no", "off"}
        try:
            gzip_min_bytes = max(0, int(os.environ.get("FUNES_HTTP_GZIP_MIN_BYTES", "8192")))
        except ValueError:
            gzip_min_bytes = 8192
        if gzip_enabled and len(body) >= gzip_min_bytes:
            compressed = gzip.compress(body, mtime=0)
            if len(compressed) < len(body):
                body = compressed
                headers["Content-Encoding"] = "gzip"
        timeout = float(os.environ.get("FUNES_REMOTE_TIMEOUT", "900"))
        last = None
        for attempt in range(4):
            req=request.Request(url,data=body,headers=headers,method="POST")
            try:
                with open_no_redirect(req,timeout=timeout) as r:
                    raw=r.read()
                    if not raw:
                        raise RuntimeError("remote returned an empty durable-ingest response")
                    result = json.loads(raw)
                    # The daemon may acknowledge a local queue row only after
                    # the remote confirms a durable commit.  A 2xx response
                    # without that contract is treated as retryable rather
                    # than silently losing the pending record.
                    if result.get("durable") is not True:
                        raise RuntimeError("remote did not confirm a durable commit")
                    accepted = int(result.get("accepted", result.get("created", 0) + result.get("updated", 0) + result.get("deduped", 0)))
                    if accepted < len(records):
                        raise RuntimeError(f"remote accepted {accepted}/{len(records)} records")
                    return result
            except error.HTTPError as exc:
                last = exc
                if exc.code not in (408, 425, 429) and exc.code < 500:
                    raise
            except RuntimeError as exc:
                last = exc
            except (error.URLError, TimeoutError, OSError) as exc:
                last = exc
            if attempt < 3:
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"remote ingest failed after retries: {type(last).__name__}") from last

    def health(self) -> bool:
        try:
            headers = self._auth_headers()
            req = request.Request(
                self.config.remote_url.rstrip("/") + "/ready",
                headers=headers,
                method="GET",
            )
            with open_no_redirect(req, timeout=8) as r:
                return 200 <= r.status < 300
        except (OSError, error.URLError, RuntimeError):
            return False

    def sync_snapshot(self) -> dict:
        headers=self._auth_headers(json_content=True)
        req=request.Request(self.config.remote_url.rstrip("/")+"/sync",data=b"{}",headers=headers,method="POST")
        with open_no_redirect(req,timeout=120) as r:
            raw=r.read(); result=json.loads(raw) if raw else {}
            if result.get("durable") is not True:
                raise RuntimeError("remote snapshot sync did not confirm a durable commit")
            return result

    def reindex(self, scope: str) -> dict:
        if scope not in {"retrieval_text", "all"}:
            raise ValueError("scope must be retrieval_text or all")
        body = json.dumps({"scope": scope}, separators=(",", ":")).encode()
        req = request.Request(
            self.config.remote_url.rstrip("/") + "/reindex",
            data=body,
            headers=self._auth_headers(json_content=True),
            method="POST",
        )
        timeout = float(os.environ.get("FUNES_REMOTE_TIMEOUT", "900"))
        with open_no_redirect(req, timeout=timeout) as response:
            raw = response.read()
            result = json.loads(raw) if raw else {}
            if response.status != 202:
                raise RuntimeError(
                    f"remote reindex returned HTTP {response.status}, expected 202"
                )
            if result.get("durable") is not True or result.get("queued") is not True:
                raise RuntimeError("remote reindex did not confirm a durable queue record")
            return result
