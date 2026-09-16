from __future__ import annotations
import gzip
import json, os, subprocess, sys, time
from email.utils import parsedate_to_datetime
from urllib import error, parse, request
from .config import Config
from .http import open_no_redirect


class _PermanentIngestError(RuntimeError):
    pass


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
        headers["Prefer"] = "respond-async"
        timeout = float(os.environ.get("FUNES_REMOTE_TIMEOUT", "900"))
        if timeout <= 0:
            raise RuntimeError("FUNES_REMOTE_TIMEOUT must be greater than zero")
        try:
            max_response_bytes = max(
                1, int(os.environ.get("FUNES_REMOTE_MAX_RESPONSE_BYTES", "1048576"))
            )
        except ValueError:
            max_response_bytes = 1048576
        deadline = time.monotonic() + timeout
        poll_url = None
        operation_id = None
        backoff = 1.0
        consecutive_failures = 0
        last = None

        def remaining() -> float:
            value = deadline - time.monotonic()
            if value <= 0:
                raise RuntimeError("remote ingest timed out before durable confirmation") from last
            return value

        def response_header(response, name: str) -> str:
            response_headers = getattr(response, "headers", None)
            if response_headers is not None:
                value = response_headers.get(name)
                if value is not None:
                    return str(value)
            getheader = getattr(response, "getheader", None)
            return str(getheader(name) or "") if getheader is not None else ""

        def retry_delay(response) -> float:
            nonlocal backoff
            value = response_header(response, "Retry-After").strip()
            delay = None
            if value:
                try:
                    delay = max(0.0, float(value))
                except ValueError:
                    try:
                        retry_at = parsedate_to_datetime(value)
                        delay = max(0.0, retry_at.timestamp() - time.time())
                    except (TypeError, ValueError, OverflowError):
                        delay = None
            if delay is None:
                delay = backoff
            backoff = min(30.0, backoff * 2.0)
            return delay

        def sleep_before_retry(response) -> None:
            time.sleep(min(retry_delay(response), remaining()))

        def note_failure(exc: Exception) -> None:
            nonlocal consecutive_failures
            consecutive_failures += 1
            if consecutive_failures >= 4:
                raise RuntimeError(
                    f"remote ingest failed after retries: {type(exc).__name__}"
                ) from exc

        def set_response_timeout(response, value: float) -> None:
            fp = getattr(response, "fp", None)
            raw = getattr(fp, "raw", None)
            sock = getattr(raw, "_sock", None) or getattr(response, "_sock", None)
            setter = getattr(sock, "settimeout", None)
            if setter is not None:
                setter(value)

        def read_result(response) -> dict:
            content_length = response_header(response, "Content-Length").strip()
            if content_length:
                try:
                    if int(content_length) > max_response_bytes:
                        raise _PermanentIngestError(
                            "remote durable-ingest response exceeded size limit"
                        )
                except ValueError:
                    pass
            reader = getattr(response, "read1", None)
            if not callable(reader):
                reader = response.read
            chunks = bytearray()
            one_shot = False
            while True:
                budget = remaining()
                set_response_timeout(response, budget)
                read_size = min(65536, max_response_bytes + 1 - len(chunks))
                try:
                    chunk = reader(read_size)
                except TypeError:
                    if chunks:
                        raise
                    chunk = reader()
                    one_shot = True
                if not chunk:
                    break
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                chunks.extend(chunk)
                if len(chunks) > max_response_bytes:
                    raise _PermanentIngestError(
                        "remote durable-ingest response exceeded size limit"
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "remote ingest timed out before durable confirmation"
                    )
                if one_shot:
                    break
            raw = bytes(chunks)
            if not raw:
                raise RuntimeError("remote returned an empty durable-ingest response")
            try:
                result = json.loads(raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("remote returned an invalid durable-ingest response") from exc
            if not isinstance(result, dict):
                raise RuntimeError("remote returned a non-object durable-ingest response")
            return result

        def validate_durable(result: dict) -> dict:
            # The daemon may acknowledge a local queue row only after
            # the remote confirms a durable commit.  A 2xx response
            # without that contract is treated as retryable rather
            # than silently losing the pending record.
            if result.get("durable") is not True:
                raise RuntimeError("remote did not confirm a durable commit")
            try:
                accepted = (
                    int(result["accepted"])
                    if "accepted" in result
                    else int(result.get("created", 0))
                    + int(result.get("updated", 0))
                    + int(result.get("deduped", 0))
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError("remote returned an invalid accepted count") from exc
            if accepted < len(records):
                raise RuntimeError(f"remote accepted {accepted}/{len(records)} records")
            remaining()
            return result

        def status_url(result: dict, response) -> tuple[str, str]:
            returned_id = str(result.get("operation_id") or "").strip()
            if not returned_id:
                raise RuntimeError("remote async ingest response omitted operation_id")
            candidate = str(result.get("status_url") or response_header(response, "Location")).strip()
            if not candidate:
                candidate = url.rstrip("/") + "/operations/" + parse.quote(returned_id, safe="")
            resolved = parse.urljoin(url + "/", candidate)
            base = parse.urlsplit(url)
            target = parse.urlsplit(resolved)
            if target.scheme not in {"http", "https"} or (
                target.scheme.lower(), target.netloc.lower()
            ) != (base.scheme.lower(), base.netloc.lower()):
                raise _PermanentIngestError(
                    "remote async ingest status URL changed origin"
                )
            return returned_id, resolved

        while True:
            polling = poll_url is not None
            req = request.Request(
                poll_url if polling else url,
                data=None if polling else body,
                headers=self._auth_headers() if polling else headers,
                method="GET" if polling else "POST",
            )
            try:
                with open_no_redirect(req,timeout=remaining()) as response:
                    status = int(getattr(response, "status", 0))
                    result = read_result(response)
                    if polling:
                        returned_id = str(result.get("operation_id") or "").strip()
                        if returned_id != operation_id:
                            raise RuntimeError("remote async ingest operation_id changed")
                    if status == 200:
                        return validate_durable(result)
                    if status == 202:
                        if not polling:
                            operation_id, poll_url = status_url(result, response)
                        consecutive_failures = 0
                        last = RuntimeError("remote ingest operation is still running")
                    elif status == 429 or status >= 500:
                        last = RuntimeError(f"remote ingest returned retryable HTTP {status}")
                        if polling and status == 503:
                            poll_url = None
                            operation_id = None
                        note_failure(last)
                    else:
                        raise RuntimeError(f"remote ingest returned unexpected HTTP {status}")
                sleep_before_retry(response)
            except error.HTTPError as exc:
                last = exc
                if poll_url is not None and exc.code == 404:
                    poll_url = None
                    operation_id = None
                    note_failure(exc)
                    sleep_before_retry(exc)
                    continue
                if exc.code not in (408, 425, 429) and exc.code < 500:
                    raise
                if poll_url is not None and exc.code == 503:
                    poll_url = None
                    operation_id = None
                note_failure(exc)
                sleep_before_retry(exc)
            except _PermanentIngestError:
                raise
            except RuntimeError as exc:
                last = exc
                poll_url = None
                operation_id = None
                note_failure(exc)
                sleep_before_retry(exc)
            except (error.URLError, TimeoutError, OSError) as exc:
                last = exc
                note_failure(exc)
                sleep_before_retry(exc)

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

    def missing_source_identities(self, identities: list[str]) -> list[str]:
        unique=list(dict.fromkeys(str(value) for value in identities))
        if not unique:
            return []
        if len(unique)>5000 or any(not value for value in unique):
            raise ValueError("source identity check requires 1-5000 non-empty identities")
        body=json.dumps({"source_identities":unique},separators=(",",":")).encode()
        req=request.Request(
            self.config.remote_url.rstrip("/")+"/sources/check",
            data=body,
            headers=self._auth_headers(json_content=True),
            method="POST",
        )
        timeout=min(60.0,float(os.environ.get("FUNES_REMOTE_TIMEOUT","900")))
        if timeout<=0:
            raise RuntimeError("FUNES_REMOTE_TIMEOUT must be greater than zero")
        with open_no_redirect(req,timeout=timeout) as response:
            raw=response.read(1024*1024+1)
            if len(raw)>1024*1024:
                raise RuntimeError("remote source check response exceeded size limit")
            result=json.loads(raw) if raw else {}
        if not isinstance(result,dict) or result.get("ok") is not True:
            raise RuntimeError("remote source check returned an invalid response")
        present=result.get("present")
        missing=result.get("missing")
        if not isinstance(present,list) or not isinstance(missing,list):
            raise RuntimeError("remote source check returned invalid identity lists")
        if any(not isinstance(value,str) for value in (*present,*missing)):
            raise RuntimeError("remote source check returned a non-string identity")
        if len(set(present))!=len(present) or len(set(missing))!=len(missing):
            raise RuntimeError("remote source check returned duplicate identities")
        expected=set(unique)
        if set(present)&set(missing) or set(present)|set(missing)!=expected:
            raise RuntimeError("remote source check returned an inconsistent inventory")
        return missing

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
