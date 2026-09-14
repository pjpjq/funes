from __future__ import annotations
import json,sys,os,time
import subprocess
from urllib import error, request
from .config import Config
from .http import open_no_redirect
from .store import Store

def _keychain(service):
    if sys.platform != "darwin":
        return ""
    try:
        return subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-a", os.environ.get("USER", ""), "-s", service, "-w"],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.rstrip("\n")
    except OSError:
        return ""


def _float_env(name, default, lower, upper):
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = float(default)
    return min(float(upper), max(float(lower), value))


def _int_env(name, default, lower, upper):
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = int(default)
    return min(int(upper), max(int(lower), value))


def _auth_headers(token, hub_token):
    headers = {"Content-Type": "application/json", "User-Agent": "funes-sync-mcp/1"}
    if hub_token:
        headers["Authorization"] = "Bearer " + hub_token
        headers["X-Funes-Authorization"] = "Bearer " + token
    else:
        headers["Authorization"] = "Bearer " + token
    return headers


_READY_ERROR_BODY_MAX = 16 * 1024


def _ready_value(value):
    """Reduce a readiness response to an internal poll state only."""
    if not isinstance(value, dict):
        return ""
    # Legacy Spaces returned HTTP 200/ok while native warm or source restore
    # was still in progress. Those gates must win over the top-level marker.
    warm = value.get("native_warm")
    if isinstance(warm, dict) and warm.get("state") == "warming":
        return "warming"
    sources = value.get("source_store")
    if (
        isinstance(sources, dict)
        and sources.get("configured")
        and not sources.get("ready")
    ):
        # Restores and unavailable source indexes are retryable readiness
        # gates, even when the native worker itself has already warmed.
        return "warming"
    return "ready" if value.get("ok") else ""


def _ready_state(base, headers, timeout):
    """Read only a bounded readiness state, never exposing response data."""
    req = request.Request(base + "/ready", headers=headers, method="GET")
    try:
        with open_no_redirect(req, timeout=timeout) as resp:
            value = json.loads(resp.read() or b"{}")
    except error.HTTPError as exc:
        if exc.code != 503:
            return ""
        try:
            raw = exc.read(_READY_ERROR_BODY_MAX + 1)
            if len(raw) > _READY_ERROR_BODY_MAX:
                return ""
            value = json.loads(raw or b"{}")
        except (OSError, TypeError, ValueError):
            return ""
        finally:
            try:
                exc.close()
            except OSError:
                pass
    except (error.URLError, TimeoutError, OSError, ValueError):
        return ""
    return _ready_value(value)


def _wait_until_ready(base, headers, deadline):
    """Avoid sending a recall into the known cold/warming window."""
    timeout = _float_env("FUNES_REMOTE_READY_TIMEOUT", 8, 1, 15)
    polls = _int_env("FUNES_REMOTE_READY_POLLS", 8, 1, 30)
    state = ""
    for _ in range(polls):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        state = _ready_state(base, headers, min(timeout, remaining))
        if state != "warming":
            break
        time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
    return state


def _retry_after(exc, attempt):
    exponential = min(30.0, float(2 ** (attempt + 1)))
    if isinstance(exc, error.HTTPError):
        try:
            value = float(exc.headers.get("Retry-After", ""))
            return max(exponential, min(30.0, max(0.0, value)))
        except (AttributeError, TypeError, ValueError):
            pass
    return exponential


def _remote_call(path, payload):
    config = Config.load()
    base=(os.environ.get("FUNES_REMOTE_URL") or config.remote_url).rstrip("/")
    token=os.environ.get("FUNES_API_TOKEN", "") or _keychain("funes-api-token")
    hub_token=os.environ.get("FUNES_HF_TOKEN", "") or os.environ.get("HF_TOKEN", "") or _keychain("funes-hf-token")
    if not base or not token:
        return None
    headers = _auth_headers(token, hub_token)
    total = _float_env("FUNES_REMOTE_TIMEOUT", 180, 10, 300)
    attempts = _int_env("FUNES_REMOTE_ATTEMPTS", 5, 1, 5)
    attempt_timeout = _float_env("FUNES_REMOTE_ATTEMPT_TIMEOUT", 50, 5, 55)
    deadline = time.monotonic() + total

    # The Space reports HTTP 200 while its native worker is warming.  Waiting
    # here is materially better than sending a request that sits behind the
    # warm lock until the HF front door closes the connection.
    if path in ("/search", "/recall"):
        state = _wait_until_ready(base, headers, deadline)
        if state == "warming" and time.monotonic() >= deadline:
            raise TimeoutError("remote native worker is still warming")

    last = None
    for attempt in range(attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        req=request.Request(
            base+path,
            data=json.dumps(payload,ensure_ascii=False).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with open_no_redirect(req, timeout=min(attempt_timeout, remaining)) as resp:
                raw = resp.read()
            if not raw:
                raise RuntimeError("remote returned an empty response")
            return json.loads(raw)
        except error.HTTPError as exc:
            # Authentication and malformed requests are caller errors; retry
            # only transient gateway/provider failures.
            if exc.code < 500 and exc.code not in (408, 425, 429):
                raise
            last = exc
        except (error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
            last = exc
        if attempt + 1 < attempts:
            delay = min(_retry_after(last, attempt), max(0.0, deadline - time.monotonic()))
            if delay:
                time.sleep(delay)
    raise RuntimeError("remote request failed after retries") from last

def serve(store=None):
    store=store or Store()
    config = Config.load()
    remote=bool((os.environ.get("FUNES_REMOTE_URL") or config.remote_url) and (os.environ.get("FUNES_API_TOKEN") or _keychain("funes-api-token")))
    for line in sys.stdin:
        try:
            msg=json.loads(line); method=msg.get("method"); ident=msg.get("id"); p=msg.get("params") or {}
            if method=="initialize": result={"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"funes-sync","version":"1"}}
            elif method=="notifications/initialized": continue
            elif method=="tools/list": result={"tools":[{"name":"recall","description":"Search the unified Codex, Pi and Claude memory; return original raw context","inputSchema":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"},"source_agent":{"type":"string"},"source_type":{"type":"string"},"project":{"type":"string"},"repo":{"type":"string"},"device_id":{"type":"string"},"role":{"type":"string"},"content_type":{"type":"string"},"since":{"type":"string"},"until":{"type":"string"}} ,"required":["query"]}},{"name":"get","description":"Get one original memory record","inputSchema":{"type":"object","properties":{"record_id":{"type":"string"}},"required":["record_id"]}},{"name":"status","description":"Show unified memory sync status","inputSchema":{"type":"object","properties":{}}}]}
            elif method=="tools/call":
                name=p.get("name"); args=p.get("arguments") or {}
                if remote:
                    if name=="recall": val=_remote_call("/search", args)
                    elif name=="get": val=_remote_call("/get", {"source_identity":args.get("record_id","")})
                    elif name=="status": val=_remote_call("/sync/status", {})
                    else: val={"error":"unknown_tool"}
                else:
                    val=store.search(args.get("query", "")) if name=="recall" else store.get(args.get("record_id", ""))
                result={"content":[{"type":"text","text":json.dumps(val,ensure_ascii=False)}]}
            else: result={}
            if ident is not None: print(json.dumps({"jsonrpc":"2.0","id":ident,"result":result}),flush=True)
        except Exception as e:
            if 'ident' in locals() and ident is not None: print(json.dumps({"jsonrpc":"2.0","id":ident,"error":{"code":-32000,"message":str(e)}}),flush=True)
