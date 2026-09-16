from __future__ import annotations
import atexit
import http.client
import io
import json,sys,os,time
import subprocess
import threading
from urllib import error, request
from urllib.parse import urlsplit, urlunsplit
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
_ERROR_BODY_MAX = _READY_ERROR_BODY_MAX
_CONNECTIONS = {}
_CONNECTIONS_LOCK = threading.Lock()
_FALLBACK_TRANSPORT = object()


class _BufferedResponse(io.BytesIO):
    """Small urllib-compatible response returned after a reusable read."""

    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False


class _ConnectionSlot:
    """Serialize access because HTTPConnection cannot pipeline safely."""

    __slots__ = ("plan", "lock", "connection")

    def __init__(self, plan):
        self.plan = plan
        self.lock = threading.Lock()
        self.connection = None


def _validated_url(url):
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("remote URL is invalid") from exc
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("remote URL must use http or https")
    if not parsed.hostname:
        raise ValueError("remote URL must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("remote URL must not include userinfo")
    return parsed, port


def _connection_plan(parsed, explicit_port):
    """Return a safe keep-alive plan, or request urllib proxy fallback."""
    scheme = parsed.scheme.lower()
    target_host = parsed.hostname
    target_port = explicit_port or (443 if scheme == "https" else 80)
    try:
        proxies = request.getproxies()
        bypass_proxy = request.proxy_bypass(parsed.netloc)
    except (OSError, TypeError, ValueError):
        return _FALLBACK_TRANSPORT
    proxy_url = None if bypass_proxy else proxies.get(scheme)
    if not proxy_url:
        return ("direct", scheme, target_host, target_port)

    # Safely support the common HTTPS-over-HTTP CONNECT case.  urllib remains
    # responsible for authenticated, HTTPS, PAC/system, and other proxy forms.
    if scheme != "https":
        return _FALLBACK_TRANSPORT
    try:
        proxy = urlsplit(proxy_url)
        proxy_port = proxy.port
    except (TypeError, ValueError):
        return _FALLBACK_TRANSPORT
    if (
        proxy.scheme.lower() != "http"
        or not proxy.hostname
        or proxy.username is not None
        or proxy.password is not None
        or proxy.query
        or proxy.fragment
        or proxy.path not in ("", "/")
    ):
        return _FALLBACK_TRANSPORT
    return (
        "http-connect",
        scheme,
        target_host,
        target_port,
        proxy.hostname,
        proxy_port or 80,
    )


def _connection_slot(plan):
    # The key deliberately contains endpoints only, never auth header values.
    with _CONNECTIONS_LOCK:
        slot = _CONNECTIONS.get(plan)
        if slot is None:
            slot = _ConnectionSlot(plan)
            _CONNECTIONS[plan] = slot
        return slot


def _new_connection(plan, timeout):
    if plan[0] == "direct":
        _, scheme, host, port = plan
        connection_type = (
            http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        )
        return connection_type(host, port, timeout=timeout)
    _, _, target_host, target_port, proxy_host, proxy_port = plan
    connection = http.client.HTTPSConnection(proxy_host, proxy_port, timeout=timeout)
    # Application auth is intentionally not passed to CONNECT. HTTPSConnection
    # also uses the tunnel host as TLS server_hostname after CONNECT succeeds.
    connection.set_tunnel(target_host, target_port)
    return connection


def _discard_connection(slot):
    connection, slot.connection = slot.connection, None
    if connection is not None:
        try:
            connection.close()
        except OSError:
            pass


def _close_connections():
    with _CONNECTIONS_LOCK:
        slots = list(_CONNECTIONS.values())
        _CONNECTIONS.clear()
    for slot in slots:
        with slot.lock:
            _discard_connection(slot)


atexit.register(_close_connections)


def _persistent_open(req, parsed, plan, timeout):
    slot = _connection_slot(plan)
    target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    headers = dict(req.header_items())
    with slot.lock:
        if slot.connection is None:
            slot.connection = _new_connection(plan, timeout)
        connection = slot.connection
        try:
            connection.timeout = timeout
            sock = getattr(connection, "sock", None)
            if sock is not None:
                sock.settimeout(timeout)
            connection.request(req.get_method(), target, body=req.data, headers=headers)
            response = connection.getresponse()
            status = response.status
            response_headers = response.headers
            if 200 <= status < 300:
                body = response.read()
            else:
                body = response.read(_ERROR_BODY_MAX + 1)
            if response.will_close or len(body) > _ERROR_BODY_MAX and status >= 300:
                _discard_connection(slot)
        except (http.client.HTTPException, OSError) as exc:
            _discard_connection(slot)
            raise error.URLError(exc) from exc
    if not 200 <= status < 300:
        raise error.HTTPError(req.full_url, status, response.reason, response_headers, io.BytesIO(body))
    return _BufferedResponse(body, status, response_headers)


def _open_remote(req, timeout):
    parsed, port = _validated_url(req.full_url)
    plan = _connection_plan(parsed, port)
    if plan is _FALLBACK_TRANSPORT:
        return open_no_redirect(req, timeout=timeout)
    return _persistent_open(req, parsed, plan, timeout)


def _ready_value(value):
    """Reduce a readiness response to an internal poll state only."""
    if not isinstance(value, dict):
        return ""
    # Legacy Spaces returned HTTP 200/ok while native warm was still in
    # progress. That search gate must win over the top-level marker.
    warm = value.get("native_warm")
    if isinstance(warm, dict) and warm.get("state") == "warming":
        return "warming"
    return "ready" if value.get("ok") else ""


def _ready_state(base, headers, timeout):
    """Read only a bounded readiness state, never exposing response data."""
    req = request.Request(base + "/ready/search", headers=headers, method="GET")
    try:
        with _open_remote(req, timeout=timeout) as resp:
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


def _wait_until_ready(base, headers, deadline, default_timeout=8, default_polls=8):
    """Avoid sending a recall into the known cold/warming window."""
    timeout = _float_env("FUNES_REMOTE_READY_TIMEOUT", default_timeout, 0.1, 15)
    polls = _int_env("FUNES_REMOTE_READY_POLLS", default_polls, 1, 30)
    state = ""
    for _ in range(polls):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        state = _ready_state(base, headers, min(timeout, remaining))
        if state != "warming":
            break
        if _ + 1 >= polls:
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
    _validated_url(base)
    headers = _auth_headers(token, hub_token)
    recall = path in ("/search", "/recall")
    total = _float_env("FUNES_REMOTE_TIMEOUT", 8 if recall else 180, 0.1, 300)
    attempts = _int_env("FUNES_REMOTE_ATTEMPTS", 2 if recall else 5, 1, 5)
    attempt_timeout = _float_env(
        "FUNES_REMOTE_ATTEMPT_TIMEOUT", 8 if recall else 50, 0.1, 55
    )
    deadline = time.monotonic() + total

    # /search owns its cold-restore/degraded behavior and must receive the
    # entire fail-open deadline. A separate readiness request can consume most
    # of that budget on a high-latency Space. Keep the legacy /recall preflight
    # for callers that still use that endpoint.
    if path == "/recall":
        state = _wait_until_ready(
            base, headers, deadline, default_timeout=2, default_polls=1
        )
        if state == "warming" and time.monotonic() >= deadline:
            return None

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
            with _open_remote(req, timeout=min(attempt_timeout, remaining)) as resp:
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
    if recall:
        return None
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
