#!/usr/bin/env python3
"""Small authenticated HTTP bridge for the Funes CLI.

The durable source of truth is the configured HF Hub dataset (FUNES_MEMORY).  The
container's /data/.funes directory is only a warm cache and may be recreated.
"""
import hashlib
import atexit
import json
import os
import re
import select
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from service.server import App as SourceApp
from service.server import ingest_documents as persist_source_ingest
from service.server import prepare_ingest_documents as prepare_source_ingest_documents


FUNES_BIN = os.getenv("FUNES_BIN", "/usr/local/bin/funes")
REMOTE = os.getenv("FUNES_MEMORY", "")
TOKEN = os.getenv("FUNES_API_TOKEN", "")
HOME = Path(os.getenv("FUNES_HOME", "/data/.funes"))
PORT = int(os.getenv("PORT", "7860"))
TRANSLATION_THRESHOLD = float(os.getenv("TRANSLATE_CHINESE_THRESHOLD", "0.15"))
INGEST_INDEX_TIMEOUT = int(os.getenv("FUNES_INGEST_INDEX_TIMEOUT", "900"))
INGEST_PUSH_TIMEOUT = int(os.getenv("FUNES_INGEST_PUSH_TIMEOUT", "1800"))
MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_TIMEOUT = float(os.getenv("FUNES_MCP_TIMEOUT", "180"))
MCP_HANDSHAKE_TIMEOUT = float(os.getenv("FUNES_MCP_HANDSHAKE_TIMEOUT", "10"))
try:
    RECALL_LOCK_TIMEOUT = min(55.0, max(0.1, float(os.getenv("FUNES_RECALL_LOCK_TIMEOUT", "2"))))
except ValueError:
    RECALL_LOCK_TIMEOUT = 2.0
try:
    HTTP_MAX_CANDIDATES = max(1, int(os.getenv("FUNES_HTTP_MAX_CANDIDATES", "12")))
except ValueError:
    HTTP_MAX_CANDIDATES = 12
PROMPT_VERSION = "funes-retrieval-v1"
LANGUAGE_MODE = os.getenv("FUNES_RETRIEVAL_LANGUAGE_MODE", "auto").lower()
INDEX_LOCK = threading.Lock()
# Writes use the native memory lock inside the `funes` process.  Keep their
# Python-level serialization separate from reads so a background ingest/push
# cannot make an HTTP recall wait for the full upload duration.
WRITE_LOCK = threading.Lock()
SOURCE_APP = None
SOURCE_APP_LOCK = threading.Lock()


def source_app():
    """Return the encrypted raw/source store, or None when not configured."""
    global SOURCE_APP
    if SOURCE_APP is not None:
        return SOURCE_APP
    if not os.getenv("FUNES_STORAGE_REPO"):
        return None
    with SOURCE_APP_LOCK:
        if SOURCE_APP is None:
            os.environ.setdefault("FUNES_DATA_DIR", str(HOME / "source-store"))
            os.environ.setdefault("FUNES_LAZY_RESTORE", "true")
            os.environ.setdefault("FUNES_REQUIRE_DURABLE_ACK", "true")
            SOURCE_APP = SourceApp()
    return SOURCE_APP


def source_state() -> dict[str, object]:
    app = source_app()
    if app is None:
        return {"configured": False, "ready": False, "documents": 0}
    if app.syncer.restoring:
        return {"configured": True, "ready": False, "restoring": True, "documents": app.store.count()}
    if app.syncer.restore_failed:
        return {"configured": True, "ready": False, "error": "restore_failed", "documents": app.store.count()}
    return {
        "configured": True,
        "ready": True,
        "documents": app.store.count(),
        "restored": app.restore_result,
        "sync": app.store.sync_status(),
    }


def close_source_app() -> None:
    global SOURCE_APP
    with SOURCE_APP_LOCK:
        app = SOURCE_APP
        SOURCE_APP = None
    if app is not None:
        try:
            app.close()
        except Exception:
            pass


def prepare_source_documents(app, docs: list[dict]) -> list[dict]:
    """Build the durable raw-first representation without provider I/O."""
    return prepare_source_ingest_documents(app, docs)


def ingest_source_documents(docs: list[dict]) -> tuple[int, dict, list[dict]] | None:
    app = source_app()
    if app is None:
        return None
    if app.syncer.restoring or app.syncer.restore_failed:
        error = "restore_in_progress" if app.syncer.restoring else "restore_failed"
        return 503, {"ok": False, "durable": False, "error": error}, []
    result = persist_source_ingest(app, docs)
    identities = [str(item.get("source_identity", "")) for item in result["items"]]
    canonical = app.store.get_many(identities)
    result["ok"] = bool(result["durable"])
    return (200 if result["durable"] else 503), result, canonical


def search_source_documents(query: str, limit: int, filters: dict[str, object]) -> tuple[str, list[dict]]:
    app = source_app()
    if app is None or app.syncer.restoring or app.syncer.restore_failed:
        return query, []
    rewritten = app.translator.rewrite(query)
    hits = app.store.search(rewritten, limit, filters=filters)
    if rewritten != query and not hits:
        hits = app.store.search(query, limit, filters=filters)
    native_filterable = set(filters).issubset({"source_agent", "repo"})
    if native_filterable:
        hits = [
            item
            for item in hits
            if str(item.get("source_type", "")).lower()
            not in {"session", "codex", "codex_session", "pi", "pi_session", "claude", "claude_session"}
        ]
    for item in hits:
        if os.getenv("RETURN_RETRIEVAL_TEXT", "false").lower() not in {"1", "true", "yes", "on"}:
            item.pop("retrieval_text", None)
    return rewritten, hits

# A remote Lance memory can take longer than the Space ingress timeout to open
# on the first recall (model + snapshot + ANN/FTS handles).  Warm it in the
# background at process start so the public search path is hot before an agent
# asks its first question.  Keep only coarse state here: never expose child
# stderr or query/source text in the readiness response.
_WARM_STATE_LOCK = threading.Lock()
_WARM_STATE = {
    "state": "not_started",
    "started_at": None,
    "finished_at": None,
}


def warm_state() -> dict[str, object]:
    with _WARM_STATE_LOCK:
        return dict(_WARM_STATE)


def _warm_native_memory(*, replace: bool = False) -> None:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _WARM_STATE_LOCK:
        if _WARM_STATE.get("state") != "warming":
            _WARM_STATE.update(state="warming", started_at=started, finished_at=None)
    try:
        # A minimal recall initializes the same remote dataset, embedding model,
        # text index, and reranker used by real requests.  It is intentionally
        # run outside the HTTP request lifecycle.  Refreshes use a second child
        # and swap it in only after it is ready, so an old worker can keep
        # serving while a newly-pushed remote snapshot is opened.
        # Always build a replacement child outside INDEX_LOCK.  The first warm
        # used to call the active worker while holding that lock, so the first
        # real HTTP recall could wait for the full cold-open duration.  A
        # replacement is safe for both startup and post-push refreshes: the
        # old worker remains available until the candidate is ready.
        _refresh_native_worker()
    except Exception:
        # Readiness remains useful when a provider/HF endpoint is temporarily
        # unavailable; the next request will retry through the normal worker
        # recovery path.  Do not retain or emit exception text.
        state = "error"
    else:
        state = "ready"
    with _WARM_STATE_LOCK:
        _WARM_STATE.update(state=state, finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def request_warm(*, force: bool = False) -> dict[str, object]:
    """Start one background refresh, optionally replacing an old remote worker."""
    with _WARM_STATE_LOCK:
        if _WARM_STATE.get("state") == "warming":
            return dict(_WARM_STATE)
        # Reserve the state before starting the thread.  Thread.start() may be
        # delayed, and a second /warm or ingest call must not launch another
        # native MCP child in that gap.
        _WARM_STATE.update(
            state="warming",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            finished_at=None,
        )
    threading.Thread(
        target=_warm_native_memory,
        kwargs={"replace": force},
        name="funes-native-warm",
        daemon=True,
    ).start()
    return warm_state()

# The default Funes embedding model is English-oriented.  Keep this small,
# deterministic fallback for installations without a translation provider so
# Chinese queries do not enter the slow CJK tokenizer path.  Technical names
# and identifiers are always collected separately and remain verbatim.
CHINESE_RETRIEVAL_TERMS = (
    ("previous_response_id", "previous_response_id"),
    ("后台任务", "background task"),
    ("保持运行", "keep running"),
    ("如何保持", "how to keep"),
    ("不能上传", "must not upload"),
    ("硬件标识", "hardware identifier"),
    ("远程 MCP", "remote MCP"),
    ("上下文丢失", "context loss"),
    ("上下文丢了", "context loss"),
    ("重启后", "after restart"),
    ("自动恢复", "automatic recovery"),
    ("断网时", "when offline"),
    ("重复扫描", "repeated scan"),
    ("如何去重", "how to deduplicate"),
    ("换 embedding", "switch embedding"),
    ("远程", "remote"),
    ("日志", "logs"),
    ("时区", "timezone"),
    ("对齐", "align"),
    ("扫描", "scan"),
    ("文件", "file"),
    ("上传", "upload"),
    ("模型", "model"),
    ("重建", "rebuild"),
    ("挂了", "unavailable failed"),
    ("怎么办", "what to do"),
    ("共存", "coexistence"),
    ("注意", "considerations"),
    ("查询", "query"),
    ("字段", "fields schema"),
    ("检索", "retrieval"),
    ("处理", "handle"),
    ("关系", "relationship"),
    ("能不能", "can"),
    ("多台", "multiple"),
    ("避免", "avoid"),
    ("泄露", "leak"),
    ("重复", "duplicate"),
    ("搜", "search"),
    ("丢吗", "lost"),
    ("第二轮", "second turn"),
    ("第二次", "second turn"),
    ("丢上下文", "context loss"),
    ("上下文", "context"),
    ("高延迟", "high latency"),
    ("延迟", "latency"),
    ("连接", "connection"),
    ("配置", "configuration settings"),
    ("设置", "configuration settings"),
    ("默认", "default"),
    ("推理", "reasoning inference"),
    ("记忆", "memory"),
    ("会话", "conversation session"),
    ("历史", "history previous"),
    ("之前", "previous prior"),
    ("上次", "previous last time"),
    ("以前", "previous earlier"),
    ("为什么", "why cause"),
    ("为何", "why cause"),
    ("是什么", "what is"),
    ("讨论", "discussion"),
    ("决定", "decision"),
    ("测试", "test result"),
    ("偏好", "preference"),
    ("部署", "deployment deployed"),
    ("丢", "loss lost"),
    ("办公室", "office"),
)


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum("\u4e00" <= c <= "\u9fff" for c in text) / max(1, len(text))


def query_text(raw: str) -> str:
    """Build an ASCII retrieval shadow while leaving the caller's raw query intact."""
    if LANGUAGE_MODE == "raw":
        return raw
    if LANGUAGE_MODE == "auto" and cjk_ratio(raw) < TRANSLATION_THRESHOLD:
        return raw
    # Keep technical entities verbatim even when the optional provider is absent.
    entities = re.findall(r"[A-Za-z][A-Za-z0-9_.*:/-]{1,}", raw)
    prompt = (
        "You are a retrieval normalization engine. Convert the natural-language Chinese "
        "portions into concise English retrieval text. Preserve technical entities exactly. "
        "Output retrieval text only.\n\nInput:\n" + raw
    )
    base = os.getenv("TRANSLATION_BASE_URL")
    key = os.getenv("TRANSLATION_API_KEY")
    model = os.getenv("TRANSLATION_MODEL", "")
    translated = ""
    if base and key and model:
        payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode()
        base_url = base.rstrip("/")
        endpoint = base_url + ("/chat/completions" if base_url.endswith("/v1") else "/v1/chat/completions")
        req = urllib.request.Request(endpoint, data=payload, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                obj = json.load(response)
            translated = obj.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except (OSError, ValueError, KeyError, IndexError):
            translated = ""
    # A configured provider is preferred.  If it is unavailable, use the
    # deterministic phrase map above; sending raw CJK to the English embedding
    # backend can take longer than the Space request deadline.
    fallback = []
    for phrase, english in CHINESE_RETRIEVAL_TERMS:
        if phrase in raw:
            fallback.append(english)
    # Do not put a CJK-only shadow back into the native CLI.  The raw query is
    # still returned to clients and the original source text remains untouched.
    translated_ascii = re.sub(r"[^\x00-\x7F]+", " ", translated).strip()
    pieces = []
    for piece in (translated_ascii, " ".join(fallback), " ".join(entities)):
        if piece and piece not in pieces:
            pieces.append(piece)
    shadow = " ".join(pieces).strip()
    return shadow or "memory context retrieval"


def run(*args: str, timeout: int = 180) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["FUNES_HOME"] = str(HOME)
    p = subprocess.run([FUNES_BIN, *args], text=True, capture_output=True, timeout=timeout, env=env)
    return p.returncode, p.stdout, p.stderr


class NativeMcpError(subprocess.SubprocessError):
    """A native MCP failure without carrying process output into logs or HTTP responses."""


class NativeMcpBusyError(NativeMcpError):
    """The single native read worker is occupied; callers should retry shortly."""


class NativeMcpWorker:
    """Keep one ``funes mcp`` process warm for model and remote-cache reuse."""

    def __init__(
        self,
        binary: str,
        remote: str,
        home: Path,
        *,
        timeout: float = MCP_TIMEOUT,
        handshake_timeout: float = MCP_HANDSHAKE_TIMEOUT,
    ) -> None:
        self.binary = binary
        self.remote = remote
        self.home = Path(home)
        self.timeout = float(timeout)
        self.handshake_timeout = float(handshake_timeout)
        self._process = None
        self._next_id = 1
        self._lock = threading.RLock()

    @property
    def process(self):
        """Expose the child for focused tests without making it part of the HTTP contract."""
        return self._process

    def close(self) -> None:
        with self._lock:
            self._stop_locked()

    def _environment(self) -> dict[str, str]:
        # Keep HF_TOKEN/HF_HOME and any other caller-provided Hub settings.  Only
        # FUNES_HOME is pinned to the Space's durable warm-cache directory.
        env = os.environ.copy()
        env["FUNES_HOME"] = str(self.home)
        return env

    @staticmethod
    def _alive(process) -> bool:
        try:
            return process.poll() is None
        except AttributeError:
            return True

    @staticmethod
    def _close_stream(stream) -> None:
        if stream is None:
            return
        try:
            stream.close()
        except (OSError, ValueError):
            pass

    def _stop_locked(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if self._alive(process):
            try:
                process.terminate()
            except (OSError, AttributeError):
                pass
            try:
                process.wait(timeout=0.5)
            except (OSError, subprocess.TimeoutExpired, AttributeError, TypeError):
                try:
                    process.kill()
                except (OSError, AttributeError):
                    pass
                try:
                    process.wait(timeout=0.5)
                except (OSError, subprocess.TimeoutExpired, AttributeError, TypeError):
                    pass
        self._close_stream(getattr(process, "stdin", None))
        self._close_stream(getattr(process, "stdout", None))

    def _start_locked(self) -> None:
        args = [self.binary, "mcp"]
        if self.remote:
            args.append(self.remote)
        try:
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=self._environment(),
            )
        except OSError as exc:
            raise NativeMcpError("native MCP unavailable") from exc
        self._process = process
        # JSON-RPC ids are local to one child.  Resetting here also makes a
        # restarted worker interoperable with strict fake/native servers.
        self._next_id = 1
        try:
            self._request_locked(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "funes-space-bridge", "version": "1"},
                },
                self.handshake_timeout,
            )
            self._send_locked({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        except NativeMcpError:
            self._stop_locked()
            raise

    def _ensure_started_locked(self) -> None:
        if self._process is not None and self._alive(self._process):
            return
        self._stop_locked()
        self._start_locked()

    def _send_locked(self, message: dict) -> None:
        process = self._process
        if process is None or not self._alive(process):
            raise NativeMcpError("native MCP process exited")
        stdin = getattr(process, "stdin", None)
        if stdin is None:
            raise NativeMcpError("native MCP stdin unavailable")
        try:
            stdin.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
            stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise NativeMcpError("native MCP write failed") from exc

    @staticmethod
    def _read_line(stream, deadline: float) -> str:
        """Read one newline-delimited frame without ever blocking past deadline."""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NativeMcpError("native MCP request timed out")
            try:
                fd = stream.fileno()
            except (AttributeError, OSError, ValueError):
                # Small in-memory fakes used by tests do not expose a file
                # descriptor.  Their readline is non-blocking by contract.
                readable = True
            else:
                try:
                    readable = bool(select.select([fd], [], [], remaining)[0])
                except (OSError, ValueError):
                    readable = True
            if not readable:
                raise NativeMcpError("native MCP request timed out")
            try:
                line = stream.readline()
            except (OSError, ValueError) as exc:
                raise NativeMcpError("native MCP read failed") from exc
            if line in ("", b""):
                raise NativeMcpError("native MCP process closed stdout")
            if isinstance(line, bytes):
                line = line.decode("utf-8", "replace")
            return line.strip()

    def _request_locked(self, method: str, params: dict, timeout: float) -> object:
        request_id = self._next_id
        self._next_id += 1
        self._send_locked({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        process = self._process
        stdout = getattr(process, "stdout", None) if process is not None else None
        if stdout is None:
            raise NativeMcpError("native MCP stdout unavailable")
        deadline = time.monotonic() + max(0.001, float(timeout))
        while True:
            line = self._read_line(stdout, deadline)
            if not line:
                continue
            try:
                message = json.loads(line)
            except (TypeError, ValueError):
                # A native server must keep stdout JSON-RPC, but ignoring a
                # stray line avoids reflecting it (which could contain data).
                continue
            if not isinstance(message, dict) or message.get("id") != request_id:
                continue
            if message.get("error") is not None:
                raise NativeMcpError("native MCP request failed")
            return message.get("result")

    def _call(self, method: str, params: dict, *, timeout: float | None = None) -> object:
        # A read-only recall/get can safely be retried once after a dead child;
        # the retry also covers a child that exits during initialization.
        last_error = None
        with self._lock:
            for attempt in range(2):
                try:
                    self._ensure_started_locked()
                    return self._request_locked(method, params, self.timeout if timeout is None else timeout)
                except NativeMcpError as exc:
                    last_error = exc
                    self._stop_locked()
                    if attempt == 0:
                        continue
                    raise
        raise last_error or NativeMcpError("native MCP request failed")

    @staticmethod
    def _text(result: object) -> str:
        if isinstance(result, str):
            return result
        if not isinstance(result, dict):
            raise NativeMcpError("native MCP response malformed")
        if result.get("isError"):
            raise NativeMcpError("native MCP tool failed")
        content = result.get("content")
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text", ""), str)
            )
        text = result.get("text")
        return text if isinstance(text, str) else ""

    def call_tool(self, name: str, arguments: dict, *, timeout: float | None = None) -> str:
        result = self._call("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)
        return self._text(result)

    def recall(
        self,
        query: str,
        *,
        k: int = 8,
        candidates: int | None = None,
        half_life: float | None = None,
        neighbors: int | None = None,
        block_type: str | None = None,
        harness: str | None = None,
        **extra,
    ) -> str:
        arguments = {"query": str(query), "k": int(k)}
        for name, value in (
            ("candidates", candidates),
            ("half_life", half_life),
            ("neighbors", neighbors),
            ("block_type", block_type),
            ("harness", harness),
        ):
            if value is not None:
                arguments[name] = value
        # Keep compatibility with callers that used the old HTTP filter names;
        # rmcp/serde ignores unknown optional fields while native recall still
        # receives the same query and bounded tuning values.
        arguments.update({name: value for name, value in extra.items() if value is not None})
        return self.call_tool("recall", arguments)

    def get(self, session_id: str, *, from_: int | None = None, to: int | None = None, **extra) -> str:
        if from_ is None and "from" in extra:
            from_ = extra.pop("from")
        arguments = {"session_id": str(session_id)}
        if from_ is not None:
            arguments["from"] = from_
        if to is not None:
            arguments["to"] = to
        arguments.update({name: value for name, value in extra.items() if value is not None})
        return self.call_tool("get", arguments)


MCP_WORKER = None
_MCP_WORKER_CONFIG = None
_MCP_WORKER_LOCK = threading.Lock()


def native_worker() -> NativeMcpWorker:
    """Return the process singleton, rebuilding it only when runtime config changes."""
    global MCP_WORKER, _MCP_WORKER_CONFIG
    # Tests and embedders may supply a small fake directly; do not replace it.
    if MCP_WORKER is not None and not isinstance(MCP_WORKER, NativeMcpWorker):
        return MCP_WORKER
    config = (FUNES_BIN, REMOTE, str(HOME), MCP_TIMEOUT, MCP_HANDSHAKE_TIMEOUT)
    with _MCP_WORKER_LOCK:
        # During the initial background warm there is intentionally no active
        # worker yet.  Do not race it by spawning a second native MCP child;
        # callers receive a bounded retryable error instead.
        with _WARM_STATE_LOCK:
            warming = _WARM_STATE.get("state") == "warming"
        if MCP_WORKER is None and warming:
            raise NativeMcpError("native MCP warming")
        if isinstance(MCP_WORKER, NativeMcpWorker) and _MCP_WORKER_CONFIG == config:
            return MCP_WORKER
        if isinstance(MCP_WORKER, NativeMcpWorker):
            MCP_WORKER.close()
        MCP_WORKER = NativeMcpWorker(
            FUNES_BIN,
            REMOTE,
            HOME,
            timeout=MCP_TIMEOUT,
            handshake_timeout=MCP_HANDSHAKE_TIMEOUT,
        )
        _MCP_WORKER_CONFIG = config
        return MCP_WORKER


def close_native_worker() -> None:
    global MCP_WORKER, _MCP_WORKER_CONFIG
    with _MCP_WORKER_LOCK:
        if isinstance(MCP_WORKER, NativeMcpWorker):
            MCP_WORKER.close()
        MCP_WORKER = None
        _MCP_WORKER_CONFIG = None


def _refresh_native_worker() -> None:
    """Warm a replacement child and atomically swap it with the active one."""
    global MCP_WORKER, _MCP_WORKER_CONFIG
    config = (FUNES_BIN, REMOTE, str(HOME), MCP_TIMEOUT, MCP_HANDSHAKE_TIMEOUT)
    candidate = NativeMcpWorker(
        FUNES_BIN,
        REMOTE,
        HOME,
        timeout=MCP_TIMEOUT,
        handshake_timeout=MCP_HANDSHAKE_TIMEOUT,
    )
    try:
        candidate.recall("memory", k=1, candidates=1, half_life=0, neighbors=0)
    except Exception:
        candidate.close()
        raise
    with INDEX_LOCK:
        with _MCP_WORKER_LOCK:
            previous = MCP_WORKER
            MCP_WORKER = candidate
            _MCP_WORKER_CONFIG = config
        if isinstance(previous, NativeMcpWorker):
            previous.close()


atexit.register(close_native_worker)
atexit.register(close_source_app)


def recall(query: str, **kwargs) -> str:
    if not INDEX_LOCK.acquire(timeout=RECALL_LOCK_TIMEOUT):
        raise NativeMcpBusyError("native MCP busy")
    try:
        return native_worker().recall(query, **kwargs)
    finally:
        INDEX_LOCK.release()


def get(session_id: str, **kwargs) -> str:
    if not INDEX_LOCK.acquire(timeout=RECALL_LOCK_TIMEOUT):
        raise NativeMcpBusyError("native MCP busy")
    try:
        return native_worker().get(session_id, **kwargs)
    finally:
        INDEX_LOCK.release()


def auth_ok(handler: BaseHTTPRequestHandler) -> bool:
    supplied = handler.headers.get("X-Funes-Authorization", "") or handler.headers.get("X-Funes-Token", "")
    if supplied:
        return bool(TOKEN) and supplied == "Bearer " + TOKEN
    # Public Spaces and local tests can use the normal Authorization header.
    # Private Spaces reserve that header for the Hub token and use the explicit
    # application header above.
    return bool(TOKEN) and handler.headers.get("Authorization", "") == "Bearer " + TOKEN


def ready_payload() -> tuple[int, dict[str, object]]:
    """Return the authenticated readiness/status payload shared by both APIs."""
    if not REMOTE:
        return 503, {"ok": False, "error": "FUNES_MEMORY is not configured", "native_warm": warm_state()}
    code, out, err = run("status", REMOTE, timeout=30)
    sources = source_state()
    source_ok = not sources.get("configured") or bool(sources.get("ready"))
    return (
        200 if code == 0 and source_ok else 503,
        {
            "ok": code == 0 and source_ok,
            "remote": REMOTE,
            "status": out[-2000:],
            "error": err[-500:],
            "native_warm": warm_state(),
            "source_store": sources,
        },
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "funes-http/1"

    def log_message(self, fmt: str, *args) -> None:
        # Never log request bodies, Authorization, or raw memory text.
        return

    def send_json(self, code: int, obj: object, headers: dict[str, str] | None = None) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length", "0"))
        max_bytes = int(os.getenv("FUNES_MAX_BODY_BYTES", "64000000"))
        if n > max_bytes:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"ok": True, "service": "funes"})
            return
        if self.path in ("/ready", "/sync/status"):
            if not auth_ok(self):
                self.send_json(401, {"error": "unauthorized"})
                return
            code, payload = ready_payload()
            self.send_json(code, payload)
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not auth_ok(self):
            self.send_json(401, {"error": "unauthorized"})
            return
        try:
            obj = self.body()
            if self.path == "/sync/status":
                code, payload = ready_payload()
                self.send_json(code, payload)
                return
            if self.path == "/sync":
                app = source_app()
                if app is not None:
                    if app.syncer.restoring or app.syncer.restore_failed:
                        error = "restore_in_progress" if app.syncer.restoring else "restore_failed"
                        self.send_json(503, {"ok": False, "durable": False, "error": error})
                        return
                    result = app.syncer.upload()
                    result["ok"] = bool(result.get("durable"))
                    self.send_json(200 if result["ok"] else 503, result)
                    return
                if not REMOTE:
                    self.send_json(503, {"ok": False, "durable": False, "error": "FUNES_MEMORY is not configured"})
                    return
                # Every successful /ingest already completes a native push.
                # This compatibility checkpoint therefore acknowledges that
                # durable state without rebuilding or pushing the index again.
                self.send_json(200, {"ok": True, "durable": True, "remote": REMOTE})
                return
            if self.path == "/warm":
                if not REMOTE:
                    self.send_json(503, {"ok": False, "error": "FUNES_MEMORY is not configured"})
                    return
                self.send_json(202, {"ok": True, "native_warm": request_warm(force=True)})
                return
            if self.path in ("/search", "/recall"):
                raw_query = str(obj.get("query", "")).strip()
                if not raw_query:
                    self.send_json(400, {"error": "query is required"})
                    return
                limit = min(int(obj.get("limit", obj.get("k", 8))), 50)
                filters = {
                    key: obj.get(key)
                    for key in (
                        "source_agent",
                        "source_type",
                        "project",
                        "repo",
                        "device_id",
                        "role",
                        "content_type",
                        "since",
                        "until",
                    )
                    if obj.get(key) is not None
                }
                source_query, source_hits = search_source_documents(raw_query, limit, filters)
                query = query_text(raw_query)
                # CJK queries use the ASCII shadow above.  Keep the native
                # search bounded so the CPU Space does not spend its entire
                # request window reranking broad generic terms.
                tuning = {}
                if LANGUAGE_MODE == "auto" and cjk_ratio(raw_query) >= TRANSLATION_THRESHOLD:
                    tuning = {
                        "candidates": max(6, min(20, limit * 3)),
                        "neighbors": 0,
                        "half_life": 0,
                    }
                for name in ("harness", "repo"):
                    if obj.get(name):
                        tuning[name] = str(obj[name])
                if obj.get("source_agent") and not obj.get("harness"):
                    tuning["harness"] = {
                        "claude_code": "claude",
                    }.get(str(obj["source_agent"]), str(obj["source_agent"]))
                # Native CLI output is intentionally human-readable.  Keep it
                # lossless while also exposing the list shape expected by MCP
                # clients; the service backend can later provide structured
                # per-chunk metadata without changing this contract.
                # Native Funes presently filters only harness/repo. For a
                # stricter metadata/date filter, returning an unfiltered native
                # block would violate the caller's contract; the authoritative
                # source store handles those filters instead.
                native_allowed = not any(
                    key in filters
                    for key in ("source_type", "project", "device_id", "role", "content_type", "since", "until")
                )
                try:
                    # The native CLI defaults to 30 fused candidates, recency
                    # weighting, and neighbor expansion. Those defaults are
                    # useful interactively but can exceed a CPU Space ingress
                    # deadline after a large remote snapshot is opened. Keep
                    # the HTTP surface bounded while allowing operators to
                    # raise the cap with FUNES_HTTP_MAX_CANDIDATES.
                    tuning.setdefault("candidates", min(HTTP_MAX_CANDIDATES, max(2, limit * 2)))
                    tuning.setdefault("neighbors", 0)
                    tuning.setdefault("half_life", 0)
                    out = recall(query, k=limit, **tuning) if native_allowed else ""
                except NativeMcpBusyError:
                    self.send_json(429, {"ok": False, "query": raw_query, "retrieval_query": query, "results": [], "results_text": "", "error": "native_mcp_busy", "retry_after": 3}, {"Retry-After": "3"})
                    return
                except NativeMcpError:
                    self.send_json(503, {"ok": False, "query": raw_query, "retrieval_query": query, "results": [], "results_text": "", "error": "native_mcp_unavailable"})
                    return
                results = list(source_hits)
                if out.strip():
                    results.append({"raw_text": out, "source_type": "session", "retrieval_backend": "native_funes"})
                source_text = "\n\n".join(str(item.get("raw_text", "")) for item in source_hits if item.get("raw_text"))
                results_text = "\n\n".join(part for part in (source_text, out) if part.strip())
                self.send_json(200, {"ok": True, "query": raw_query, "retrieval_query": source_query if source_query != raw_query else query, "results": results, "results_text": results_text, "error": ""})
                return
            if self.path == "/get":
                sid = str(obj.get("session_id", obj.get("id", ""))).strip()
                if not sid:
                    self.send_json(400, {"error": "session_id is required"})
                    return
                app = source_app()
                item = app.store.get(sid) if app is not None and not app.syncer.restoring and not app.syncer.restore_failed else None
                if item is not None:
                    if os.getenv("RETURN_RETRIEVAL_TEXT", "false").lower() not in {"1", "true", "yes", "on"}:
                        item.pop("retrieval_text", None)
                    self.send_json(200, {"ok": True, "result": item, "error": ""})
                    return
                try:
                    out = get(sid, from_=obj.get("from"), to=obj.get("to"))
                except NativeMcpBusyError:
                    self.send_json(429, {"ok": False, "result": "", "error": "native_mcp_busy", "retry_after": 3}, {"Retry-After": "3"})
                    return
                except NativeMcpError:
                    self.send_json(503, {"ok": False, "result": "", "error": "native_mcp_unavailable"})
                    return
                self.send_json(200, {"ok": True, "result": out, "error": ""})
                return
            if self.path == "/ingest":
                docs = obj.get("documents", obj.get("records", obj.get("items")))
                if docs is None:
                    docs = [obj]
                if not isinstance(docs, list) or not docs:
                    self.send_json(400, {"error": "documents must be a non-empty list"})
                    return
                if not all(isinstance(doc, dict) for doc in docs):
                    self.send_json(400, {"error": "each document must be an object"})
                    return
                source_result = ingest_source_documents(docs)
                if source_result is not None:
                    status, result, _canonical = source_result
                    self.send_json(status, result)
                    return
                if not REMOTE:
                    self.send_json(503, {"error": "FUNES_MEMORY is not configured", "durable": False})
                    return
                now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                source = Path(tempfile.mkdtemp(prefix="funes-ingest-", dir=HOME / "sources"))
                try:
                    harnesses = set()
                    session_ids = []
                    for index, doc in enumerate(docs):
                        if not isinstance(doc, dict):
                            self.send_json(400, {"error": "each document must be an object"})
                            return
                        raw = str(doc.get("raw_text", doc.get("text", "")))
                        if not raw:
                            self.send_json(400, {"error": "raw_text is required"})
                            return
                        sid = str(doc.get("source_identity") or doc.get("session_id") or hashlib.sha256(raw.encode()).hexdigest()[:32])
                        # Native Funes owns parsing, chunking, embedding, and the
                        # TruffleHog push gate.  Keep one synthetic transcript per
                        # source identity so retries remain idempotent.
                        agent = str(doc.get("source_agent", "codex")).lower()
                        harness = {"claude_code": "claude"}.get(agent, agent if agent in {"codex", "pi", "claude", "hermes"} else "codex")
                        harnesses.add(harness)
                        session_ids.append(sid)
                        metadata = {
                            k: doc[k]
                            for k in (
                                "source_identity",
                                "source_agent",
                                "source_type",
                                "device_id",
                                "project",
                                "repo",
                                "worktree",
                                "session_id",
                                "message_id",
                                "role",
                                "timestamp",
                                "source_path",
                                "content_hash",
                                "ingested_at",
                                "updated_at",
                                "source_missing",
                                "content_type",
                                "agent_type",
                                "parent_session_id",
                                "agent_id",
                                "translation_hash",
                                "translation_version",
                                "translation_status",
                            )
                            if doc.get(k) is not None
                        }
                        metadata.setdefault("source_agent", agent)
                        metadata.setdefault("ingested_at", now)
                        cwd = str(doc.get("worktree", doc.get("project", "remote")))
                        role = str(doc.get("role", "user"))
                        source_timestamp = str(doc.get("timestamp") or now)
                        if harness == "pi":
                            line = {"type": "session", "id": sid, "cwd": cwd, "timestamp": source_timestamp, "metadata": metadata}
                            msg = {"type": "message", "id": str(doc.get("message_id") or hashlib.sha256((sid + raw).encode()).hexdigest()[:24]), "timestamp": source_timestamp, "message": {"role": role, "content": [{"type": "text", "text": raw}]}}
                        elif harness == "claude":
                            line = {"type": role if role in {"user", "assistant"} else "user", "uuid": str(doc.get("message_id") or hashlib.sha256((sid + raw).encode()).hexdigest()[:24]), "timestamp": source_timestamp, "cwd": cwd, "metadata": metadata}
                            msg = {"type": line["type"], "uuid": line["uuid"], "timestamp": source_timestamp, "cwd": cwd, "message": {"role": role, "content": [{"type": "text", "text": raw}]}}
                        else:
                            line = {"type": "session_meta", "timestamp": source_timestamp, "payload": {"id": sid, "cwd": cwd, "metadata": metadata}}
                            msg = {"type": "response_item", "timestamp": source_timestamp, "payload": {"type": "message", "role": role, "content": [{"type": "input_text", "text": raw}]}}
                        # Keep each harness in its own directory.  A Pi/Claude
                        # parser must never rescan a Codex envelope from the
                        # same batch.
                        harness_dir = source / harness
                        harness_dir.mkdir(exist_ok=True)
                        (harness_dir / f"{index:08d}-{hashlib.sha256(sid.encode()).hexdigest()[:16]}.jsonl").write_text(json.dumps(line, ensure_ascii=False) + "\n" + json.dumps(msg, ensure_ascii=False) + "\n", encoding="utf-8")
                    outputs = []
                    errors = []
                    with WRITE_LOCK:
                        # The long-lived MCP process keeps its own model/index handles; leave it
                        # alive while Lance appends. Readers use consistent snapshots and can
                        # continue serving the previous remote head while this durable write runs.
                        for harness in sorted(harnesses):
                            code, out, err = run("index", str(source / harness), "--harness", harness, "--yes", timeout=INGEST_INDEX_TIMEOUT)
                            outputs.append(out)
                            errors.append(err)
                            if code != 0:
                                self.send_json(503, {"ok": False, "durable": False, "session_ids": session_ids, "error": "native_index_failed"})
                                return
                        # Keep the durable chunk push incremental.  Forcing a
                        # full remote reindex for every tiny memory-file batch
                        # repeatedly invalidates the read worker; native Funes
                        # can search newly-pushed deltas until its normal index
                        # threshold is reached.
                        code, pout, perr = run("push", REMOTE, "--yes", timeout=INGEST_PUSH_TIMEOUT)
                    outputs.append(pout)
                    errors.append(perr)
                    durable = code == 0
                    self.send_json(200 if durable else 503, {"ok": durable, "durable": durable, "accepted": len(docs) if durable else 0, "session_ids": session_ids, "output": "".join(outputs)[-3000:] if durable else "", "error": "" if durable else "native_push_failed"})
                finally:
                    shutil.rmtree(source, ignore_errors=True)
                return
            self.send_json(404, {"error": "not found"})
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            self.send_json(500, {"error": str(exc)})


def serve(host: str = "0.0.0.0", port: int = PORT) -> None:
    (HOME / "sources").mkdir(parents=True, exist_ok=True)
    source_app()
    request_warm()
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    serve()
