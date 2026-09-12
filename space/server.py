#!/usr/bin/env python3
"""Small authenticated HTTP bridge for the Funes CLI.

The durable source of truth is the configured HF Hub dataset (FUNES_MEMORY).  The
container's /data/.funes directory is only a warm cache and may be recreated.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


FUNES_BIN = os.getenv("FUNES_BIN", "/usr/local/bin/funes")
REMOTE = os.getenv("FUNES_MEMORY", "")
TOKEN = os.getenv("FUNES_API_TOKEN", "")
HOME = Path(os.getenv("FUNES_HOME", "/data/.funes"))
PORT = int(os.getenv("PORT", "7860"))
TRANSLATION_THRESHOLD = float(os.getenv("TRANSLATE_CHINESE_THRESHOLD", "0.15"))
INGEST_INDEX_TIMEOUT = int(os.getenv("FUNES_INGEST_INDEX_TIMEOUT", "900"))
INGEST_PUSH_TIMEOUT = int(os.getenv("FUNES_INGEST_PUSH_TIMEOUT", "1800"))
PROMPT_VERSION = "funes-retrieval-v1"
LANGUAGE_MODE = os.getenv("FUNES_RETRIEVAL_LANGUAGE_MODE", "auto").lower()
INDEX_LOCK = threading.Lock()

# The default Funes embedding model is English-oriented.  Keep this small,
# deterministic fallback for installations without a translation provider so
# Chinese queries do not enter the slow CJK tokenizer path.  Technical names
# and identifiers are always collected separately and remain verbatim.
CHINESE_RETRIEVAL_TERMS = (
    ("previous_response_id", "previous_response_id"),
    ("第二轮", "second turn"),
    ("第二次", "second turn"),
    ("上下文丢失", "context loss"),
    ("上下文丢了", "context loss"),
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
        req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=payload, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
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
    shadow = " ".join(x for x in (translated_ascii, " ".join(fallback), " ".join(entities)) if x).strip()
    return shadow or "memory context retrieval"


def run(*args: str, timeout: int = 180) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["FUNES_HOME"] = str(HOME)
    p = subprocess.run([FUNES_BIN, *args], text=True, capture_output=True, timeout=timeout, env=env)
    return p.returncode, p.stdout, p.stderr


def auth_ok(handler: BaseHTTPRequestHandler) -> bool:
    supplied = handler.headers.get("X-Funes-Authorization", "") or handler.headers.get("X-Funes-Token", "")
    if supplied:
        return bool(TOKEN) and supplied == "Bearer " + TOKEN
    # Public Spaces and local tests can use the normal Authorization header.
    # Private Spaces reserve that header for the Hub token and use the explicit
    # application header above.
    return bool(TOKEN) and handler.headers.get("Authorization", "") == "Bearer " + TOKEN


class Handler(BaseHTTPRequestHandler):
    server_version = "funes-http/1"

    def log_message(self, fmt: str, *args) -> None:
        # Never log request bodies, Authorization, or raw memory text.
        return

    def send_json(self, code: int, obj: object) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
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
        if self.path == "/ready":
            if not auth_ok(self):
                self.send_json(401, {"error": "unauthorized"})
                return
            if not REMOTE:
                self.send_json(503, {"ok": False, "error": "FUNES_MEMORY is not configured"})
                return
            code, out, err = run("status", REMOTE, timeout=30)
            self.send_json(200 if code == 0 else 503, {"ok": code == 0, "remote": REMOTE, "status": out[-2000:], "error": err[-500:]})
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not auth_ok(self):
            self.send_json(401, {"error": "unauthorized"})
            return
        try:
            obj = self.body()
            if self.path in ("/search", "/recall"):
                query = query_text(str(obj.get("query", "")).strip())
                if not query:
                    self.send_json(400, {"error": "query is required"})
                    return
                args = ["recall", query, "--k", str(min(int(obj.get("limit", obj.get("k", 8))), 50))]
                if REMOTE:
                    args += ["--memory", REMOTE]
                for name in ("harness", "repo"):
                    if obj.get(name):
                        args += ["--" + name, str(obj[name])]
                code, out, err = run(*args)
                # Native CLI output is intentionally human-readable.  Keep it
                # lossless while also exposing the list shape expected by MCP
                # clients; the service backend can later provide structured
                # per-chunk metadata without changing this contract.
                results = ([{"raw_text": out}] if code == 0 and out.strip() else [])
                self.send_json(200 if code == 0 else 503, {"ok": code == 0, "query": query, "results": results, "results_text": out, "error": err[-1000:]})
                return
            if self.path == "/get":
                sid = str(obj.get("session_id", obj.get("id", ""))).strip()
                if not sid:
                    self.send_json(400, {"error": "session_id is required"})
                    return
                args = ["get", sid]
                if obj.get("from") is not None:
                    args += ["--from", str(obj["from"])]
                if obj.get("to") is not None:
                    args += ["--to", str(obj["to"])]
                if REMOTE:
                    args += ["--memory", REMOTE]
                code, out, err = run(*args)
                self.send_json(200 if code == 0 else 503, {"ok": code == 0, "result": out, "error": err[-1000:]})
                return
            if self.path == "/ingest":
                docs = obj.get("documents", obj.get("records", obj.get("items")))
                if docs is None:
                    docs = [obj]
                if not isinstance(docs, list) or not docs:
                    self.send_json(400, {"error": "documents must be a non-empty list"})
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
                        metadata = {k: doc[k] for k in ("source_identity", "source_type", "project", "repo", "worktree", "message_id", "content_type") if doc.get(k) is not None}
                        cwd = str(doc.get("worktree", doc.get("project", "remote")))
                        role = str(doc.get("role", "user"))
                        if harness == "pi":
                            line = {"type": "session", "id": sid, "cwd": cwd, "timestamp": now, "metadata": metadata}
                            msg = {"type": "message", "id": str(doc.get("message_id") or hashlib.sha256((sid + raw).encode()).hexdigest()[:24]), "timestamp": now, "message": {"role": role, "content": [{"type": "text", "text": raw}]}}
                        elif harness == "claude":
                            line = {"type": role if role in {"user", "assistant"} else "user", "uuid": str(doc.get("message_id") or hashlib.sha256((sid + raw).encode()).hexdigest()[:24]), "timestamp": now, "cwd": cwd, "metadata": metadata}
                            msg = {"type": line["type"], "uuid": line["uuid"], "timestamp": now, "cwd": cwd, "message": {"role": role, "content": [{"type": "text", "text": raw}]}}
                        else:
                            line = {"type": "session_meta", "timestamp": now, "payload": {"id": sid, "cwd": cwd, "metadata": metadata}}
                            msg = {"type": "response_item", "timestamp": now, "payload": {"type": "message", "role": role, "content": [{"type": "input_text", "text": raw}]}}
                        # Keep each harness in its own directory.  A Pi/Claude
                        # parser must never rescan a Codex envelope from the
                        # same batch.
                        harness_dir = source / harness
                        harness_dir.mkdir(exist_ok=True)
                        (harness_dir / f"{index:08d}-{hashlib.sha256(sid.encode()).hexdigest()[:16]}.jsonl").write_text(json.dumps(line, ensure_ascii=False) + "\n" + json.dumps(msg, ensure_ascii=False) + "\n", encoding="utf-8")
                    outputs = []
                    errors = []
                    with INDEX_LOCK:
                        for harness in sorted(harnesses):
                            code, out, err = run("index", str(source / harness), "--harness", harness, "--yes", timeout=INGEST_INDEX_TIMEOUT)
                            outputs.append(out)
                            errors.append(err)
                            if code != 0:
                                self.send_json(503, {"ok": False, "durable": False, "session_ids": session_ids, "error": "native_index_failed"})
                                return
                        code, pout, perr = run("push", REMOTE, "--yes", "--force-reindex", timeout=INGEST_PUSH_TIMEOUT)
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
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    serve()
