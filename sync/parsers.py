from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
from .discovery import Source

@dataclass
class Chunk:
    record_id: str
    source_key: str
    kind: str
    path: str
    session_id: str
    ordinal: int
    role: str
    text: str
    raw_text: str
    timestamp: str = ""
    metadata: dict[str,Any] | None = None
    subagent_id: str = ""
    parent_session_id: str = ""
    message_id: str = ""
    source_agent: str = ""
    source_type: str = ""
    content_type: str = "text/plain"
    project: str = ""
    repo: str = ""
    worktree: str = ""
    agent_type: str = ""
    device_id: str = ""
    def as_dict(self):
        value = asdict(self)
        # The HTTP service consumes raw_text and source_identity, while the local
        # database retains the compact record_id too.
        value.pop("text", None)
        value["source_identity"] = value["record_id"]
        value["source_version"] = value["content_hash"] = hashlib.sha256(self.raw_text.encode()).hexdigest()
        value["source_path"] = value["path"]
        value["source_agent"] = self.source_agent or self.kind
        value["source_type"] = self.source_type or self.kind
        value["agent_id"] = self.subagent_id
        return value

def _text(v: Any) -> str:
    if isinstance(v,str): return v
    if isinstance(v,list): return "\n".join(_text(x) for x in v)
    if isinstance(v,dict):
        for k in ("text","content","message","value","prompt","output"):
            if k in v:
                x=_text(v[k])
                if x: return x
    return "" if v is None else str(v)

def _find(obj: dict, keys: tuple[str,...], default=""):
    for k in keys:
        if k in obj and obj[k] not in (None,""): return obj[k]
    return default

def _record(source: Source, n: int, obj: Any, raw: str, session="") -> Chunk|None:
    if isinstance(obj,dict):
        role=str(_find(obj,("role","author_role","sender","type"),""))
        msg=obj.get("message", obj)
        text=_text(_find(obj,("text","content","prompt","output"),"")) or _text(msg)
        # envelope payloads are common in Codex rollout files
        if isinstance(obj.get("payload"),dict):
            p=obj["payload"]; text=text or _text(p.get("message") or p.get("content") or p.get("text")); role=role or str(p.get("role", "")); msg=p
        sid=str(_find(obj,("session_id","sessionId","conversation_id","thread_id","id"),session))
        ts=str(_find(obj,("timestamp","created_at","createdAt","time"),""))
        sub=str(_find(obj,("subagent_id","subagentId","agent_id","agentId"),""))
        parent=str(_find(obj,("parent_session_id","parentSessionId","parent_id"),""))
        message_id=str(_find(obj,("message_id","messageId","event_id","uuid"),""))
        project=str(_find(obj,("project","project_name"),source.project))
        repo=str(_find(obj,("repo","repository"),"")); worktree=str(_find(obj,("worktree","workspace"),""))
        agent_type=str(_find(obj,("agent_type","agentType","harness"),source.kind))
        meta={k:v for k,v in obj.items() if k not in ("text","content","message","payload")}
    else:
        role=""; text=_text(obj); sid=session; ts=""; sub=""; parent=""; message_id=""; project=source.project; repo=""; worktree=""; agent_type=source.kind; meta={}
    if not text and not raw.strip(): return None
    # Identity is based on location, not content, so edits update one remote
    # document instead of creating a stale duplicate.
    locator=f"{source.source_key}:{sid}:{message_id or n}"
    rid=hashlib.sha256(locator.encode()).hexdigest()
    return Chunk(rid,source.source_key,source.kind,str(source.path),sid,n,role,text,raw,ts,meta,sub,parent,message_id,source.kind,source.kind,"text/plain",project,repo,worktree,agent_type)

def _parse_jsonl(source: Source, lines: Iterable[str], start=0) -> list[Chunk]:
    out=[]; session=""
    for n,raw in enumerate(lines,start):
        raw=raw.rstrip("\n")
        if not raw.strip(): continue
        try: obj=json.loads(raw)
        except Exception: obj=raw
        c=_record(source,n,obj,raw,session)
        if c:
            session=session or c.session_id
            out.append(c)
    return out

def parse_file(source: Source, start=0) -> list[Chunk]:
    p=Path(source.path)
    try:
        data=p.read_bytes()
    except OSError: return []
    if start and start < len(data):
        # Cursor is a byte offset. Discard the possibly partial UTF-8/JSONL line
        # before decoding, then parse only complete appended lines.
        tail=data[start:]
        if start and not data[:start].endswith(b"\n"):
            cut=tail.find(b"\n")
            tail=tail[cut+1:] if cut >= 0 else b""
        data=tail
    raw=data.decode("utf-8",errors="replace")
    if p.suffix.lower() in (".jsonl", ".ndjson") or source.kind in ("codex","pi","claude") and "\n" in raw:
        # offset is only used as an append cursor; ordinals remain stable enough for new records.
        return _parse_jsonl(source,raw.splitlines(),start)
    if p.suffix.lower()==".json":
        try:
            obj=json.loads(raw)
            vals=obj if isinstance(obj,list) else [obj]
            return [c for i,v in enumerate(vals,start) if (c:=_record(source,i,v,json.dumps(v,ensure_ascii=False),""))]
        except Exception: pass
    # Markdown/plain persistent files are represented as one stable raw chunk.
    c=_record(source,0,raw,raw,"")
    return [c] if c else []

# Explicit adapters are useful to callers that already know the harness; they all
# share the envelope parser so chunk shape stays identical.
def parse_codex(source: Source, start=0): return parse_file(source, start)
def parse_pi(source: Source, start=0): return parse_file(source, start)
def parse_claude(source: Source, start=0): return parse_file(source, start)
def parse_generic(source: Source, start=0): return parse_file(source, start)
def iter_chunks(source: Source, start=0):
    yield from parse_file(source, start)


# The compact helpers above remain as a permissive fallback for third-party JSONL
# exports.  Native agent files need a little more structure: their envelope lines
# carry the session id and role outside the human text.  These adapters keep that
# metadata stable while still retaining the exact source line in `metadata.raw_record`.
def _native_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(x for part in value if (x := _native_text(part)))
    if isinstance(value, dict):
        typ = str(value.get("type", ""))
        if typ in {"input_text", "output_text", "text"}:
            return str(value.get("text", ""))
        if typ in {"tool_use", "toolCall", "function_call", "custom_tool_call"}:
            body = value.get("input", value.get("arguments", ""))
            return body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, sort_keys=True)
        if typ in {"tool_result", "toolResult", "function_call_output", "custom_tool_call_output"}:
            return _native_text(value.get("content", value.get("output", "")))
        for key in ("content", "message", "text", "summary", "output", "data"):
            if key in value:
                text = _native_text(value[key])
                if text:
                    return text
    return "" if value is None else str(value)


def _stable_identity(source: Source, session: str, message: str, ordinal: int) -> str:
    # Device id is deliberately absent: the same session converges when copied to
    # another Mac.  A path is only the final fallback for records without a native id.
    agent = source.source_agent
    kind = source.source_type
    locator = f"{agent}|{kind}|{session}|{message or ordinal}"
    return hashlib.sha256(locator.encode("utf-8")).hexdigest()


def _chunk(source: Source, *, ordinal: int, session: str, message: str, role: str,
           text: str, raw: str, timestamp: str = "", content_type: str = "assistant_message",
           metadata: dict[str, Any] | None = None, parent_session: str = "", agent_id: str = "", worktree: str = "") -> Chunk | None:
    if not text.strip():
        return None
    meta = dict(metadata or {})
    # The exact human/tool text is already `raw_text`.  Retain only a fingerprint
    # of the JSON envelope so large tool records are not duplicated in every
    # split chunk and the local pending queue stays bounded.
    meta.setdefault("raw_record_hash", hashlib.sha256(raw.encode("utf-8")).hexdigest())
    identity = _stable_identity(source, session, message, ordinal)
    return Chunk(
        record_id=identity,
        source_key=source.source_key,
        kind=source.kind,
        path=str(source.path),
        session_id=session,
        ordinal=ordinal,
        role=role,
        text=text,
        raw_text=text,
        timestamp=timestamp,
        metadata=meta,
        subagent_id=agent_id,
        parent_session_id=parent_session,
        message_id=message,
        source_agent=source.source_agent,
        source_type=source.source_type,
        content_type=content_type,
        project=source.project,
        worktree=worktree,
        agent_type="subagent" if agent_id or "subagent" in str(source.path) or "agent-" in str(source.path) else "main",
        device_id=source.device_id,
    )


def _read_lines(path: Path, start: int = 0) -> list[str]:
    data = path.read_bytes()
    if start and start < len(data):
        tail = data[start:]
        # A cursor can land in the middle of a UTF-8/JSONL record.  Drop that
        # partial line; the next reconciliation will see it again if needed.
        if not data[:start].endswith(b"\n"):
            cut = tail.find(b"\n")
            tail = tail[cut + 1:] if cut >= 0 else b""
        data = tail
    return data.decode("utf-8", errors="replace").splitlines()


def _parse_codex_native(source: Source, start: int = 0) -> list[Chunk]:
    # Read the small prefix for session_meta even during an append-only pass.
    path = Path(source.path)
    all_lines = _read_lines(path, 0 if not start else 0)
    session = path.stem
    parent = ""
    worktree = ""
    for line in all_lines[:256]:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("type") == "session_meta":
            payload = obj.get("payload") or {}
            session = str(payload.get("id") or payload.get("session_id") or session)
            parent = str(payload.get("parent_thread_id") or "")
            worktree = str(payload.get("cwd") or "")
            break
    lines = _read_lines(path, start) if start else all_lines
    # `start` is a byte cursor, not a line number; use the source offset only as a
    # deterministic fallback ordinal.  Native message ids make identities stable.
    out: list[Chunk] = []
    for ordinal, line in enumerate(lines, start if start else 0):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("type") != "response_item":
            continue
        payload = obj.get("payload") or {}
        typ = str(payload.get("type") or "")
        role = str(payload.get("role") or ("tool" if typ.endswith("_output") else "assistant"))
        if typ == "message":
            text = _native_text(payload.get("content", ""))
            ctype = "user_message" if role == "user" else "assistant_message"
        elif typ == "reasoning":
            text = _native_text(payload.get("summary", ""))
            ctype = "summary"
        elif typ in {"function_call", "custom_tool_call"}:
            text = _native_text(payload.get("arguments", payload.get("input", "")))
            ctype = "tool_call"
        elif typ in {"function_call_output", "custom_tool_call_output"}:
            text = _native_text(payload.get("output", ""))
            ctype = "tool_result"
            role = "tool"
        else:
            continue
        message = str(payload.get("id") or payload.get("call_id") or f"{session}:{ordinal}")
        if typ in {"function_call", "custom_tool_call", "function_call_output", "custom_tool_call_output"}:
            # A call and its output share `call_id`, but are distinct retrievable
            # records; include the direction in the stable identity.
            message = f"{message}:{'out' if typ.endswith('_output') else 'call'}"
        agent_id = str(payload.get("agent_id") or payload.get("agentId") or "")
        chunk = _chunk(source, ordinal=ordinal, session=session, message=message, role=role,
                       text=text, raw=line, timestamp=str(obj.get("timestamp") or ""),
                       content_type=ctype, parent_session=parent, agent_id=agent_id,
                       metadata={"record_type": typ, "tool_name": payload.get("name")}, worktree=worktree)
        if chunk:
            out.append(chunk)
    return out


def _parse_claude_native(source: Source, start: int = 0) -> list[Chunk]:
    out: list[Chunk] = []
    for ordinal, line in enumerate(_read_lines(Path(source.path), start), start if start else 0):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        typ = str(obj.get("type") or "")
        if typ not in {"user", "assistant", "system", "summary"}:
            continue
        msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
        role = str(msg.get("role") or typ)
        text = _native_text(msg.get("content", obj.get("summary", "")))
        ctype = {"user": "user_message", "assistant": "assistant_message", "system": "system", "summary": "summary"}.get(typ, typ)
        sid = str(obj.get("sessionId") or obj.get("session_id") or Path(source.path).stem)
        mid = str(obj.get("uuid") or obj.get("id") or f"{sid}:{ordinal}")
        chunk = _chunk(source, ordinal=ordinal, session=sid, message=mid, role=role, text=text,
                       raw=line, timestamp=str(obj.get("timestamp") or ""), content_type=ctype,
                       parent_session=str(obj.get("parentUuid") or ""),
                       agent_id=str(obj.get("agentId") or ""),
                       metadata={"is_sidechain": bool(obj.get("isSidechain")), "git_branch": obj.get("gitBranch")},
                       worktree=str(obj.get("cwd") or ""))
        if chunk:
            out.append(chunk)
    return out


def _parse_pi_native(source: Source, start: int = 0) -> list[Chunk]:
    out: list[Chunk] = []
    session = Path(source.path).stem
    worktree = ""
    for ordinal, line in enumerate(_read_lines(Path(source.path), start), start if start else 0):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("type") == "session":
            session = str(obj.get("id") or obj.get("sessionId") or session)
            worktree = str(obj.get("cwd") or "")
            continue
        if obj.get("type") != "message":
            continue
        msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
        role = str(msg.get("role") or "")
        text = _native_text(msg.get("content", msg.get("text", "")))
        ctype = {"user": "user_message", "assistant": "assistant_message", "toolResult": "tool_result"}.get(role, role or "message")
        if role == "toolResult":
            role = "tool"
        mid = str(obj.get("id") or msg.get("id") or f"{session}:{ordinal}")
        chunk = _chunk(source, ordinal=ordinal, session=session, message=mid, role=role, text=text,
                       raw=line, timestamp=str(obj.get("timestamp") or ""), content_type=ctype,
                       parent_session=str(obj.get("parentId") or ""),
                       metadata={"tool_name": msg.get("toolName")}, worktree=worktree)
        if chunk:
            out.append(chunk)
    return out


def _parse_memory_native(source: Source) -> list[Chunk]:
    raw = Path(source.path).read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return []
    # Keep headings/paragraphs together and cap individual requests so a large
    # memory file never becomes one unbounded translation call.
    pieces = [p.strip() for p in re.split(r"\n(?=\s*#{1,6}\s)", raw) if p.strip()]
    if not pieces:
        pieces = [raw]
    out = []
    for i, piece in enumerate(pieces):
        if len(piece) > 6000:
            pieces[i:i + 1] = [piece[j:j + 6000] for j in range(0, len(piece), 6000)]
    for i, piece in enumerate(pieces):
        c = _chunk(source, ordinal=i, session="", message=f"{source.source_key}:{i}", role="system",
                   text=piece, raw=piece, content_type="agents_md" if source.kind == "agents_md" else "memory",
                   metadata={"source_file": str(source.path)})
        if c:
            out.append(c)
    return out


def _split_large(chunks: list[Chunk], max_chars: int = 6000) -> list[Chunk]:
    """Bound transport/translation size without losing turn identity."""
    out: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.raw_text) <= max_chars:
            out.append(chunk)
            continue
        pieces = [p for p in re.split(r"\n\s*\n", chunk.raw_text) if p]
        if not pieces:
            pieces = [chunk.raw_text]
        bounded: list[str] = []
        for piece in pieces:
            bounded.extend(piece[i:i + max_chars] for i in range(0, len(piece), max_chars))
        for index, piece in enumerate(bounded):
            child = Chunk(**{**chunk.__dict__,
                             "record_id": hashlib.sha256(f"{chunk.record_id}:part:{index}".encode()).hexdigest(),
                             "ordinal": chunk.ordinal * 100000 + index,
                             "message_id": f"{chunk.message_id}:part:{index}",
                             "text": piece,
                             "raw_text": piece,
                             "metadata": {**(chunk.metadata or {}), "chunk_index": index, "chunk_count": len(bounded)}})
            out.append(child)
    return out


def parse_file(source: Source, start: int = 0) -> list[Chunk]:
    """Parse one discovered source with the native adapter selected by its kind."""
    try:
        if source.kind in {"codex", "codex_session"}:
            native = _parse_codex_native(source, start)
            return _split_large(native or _parse_jsonl(source, _read_lines(Path(source.path), start), start))
        if source.kind in {"claude", "claude_session"}:
            native = _parse_claude_native(source, start)
            return _split_large(native or _parse_jsonl(source, _read_lines(Path(source.path), start), start))
        if source.kind in {"pi", "pi_session"}:
            native = _parse_pi_native(source, start)
            return _split_large(native or _parse_jsonl(source, _read_lines(Path(source.path), start), start))
        if source.kind in {"codex_memory", "pi_memory", "claude_memory", "agents_md", "persistent"}:
            return _split_large(_parse_memory_native(source))
        return _split_large(_parse_jsonl(source, _read_lines(Path(source.path), start), start))
    except OSError:
        return []


def parse_codex(source: Source, start=0): return _parse_codex_native(source, start)
def parse_pi(source: Source, start=0): return _parse_pi_native(source, start)
def parse_claude(source: Source, start=0): return _parse_claude_native(source, start)
