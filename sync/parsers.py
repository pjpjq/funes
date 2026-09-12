from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit
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


def _stable_digest(*parts: Any) -> str:
    """Hash canonical scalar/list values without depending on local paths."""
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _payload(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    value = obj.get("payload")
    return value if isinstance(value, dict) else None


def _value(obj: Any, keys: tuple[str, ...], default: Any = "") -> Any:
    """Read an envelope field before its payload, while tolerating both shapes."""
    if not isinstance(obj, dict):
        return default
    payload = _payload(obj)
    for mapping in (obj, payload):
        if not mapping:
            continue
        for key in keys:
            value = mapping.get(key)
            if value not in (None, ""):
                return value
    return default


def _session_value(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    value = _value(obj, ("session_id", "sessionId", "conversation_id", "conversationId", "thread_id", "threadId"))
    if value not in (None, ""):
        return str(value)
    payload = _payload(obj)
    typ = str((payload or {}).get("type") or obj.get("type") or "")
    # Session envelopes are the one place where a bare `id` is a session id.
    if typ in {"session", "session_meta", "sessionMetadata"}:
        value = _value(obj, ("id",))
        if value not in (None, ""):
            return str(value)
    return ""


def _native_value(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    payload = _payload(obj)
    typ = str((payload or {}).get("type") or obj.get("type") or "")
    for mapping in (payload, obj):
        if not mapping:
            continue
        for key in ("message_id", "messageId", "event_id", "eventId", "uuid", "id", "call_id", "callId"):
            value = mapping.get(key)
            if value in (None, ""):
                continue
            # Do not mistake a session envelope id for a message identity.
            if key == "id" and typ in {"session", "session_meta", "sessionMetadata"}:
                continue
            return str(value)
    message = obj.get("message")
    if isinstance(message, dict):
        return _native_value(message)
    return ""


def _timestamp_value(obj: Any) -> str:
    value = _value(obj, ("timestamp", "created_at", "createdAt", "time"))
    if value in (None, "") and isinstance(obj, dict) and isinstance(obj.get("message"), dict):
        value = _value(obj["message"], ("timestamp", "created_at", "createdAt", "time"))
    return "" if value in (None, "") else str(value)


def _record_type_value(obj: Any, default: str = "record") -> str:
    if not isinstance(obj, dict):
        return default
    payload = _payload(obj)
    value = (payload or {}).get("record_type") or (payload or {}).get("type")
    value = value or obj.get("record_type") or obj.get("type")
    if not value and isinstance(obj.get("message"), dict):
        value = "message"
    return str(value or default)


def _role_value(obj: Any, default: str = "") -> str:
    value = _value(obj, ("role", "author_role", "sender"))
    if value in (None, "") and isinstance(obj, dict) and isinstance(obj.get("message"), dict):
        value = _value(obj["message"], ("role", "author_role", "sender"))
    if value not in (None, ""):
        return str(value)
    typ = _record_type_value(obj)
    if typ in {"function_call", "custom_tool_call", "reasoning"}:
        return "assistant"
    if typ in {"function_call_output", "custom_tool_call_output", "tool_result", "toolResult"}:
        return "tool"
    return default


def _object_text(obj: Any) -> str:
    if not isinstance(obj, dict):
        return _native_text(obj)
    payload = _payload(obj)
    for mapping in (payload, obj):
        if not mapping:
            continue
        for key in ("text", "content", "prompt", "output", "message", "summary", "arguments", "input", "data", "value"):
            if key in mapping:
                text = _native_text(mapping[key])
                if text:
                    return text
    # Do not stringify the whole envelope here: it may contain a device-local
    # cwd/path and would poison the implicit session anchor.
    return ""


def _session_fallback(lines: Iterable[str], source: Source) -> str:
    """Derive an implicit session from stable first-record fields.

    Native session envelopes win.  When an export omits one, the first record's
    semantic fields provide a path/device-independent anchor for append scans.
    """
    parsed: list[dict[str, Any]] = []
    for raw in lines:
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        sid = _session_value(obj)
        if sid:
            return sid
        typ = _record_type_value(obj)
        # A session envelope may contain a device-local cwd but no usable id;
        # never anchor cross-device identity to that path-bearing metadata.
        if typ in {"session", "session_meta", "sessionMetadata"}:
            continue
        parsed.append({
            "timestamp": _timestamp_value(obj),
            "role": _role_value(obj),
            "record_type": typ,
            "text": _object_text(obj),
            "native_id": _native_value(obj),
        })
    for item in parsed:
        if item["native_id"]:
            return "implicit:" + _stable_digest(
                source.source_agent, source.source_type, "native", item["native_id"]
            )[:32]
        text = item["text"]
        if text or item["record_type"]:
            return "implicit:" + _stable_digest(
                source.source_agent, source.source_type, item["timestamp"], item["role"],
                item["record_type"], hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )[:32]
    return "implicit:" + _stable_digest(source.source_agent, source.source_type)[:32]


def _fallback_identity_key(session: str, timestamp: str, role: str, record_type: str,
                           text: str, extra: Any = "") -> str:
    return _stable_digest(
        session,
        timestamp,
        role,
        record_type,
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        extra,
    )


def _git_remote_identity(root: Path) -> str:
    """Return a path-independent origin identity without retaining credentials."""
    try:
        git_dir = root / ".git"
        if git_dir.is_file():
            pointer = git_dir.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"gitdir:\s*(.+)", pointer, re.IGNORECASE)
            if match:
                git_dir = (root / match.group(1).strip()).resolve()
        config = (git_dir / "config").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    block = re.search(r'\[remote\s+"origin"\](.*?)(?=\n\[|\Z)', config, re.DOTALL)
    if not block:
        return ""
    match = re.search(r"^\s*url\s*=\s*(\S+)", block.group(1), re.MULTILINE)
    if not match:
        return ""
    value = match.group(1).strip()
    if "://" in value:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower()
        path = parsed.path.strip("/")
    elif ":" in value:
        host, path = value.split(":", 1)
        host = host.rsplit("@", 1)[-1].lower()
        path = path.strip("/")
    else:
        host, path = "", value.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return "/".join(part for part in (host, path) if part)


def _memory_identity_context(source: Source) -> tuple[str, str]:
    """Build a repo/path identity that survives different checkout prefixes."""
    path = Path(source.path).expanduser()
    project_text = str(source.project or "").strip()
    project = Path(project_text).expanduser() if project_text and "://" not in project_text else None
    roots: list[Path] = []
    if project:
        roots.append(project)
    roots.extend([path.parent, *path.parents])
    repo_root: Path | None = None
    remote = ""
    for root in roots:
        candidate = root if (root / ".git").exists() else None
        if candidate is not None:
            repo_root = root
            remote = _git_remote_identity(root)
            break
    if remote:
        repo_identity = "remote:" + remote
    elif project and project.name:
        repo_identity = "repo:" + project.name
    elif repo_root and repo_root.name:
        repo_identity = "repo:" + repo_root.name
    else:
        repo_identity = "global:" + source.kind
    base = project or repo_root
    relative = ""
    if base:
        try:
            relative = str(path.resolve().relative_to(base.resolve()))
        except ValueError:
            relative = ""
    if not relative and project and project.name:
        # Test fixtures and exported source manifests may carry a logical
        # source_key whose absolute checkout prefix is not present locally.
        key_path = source.source_key.split(":", 1)[-1].replace("\\", "/")
        marker = "/" + project.name.strip("/") + "/"
        if marker in key_path:
            relative = key_path.rsplit(marker, 1)[-1]
    relative = relative or path.name
    return repo_identity, relative.replace("\\", "/")


def _record(source: Source, n: int, obj: Any, raw: str, session="") -> Chunk|None:
    if isinstance(obj,dict):
        role=_role_value(obj)
        msg=obj.get("message", obj)
        text=_object_text(obj) or _text(msg)
        # envelope payloads are common in Codex rollout files
        if isinstance(obj.get("payload"),dict):
            p=obj["payload"]; text=text or _text(p.get("message") or p.get("content") or p.get("text")); role=role or str(p.get("role", "")); msg=p
        sid=_session_value(obj) or session
        ts=_timestamp_value(obj)
        sub=str(_find(obj,("subagent_id","subagentId","agent_id","agentId"),""))
        parent=str(_find(obj,("parent_session_id","parentSessionId","parent_id"),""))
        message_id=_native_value(obj)
        project=str(_find(obj,("project","project_name"),source.project))
        repo=str(_find(obj,("repo","repository"),"")); worktree=str(_find(obj,("worktree","workspace"),""))
        agent_type=str(_find(obj,("agent_type","agentType","harness"),source.kind))
        meta={k:v for k,v in obj.items() if k not in ("text","content","message","payload")}
        record_type=_record_type_value(obj)
    else:
        role=""; text=_text(obj); sid=session; ts=""; sub=""; parent=""; message_id=""; project=source.project; repo=""; worktree=""; agent_type=source.kind; meta={}; record_type="record"
    if not text and not raw.strip(): return None
    meta.setdefault("record_type", record_type)
    return _chunk(source, ordinal=n, session=sid, message=message_id, role=role,
                  text=text, raw=raw, timestamp=ts, content_type="text/plain",
                  metadata=meta, parent_session=parent, agent_id=sub,
                  worktree=worktree, record_type=record_type, project=project,
                  repo=repo, agent_type=agent_type)

def _parse_jsonl(source: Source, lines: Iterable[str], start=0, session_id="") -> list[Chunk]:
    out=[]; session=session_id
    lines=list(lines)
    if not session:
        session=_session_fallback(lines, source)
    for n,raw in enumerate(lines):
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
        # offset is only used as an append cursor; record identity is semantic.
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
        # Canonicalize opaque objects so equivalent JSON with different key
        # order/whitespace hashes to the same fallback identity.
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "" if value is None else str(value)


def _stable_identity(source: Source, session: str, message: str = "", *, timestamp: str = "",
                     role: str = "", record_type: str = "", content_hash: str = "",
                     fallback_key: str = "") -> str:
    # Device id and absolute path are deliberately absent: the same session
    # converges when copied to another Mac.  Native ids win; fallback records use
    # semantic fields rather than a byte offset or line ordinal.
    agent = source.source_agent or source.kind
    kind = source.source_type or source.kind
    if message:
        locator = ("native", agent, kind, session, record_type, message)
    else:
        locator = ("fallback", agent, kind, session, timestamp, role,
                   record_type, content_hash, fallback_key)
    return _stable_digest(*locator)


def _chunk(source: Source, *, ordinal: int, session: str, message: str, role: str,
           text: str, raw: str, timestamp: str = "", content_type: str = "assistant_message",
           metadata: dict[str, Any] | None = None, parent_session: str = "", agent_id: str = "", worktree: str = "",
           record_type: str = "", fallback_key: str = "", project: str | None = None,
           repo: str = "", agent_type: str = "") -> Chunk | None:
    if not text.strip():
        return None
    meta = dict(metadata or {})
    # The exact human/tool text is already `raw_text`.  Retain only a fingerprint
    # of the JSON envelope so large tool records are not duplicated in every
    # split chunk and the local pending queue stays bounded.
    meta.setdefault("raw_record_hash", hashlib.sha256(raw.encode("utf-8")).hexdigest())
    stable_type = record_type or (metadata or {}).get("record_type") or content_type or "record"
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = _stable_identity(source, session, message, timestamp=timestamp, role=role,
                                record_type=str(stable_type), content_hash=content_hash,
                                fallback_key=fallback_key)
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
        project=source.project if project is None else project,
        repo=repo,
        worktree=worktree,
        agent_type=agent_type or ("subagent" if agent_id or "subagent" in str(source.path) or "agent-" in str(source.path) else "main"),
        device_id=source.device_id,
    )


def _read_lines(path: Path, start: int = 0) -> list[str]:
    data = path.read_bytes()
    if start:
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
    session = _session_fallback(all_lines, source)
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
    # `start` is a byte cursor only; identities never depend on that cursor.
    out: list[Chunk] = []
    for ordinal, line in enumerate(lines):
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
        message = str(payload.get("id") or payload.get("call_id") or obj.get("id") or "")
        if typ in {"function_call", "custom_tool_call", "function_call_output", "custom_tool_call_output"}:
            # A call and its output share `call_id`, but are distinct retrievable
            # records; include the direction in the stable identity.
            if message:
                message = f"{message}:{'out' if typ.endswith('_output') else 'call'}"
        agent_id = str(payload.get("agent_id") or payload.get("agentId") or "")
        chunk = _chunk(source, ordinal=ordinal, session=session, message=message, role=role,
                       text=text, raw=line, timestamp=_timestamp_value(obj),
                       content_type=ctype, parent_session=parent, agent_id=agent_id,
                       metadata={"record_type": typ, "tool_name": payload.get("name")}, worktree=worktree,
                       record_type=typ)
        if chunk:
            out.append(chunk)
    return out


def _parse_claude_native(source: Source, start: int = 0) -> list[Chunk]:
    out: list[Chunk] = []
    path = Path(source.path)
    all_lines = _read_lines(path, 0)
    session = _session_fallback(all_lines, source)
    lines = _read_lines(path, start) if start else all_lines
    for ordinal, line in enumerate(lines):
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
        sid = _session_value(obj) or session
        mid = _native_value(obj)
        record_type = typ
        chunk = _chunk(source, ordinal=ordinal, session=sid, message=mid, role=role, text=text,
                       raw=line, timestamp=_timestamp_value(obj), content_type=ctype,
                       parent_session=str(obj.get("parentUuid") or ""),
                       agent_id=str(obj.get("agentId") or ""),
                       metadata={"is_sidechain": bool(obj.get("isSidechain")), "git_branch": obj.get("gitBranch")},
                       worktree=str(obj.get("cwd") or ""), record_type=record_type)
        if chunk:
            out.append(chunk)
    return out


def _parse_pi_native(source: Source, start: int = 0) -> list[Chunk]:
    out: list[Chunk] = []
    path = Path(source.path)
    # Append scans must retain the session envelope from the beginning of the
    # file; otherwise the fallback stem becomes a new identity for every tail.
    prefix = _read_lines(path, 0)
    session = _session_fallback(prefix, source)
    worktree = ""
    for line in prefix[:256]:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("type") == "session":
            session = str(obj.get("id") or obj.get("sessionId") or session)
            worktree = str(obj.get("cwd") or "")
            break
    lines = _read_lines(path, start) if start else prefix
    for ordinal, line in enumerate(lines):
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
        mid = _native_value(obj) or _native_value(msg)
        record_type = str(obj.get("record_type") or obj.get("type") or "message")
        chunk = _chunk(source, ordinal=ordinal, session=session, message=mid, role=role, text=text,
                       raw=line, timestamp=_timestamp_value(obj), content_type=ctype,
                       parent_session=str(obj.get("parentId") or ""),
                       metadata={"tool_name": msg.get("toolName"), "record_type": record_type}, worktree=worktree,
                       record_type=record_type)
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
    repo_identity, relative_path = _memory_identity_context(source)
    for i, piece in enumerate(pieces):
        # Memory identity is semantic rather than path/index based so copies of
        # the same repository memory converge across devices and mount points.
        lines = piece.splitlines()
        heading = lines[0].strip() if lines else ""
        # A heading is the durable section key: content may be edited in place,
        # while an absolute checkout path and byte/line position can change.
        section = heading or "__preamble__"
        semantic_id = f"memory:{source.kind}:{repo_identity}:{relative_path}:{section}"
        c = _chunk(source, ordinal=i, session="", message=semantic_id, role="system",
                   text=piece, raw=piece, content_type="agents_md" if source.kind == "agents_md" else "memory",
                   metadata={"source_file": str(source.path)}, record_type="agents_md" if source.kind == "agents_md" else "memory",
                   fallback_key=heading)
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


def _parse_jsonl_path(source: Source, start: int = 0) -> list[Chunk]:
    """Parse a generic JSONL tail with a session anchor from the full file."""
    path = Path(source.path)
    all_lines = _read_lines(path, 0)
    session = _session_fallback(all_lines, source)
    lines = _read_lines(path, start) if start else all_lines
    return _parse_jsonl(source, lines, start, session)


def parse_file(source: Source, start: int = 0) -> list[Chunk]:
    """Parse one discovered source with the native adapter selected by its kind."""
    try:
        if source.kind in {"codex", "codex_session"}:
            native = _parse_codex_native(source, start)
            return _split_large(native or _parse_jsonl_path(source, start))
        if source.kind in {"claude", "claude_session"}:
            native = _parse_claude_native(source, start)
            return _split_large(native or _parse_jsonl_path(source, start))
        if source.kind in {"pi", "pi_session"}:
            native = _parse_pi_native(source, start)
            return _split_large(native or _parse_jsonl_path(source, start))
        if source.kind in {"codex_memory", "pi_memory", "claude_memory", "agents_md", "persistent"}:
            return _split_large(_parse_memory_native(source))
        return _split_large(_parse_jsonl_path(source, start))
    except OSError:
        return []


def parse_codex(source: Source, start=0): return _parse_codex_native(source, start)
def parse_pi(source: Source, start=0): return _parse_pi_native(source, start)
def parse_claude(source: Source, start=0): return _parse_claude_native(source, start)
