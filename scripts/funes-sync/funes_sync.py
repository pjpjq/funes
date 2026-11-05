#!/usr/bin/env python3
"""Local, lossless sync bridge for Codex/Claude/pi artifacts and funes.

The bridge deliberately writes only under FUNES_HOME (default ``~/.funes``).  It does not
edit agent configuration files.  Source artifacts are represented as Claude-compatible JSONL
sessions so the existing ``funes index`` pipeline remains the single indexing implementation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import plistlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

VERSION = 1
LABEL = "com.funes.sync"
DEFAULT_INTERVAL = 300
MAX_FILES = 20_000


def configured_memory() -> str:
    """Read only the non-secret remote memory name from the optional TOML config."""
    value = os.environ.get("FUNES_MEMORY", "").strip()
    if value:
        return value
    path = Path(os.environ.get("FUNES_CONFIG", "~/.config/funes/config.toml")).expanduser()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("memory") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _iter_files(root: Path, suffixes: tuple[str, ...], limit: int = MAX_FILES) -> Iterator[Path]:
    if not root.is_dir():
        return
    count = 0
    try:
        for path in sorted(root.rglob("*")):
            if count >= limit:
                break
            if path.is_file() and path.suffix.lower() in suffixes:
                count += 1
                yield path
    except OSError:
        return


@dataclass(frozen=True)
class Source:
    path: Path
    kind: str
    harness: str = "claude"

    @property
    def key(self) -> str:
        return str(self.path)


class Discoverer:
    """Discover known session trees, memory files, and repository instructions."""

    def __init__(self, home: Optional[Path] = None, repos: Optional[Iterable[Path]] = None):
        self.home = (home or Path(os.environ.get("HOME", "~")).expanduser()).resolve()
        if repos is not None:
            explicit = list(repos)
        else:
            explicit = self._env_repos()
            if not explicit:
                explicit = self._cwd_repos()
        self.repos = [Path(p).expanduser().resolve() for p in explicit]

    def _env_repos(self) -> list[Path]:
        raw = os.environ.get("FUNES_SYNC_REPOS", "")
        return [Path(x) for x in raw.split(os.pathsep) if x]

    def _cwd_repos(self) -> list[Path]:
        cwd = Path.cwd().resolve()
        candidates = [cwd]
        candidates.extend(cwd.parents)
        return [p for p in candidates if (p / ".git").exists()][:8]

    def _session_roots(self) -> list[tuple[Path, str]]:
        candidates = [
            (self.home / ".codex" / "sessions", "codex"),
            (self.home / ".codex" / "archived_sessions", "codex"),
            (self.home / ".codex" / "archive", "codex"),
            (self.home / ".codex" / "sessions" / "archive", "codex"),
            (self.home / ".claude" / "projects", "claude"),
            (self.home / ".pi" / "agent" / "sessions", "pi"),
            (self.home / ".pi" / "sessions", "pi"),
        ]
        seen: set[Path] = set()
        out: list[tuple[Path, str]] = []
        for root, harness in candidates:
            try:
                root = root.resolve()
            except OSError:
                continue
            if root in seen or not root.is_dir():
                continue
            seen.add(root)
            out.append((root, harness))
        return out

    def discover(self) -> list[Source]:
        out: dict[str, Source] = {}
        for root, harness in self._session_roots():
            for path in _iter_files(root, (".jsonl",)):
                out[str(path)] = Source(path, f"{harness}_session", harness)

        # User-wide instructions and memory.  Keep raw files untouched; conversion is downstream.
        explicit = [self.home / ".codex" / "AGENTS.md"]
        explicit.extend(_iter_files(self.home / ".codex" / "memories", (".md",)))
        explicit.extend(_iter_files(self.home / ".claude", (".md",)))
        explicit.extend(_iter_files(self.home / ".pi", (".md",)))
        for path in explicit:
            if path.is_file():
                out[str(path.resolve())] = Source(path.resolve(), "memory", "claude")

        # Repository AGENTS/MEMORY files, including nested package guidance.
        for repo in self.repos:
            if not repo.is_dir():
                continue
            for path in _iter_files(repo, (".md",), limit=MAX_FILES):
                if path.name.upper() in {"AGENTS.MD", "MEMORY.MD"}:
                    out[str(path.resolve())] = Source(path.resolve(), "repo_memory", "claude")
        return sorted(out.values(), key=lambda s: s.key)


class SyncStore:
    def __init__(self, root: Optional[Path] = None):
        configured = root or Path(os.environ.get("FUNES_HOME", "~/.funes")).expanduser()
        self.root = configured.resolve()
        self.sources = self.root / "sources"
        self.state_path = self.root / "sync-state.json"
        self.log_path = self.root / "funes-sync.log"
        self.root.mkdir(parents=True, exist_ok=True)
        self.sources.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value.setdefault("files", {})
                value.setdefault("pending", {})
                return value
        except (OSError, ValueError):
            pass
        return {"version": VERSION, "files": {}, "pending": {}, "created_at": _now()}

    def save(self, state: dict) -> None:
        state["version"] = VERSION
        _atomic_write(self.state_path, (json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())

    def log(self, message: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{_now()} {message}\n")


def _source_identity(source: Source, raw: Optional[str] = None) -> str:
    """Stable identity across devices for a transcript; path identity for memory files."""
    if raw and source.kind.endswith("_session"):
        for line in raw.splitlines()[:32]:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            payload = obj.get("payload", {}) if isinstance(obj, dict) else {}
            sid = payload.get("id") or obj.get("sessionId") or obj.get("session_id")
            if sid:
                return _sha256(f"session\0{source.harness}\0{sid}".encode())[:32]
    return _sha256(f"{source.kind}\0{source.path.resolve()}".encode())[:32]


def _version_id(source: Source, raw: str) -> tuple[str, str]:
    identity = _source_identity(source, raw)
    return identity, _sha256(f"{identity}\0{_sha256(raw.encode())}".encode())[:32]


def _jsonl_record(source: Source, raw: str, seq: int, source_id: str, role: str, content: str, identity: str) -> dict:
    # Extra fields preserve provenance/raw data for future consumers; Claude parser ignores them.
    return {
        "type": role,
        "uuid": f"{source_id}-{seq}",
        "timestamp": _now(),
        "cwd": str(source.path.parent),
        "funes_sync": {"source": str(source.path), "source_id": source_id, "source_identity": identity, "kind": source.kind, "content_hash": _sha256(raw.encode())},
        "message": {"role": role, "content": content},
    }


def convert_source(source: Source, destination: Path, raw: Optional[str] = None) -> None:
    """Write one stable synthetic Claude session, retaining the complete source as raw text."""
    if raw is None:
        raw = source.path.read_text(encoding="utf-8", errors="replace")
    identity, source_id = _version_id(source, raw)
    title = f"[funes-sync] {source.kind}: {source.path}"
    records = [
        _jsonl_record(source, raw, 0, source_id, "user", title, identity),
        _jsonl_record(source, raw, 1, source_id, "assistant", raw, identity),
    ]
    payload = "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in records).encode()
    _atomic_write(destination, payload)


class SyncEngine:
    def __init__(self, store: SyncStore, discoverer: Optional[Discoverer] = None, runner: Optional[Callable[..., subprocess.CompletedProcess]] = None):
        self.store = store
        self.discoverer = discoverer or Discoverer()
        self.runner = runner or subprocess.run

    def _destination(self, source: Source, version_id: Optional[str] = None) -> Path:
        return self.store.sources / f"{version_id or _source_identity(source)}.jsonl"

    def sync_once(self) -> dict:
        state = self.store.load()
        files = state.setdefault("files", {})
        pending = state.setdefault("pending", {})
        changed = 0
        missing = 0
        sources = self.discoverer.discover()
        seen: set[str] = set()
        for source in sources:
            key = source.key
            seen.add(key)
            try:
                stat = source.path.stat()
                mtime_ns = stat.st_mtime_ns
                size = stat.st_size
            except OSError as exc:
                entry = files.setdefault(key, {"source_identity": _source_identity(source)})
                entry.update({"missing": True, "missing_at": _now(), "error": str(exc)})
                missing += 1
                continue
            old = files.get(key, {})
            if old.get("mtime_ns") == mtime_ns and old.get("size") == size and (old.get("sha256") or source.kind.endswith("_session")):
                entry = old
                entry["missing"] = False
                continue
            # Native transcript parsers already preserve session/message metadata and perform
            # chunk-level deduplication. Do not materialize multi-GB JSONL into synthetic files;
            # the index step below reads these roots directly.
            if source.kind.endswith("_session"):
                digest = _sha256(f"{source.harness}\0{source.path.resolve()}".encode())
                files[key] = {"source_identity": digest[:32], "source_id": digest[:32], "mtime_ns": mtime_ns, "size": size, "kind": source.kind, "missing": False, "updated_at": _now()}
                pending[digest[:32]] = {"source": key, "source_id": digest[:32], "source_identity": digest[:32], "updated_at": _now()}
                changed += 1
                continue
            try:
                raw_bytes = source.path.read_bytes()
                raw = raw_bytes.decode("utf-8", errors="replace")
            except OSError as exc:
                old.update({"missing": True, "missing_at": _now(), "error": str(exc)})
                files[key] = old
                missing += 1
                continue
            digest = _sha256(raw_bytes)
            identity, version_id = _version_id(source, raw)
            if old.get("sha256") == digest and self._destination(source, version_id).is_file():
                old.update({"mtime_ns": mtime_ns, "size": size, "missing": False})
                files[key] = old
                continue
            destination = self._destination(source, version_id)
            convert_source(source, destination, raw)
            files[key] = {"source_identity": identity, "source_id": version_id, "sha256": digest, "mtime_ns": mtime_ns, "size": size, "kind": source.kind, "generated": str(destination), "missing": False, "updated_at": _now()}
            pending[version_id] = {"source": key, "source_id": version_id, "source_identity": identity, "updated_at": _now()}
            changed += 1
            # Large historical rollouts can be tens of MB. Release transient byte/string and
            # JSON encoder allocations before the next source so the daemon stays bounded.
            del raw_bytes, raw
            if changed % 16 == 0:
                gc.collect()

        # Preserve records for disappeared files, but mark them softly instead of deleting history.
        for key, entry in files.items():
            if key not in seen and not entry.get("missing"):
                entry.update({"missing": True, "missing_at": _now()})
                missing += 1

        result = {"changed": changed, "missing": missing, "sources": len(sources), "indexed": False, "pushed": False, "pending": len(pending)}
        if changed or pending:
            index_ok = self._run_index()
            result["indexed"] = index_ok
            if index_ok:
                memory = configured_memory()
                if memory and pending:
                    push_ok = self._run_push(memory)
                    result["pushed"] = push_ok
                    if push_ok:
                        pending.clear()
                elif not os.environ.get("FUNES_MEMORY"):
                    # Local-only indexing has no remote acknowledgement to wait for.
                    pending.clear()
        result["pending"] = len(pending)
        state["last_sync"] = _now()
        state["last_result"] = result
        state["pending"] = pending
        state["files"] = files
        if result.get("indexed") is False and (changed or pending):
            state["last_error"] = "funes index failed or unavailable"
        elif result.get("pushed") is False and configured_memory() and pending:
            state["last_error"] = "remote push failed; pending queue retained"
        else:
            state.pop("last_error", None)
        self.store.save(state)
        self.store.log(f"sync result={json.dumps(result, ensure_ascii=False, sort_keys=True)}")
        return result

    def _run_index(self) -> bool:
        binary = os.environ.get("FUNES_BIN", "funes")
        roots = [(root, harness) for root, harness in self.discoverer._session_roots()]
        if any(s.kind in {"memory", "repo_memory"} for s in self.discoverer.discover()):
            roots.append((self.store.sources, "claude"))
        for root, harness in roots:
            cmd = [binary, "index", str(root), "--harness", harness, "--yes"]
            try:
                completed = self.runner(cmd, check=False, capture_output=True, text=True)
            except (OSError, subprocess.SubprocessError) as exc:
                self.store.log(f"index error={exc}")
                return False
            if completed.returncode != 0:
                self.store.log(f"index failed rc={completed.returncode} stderr={completed.stderr[-1000:]}")
                return False
        return True

    def _run_push(self, memory: str) -> bool:
        binary = os.environ.get("FUNES_BIN", "funes")
        try:
            completed = self.runner([binary, "push", memory], check=False, capture_output=True, text=True)
        except (OSError, subprocess.SubprocessError) as exc:
            self.store.log(f"push error={exc}")
            return False
        if completed.returncode != 0:
            self.store.log(f"push failed rc={completed.returncode} stderr={completed.stderr[-1000:]}")
            return False
        return True


def launch_agent_path(home: Optional[Path] = None) -> Path:
    configured = os.environ.get("FUNES_SYNC_LAUNCH_AGENT")
    if configured:
        return Path(configured).expanduser()
    return (home or Path.home()) / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def install_launch_agent(force: bool = False, script: Optional[Path] = None, home: Optional[Path] = None) -> Path:
    path = launch_agent_path(home)
    script = script or Path(__file__).resolve()
    log_root = Path(os.environ.get("FUNES_HOME", "~/.funes")).expanduser()
    log_path = log_root / "funes-sync.log"
    payload = {
        "Label": LABEL,
        "ProgramArguments": [sys.executable, str(script), "daemon"],
        "EnvironmentVariables": {
            "FUNES_BIN": os.environ.get("FUNES_BIN") or shutil.which("funes") or str(Path.home() / ".local/bin/funes"),
            "FUNES_HOME": str(log_root),
        },
        "RunAtLoad": True,
        "KeepAlive": True,
        "StartInterval": int(os.environ.get("FUNES_SYNC_INTERVAL", str(DEFAULT_INTERVAL))),
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
        "ProcessType": "Background",
    }
    if path.exists() and not force:
        try:
            existing = plistlib.loads(path.read_bytes())
        except Exception as exc:
            raise RuntimeError(f"refusing to overwrite unreadable LaunchAgent {path}: {exc}")
        if existing.get("Label") != LABEL:
            raise RuntimeError(f"refusing to overwrite unrelated LaunchAgent {path}")
    _atomic_write(path, plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=False))
    return path


def _launchctl(action: str, path: Path) -> int:
    if sys.platform != "darwin":
        print("launchctl is only available on macOS", file=sys.stderr)
        return 2
    uid = os.getuid()
    if action == "start":
        cmd = ["launchctl", "bootstrap", f"gui/{uid}", str(path)]
    elif action == "stop":
        cmd = ["launchctl", "bootout", f"gui/{uid}", str(path)]
    else:
        raise ValueError(action)
    return subprocess.run(cmd, check=False).returncode


def daemon(engine: SyncEngine, interval: int) -> int:
    stop = False
    def handler(_sig: int, _frame: object) -> None:
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    while not stop:
        try:
            engine.sync_once()
        except Exception as exc:  # daemon must survive a transient unreadable source
            engine.store.log(f"sync exception={exc!r}")
        for _ in range(max(1, interval)):
            if stop:
                break
            time.sleep(1)
    return 0


def doctor(store: SyncStore) -> int:
    binary = os.environ.get("FUNES_BIN", "funes")
    found = shutil.which(binary) or (binary if Path(binary).is_file() else None)
    checks = {"funes_bin": found, "funes_home": str(store.root), "sources": str(store.sources), "launch_agent": str(launch_agent_path())}
    print(json.dumps(checks, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if found else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync Codex/Claude/pi memory artifacts into funes")
    parser.add_argument("command", nargs="?", default="once", choices=["once", "sync", "daemon", "status", "sources", "doctor", "install", "start", "stop", "restart", "logs"])
    parser.add_argument("--root", type=Path, help="override FUNES_HOME for this invocation")
    parser.add_argument("--interval", type=int, default=int(os.environ.get("FUNES_SYNC_INTERVAL", str(DEFAULT_INTERVAL))))
    parser.add_argument("--follow", action="store_true", help="follow logs")
    parser.add_argument("--force", action="store_true", help="allow replacing our own LaunchAgent")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Accept both ``sync once`` (documented service form) and the concise ``once`` form.
    if raw and raw[0] == "sync":
        raw = raw[1:]
        if not raw or raw[0].startswith("-"):
            raw.insert(0, "once")
    args = build_parser().parse_args(raw)
    store = SyncStore(args.root)
    discoverer = Discoverer()
    engine = SyncEngine(store, discoverer)
    if args.command in {"once", "sync"}:
        print(json.dumps(engine.sync_once(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "daemon":
        return daemon(engine, args.interval)
    if args.command == "status":
        print(json.dumps(store.load(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "sources":
        for source in discoverer.discover():
            print(f"{source.kind}\t{source.path}")
        return 0
    if args.command == "doctor":
        return doctor(store)
    if args.command == "install":
        print(install_launch_agent(args.force))
        return 0
    if args.command in {"start", "stop"}:
        return _launchctl(args.command, launch_agent_path())
    if args.command == "restart":
        _launchctl("stop", launch_agent_path())
        return _launchctl("start", launch_agent_path())
    if args.command == "logs":
        if not store.log_path.exists():
            return 0
        if args.follow:
            with store.log_path.open(encoding="utf-8") as fh:
                fh.seek(0, os.SEEK_END)
                while True:
                    line = fh.readline()
                    if line:
                        print(line, end="")
                    else:
                        time.sleep(0.5)
        else:
            print(store.log_path.read_text(encoding="utf-8", errors="replace"), end="")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
