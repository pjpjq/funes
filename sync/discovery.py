from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import Config


@dataclass(frozen=True)
class Source:
    source_key: str
    kind: str
    path: Path
    device_id: str
    project: str = ""
    @property
    def source_agent(self) -> str:
        if self.kind.startswith("codex"):
            return "codex"
        if self.kind.startswith("pi"):
            return "pi"
        if self.kind.startswith("claude"):
            return "claude_code"
        if self.path.name.upper() == "AGENTS.MD":
            return "codex"
        if self.path.name.upper() == "CLAUDE.MD":
            return "claude_code"
        if self.kind == "persistent":
            return "shared"
        return "unknown"

    @property
    def source_type(self) -> str:
        if self.kind in {"codex", "pi", "claude", "codex_session", "pi_session", "claude_session"} or self.kind.endswith("session"):
            return "session"
        if self.path.name.upper() == "AGENTS.MD" or self.kind == "agents_md":
            return "agents_md"
        if self.path.name.upper() == "CLAUDE.MD":
            return "project_instruction"
        if self.kind.endswith("memory") or self.kind == "persistent":
            return "memory"
        return self.kind

    def as_dict(self): return {"source_key": self.source_key, "kind": self.kind, "source_agent": self.source_agent, "source_type": self.source_type, "path": str(self.path), "device_id": self.device_id, "project": self.project}

def _canonical(path: Path, home: Path) -> str:
    try: return "~/{0}".format(path.resolve().relative_to(home.resolve()))
    except ValueError: return str(path.resolve())

def _source(kind: str, path: Path, cfg: Config, project="") -> Source:
    # The key deliberately excludes device id: the same logical source converges across devices.
    key = f"{kind}:{_canonical(path, cfg.home)}"
    return Source(key, kind, path, cfg.device_id, project)

def _files(root: Path, patterns: tuple[str,...], max_files=10000) -> Iterable[Path]:
    if not root.exists(): return
    n=0
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file() and p.name.lower() not in {"auth.json", "credentials.json", ".env", "secrets.json"}:
                yield p; n += 1
                if n >= max_files: return


def _project_roots(cfg: Config) -> list[Path]:
    roots = [Path.cwd()]
    configured = os.environ.get("FUNES_PROJECT_ROOTS", "")
    if configured:
        roots.extend(Path(value).expanduser() for value in configured.split(os.pathsep) if value)
    single = os.environ.get("FUNES_PROJECT_ROOT")
    if single:
        roots.append(Path(single).expanduser())
    roots.extend(path for path in (cfg.home / "code", cfg.home / "Projects", cfg.home / "Developer") if path.exists())
    result = []
    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _project_memory_files(
    root: Path, max_depth: int = 5, max_files: int = 5000, max_directories: int = 10000
) -> Iterable[Path]:
    wanted = {"AGENTS.md", "CLAUDE.md", "MEMORY.md", "memory.md"}
    excluded = {".git", ".venv", "node_modules", "target", "logs", "__pycache__", ".cache"}
    count = 0
    directories = 0
    for directory, children, names in os.walk(root):
        directories += 1
        if directories > max_directories:
            return
        current = Path(directory)
        depth = len(current.parts) - len(root.parts)
        children[:] = [] if depth >= max_depth else [name for name in children if name not in excluded]
        for name in names:
            if name not in wanted:
                continue
            yield current / name
            count += 1
            if count >= max_files:
                return


def _project_for(path: Path, scan_root: Path) -> str:
    for parent in (path.parent, *path.parents):
        if (parent / ".git").exists():
            return str(parent)
        if parent == scan_root:
            break
    return str(scan_root)

def _pi_project_for(path: Path) -> str:
    """Use Pi's session header, never guess an ambiguous encoded directory."""
    if path.suffix.lower() == ".jsonl":
        try:
            with path.open("r", encoding="utf-8") as stream:
                header = json.loads(stream.readline(16384))
            cwd = header.get("cwd") if isinstance(header, dict) and header.get("type") == "session" else None
            if isinstance(cwd, str) and Path(cwd).is_absolute():
                directory = Path(cwd)
                for parent in (directory, *directory.parents):
                    if (parent / ".git").exists():
                        return str(parent)
                return str(directory)
        except (OSError, UnicodeError, ValueError):
            pass
    for parent in path.parents:
        if (parent / ".git").exists():
            return str(parent)
    return ""


def discover_sources(cfg: Config|None=None) -> list[Source]:
    cfg = cfg or Config.load(); h=cfg.home; out=[]; seen=set(); seen_keys=set()
    def add(kind,p,project=""):
        try:
            p = Path(p)
            if not p.is_file():
                return
            resolved = p.resolve()
            if resolved in seen:
                return
            source = _source(kind, p, cfg, project)
        except (TypeError, OSError, RuntimeError):
            return
        if source.source_key in seen_keys:
            return
        seen.add(resolved)
        seen_keys.add(source.source_key)
        out.append(source)
    codex = Path(os.environ.get("CODEX_HOME", h/".codex")).expanduser()
    if cfg.source_codex:
        for root in (codex/"sessions", codex/"archived_sessions", codex/"subagents", codex/"sessions"/"archive"):
            for p in _files(root,("**/*.jsonl",)): add("codex",p)
        for p in _files(codex / "memories", ("**/*.md", "**/*.jsonl", "**/*.txt")):
            add("codex_memory", p)
        # Automation run logs/evaluations are operational output, not durable
        # agent memory; importing them would swamp retrieval with shell output.
        # Keep only the automation's instructions/config and its explicit memory.md.
        for p in _files(codex / "automations", ("**/*.md", "**/*.toml")):
            add("codex_memory", p)
    # Codex top-level persistent instruction files
    for p in (codex/"AGENTS.md", codex/"MEMORY.md", codex/"memory.md"):
        add("codex_memory",p)
    pi_roots=[]
    for key in ("PI_CODING_AGENT_SESSION_DIR", "PI_CODING_AGENT_DIR","PI_SESSION_DIR"):
        if os.environ.get(key): pi_roots.append(Path(os.environ[key]).expanduser())
    pi_roots += [h/".pi/agent/sessions", h/".pi/sessions"]
    if cfg.source_pi:
        for root in pi_roots:
            for p in _files(root,("**/*.jsonl",)): add("pi", p, _pi_project_for(p))
        # Older pi releases allowed a session directory directly under ~/.pi;
        # retain that fallback for existing installations and test fixtures.
        for p in _files(h/".pi", ("**/*.jsonl",)):
            add("pi", p, _pi_project_for(p))
        for root in (h/".pi/agent", h/".pi"):
            for p in _files(root,("**/*.md",)): add("pi_memory", p, _pi_project_for(p))
    claude = Path(os.environ.get("CLAUDE_CONFIG_DIR", h/".claude")).expanduser()
    for root in (claude/"projects", claude/"history"):
        for p in _files(root,("**/*.jsonl",)): add("claude",p)
    for p in (claude/"history.jsonl", h/".claude"/"history.jsonl"):
        add("claude", p)
    for root in (claude/"memory", claude/"memories"):
        for p in _files(root,("**/*.md",)): add("claude_memory",p)
    for p in (claude/"CLAUDE.md", h/"CLAUDE.md", h/"MEMORY.md", h/"memory.md"):
        add("persistent",p)
    # Repository instruction files: bounded roots, avoiding home-wide scans.
    for root in _project_roots(cfg):
        for p in _project_memory_files(root):
            add("agents_md" if p.name.upper() == "AGENTS.MD" else "persistent", p, _project_for(p, root))
    return out
