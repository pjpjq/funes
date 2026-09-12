from __future__ import annotations
import hashlib, os, socket
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
        return "unknown"

    @property
    def source_type(self) -> str:
        if self.kind in {"codex", "pi", "claude", "codex_session", "pi_session", "claude_session"} or self.kind.endswith("session"):
            return "session"
        if self.kind.endswith("memory"):
            return "memory"
        if self.kind == "agents_md":
            return "agents_md"
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

def discover_sources(cfg: Config|None=None) -> list[Source]:
    cfg = cfg or Config.load(); h=cfg.home; out=[]; seen=set()
    def add(kind,p,project=""):
        try: p=Path(p)
        except TypeError: return
        if p.is_file() and p not in seen:
            seen.add(p); out.append(_source(kind,p,cfg,project))
    codex = Path(os.environ.get("CODEX_HOME", h/".codex")).expanduser()
    if cfg.source_codex:
        for root in (codex/"sessions", codex/"archived_sessions", codex/"subagents", codex/"sessions"/"archive"):
            for p in _files(root,("**/*.jsonl",)): add("codex",p)
        for root in (codex/"memories", codex/"automations"):
            for p in _files(root,("**/*.md","**/*.jsonl","**/*.txt")): add("codex_memory",p)
    # Codex top-level persistent instruction files
    for p in (codex/"AGENTS.md", codex/"MEMORY.md", codex/"memory.md"):
        add("codex_memory",p)
    pi_roots=[]
    for key in ("PI_CODING_AGENT_SESSION_DIR", "PI_CODING_AGENT_DIR","PI_SESSION_DIR"):
        if os.environ.get(key): pi_roots.append(Path(os.environ[key]).expanduser())
    pi_roots += [h/".pi/agent/sessions", h/".pi/sessions"]
    if cfg.source_pi:
        for root in pi_roots:
            for p in _files(root,("**/*.jsonl",)): add("pi",p)
        # Older pi releases allowed a session directory directly under ~/.pi;
        # retain that fallback for existing installations and test fixtures.
        for p in _files(h/".pi", ("**/*.jsonl",)):
            add("pi", p)
        for root in (h/".pi/agent", h/".pi"):
            for p in _files(root,("**/*.md",)): add("pi_memory",p)
    claude = Path(os.environ.get("CLAUDE_CONFIG_DIR", h/".claude")).expanduser()
    for root in (claude/"projects", claude/"history"):
        for p in _files(root,("**/*.jsonl",)): add("claude",p)
    for root in (claude/"memory", claude/"memories"):
        for p in _files(root,("**/*.md",)): add("claude_memory",p)
    for p in (claude/"CLAUDE.md", h/"CLAUDE.md", h/"MEMORY.md", h/"memory.md"):
        add("persistent",p)
    # Repository instruction files: bounded roots, avoiding home-wide scans.
    roots=[Path.cwd(), Path(os.environ.get("FUNES_PROJECT_ROOT", Path.cwd()))]
    for root in roots:
        if root.exists():
            for name in ("AGENTS.md","CLAUDE.md","MEMORY.md","memory.md"):
                for p in root.rglob(name):
                    if p.is_file() and len(p.parts)-len(root.parts) <= 5: add("agents_md" if name.upper() == "AGENTS.MD" else "persistent",p,str(root))
    return out
