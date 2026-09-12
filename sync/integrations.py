from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

MARKER = "<!-- funes-unified-memory -->"
INSTRUCTIONS = f"""{MARKER}
When a question refers to prior work, decisions, history, previous tests, or user preferences, use the `funes-remote` MCP tools (`recall`/`get`) before re-deriving an answer. Search across Codex, Pi, and Claude; return original raw context and preserve source metadata. Do not call it for trivial self-contained questions.
{MARKER}
"""


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def install_pi(home: Path) -> Path:
    target = home / ".pi" / "agent" / "extensions" / "funes-remote.ts"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text((root() / "integrations/pi/funes-remote.ts").read_text(), encoding="utf-8")
    return target


def append_instruction(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if MARKER not in current:
        with path.open("a", encoding="utf-8") as fh:
            if current and not current.endswith("\n"):
                fh.write("\n")
            fh.write("\n" + INSTRUCTIONS)


def _run(args: list[str]) -> bool:
    try:
        return subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except OSError:
        return False


def install_all(home: Path | None = None) -> dict[str, str | bool]:
    home = home or Path(os.environ.get("HOME", "~")).expanduser()
    bin_path = Path(os.environ.get("FUNES_SYNC_BIN", root() / "bin/funes-sync")).expanduser()
    result: dict[str, str | bool] = {"pi_extension": str(install_pi(home))}
    append_instruction(home / ".codex" / "AGENTS.md")
    append_instruction(home / ".claude" / "CLAUDE.md")
    result["codex_instructions"] = True
    result["claude_instructions"] = True
    if shutil.which("codex"):
        result["codex_mcp"] = _run(["codex", "mcp", "add", "funes-remote", "--", str(bin_path), "mcp"])
    else:
        result["codex_mcp"] = False
    if shutil.which("claude"):
        result["claude_mcp"] = _run(["claude", "mcp", "add", "--scope", "user", "funes-remote", "--", str(bin_path), "mcp"])
    else:
        result["claude_mcp"] = False
    return result
