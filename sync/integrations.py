from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
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


def remove_legacy_pi_package(home: Path) -> bool:
    """Stop the old local-index Pi package when installing the remote bridge.

    The continuous sync daemon owns ingestion for this deployment.  Leaving the
    historical ``funes add pi`` package enabled starts a second index/push loop
    on every Pi session and can contend for the same memory lock.  Only the
    exact user-wide Funes package is removed; its files and all unrelated Pi
    settings remain untouched.
    """
    settings = home / ".pi" / "agent" / "settings.json"
    if not settings.exists():
        return False

    # Pi uses proper-lockfile with ``realpath: false``.  Its lock is an atomic
    # sibling directory named ``settings.json.lock``; using the same protocol
    # prevents this read-modify-write from racing a live Pi process.
    lock = settings.with_name(settings.name + ".lock")
    acquired = False
    for attempt in range(10):
        try:
            lock.mkdir()
            acquired = True
            break
        except FileExistsError:
            if attempt < 9:
                time.sleep(0.02)
        except OSError:
            return False
    if not acquired:
        return False

    try:
        try:
            data = json.loads(settings.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
            return False
        if not isinstance(data, dict):
            return False
        packages = data.get("packages")
        if not isinstance(packages, list):
            return False

        def lexical(path: Path) -> Path:
            return Path(os.path.abspath(os.path.normpath(os.fspath(path))))

        legacy = lexical(home / ".funes" / "integrations" / "pi")

        def source_path(entry: object) -> Path | None:
            if isinstance(entry, str):
                source = entry
            elif isinstance(entry, dict):
                source = entry.get("source")
            else:
                source = None
            if not isinstance(source, str) or not source.strip():
                return None
            source = source.strip()
            if source == "~":
                candidate = home
            elif source.startswith("~/"):
                candidate = home / source[2:]
            else:
                candidate = Path(source)
                if not candidate.is_absolute():
                    candidate = settings.parent / candidate
            return lexical(candidate)

        retained = [entry for entry in packages if source_path(entry) != legacy]
        if len(retained) == len(packages):
            return False
        data["packages"] = retained

        # Replacing the symlink itself would detach dotfile-managed settings.
        # Atomically replace its final target instead, while holding Pi's lock
        # at the public settings path.
        write_path = settings.resolve() if settings.is_symlink() else settings
        mode = write_path.stat().st_mode & 0o777
        fd, temporary = tempfile.mkstemp(prefix=f".{write_path.name}.tmp-", dir=write_path.parent)
        temporary_path = Path(temporary)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_path, mode)
            os.replace(temporary_path, write_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return True
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


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
    result: dict[str, str | bool] = {
        "pi_extension": str(install_pi(home)),
        "pi_legacy_package_removed": remove_legacy_pi_package(home),
    }
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
