import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "bin" / "funes-sync"


def _fake_python(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nprintf "%s\\n" "$0" "$@" > "$CAPTURE"\n')
    path.chmod(0o755)
    return path


def _run(tmp_path: Path, *, python: Path | None = None) -> list[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    capture = tmp_path / "capture"
    env = {
        **os.environ,
        "HOME": str(home),
        "CAPTURE": str(capture),
        "FUNES_API_TOKEN": "test-app-token",
        "FUNES_HF_TOKEN": "test-hub-token",
    }
    env.pop("PYTHON", None)
    if python is not None:
        env["PYTHON"] = str(python)
    subprocess.run([str(LAUNCHER), "status"], env=env, check=True)
    return capture.read_text().splitlines()


def test_launcher_prefers_managed_venv_for_noninteractive_shell(tmp_path):
    venv_python = _fake_python(
        tmp_path / "home" / ".local" / "share" / "funes-sync" / "venv" / "bin" / "python"
    )

    assert _run(tmp_path) == [str(venv_python), "-m", "sync", "status"]


def test_launcher_honors_explicit_python(tmp_path):
    _fake_python(
        tmp_path / "home" / ".local" / "share" / "funes-sync" / "venv" / "bin" / "python"
    )
    explicit_python = _fake_python(tmp_path / "explicit" / "python")

    assert _run(tmp_path, python=explicit_python) == [
        str(explicit_python),
        "-m",
        "sync",
        "status",
    ]
