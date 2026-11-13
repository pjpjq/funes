"""Static verification for Space entrypoint and Dockerfile consistency."""
import hashlib
from pathlib import Path
import py_compile
import shlex
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILES = (
    REPO_ROOT / "deploy" / "funes-space" / "Dockerfile",
    REPO_ROOT / "space" / "Dockerfile",
    REPO_ROOT / "service" / "Dockerfile",
)


def test_dockerfiles_copy_canonical_space_server():
    """Verify all Space Dockerfiles copy space/server.py and not root server.py."""
    for path in DOCKERFILES:
        assert path.exists(), f"Missing Dockerfile: {path}"
        text = path.read_text(encoding="utf-8")
        assert (
            "COPY space/server.py /app/server.py" in text
        ), f"{path.relative_to(REPO_ROOT)} must COPY space/server.py /app/server.py"
        assert (
            "COPY server.py /app/server.py" not in text
        ), f"{path.relative_to(REPO_ROOT)} should not COPY root server.py directly"


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_runtime_copy_contains_importable_canonical_ab_and_source_package(dockerfile, tmp_path):
    """Stage only final-image COPY files, then import without the checkout/PYTHONPATH.

    space is an implicit namespace package: copying its controller under
    /app/space is sufficient, without inventing an __init__.py for the package.
    This test never builds Rust/Docker, starts a server, or calls a provider.
    """
    text = dockerfile.read_text(encoding="utf-8")
    runtime_stage = text.rsplit("\nFROM ", 1)[-1]
    assert "COPY space/canonical_ab.py /app/space/canonical_ab.py" in runtime_stage
    assert "COPY service /app/service" in runtime_stage
    assert "pip install --no-cache-dir -r /app/service/requirements.txt" in runtime_stage
    runtime = tmp_path / "app"
    runtime.mkdir()
    for line in runtime_stage.splitlines():
        if not line.lstrip().startswith("COPY "):
            continue
        fields = shlex.split(line)
        if fields[1].startswith("--from="):
            continue
        assert len(fields) == 3, f"Extend staging for this COPY directive: {line}"
        source = REPO_ROOT / fields[1]
        assert source.exists(), f"Docker build context is missing {fields[1]}"
        target = runtime / Path(fields[2]).relative_to("/app")
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    result = subprocess.run(
        [
            sys.executable, "-I", "-c",
            "import pathlib, runpy, sys; "
            f"runtime = pathlib.Path({str(runtime)!r}); "
            "sys.path.insert(0, str(runtime)); "
            "runpy.run_path(str(runtime / 'server.py'), run_name='_image_import_test'); "
            "import space.canonical_ab, service; "
            "assert pathlib.Path(space.canonical_ab.__file__) == runtime / 'space/canonical_ab.py'; "
            "assert pathlib.Path(service.__file__) == runtime / 'service/__init__.py'; "
            "assert space.canonical_ab.CanonicalABController().status()['phase'] == 'idle'",
        ],
        cwd=runtime, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_production_space_enables_safe_voyage_batching_and_pacing():
    """Keep paid-tier production batching bounded after the live A/B probe."""
    required = (
        "FUNES_CANONICAL_INDEX_BATCH=512",
        "FUNES_CANONICAL_INDEX_REQUEST_ROWS=64",
        "FUNES_CANONICAL_INDEX_MAX_CHARS=192000",
        "FUNES_CANONICAL_INDEX_MIN_REQUEST_INTERVAL=1",
        "FUNES_VOYAGE_CONCURRENCY=4",
        "FUNES_VOYAGE_MAX_REQUEST_TOKENS=9000",
        "FUNES_VOYAGE_TOKENS_PER_MINUTE=120000",
        "FUNES_VOYAGE_MIN_REQUEST_INTERVAL=0",
    )
    for path in (
        REPO_ROOT / "deploy" / "funes-space" / "Dockerfile",
        REPO_ROOT / "space" / "Dockerfile",
    ):
        text = path.read_text(encoding="utf-8")
        for value in required:
            assert value in text, f"{value} missing from {path.relative_to(REPO_ROOT)}"


def test_root_server_synced_with_space_server():
    """Verify root server.py matches space/server.py byte-for-byte to prevent drift."""
    space_server = (REPO_ROOT / "space" / "server.py").read_bytes()
    root_server = (REPO_ROOT / "server.py").read_bytes()
    assert hashlib.sha256(root_server).hexdigest() == hashlib.sha256(space_server).hexdigest(), (
        "server.py drifted from space/server.py; run 'cp space/server.py server.py'"
    )


def test_server_compilation():
    """Verify space/server.py and server.py compile cleanly without syntax errors."""
    py_compile.compile(str(REPO_ROOT / "space" / "server.py"), doraise=True)
    py_compile.compile(str(REPO_ROOT / "server.py"), doraise=True)
