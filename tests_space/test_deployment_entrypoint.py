"""Static verification for Space entrypoint and Dockerfile consistency."""
import hashlib
from pathlib import Path
import py_compile

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dockerfiles_copy_canonical_space_server():
    """Verify all Space Dockerfiles copy space/server.py and not root server.py."""
    dockerfiles = [
        REPO_ROOT / "deploy" / "funes-space" / "Dockerfile",
        REPO_ROOT / "space" / "Dockerfile",
        REPO_ROOT / "service" / "Dockerfile",
    ]
    for path in dockerfiles:
        assert path.exists(), f"Missing Dockerfile: {path}"
        text = path.read_text(encoding="utf-8")
        assert (
            "COPY space/server.py /app/server.py" in text
        ), f"{path.relative_to(REPO_ROOT)} must COPY space/server.py /app/server.py"
        assert (
            "COPY server.py /app/server.py" not in text
        ), f"{path.relative_to(REPO_ROOT)} should not COPY root server.py directly"


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
