from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).parents[1] / "deploy" / "funes-sync" / "native-backfill.sh"


@unittest.skipUnless(
    sys.platform == "darwin" and shutil.which("lockf"), "macOS lockf required"
)
class NativeBackfillLockTest(unittest.TestCase):
    def test_second_process_cannot_enter_and_signal_releases_lock(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            invocations = root / "invocations"
            fake = root / "funes"
            fake.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = index ]; then\n'
                f"  printf '%s\\n' \"$PPID\" >>'{invocations}'\n"
                "  sleep 30\n"
                "  printf 'indexed sessions=0 skipped=0 chunks=0\\n'\n"
                'elif [ "$1" = status ]; then\n'
                "  printf 'memory: test\\n'\n"
                "fi\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            env = {
                **os.environ,
                "HOME": str(root),
                "USER": "test",
                "HF_TOKEN": "synthetic-hf-token",
                "FUNES_NATIVE_MEMORY": "test/memory",
                "FUNES_SYNC_STATE_DIR": str(root / "state"),
                "FUNES_NATIVE_BACKFILL_LOG": str(root / "backfill.log"),
                "FUNES_BIN": str(fake),
                "PYTHON": sys.executable,
            }
            first = subprocess.Popen(
                [str(SCRIPT)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 5
                while not invocations.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(invocations.exists())
                second = subprocess.run(
                    [str(SCRIPT)],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                    check=False,
                )
                self.assertEqual(second.returncode, 0)
                self.assertEqual(
                    len(invocations.read_text(encoding="utf-8").splitlines()), 1
                )
            finally:
                if first.poll() is None:
                    os.killpg(first.pid, signal.SIGTERM)
                    first.wait(timeout=5)

            probe = subprocess.run(
                [
                    "/usr/bin/lockf",
                    "-s",
                    "-t",
                    "0",
                    str(root / "state/native-backfill.lockfile"),
                    "/usr/bin/true",
                ],
                check=False,
            )
            self.assertEqual(probe.returncode, 0)

    def test_indexes_archived_subagent_and_legacy_roots(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for path in (
                root / ".codex/archived_sessions",
                root / ".codex/subagents",
                root / ".pi/sessions",
                root / ".claude/history",
                root / "custom-pi",
            ):
                path.mkdir(parents=True)
            invocations = root / "invocations"
            fake = root / "funes"
            fake.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = index ]; then\n'
                f"  printf '%s\\n' \"$*\" >>'{invocations}'\n"
                "  printf 'indexed sessions=0 skipped=0 chunks=0\\n'\n"
                'elif [ "$1" = status ]; then\n'
                "  printf 'memory: test\\n'\n"
                "fi\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            env = {
                **os.environ,
                "HOME": str(root),
                "USER": "test",
                "HF_TOKEN": "synthetic-hf-token",
                "FUNES_NATIVE_MEMORY": "test/memory",
                "FUNES_SYNC_STATE_DIR": str(root / "state"),
                "FUNES_NATIVE_BACKFILL_LOG": str(root / "backfill.log"),
                "FUNES_NATIVE_BACKFILL_RECONCILE_INTERVAL": "30",
                "FUNES_BIN": str(fake),
                "PI_CODING_AGENT_SESSION_DIR": str(root / "custom-pi"),
                "PYTHON": sys.executable,
            }
            process = subprocess.Popen(
                [str(SCRIPT)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    lines = (
                        invocations.read_text(encoding="utf-8").splitlines()
                        if invocations.exists()
                        else []
                    )
                    if len(lines) >= 9:
                        break
                    time.sleep(0.05)
                self.assertIn("index --harness codex --yes", lines)
                self.assertIn(
                    f"index {root}/.codex/archived_sessions --harness codex --yes",
                    lines,
                )
                self.assertIn(
                    f"index {root}/.codex/subagents --harness codex --yes", lines
                )
                self.assertIn(f"index {root}/.pi/sessions --harness pi --yes", lines)
                self.assertIn(f"index {root}/custom-pi --harness pi --yes", lines)
                self.assertIn(
                    f"index {root}/.claude/history --harness claude --yes", lines
                )
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
