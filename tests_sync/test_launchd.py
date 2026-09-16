from __future__ import annotations

import os
import plistlib
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sync.config import Config
from sync.launchd import (
    LABEL,
    NATIVE_LABEL,
    StateMigrationBlocked,
    helper_paths,
    install,
    native_plist_path,
    plist_path,
    plist_paths,
    persist_keychain_credentials,
    restart,
    start,
    status,
    stop,
    uninstall,
)


def configured(tmp_path: Path) -> Config:
    return Config(
        tmp_path,
        tmp_path / "state",
        tmp_path / "config.toml",
        remote_url="https://memory.example",
        interval=37,
        batch_size=7,
        native_memory="owner/memory",
        native_primary=True,
        native_bin="/opt/funes/bin/funes",
    )


def successful_run(calls):
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="")

    return run


class LaunchctlRunner:
    def __init__(self, calls, fail_bootstrap_label=None, fail_bootstrap_times=1):
        self.calls = calls
        self.loaded = set()
        self.fail_bootstrap_label = fail_bootstrap_label
        self.fail_bootstrap_times = fail_bootstrap_times

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[0] != "launchctl":
            return SimpleNamespace(returncode=0, stdout="")
        if args[1] == "print":
            label = args[-1].rsplit("/", 1)[-1]
            code = 0 if label in self.loaded else 113
            return SimpleNamespace(returncode=code, stdout="")
        target = args[-1]
        if args[1] == "bootout" and target.startswith("gui/"):
            label = target.rsplit("/", 1)[-1]
        else:
            path = Path(target)
            label = plistlib.loads(path.read_bytes())["Label"]
        if args[1] == "bootstrap":
            if label == self.fail_bootstrap_label and self.fail_bootstrap_times:
                self.fail_bootstrap_times -= 1
                return SimpleNamespace(returncode=5, stdout="")
            self.loaded.add(label)
        elif args[1] == "bootout":
            self.loaded.discard(label)
        return SimpleNamespace(returncode=0, stdout="")


def test_install_writes_dual_plists_and_atomic_secret_free_helpers(tmp_path, monkeypatch):
    config = configured(tmp_path)
    calls = []
    runner = LaunchctlRunner(calls)
    replacements = []
    real_replace = os.replace
    monkeypatch.setenv("FUNES_API_TOKEN", "api-secret-must-not-be-written")
    monkeypatch.setenv("FUNES_HF_TOKEN", "hub-secret-must-not-be-written")
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)

    def replace(source, destination):
        replacements.append(Path(destination))
        real_replace(source, destination)

    monkeypatch.setattr("sync.launchd.os.replace", replace)

    installed = install(config=config, python="/usr/bin/python3")

    main_path = plist_path(config.home)
    native_path = native_plist_path(config.home)
    native_script, warm_helper = helper_paths(config)
    assert installed == (main_path, native_path)
    assert set(replacements) == {main_path, native_path, native_script, warm_helper}
    assert not list(config.state_dir.glob(".*.tmp-*"))

    main = plistlib.loads(main_path.read_bytes())
    native = plistlib.loads(native_path.read_bytes())
    assert main["Label"] == LABEL
    assert main["EnvironmentVariables"]["FUNES_MEMORY_ONLY"] == "1"
    assert main["EnvironmentVariables"]["FUNES_CONFIG"] == str(config.config_path)
    assert main["EnvironmentVariables"]["FUNES_STATE_DIR"] == str(config.state_dir)
    assert main["EnvironmentVariables"]["FUNES_REMOTE_URL"] == config.remote_url
    assert native["Label"] == NATIVE_LABEL
    assert native["ProgramArguments"] == [str(native_script)]
    assert native["ThrottleInterval"] == 30
    assert native["EnvironmentVariables"] == {
        "FUNES_BIN": config.native_bin,
        "FUNES_NATIVE_BACKFILL_RECONCILE_INTERVAL": str(config.interval),
        "FUNES_NATIVE_MEMORY": config.native_memory,
        "FUNES_NATIVE_WARM_HELPER": str(warm_helper),
        "FUNES_REMOTE_URL": config.remote_url,
        "FUNES_SYNC_STATE_DIR": str(config.state_dir),
        "HOME": str(config.home),
        "PYTHON": "/usr/bin/python3",
        "USER": os.environ.get("USER") or str(os.getuid()),
    }
    assert native_script.read_bytes() == (
        Path(__file__).parents[1] / "deploy/funes-sync/native-backfill.sh"
    ).read_bytes()
    assert warm_helper.read_bytes() == (
        Path(__file__).parents[1] / "deploy/funes-sync/warm-space.py"
    ).read_bytes()
    assert native_script.stat().st_mode & 0o777 == 0o700
    assert warm_helper.stat().st_mode & 0o777 == 0o700
    installed_bytes = b"".join(
        path.read_bytes() for path in (main_path, native_path, native_script, warm_helper)
    )
    assert b"api-secret-must-not-be-written" not in installed_bytes
    assert b"hub-secret-must-not-be-written" not in installed_bytes
    commands = [entry[0] for entry in calls]
    assert sum(command[1] == "print" for command in commands) == 6
    assert sum(command[1] == "bootstrap" for command in commands) == 2


def test_install_without_native_primary_removes_legacy_helper(tmp_path, monkeypatch):
    config = configured(tmp_path)
    config.native_primary = False
    main_path, native_path = plist_paths(config.home)
    native_script, warm_helper = helper_paths(config)
    old_plists = owned_plist_bytes(config)
    native_script.parent.mkdir(parents=True, exist_ok=True)
    native_script.write_text("legacy native helper\n", encoding="utf-8")
    warm_helper.write_text("legacy warm helper\n", encoding="utf-8")
    calls = []
    runner = LaunchctlRunner(calls)
    runner.loaded.update({LABEL, NATIVE_LABEL})
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)

    installed = install(config=config, python="/usr/bin/python3")

    assert installed == (main_path,)
    assert main_path.exists()
    assert not native_path.exists()
    assert not native_script.exists()
    assert not warm_helper.exists()
    assert runner.loaded == {LABEL}
    assert old_plists[native_path] != main_path.read_bytes()
    commands = [entry[0] for entry in calls]
    assert sum(command[1] == "bootout" for command in commands) == 2
    assert sum(command[1] == "bootstrap" for command in commands) == 1


def test_lifecycle_without_native_primary_manages_only_main(tmp_path, monkeypatch):
    config = configured(tmp_path)
    config.native_primary = False
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)
    install(config=config)
    calls.clear()

    stop(config=config)
    start(config=config)
    restart(config=config)

    commands = [entry[0] for entry in calls]
    mutated = [command for command in commands if command[1] in {"bootout", "bootstrap"}]
    assert all(command[-1] == str(plist_path(config.home)) for command in mutated)
    assert sum(command[1] == "bootout" for command in mutated) == 2
    assert sum(command[1] == "bootstrap" for command in mutated) == 2
    assert runner.loaded == {LABEL}


def test_disabling_native_primary_removes_helpers_from_old_state_dir(
    tmp_path, monkeypatch
):
    config = configured(tmp_path)
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)
    install(config=config)
    old_helpers = helper_paths(config)
    assert all(path.exists() for path in old_helpers)

    config.state_dir = tmp_path / "new-state"
    config.native_primary = False
    installed = install(config=config)

    assert installed == (plist_path(config.home),)
    assert all(not path.exists() for path in old_helpers)
    assert all(not path.exists() for path in helper_paths(config))
    assert runner.loaded == {LABEL}


def test_disabling_native_primary_rolls_back_files_and_agents_on_failure(
    tmp_path, monkeypatch
):
    config = configured(tmp_path)
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)
    install(config=config)
    artifacts = (*plist_paths(config.home), *helper_paths(config))
    before = {path: path.read_bytes() for path in artifacts}
    config.native_primary = False
    runner.fail_bootstrap_label = LABEL
    runner.fail_bootstrap_times = 1

    with pytest.raises(RuntimeError, match="launchctl bootstrap failed"):
        install(config=config)

    assert runner.loaded == {LABEL, NATIVE_LABEL}
    assert {path: path.read_bytes() for path in artifacts} == before


def test_disabling_native_primary_stops_loaded_helper_with_missing_plist(
    tmp_path, monkeypatch
):
    config = configured(tmp_path)
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)
    install(config=config)
    native_plist_path(config.home).unlink()
    config.native_primary = False
    calls.clear()

    started = start(config=config)

    assert started == (plist_path(config.home),)
    assert runner.loaded == {LABEL}
    assert any(
        command[:2] == ["launchctl", "bootout"]
        and command[-1] == f"gui/{os.getuid()}/{NATIVE_LABEL}"
        for command, _kwargs in calls
    )


def test_restarting_without_native_primary_stops_loaded_helper_with_missing_plist(
    tmp_path, monkeypatch
):
    config = configured(tmp_path)
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)
    install(config=config)
    native_plist_path(config.home).unlink()
    config.native_primary = False
    calls.clear()

    restarted = restart(config=config)

    assert restarted == (plist_path(config.home),)
    assert runner.loaded == {LABEL}
    assert any(
        command[:2] == ["launchctl", "bootout"]
        and command[-1] == f"gui/{os.getuid()}/{NATIVE_LABEL}"
        for command, _kwargs in calls
    )


def test_native_reinstall_preserves_helpers_when_state_paths_alias(
    tmp_path, monkeypatch
):
    real_state = tmp_path / "real-state"
    real_state.mkdir()
    alias_state = tmp_path / "alias-state"
    alias_state.symlink_to(real_state, target_is_directory=True)
    config = configured(tmp_path)
    config.state_dir = alias_state
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)
    install(config=config)

    config.state_dir = real_state
    install(config=config)

    assert all(path.exists() for path in helper_paths(config))
    assert runner.loaded == {LABEL, NATIVE_LABEL}


def test_uninstall_removes_helpers_from_old_state_dir(tmp_path, monkeypatch):
    config = configured(tmp_path)
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)
    install(config=config)
    old_helpers = helper_paths(config)
    config.state_dir = tmp_path / "new-state"
    config.native_primary = False

    uninstall(config=config)

    assert all(not path.exists() for path in old_helpers)
    assert all(not path.exists() for path in helper_paths(config))
    assert runner.loaded == set()


def test_install_conflict_is_detected_before_any_change(tmp_path, monkeypatch):
    config = configured(tmp_path)
    main_path = plist_path(config.home)
    native_path = native_plist_path(config.home)
    main_path.parent.mkdir(parents=True)
    original = plistlib.dumps({"Label": LABEL, "sentinel": "unchanged"})
    main_path.write_bytes(original)
    native_path.write_bytes(plistlib.dumps({"Label": "com.example.unrelated"}))
    keychain_calls = []
    launchctl_calls = []
    monkeypatch.setattr(
        "sync.launchd.persist_keychain_credentials", lambda: keychain_calls.append(True)
    )
    monkeypatch.setattr(
        "sync.launchd.subprocess.run", successful_run(launchctl_calls)
    )

    with pytest.raises(RuntimeError, match="unrelated LaunchAgent"):
        install(config=config)

    assert main_path.read_bytes() == original
    assert keychain_calls == []
    assert launchctl_calls == []
    assert not any(path.exists() for path in helper_paths(config))


def test_lifecycle_manages_both_labels_and_preserves_state_and_keychain(
    tmp_path, monkeypatch
):
    config = configured(tmp_path)
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)
    install(config=config)
    sentinel = config.state_dir / "sync.db"
    sentinel.write_text("state survives", encoding="utf-8")
    calls.clear()

    stop(config=config)
    start(config=config)
    restart(config=config)
    observed = status(config=config)

    commands = [entry[0] for entry in calls]
    assert sum(command[1] == "bootout" for command in commands) == 4
    assert sum(command[1] == "bootstrap" for command in commands) == 4
    assert [command[-1] for command in commands[-2:]] == [
        f"gui/{os.getuid()}/{LABEL}",
        f"gui/{os.getuid()}/{NATIVE_LABEL}",
    ]
    assert observed == {
        LABEL: {
            "path": str(plist_path(config.home)),
            "installed": True,
            "loaded": True,
        },
        NATIVE_LABEL: {
            "path": str(native_plist_path(config.home)),
            "installed": True,
            "loaded": True,
        },
    }

    calls.clear()
    removed = uninstall(config=config)
    assert removed == (plist_path(config.home), native_plist_path(config.home))
    assert all(not path.exists() for path in removed)
    assert all(not path.exists() for path in helper_paths(config))
    assert sentinel.read_text(encoding="utf-8") == "state survives"
    assert all("security" not in command[0] for command, _kwargs in calls)
    assert sum(command[1] == "bootout" for command, _kwargs in calls) == 2


@pytest.mark.parametrize("action", [start, stop, restart])
def test_lifecycle_rejects_conflicting_plist_before_launchctl(
    action, tmp_path, monkeypatch
):
    config = configured(tmp_path)
    main_path = plist_path(config.home)
    native_path = native_plist_path(config.home)
    main_path.parent.mkdir(parents=True)
    main_path.write_bytes(plistlib.dumps({"Label": LABEL}))
    native_path.write_bytes(plistlib.dumps({"Label": "com.example.unrelated"}))
    calls = []
    monkeypatch.setattr("sync.launchd.subprocess.run", successful_run(calls))

    with pytest.raises(RuntimeError, match="unrelated LaunchAgent"):
        action(config=config)

    assert calls == []


def test_uninstall_preserves_files_when_bootout_fails(tmp_path, monkeypatch):
    config = configured(tmp_path)
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)
    install(config=config)
    installed = (*plist_paths(config.home), *helper_paths(config))

    def fail_bootout(args, **kwargs):
        calls.append((args, kwargs))
        if args[1] == "print":
            return SimpleNamespace(returncode=0, stdout="")
        if args[1] == "bootout":
            return SimpleNamespace(returncode=5, stdout="")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr("sync.launchd.subprocess.run", fail_bootout)

    with pytest.raises(RuntimeError, match="launchctl bootout failed"):
        uninstall(config=config)

    assert all(path.exists() for path in installed)


def test_native_binary_falls_back_to_path_then_user_local(tmp_path, monkeypatch):
    from sync.launchd import render_native_plist

    config = configured(tmp_path)
    config.native_bin = ""
    monkeypatch.setattr("sync.launchd.shutil.which", lambda name: "/opt/homebrew/bin/funes")
    assert render_native_plist(config)["EnvironmentVariables"]["FUNES_BIN"] == "/opt/homebrew/bin/funes"
    monkeypatch.setattr("sync.launchd.shutil.which", lambda name: None)
    assert render_native_plist(config)["EnvironmentVariables"]["FUNES_BIN"] == str(
        config.home / ".local/bin/funes"
    )


def test_second_bootstrap_failure_rolls_back_first_agent(tmp_path, monkeypatch):
    config = configured(tmp_path)
    calls = []
    runner = LaunchctlRunner(calls, fail_bootstrap_label=NATIVE_LABEL)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)

    with pytest.raises(RuntimeError, match="launchctl bootstrap failed"):
        install(config=config)

    assert runner.loaded == set()
    assert any(
        command[1] == "bootout" and command[-1] == str(plist_path(config.home))
        for command, _kwargs in calls
    )
    assert all(
        not path.exists() for path in (*plist_paths(config.home), *helper_paths(config))
    )


def owned_plist_bytes(
    config, main_env=None, native_env=None, main_extra=None, native_extra=None
):
    main_data={"Label": LABEL, "EnvironmentVariables": main_env or {}}
    native_data={"Label": NATIVE_LABEL, "EnvironmentVariables": native_env or {}}
    main_data.update(main_extra or {})
    native_data.update(native_extra or {})
    main = plistlib.dumps(main_data)
    native = plistlib.dumps(native_data)
    main_path, native_path = plist_paths(config.home)
    main_path.parent.mkdir(parents=True, exist_ok=True)
    main_path.write_bytes(main)
    native_path.write_bytes(native)
    return {main_path: main, native_path: native}


def test_install_failure_restores_existing_artifacts_and_two_loaded_agents(
    tmp_path, monkeypatch
):
    config = configured(tmp_path)
    old_plists = owned_plist_bytes(
        config,
        {"FUNES_STATE_DIR": str(config.state_dir)},
        {"FUNES_SYNC_STATE_DIR": str(config.state_dir)},
    )
    native_script, warm_helper = helper_paths(config)
    native_script.parent.mkdir(parents=True, exist_ok=True)
    native_script.write_bytes(b"old native helper\n")
    warm_helper.write_bytes(b"old warm helper\n")
    old_helpers = {
        native_script: native_script.read_bytes(),
        warm_helper: warm_helper.read_bytes(),
    }
    calls = []
    runner = LaunchctlRunner(calls, fail_bootstrap_label=NATIVE_LABEL)
    runner.loaded.update({LABEL, NATIVE_LABEL})
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)

    with pytest.raises(RuntimeError, match="launchctl bootstrap failed"):
        install(config=config)

    assert runner.loaded == {LABEL, NATIVE_LABEL}
    for path, content in {**old_plists, **old_helpers}.items():
        assert path.read_bytes() == content


def test_restart_failure_restores_original_two_loaded_agents(tmp_path, monkeypatch):
    config = configured(tmp_path)
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)
    install(config=config)
    before = {
        path: path.read_bytes()
        for path in (*plist_paths(config.home), *helper_paths(config))
    }
    runner.fail_bootstrap_label = NATIVE_LABEL
    runner.fail_bootstrap_times = 1

    with pytest.raises(RuntimeError, match="launchctl bootstrap failed"):
        restart(config=config)

    assert runner.loaded == {LABEL, NATIVE_LABEL}
    assert {path: path.read_bytes() for path in before} == before


def test_all_keychain_commands_have_bounded_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("FUNES_API_TOKEN", "api-value")
    monkeypatch.setenv("FUNES_HF_TOKEN", "hub-value")
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setattr("sync.launchd.shutil.which", lambda name: "/usr/bin/security")
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        service = args[args.index("-s") + 1]
        value = "api-value\n" if service == "funes-api-token" else "hub-value\n"
        return SimpleNamespace(returncode=0, stdout=value)

    monkeypatch.setattr("sync.launchd.subprocess.run", run)

    persist_keychain_credentials()

    assert len(calls) == 4
    assert all(kwargs["timeout"] == 8 for _args, kwargs in calls)


@pytest.mark.parametrize(
    ("env_name", "secret"),
    [("FUNES_API_TOKEN", "api-timeout-secret"), ("FUNES_HF_TOKEN", "hub-timeout-secret")],
)
def test_keychain_timeout_is_safe(env_name, secret, monkeypatch):
    for name in ("FUNES_API_TOKEN", "FUNES_HF_TOKEN", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_name, secret)
    monkeypatch.setattr("sync.launchd.shutil.which", lambda name: "/usr/bin/security")

    def timeout(args, **kwargs):
        assert kwargs["timeout"] == 8
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr("sync.launchd.subprocess.run", timeout)

    with pytest.raises(RuntimeError, match="Keychain command timed out") as exc:
        persist_keychain_credentials()

    assert secret not in str(exc.value)


def test_upgrade_preserves_only_allowlisted_nonsecret_tuning(tmp_path, monkeypatch):
    config = configured(tmp_path)
    config.batch_size = 50
    owned_plist_bytes(
        config,
        {
            "FUNES_STATE_DIR": str(config.state_dir),
            "FUNES_SYNC_BATCH": "5",
            "FUNES_REMOTE_TIMEOUT": "180",
            "FUNES_API_TOKEN": "old-api-secret",
        },
        {
            "FUNES_SYNC_STATE_DIR": str(config.state_dir),
            "FUNES_NATIVE_BACKFILL_PUSH_EVERY": "20",
            "FUNES_NATIVE_BACKFILL_SLEEP": "9",
            "HF_TOKEN": "old-hub-secret",
        },
    )
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)

    install(config=config)

    main = plistlib.loads(plist_path(config.home).read_bytes())["EnvironmentVariables"]
    native = plistlib.loads(native_plist_path(config.home).read_bytes())["EnvironmentVariables"]
    assert main["FUNES_SYNC_BATCH"] == "5"
    assert main["FUNES_REMOTE_TIMEOUT"] == "180"
    assert native["FUNES_NATIVE_BACKFILL_PUSH_EVERY"] == "20"
    assert native["FUNES_NATIVE_BACKFILL_SLEEP"] == "9"
    assert "FUNES_API_TOKEN" not in main
    assert "HF_TOKEN" not in native


def test_explicit_tuning_overrides_previous_plist(tmp_path, monkeypatch):
    config = configured(tmp_path)
    owned_plist_bytes(
        config,
        {
            "FUNES_STATE_DIR": str(config.state_dir),
            "FUNES_SYNC_BATCH": "5",
            "FUNES_REMOTE_TIMEOUT": "180",
        },
        {
            "FUNES_SYNC_STATE_DIR": str(config.state_dir),
            "FUNES_NATIVE_BACKFILL_PUSH_EVERY": "20",
            "FUNES_NATIVE_BACKFILL_SLEEP": "9",
        },
    )
    monkeypatch.setenv("FUNES_SYNC_BATCH", "11")
    monkeypatch.setenv("FUNES_REMOTE_TIMEOUT", "44")
    monkeypatch.setenv("FUNES_NATIVE_BACKFILL_PUSH_EVERY", "3")
    monkeypatch.setenv("FUNES_NATIVE_BACKFILL_SLEEP", "4")
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)

    install(config=config)

    main = plistlib.loads(plist_path(config.home).read_bytes())["EnvironmentVariables"]
    native = plistlib.loads(native_plist_path(config.home).read_bytes())["EnvironmentVariables"]
    assert main["FUNES_SYNC_BATCH"] == "11"
    assert main["FUNES_REMOTE_TIMEOUT"] == "44"
    assert native["FUNES_NATIVE_BACKFILL_PUSH_EVERY"] == "3"
    assert native["FUNES_NATIVE_BACKFILL_SLEEP"] == "4"


def test_configured_tuning_overrides_previous_plist(tmp_path, monkeypatch):
    config = configured(tmp_path)
    config.config_path.write_text(
        "[sync]\n"
        "batch_size = 13\n"
        "remote_timeout = 91\n"
        "native_backfill_push_every = 6\n"
        "native_backfill_sleep = 8\n",
        encoding="utf-8",
    )
    owned_plist_bytes(
        config,
        {
            "FUNES_STATE_DIR": str(config.state_dir),
            "FUNES_SYNC_BATCH": "5",
            "FUNES_REMOTE_TIMEOUT": "180",
        },
        {
            "FUNES_SYNC_STATE_DIR": str(config.state_dir),
            "FUNES_NATIVE_BACKFILL_PUSH_EVERY": "20",
            "FUNES_NATIVE_BACKFILL_SLEEP": "9",
        },
    )
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)

    install(config=config)

    main = plistlib.loads(plist_path(config.home).read_bytes())["EnvironmentVariables"]
    native = plistlib.loads(native_plist_path(config.home).read_bytes())["EnvironmentVariables"]
    assert main["FUNES_SYNC_BATCH"] == "13"
    assert main["FUNES_REMOTE_TIMEOUT"] == "91"
    assert native["FUNES_NATIVE_BACKFILL_PUSH_EVERY"] == "6"
    assert native["FUNES_NATIVE_BACKFILL_SLEEP"] == "8"


def queue_database(state_dir, pending):
    state_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(state_dir / "sync.db") as database:
        database.execute("CREATE TABLE queue(record_id TEXT PRIMARY KEY)")
        database.executemany(
            "INSERT INTO queue(record_id) VALUES(?)",
            ((f"record-{index}",) for index in range(pending)),
        )


def test_cross_state_pending_queue_blocks_install_before_side_effects(
    tmp_path, monkeypatch
):
    config = configured(tmp_path)
    old_state = tmp_path / "old-state"
    queue_database(old_state, 2)
    old_plists = owned_plist_bytes(
        config,
        {"FUNES_STATE_DIR": str(old_state)},
        {"FUNES_SYNC_STATE_DIR": str(old_state)},
    )
    keychain_calls = []
    launchctl_calls = []
    monkeypatch.setattr(
        "sync.launchd.persist_keychain_credentials", lambda: keychain_calls.append(True)
    )
    monkeypatch.setattr("sync.launchd.subprocess.run", successful_run(launchctl_calls))

    with pytest.raises(StateMigrationBlocked) as exc:
        install(config=config)

    warning = exc.value.as_dict()
    assert warning["error"] == "state_migration_blocked"
    assert warning["pending"] == 2
    assert warning["old_state_dir"] == str(old_state)
    assert warning["new_state_dir"] == str(config.state_dir)
    assert keychain_calls == []
    assert launchctl_calls == []
    assert {path: path.read_bytes() for path in old_plists} == old_plists
    assert not any(path.exists() for path in helper_paths(config))


def test_cross_state_zero_queue_moves_only_helpers(tmp_path, monkeypatch):
    config = configured(tmp_path)
    old_state = tmp_path / "old-state"
    queue_database(old_state, 0)
    owned_plist_bytes(
        config,
        {"FUNES_STATE_DIR": str(old_state)},
        {"FUNES_SYNC_STATE_DIR": str(old_state)},
    )
    calls = []
    runner = LaunchctlRunner(calls)
    monkeypatch.setattr("sync.launchd.persist_keychain_credentials", lambda: None)
    monkeypatch.setattr("sync.launchd.subprocess.run", runner)

    install(config=config)

    assert all(path.exists() for path in helper_paths(config))
    assert (old_state / "sync.db").exists()
    assert not (config.state_dir / "sync.db").exists()


def test_cross_state_queue_is_rechecked_after_stopping_old_agents(
    tmp_path, monkeypatch
):
    config = configured(tmp_path)
    old_state = tmp_path / "old-state"
    queue_database(old_state, 0)
    old_plists = owned_plist_bytes(
        config,
        {"FUNES_STATE_DIR": str(old_state)},
        {"FUNES_SYNC_STATE_DIR": str(old_state)},
    )
    calls = []
    runner = LaunchctlRunner(calls)
    runner.loaded.update({LABEL, NATIVE_LABEL})
    inserted = False
    keychain_calls = []

    def race(args, **kwargs):
        nonlocal inserted
        if args[1] == "print" and not inserted:
            inserted = True
            with sqlite3.connect(old_state / "sync.db") as database:
                database.execute("INSERT INTO queue(record_id) VALUES('late-record')")
        return runner(args, **kwargs)

    monkeypatch.setattr(
        "sync.launchd.persist_keychain_credentials", lambda: keychain_calls.append(True)
    )
    monkeypatch.setattr("sync.launchd.subprocess.run", race)

    with pytest.raises(StateMigrationBlocked) as exc:
        install(config=config)

    assert exc.value.pending == 1
    assert runner.loaded == {LABEL, NATIVE_LABEL}
    assert keychain_calls == []
    assert {path: path.read_bytes() for path in old_plists} == old_plists
    assert not any(path.exists() for path in helper_paths(config))


def test_relative_old_state_uses_plist_working_directory(tmp_path, monkeypatch):
    config = configured(tmp_path)
    working = tmp_path / "old-working"
    old_state = working / "relative-state"
    queue_database(old_state, 1)
    owned_plist_bytes(
        config,
        {"FUNES_STATE_DIR": "relative-state"},
        {"FUNES_SYNC_STATE_DIR": str(config.state_dir)},
        main_extra={"WorkingDirectory": str(working)},
    )
    keychain_calls = []
    launchctl_calls = []
    monkeypatch.setattr(
        "sync.launchd.persist_keychain_credentials", lambda: keychain_calls.append(True)
    )
    monkeypatch.setattr("sync.launchd.subprocess.run", successful_run(launchctl_calls))

    with pytest.raises(StateMigrationBlocked) as exc:
        install(config=config)

    assert exc.value.as_dict()["old_state_dir"] == str(old_state)
    assert keychain_calls == []
    assert launchctl_calls == []


@pytest.mark.parametrize(
    ("command", "action_name"),
    [("install", "install"), ("start", "start_agents"), ("restart", "restart_agents")],
)
def test_cli_displays_cross_state_pending_warning(
    command, action_name, tmp_path, monkeypatch, capsys
):
    import sync.cli as cli

    config = configured(tmp_path)
    blocked = StateMigrationBlocked(
        [
            {
                "old_state_dir": str(tmp_path / "old-state"),
                "new_state_dir": str(config.state_dir),
                "pending": 545000,
                "reason": "pending_queue",
            }
        ]
    )

    monkeypatch.setattr(cli.Config, "load", classmethod(lambda cls, home=None: config))
    monkeypatch.setattr(
        cli,
        "Store",
        lambda config: (_ for _ in ()).throw(AssertionError("Store opened before gate")),
    )
    monkeypatch.setattr(
        cli, action_name, lambda **kwargs: (_ for _ in ()).throw(blocked)
    )

    assert cli.main([command]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert '"error": "state_migration_blocked"' in output.err
    assert '"pending": 545000' in output.err


@pytest.mark.parametrize("database_kind", ["corrupt", "dangling"])
def test_unreadable_old_queue_blocks_install(
    database_kind, tmp_path, monkeypatch
):
    config = configured(tmp_path)
    old_state = tmp_path / "old-state"
    old_state.mkdir()
    database_path = old_state / "sync.db"
    if database_kind == "corrupt":
        database_path.write_bytes(b"not a sqlite database")
    else:
        database_path.symlink_to(old_state / "missing-sync.db")
    owned_plist_bytes(
        config,
        {"FUNES_STATE_DIR": str(old_state)},
        {"FUNES_SYNC_STATE_DIR": str(config.state_dir)},
    )
    keychain_calls = []
    launchctl_calls = []
    monkeypatch.setattr(
        "sync.launchd.persist_keychain_credentials", lambda: keychain_calls.append(True)
    )
    monkeypatch.setattr("sync.launchd.subprocess.run", successful_run(launchctl_calls))

    with pytest.raises(StateMigrationBlocked) as exc:
        install(config=config)

    assert exc.value.pending is None
    assert exc.value.as_dict()["migrations"][0]["reason"] == "queue_unreadable"
    assert keychain_calls == []
    assert launchctl_calls == []


def test_dangling_old_state_directory_blocks_install(tmp_path, monkeypatch):
    config = configured(tmp_path)
    old_state = tmp_path / "old-state"
    old_state.symlink_to(tmp_path / "missing-volume", target_is_directory=True)
    owned_plist_bytes(
        config,
        {"FUNES_STATE_DIR": str(old_state)},
        {"FUNES_SYNC_STATE_DIR": str(config.state_dir)},
    )
    keychain_calls = []
    launchctl_calls = []
    monkeypatch.setattr(
        "sync.launchd.persist_keychain_credentials", lambda: keychain_calls.append(True)
    )
    monkeypatch.setattr("sync.launchd.subprocess.run", successful_run(launchctl_calls))

    with pytest.raises(StateMigrationBlocked) as exc:
        install(config=config)

    assert exc.value.pending is None
    assert exc.value.as_dict()["migrations"][0]["reason"] == "queue_unreadable"
    assert keychain_calls == []
    assert launchctl_calls == []
