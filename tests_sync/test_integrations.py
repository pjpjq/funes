import json

import shutil
import subprocess

from sync.integrations import (
    install_all,
    remove_legacy_pi_package,
    retire_legacy_claude_plugin,
)


def test_remove_legacy_pi_package_preserves_unrelated_settings(tmp_path):
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "packages": [
                    "../../.funes/integrations/pi",
                    {"source": "../../.funes/integrations/pi", "autoload": True},
                    "github:user/keep-me",
                ],
                "theme": "dark",
            }
        ),
        encoding="utf-8",
    )

    assert remove_legacy_pi_package(tmp_path) is True
    assert json.loads(settings.read_text(encoding="utf-8")) == {
        "packages": ["github:user/keep-me"],
        "theme": "dark",
    }


def test_remove_legacy_pi_package_is_idempotent_and_preserves_invalid_json(tmp_path):
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text('{"packages":["github:user/keep-me"]}\n', encoding="utf-8")
    assert remove_legacy_pi_package(tmp_path) is False
    original = settings.read_bytes()
    settings.write_text("not-json\n", encoding="utf-8")
    assert remove_legacy_pi_package(tmp_path) is False
    assert settings.read_bytes() == b"not-json\n"
    assert original != settings.read_bytes()


def test_remove_legacy_pi_package_honors_pi_lock(tmp_path):
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text('{"packages":["../../.funes/integrations/pi"]}\n', encoding="utf-8")
    lock = settings.with_name("settings.json.lock")
    lock.mkdir()

    assert remove_legacy_pi_package(tmp_path) is False
    assert json.loads(settings.read_text(encoding="utf-8"))["packages"] == [
        "../../.funes/integrations/pi"
    ]


def test_remove_legacy_pi_package_preserves_settings_symlink(tmp_path):
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    target = tmp_path / "dotfiles" / "pi-settings.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        '{"packages":["../../.funes/integrations/pi","github:user/keep-me"]}\n',
        encoding="utf-8",
    )
    settings.parent.mkdir(parents=True)
    settings.symlink_to(target)

    assert remove_legacy_pi_package(tmp_path) is True
    assert settings.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8"))["packages"] == [
        "github:user/keep-me"
    ]


def test_remove_legacy_pi_package_does_not_follow_package_aliases(tmp_path):
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    settings.parent.mkdir(parents=True)
    legacy = tmp_path / ".funes" / "integrations" / "pi"
    actual = tmp_path / "packages" / "different-source"
    actual.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    legacy.symlink_to(actual, target_is_directory=True)
    settings.write_text(
        json.dumps({"packages": ["../../.funes/integrations/pi", str(actual)]}),
        encoding="utf-8",
    )

    assert remove_legacy_pi_package(tmp_path) is True
    assert json.loads(settings.read_text(encoding="utf-8"))["packages"] == [
        str(actual)
    ]


def test_remove_legacy_pi_package_ignores_non_object_json(tmp_path):
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text("[]\n", encoding="utf-8")
    assert remove_legacy_pi_package(tmp_path) is False
    assert settings.read_text(encoding="utf-8") == "[]\n"



def test_retire_legacy_claude_plugin_noop_when_not_installed(tmp_path):
    # Case 1: no settings file exists
    assert retire_legacy_claude_plugin(tmp_path) == "noop"

    # Case 2: settings exists without enabledPlugins
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
    assert retire_legacy_claude_plugin(tmp_path) == "noop"
    assert json.loads(settings.read_text(encoding="utf-8")) == {"theme": "dark"}

    # Case 3: enabledPlugins exists with other plugins but not funes@huggingface
    settings.write_text(
        json.dumps({"enabledPlugins": {"other@marketplace": True}, "theme": "dark"}),
        encoding="utf-8",
    )
    assert retire_legacy_claude_plugin(tmp_path) == "noop"
    assert json.loads(settings.read_text(encoding="utf-8")) == {
        "enabledPlugins": {"other@marketplace": True},
        "theme": "dark",
    }


def test_retire_legacy_claude_plugin_noop_when_already_disabled(tmp_path):
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "enabledPlugins": {
                    "funes@huggingface": False,
                    "other@marketplace": True,
                },
                "theme": "dark",
            }
        ),
        encoding="utf-8",
    )
    mtime_before = settings.stat().st_mtime_ns
    assert retire_legacy_claude_plugin(tmp_path) == "noop"
    assert settings.stat().st_mtime_ns == mtime_before
    assert json.loads(settings.read_text(encoding="utf-8")) == {
        "enabledPlugins": {
            "funes@huggingface": False,
            "other@marketplace": True,
        },
        "theme": "dark",
    }


def test_retire_legacy_claude_plugin_calls_cli_and_preserves_other_plugins(tmp_path, monkeypatch):
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "enabledPlugins": {
                    "funes@huggingface": True,
                    "keep-me@marketplace": True,
                },
                "theme": "dark",
            }
        ),
        encoding="utf-8",
    )

    recorded = {}

    def fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        # Simulate official CLI: disable funes@huggingface while keeping all other plugins/settings
        cur = json.loads(settings.read_text(encoding="utf-8"))
        cur["enabledPlugins"]["funes@huggingface"] = False
        settings.write_text(json.dumps(cur), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/local/bin/claude" if cmd == "claude" else None)
    monkeypatch.setattr(subprocess, "run", fake_run)

    outcome = retire_legacy_claude_plugin(tmp_path)
    assert outcome == "disabled"
    assert recorded["cmd"] == ["claude", "plugin", "disable", "--scope", "user", "funes@huggingface"]
    assert recorded["kwargs"]["env"]["HOME"] == str(tmp_path)
    assert recorded["kwargs"]["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path / ".claude")
    assert recorded["kwargs"]["timeout"] <= 15.0

    # Ensure other plugins and settings are preserved, never deleted/uninstalled
    saved = json.loads(settings.read_text(encoding="utf-8"))
    assert saved["enabledPlugins"]["keep-me@marketplace"] is True
    assert saved["enabledPlugins"]["funes@huggingface"] is False
    assert saved["theme"] == "dark"


def test_retire_legacy_claude_plugin_reports_cli_failure(tmp_path, monkeypatch):
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps({"enabledPlugins": {"funes@huggingface": True}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/local/bin/claude" if cmd == "claude" else None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="failed to disable"),
    )
    assert retire_legacy_claude_plugin(tmp_path) == "failed"


def test_retire_legacy_claude_plugin_reports_missing_cli(tmp_path, monkeypatch):
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps({"enabledPlugins": {"funes@huggingface": True}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    assert retire_legacy_claude_plugin(tmp_path) == "failed"


def test_retire_legacy_claude_plugin_handles_timeout(tmp_path, monkeypatch):
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps({"enabledPlugins": {"funes@huggingface": True}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/local/bin/claude" if cmd == "claude" else None)

    def fake_timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kw.get("timeout", 15.0))

    monkeypatch.setattr(subprocess, "run", fake_timeout)
    assert retire_legacy_claude_plugin(tmp_path) == "timeout"


def test_retire_legacy_claude_plugin_honors_explicit_config_dir(tmp_path, monkeypatch):
    custom_dir = tmp_path / "custom_claude_dir"
    settings = custom_dir / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps({"enabledPlugins": {"funes@huggingface": True}}),
        encoding="utf-8",
    )
    recorded = {}

    def fake_run(cmd, **kwargs):
        recorded["kwargs"] = kwargs
        cur = json.loads(settings.read_text(encoding="utf-8"))
        cur["enabledPlugins"]["funes@huggingface"] = False
        settings.write_text(json.dumps(cur), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/local/bin/claude" if cmd == "claude" else None)
    monkeypatch.setattr(subprocess, "run", fake_run)

    outcome = retire_legacy_claude_plugin(tmp_path, claude_config_dir=custom_dir)
    assert outcome == "disabled"
    assert recorded["kwargs"]["env"]["CLAUDE_CONFIG_DIR"] == str(custom_dir)


def test_retire_legacy_claude_plugin_reports_failed_when_cli_succeeds_but_not_written_to_disk(tmp_path, monkeypatch):
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps({"enabledPlugins": {"funes@huggingface": True}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/local/bin/claude" if cmd == "claude" else None)
    # CLI returns 0 without writing changes to settings.json
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))

    assert retire_legacy_claude_plugin(tmp_path) == "failed"


def test_retire_legacy_claude_plugin_reports_invalid_settings(tmp_path):
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)

    # Corrupt JSON syntax
    settings.write_text("{bad json\n", encoding="utf-8")
    assert retire_legacy_claude_plugin(tmp_path) == "invalid_settings"

    # Non-dict JSON root
    settings.write_text("[]\n", encoding="utf-8")
    assert retire_legacy_claude_plugin(tmp_path) == "invalid_settings"

    # Non-dict enabledPlugins
    settings.write_text(json.dumps({"enabledPlugins": "invalid"}), encoding="utf-8")
    assert retire_legacy_claude_plugin(tmp_path) == "invalid_settings"


def test_retire_legacy_claude_plugin_reports_read_failed(tmp_path):
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.mkdir()
    assert retire_legacy_claude_plugin(tmp_path) == "read_failed"


def test_retire_legacy_claude_plugin_expands_tilde_in_config_path(tmp_path, monkeypatch):
    custom_claude = tmp_path / "tilde_claude"
    settings = custom_claude / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps({"enabledPlugins": {"funes@huggingface": True}}),
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        cur = json.loads(settings.read_text(encoding="utf-8"))
        cur["enabledPlugins"]["funes@huggingface"] = False
        settings.write_text(json.dumps(cur), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/local/bin/claude" if cmd == "claude" else None)
    monkeypatch.setattr(subprocess, "run", fake_run)

    # Test expanding ~ in claude_config_dir
    outcome = retire_legacy_claude_plugin(tmp_path, claude_config_dir="~/tilde_claude")
    assert outcome == "disabled"
    assert json.loads(settings.read_text(encoding="utf-8"))["enabledPlugins"]["funes@huggingface"] is False

    # Also test expanding ~ in CLAUDE_CONFIG_DIR env var
    settings.write_text(
        json.dumps({"enabledPlugins": {"funes@huggingface": True}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "~/tilde_claude")
    outcome = retire_legacy_claude_plugin(tmp_path)
    assert outcome == "disabled"
    assert json.loads(settings.read_text(encoding="utf-8"))["enabledPlugins"]["funes@huggingface"] is False


def test_install_all_returns_explicit_claude_outcome(tmp_path, monkeypatch):
    import sync.integrations as integrations_mod

    assert not hasattr(integrations_mod, "disable_legacy_claude_plugin")
    assert not hasattr(integrations_mod, "remove_legacy_claude_plugin")

    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    res = install_all(tmp_path)
    assert "claude_legacy_plugin_retired" in res
    assert "claude_legacy_plugin_disabled" not in res
    assert res["claude_legacy_plugin_retired"] == "noop"
