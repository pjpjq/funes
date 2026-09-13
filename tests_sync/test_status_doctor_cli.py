import json

import sync.cli as cli
import sync.client as client_module
from sync.config import Config
from sync.discovery import Source
from sync.parsers import Chunk
from sync.store import Store


def _config(tmp_path):
    return Config(
        tmp_path,
        tmp_path / "state",
        tmp_path / "config.toml",
        remote_url="https://remote-user:remote-secret@memory.example/api?token=query-secret",
        retrieval_language_mode="translate",
        native_memory="https://native-user:native-secret@hub.example/owner/memory?token=native-query",
        native_primary=True,
        native_bin=str(tmp_path / "bin/funes"),
    )


def _seed_synced_source(config):
    store = Store(config=config)
    source = Source(
        "codex:session",
        "codex",
        config.home / ".codex/sessions/session.jsonl",
        config.device_id,
    )
    store.register_source(source)
    store.upsert_chunks(
        [
            Chunk(
                record_id="record",
                source_key=source.source_key,
                kind=source.kind,
                path=str(source.path),
                session_id="session",
                ordinal=0,
                role="user",
                text="raw",
                raw_text="raw",
                source_agent="codex",
                source_type="session",
            )
        ]
    )
    store.ack(["record"])
    store.close()


def test_status_reports_exact_counts_and_safe_remote_state(
    tmp_path, monkeypatch, capsys
):
    config = _config(tmp_path)
    _seed_synced_source(config)
    monkeypatch.setattr(
        cli.Config, "load", classmethod(lambda cls, home=None: config)
    )

    class Client:
        def __init__(self, selected):
            assert selected is config

        def health(self):
            return True

    monkeypatch.setattr(client_module, "SyncClient", Client)
    monkeypatch.setattr(
        cli,
        "_launch_agent_diagnostics",
        lambda cfg: {"com.funes.sync": {"installed": True, "loaded": True}},
    )

    assert cli.main(["status"]) == 0
    raw = capsys.readouterr().out
    status = json.loads(raw)

    assert status["pending"] == status["pending_uploads"] == 0
    assert status["failed_uploads"] == 0
    assert status["last_successful_sync"]
    assert status["discovered"]["codex_sessions"] == 1
    assert status["synced"]["codex_sessions"] == 1
    assert status["remote_ready"] is True
    assert status["remote_status"] == {
        "url": "https://memory.example/api",
        "ready": True,
    }
    assert status["launch_agents"]["com.funes.sync"]["loaded"] is True
    assert status["backfill"]["remote_source_reconciliation"] == {
        "complete": False,
        "cursor_saved": False,
    }
    for secret in ("remote-secret", "query-secret", "native-secret", "native-query"):
        assert secret not in raw


def test_doctor_checks_paths_config_auth_translation_watcher_launchd_and_native(
    tmp_path, monkeypatch, capsys
):
    config = _config(tmp_path)
    config.config_path.write_text(
        '[remote]\nurl = "https://memory.example"\n', encoding="utf-8"
    )
    (tmp_path / ".codex/sessions").mkdir(parents=True)
    pi_sessions = tmp_path / "custom-pi"
    pi_sessions.mkdir()
    claude_home = tmp_path / "custom-claude"
    (claude_home / "projects").mkdir(parents=True)
    binary = tmp_path / "bin/funes"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)
    funes_home = tmp_path / "native-home"
    (funes_home / "memory/chunks.lance").mkdir(parents=True)
    _seed_synced_source(config)

    monkeypatch.setenv("PI_CODING_AGENT_SESSION_DIR", str(pi_sessions))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))
    monkeypatch.setenv("FUNES_HOME", str(funes_home))
    monkeypatch.setenv("TRANSLATION_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("TRANSLATION_API_KEY", "translation-secret")
    monkeypatch.setenv("TRANSLATION_MODEL", "test-model")
    monkeypatch.setattr(
        cli.Config, "load", classmethod(lambda cls, home=None: config)
    )

    class Client:
        def __init__(self, selected):
            assert selected is config

        def _auth_headers(self):
            return {"Authorization": "Bearer keychain-only-secret"}

        def health(self):
            return True

    monkeypatch.setattr(client_module, "SyncClient", Client)
    monkeypatch.setattr(
        cli,
        "_launch_agent_diagnostics",
        lambda cfg: {"com.funes.sync": {"installed": True, "loaded": True}},
    )

    assert cli.main(["doctor"]) == 0
    raw = capsys.readouterr().out
    doctor = json.loads(raw)

    assert doctor["config"]["exists"] is True
    assert doctor["config"]["parseable"] is True
    assert doctor["auth_ready"] is doctor["token_configured"] is True
    assert doctor["remote_status"]["ready"] is True
    assert doctor["translation"] == {
        "mode": "translate",
        "mode_valid": True,
        "provider_required": True,
        "provider_configured": True,
        "provider_partial": False,
        "settings": {"base_url": True, "api_key": True, "model": True},
    }
    assert doctor["watcher"]["dependency"] == "watchdog"
    assert any(item["exists"] for item in doctor["source_paths"]["codex"])
    assert any(
        item["path"] == str(pi_sessions) and item["readable"]
        for item in doctor["source_paths"]["pi"]
    )
    assert any(item["exists"] for item in doctor["source_paths"]["claude"])
    assert doctor["launch_agent_status"]["com.funes.sync"]["loaded"] is True
    assert doctor["native"]["binary_found"] is True
    assert doctor["native"]["binary_executable"] is True
    assert doctor["native"]["index_exists"] is True
    assert doctor["backfill"]["remote_source_reconciliation"]["complete"] is False
    for secret in (
        "keychain-only-secret",
        "translation-secret",
        "remote-secret",
        "query-secret",
        "native-secret",
        "native-query",
    ):
        assert secret not in raw


def test_doctor_helpers_report_invalid_config_and_launchctl_failure(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    config.config_path.write_text("[invalid", encoding="utf-8")
    parsed = cli._config_diagnostic(config.config_path)
    assert parsed["exists"] is True
    assert parsed["parseable"] is False
    assert parsed["error"] == "TOMLDecodeError"

    def unavailable(**_kwargs):
        raise RuntimeError("synthetic launchctl detail")

    monkeypatch.setattr(cli, "launchd_status", unavailable)
    result = cli._launch_agent_diagnostics(config)
    assert set(result) == {"com.funes.sync", "com.funes.native-backfill"}
    assert all(item["loaded"] is None for item in result.values())
    assert all(item["error"] == "RuntimeError" for item in result.values())
    assert "synthetic launchctl detail" not in json.dumps(result)


def test_remote_diagnostic_degrades_unexpected_health_failure():
    class Client:
        def health(self):
            raise ValueError("must not escape")

    assert cli._remote_ready(Client()) is False


def test_safe_locator_removes_query_from_scheme_less_values():
    assert cli._safe_locator("memory.example/api?token=query-secret") == "memory.example/api"
    assert cli._safe_locator("owner/memory?token=native-secret") == "owner/memory"
    assert cli._safe_locator("user:password@memory.example/api") == "memory.example/api"
    assert cli._safe_locator("//user:secret@host/api?token=x") == "//host/api"
    assert cli._safe_locator("https:///user:secret@host/api?token=x") == "<configured>"
    assert cli._safe_locator("https:/remote-user:remote-secret@memory.example/api?token=x") == "<configured>"
    assert cli._safe_locator("/remote-user:remote-secret@memory.example/api?token=x") == "memory.example/api"


def test_doctor_reports_invalid_numeric_config_without_traceback(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / ".config/funes/config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('[sync]\ninterval = "not-a-number"\n', encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert cli.main(["doctor"]) == 2

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert result["config"]["parseable"] is True
    assert result["config"]["loadable"] is False
    assert result["config"]["load_error"] == "ValueError"


def test_doctor_reports_wrong_shape_config_without_traceback(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / ".config/funes/config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("[sources]\ncodex = true\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert cli.main(["doctor"]) == 2

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert result["config"]["parseable"] is True
    assert result["config"]["loadable"] is False
    assert result["config"]["load_error"] == "AttributeError"


def test_reconcile_remote_error_does_not_drain_forever(
    tmp_path, monkeypatch, capsys
):
    config=_config(tmp_path)
    monkeypatch.setattr(cli.Config,"load",classmethod(lambda cls,home=None: config))

    class Daemon:
        def __init__(self,_config,_store):
            pass

        def reconcile_remote_sources(self,_batches,force=False):
            assert force is True
            return {"complete":False,"checked":0,"queued":0,"error":"HTTPError"}

        def drain(self,wait=True):
            raise AssertionError("error path must not drain")

    monkeypatch.setattr(cli,"SyncDaemon",Daemon)

    assert cli.main(["reconcile"]) == 2
    result=json.loads(capsys.readouterr().out)
    assert result["error"] == "HTTPError"
    assert result["remaining"] == 0


def test_reconcile_preserves_inventory_totals_after_completion_check(
    tmp_path, monkeypatch, capsys
):
    config=_config(tmp_path)
    monkeypatch.setattr(cli.Config,"load",classmethod(lambda cls,home=None: config))

    class Daemon:
        def __init__(self,_config,_store):
            self.calls=0

        def reconcile_remote_sources(self,_batches,force=False):
            self.calls+=1
            if self.calls == 1:
                assert force is True
                return {"complete":False,"checked":4,"queued":2}
            return {"complete":True,"checked":0,"queued":0}

        def drain(self,wait=True):
            assert wait is False
            return 2

    monkeypatch.setattr(cli,"SyncDaemon",Daemon)

    assert cli.main(["reconcile"]) == 0
    result=json.loads(capsys.readouterr().out)
    assert result["complete"] is True
    assert result["checked"] == 4
    assert result["queued"] == 2
    assert result["flushed"] == 2
