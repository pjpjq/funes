import json
from dataclasses import replace

from sync.config import Config
from sync.daemon import SyncDaemon
from sync.discovery import Source, _pi_project_for
from sync.store import Store


def test_pi_project_comes_from_header_not_encoded_folder(tmp_path):
    repo = tmp_path / "code" / "a-b"
    (repo / ".git").mkdir(parents=True)
    cwd = repo / "subdirectory"
    cwd.mkdir()
    session = tmp_path / "--ambiguous-encoded-path--" / "session.jsonl"
    session.parent.mkdir()
    session.write_text(json.dumps({"type": "session", "id": "pi-test", "cwd": str(cwd)}) + "\n")
    assert _pi_project_for(session) == str(repo)
    session.write_text('{"type":"session","id":"no-cwd"}\n')
    assert _pi_project_for(session) == ""


def test_pi_does_not_interpret_relative_or_message_cwd_as_project(tmp_path):
    session = tmp_path / "session.jsonl"
    for header in ({"type": "session", "cwd": "relative"}, {"type": "message", "cwd": str(tmp_path)}):
        session.write_text(json.dumps(header) + "\n")
        assert _pi_project_for(session) == ""


def test_source_project_update_reconciles_once_without_duplicate_records(tmp_path, monkeypatch):
    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    path = tmp_path / "pi.jsonl"
    path.write_text(
        '{"type":"session","id":"pi-context"}\n'
        '{"type":"message","id":"message-1","message":{"role":"user","content":"保留原文"}}\n'
    )
    source = Source("pi:~/pi.jsonl", "pi", path, cfg.device_id)
    current = [source]
    monkeypatch.setattr("sync.daemon.discover_sources", lambda _: current)
    store = Store(config=cfg)
    try:
        daemon = SyncDaemon(cfg, store, object())
        daemon.scan_once()
        record = store.pending()[0]
        record_id = record["record_id"]
        old_version = json.loads(record["payload"])["source_version"]
        store.ack([record_id])
        current[:] = [replace(source, project=str(tmp_path / "project"))]
        daemon._invalidate_source_cache()
        assert daemon.scan_once() == 1
        updated = store.pending()[0]
        payload = json.loads(updated["payload"])
        assert updated["record_id"] == record_id
        assert payload["project"] == str(tmp_path / "project")
        assert payload["raw_text"] == "保留原文"
        assert payload["source_version"] != old_version
        assert store.db.execute("SELECT count(*) FROM records").fetchone()[0] == 1
        store.ack([record_id])
        for _ in range(10):
            assert daemon.scan_once() == 0
        assert store.pending_count() == 0
    finally:
        store.close()
