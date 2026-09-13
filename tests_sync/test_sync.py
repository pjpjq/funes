import json, os, tempfile, time
from unittest import mock
from pathlib import Path
import pytest
from sync.config import Config
from sync.client import SyncClient
from sync.discovery import discover_sources
from sync.parsers import parse_file
from sync.store import Store
from sync.daemon import SyncDaemon
from sync.launchd import render_plist, persist_keychain_token

def cfg(tmp):
    h=Path(tmp); return Config(h,h/'.state',h/'config.toml',interval=1)
def test_discover_and_parse_all(tmp_path, monkeypatch):
    c=cfg(tmp_path); (tmp_path/'.codex/sessions').mkdir(parents=True); (tmp_path/'.pi').mkdir(); (tmp_path/'.claude/projects').mkdir(parents=True)
    (tmp_path/'.codex/sessions/a.jsonl').write_text(json.dumps({'type':'message','role':'user','content':'hello','session_id':'s'})+'\n')
    (tmp_path/'.pi/p.jsonl').write_text(json.dumps({'role':'assistant','text':'pi answer','subagent_id':'a'})+'\n')
    (tmp_path/'.claude/projects/c.jsonl').write_text(json.dumps({'message':{'role':'assistant','content':'claude'}})+'\n')
    monkeypatch.setenv('HOME',str(tmp_path)); ss=discover_sources(c); assert {x.kind for x in ss}>={'codex','pi','claude'}
    for s in ss:
        if s.path.suffix=='.jsonl': assert parse_file(s)[0].text

def test_store_dedupe_queue(tmp_path):
    c=cfg(tmp_path); s=Store(config=c); p=tmp_path/'x.jsonl'; p.write_text('{"role":"user","text":"x"}\n')
    from sync.discovery import Source
    src=Source('pi:~/x','pi',p,'d'); chunks=parse_file(src); assert s.upsert_chunks(chunks)==1; assert s.upsert_chunks(chunks)==1; assert s.stats()['pending']==1; assert s.pending_count()==1; s.ack([chunks[0].record_id]); assert s.stats()['pending']==0; assert s.pending_count()==0

def test_launchagent(monkeypatch):
    monkeypatch.setenv("FUNES_API_TOKEN", "redacted-test-token")
    d=render_plist('/usr/bin/python3'); assert d['Label']=='com.funes.sync'; assert d['RunAtLoad']
    assert "FUNES_API_TOKEN" not in d["EnvironmentVariables"]


def test_state_dir_can_be_selected_in_toml(tmp_path, monkeypatch):
    cfg_path=tmp_path / "config.toml"
    cfg_path.write_text('[sync]\nstate_dir = "~/.local/share/funes-memory-sync-v2"\n', encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FUNES_CONFIG", str(cfg_path))
    assert Config.load().state_dir == tmp_path / ".local/share/funes-memory-sync-v2"


def test_memory_only_daemon_does_not_require_native_binary(tmp_path):
    c=cfg(tmp_path); c.native_primary=True; c.memory_only=True
    s=Store(config=c)
    d=SyncDaemon(c, s, type("Client", (), {})())
    assert d.native is None
    s.close()


def test_memory_only_companion_still_discovers_raw_agent_sessions(tmp_path):
    c = cfg(tmp_path)
    c.memory_only = True
    for path in (
        tmp_path / ".codex/sessions/codex.jsonl",
        tmp_path / ".pi/agent/sessions/pi.jsonl",
        tmp_path / ".claude/projects/claude.jsonl",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"type":"message","message":{"role":"user","content":"raw"}}\n')

    kinds = {source.kind for source in discover_sources(c)}

    assert {"codex", "pi", "claude"}.issubset(kinds)


def test_native_primary_daemon_does_not_queue_http_records(tmp_path):
    class Native:
        def __init__(self):
            self.calls = 0

        def sync(self):
            self.calls += 1
            return type("Result", (), {"ok": True, "error": ""})()

    c=cfg(tmp_path); c.native_primary=True; c.memory_only=False
    (tmp_path/'.pi').mkdir()
    (tmp_path/'.pi/session.jsonl').write_text(
        '{"role":"user","text":"native only"}\n', encoding="utf-8"
    )
    s=Store(config=c)
    d=SyncDaemon(c, s, type("Client", (), {})())
    native=Native(); d.native=native

    d.run(once=True)

    assert native.calls == 1
    assert s.pending_count() == 0
    assert s.stats()["sources"] == 0
    assert not (c.state_dir / "source-schema-v2.complete").exists()
    s.close()


def test_schema_epoch_reprocesses_unchanged_source_once(tmp_path, monkeypatch):
    from sync.discovery import Source

    c=cfg(tmp_path)
    p=tmp_path/'memory.md'; p.write_text('stable memory', encoding='utf-8')
    source=Source('memory:~/memory.md','persistent',p,c.device_id)
    monkeypatch.setattr('sync.daemon.discover_sources', lambda _config: [source])
    calls=[]
    real_parse=parse_file

    def tracked_parse(item, start=0):
        calls.append(item.source_key)
        return real_parse(item, start)

    monkeypatch.setattr('sync.daemon.parse_file', tracked_parse)
    s=Store(config=c); d=SyncDaemon(c, s, type("Client", (), {})())
    d.scan_once()
    marker=c.state_dir/'source-schema-v2.complete'
    marker.unlink(missing_ok=True)
    calls.clear()

    d.scan_once()

    assert calls == [source.source_key]
    assert marker.exists()
    calls.clear()
    d.scan_once()
    assert calls == []
    s.close()


def test_schema_epoch_bypasses_initial_backfill_eof_seed(tmp_path, monkeypatch):
    from sync.discovery import Source

    c=cfg(tmp_path); c.initial_backfill=False
    p=tmp_path/'memory.md'; p.write_text('migrate metadata', encoding='utf-8')
    source=Source('memory:~/memory.md','persistent',p,c.device_id)
    monkeypatch.setattr('sync.daemon.discover_sources', lambda _config: [source])
    starts=[]
    real_parse=parse_file

    def tracked_parse(item, start=0):
        starts.append(start)
        return real_parse(item, start)

    monkeypatch.setattr('sync.daemon.parse_file', tracked_parse)
    s=Store(config=c)
    s.db.execute(
        "INSERT INTO sources(source_key,kind,path) VALUES(?,?,?)",
        ("legacy", "persistent", str(p)),
    )
    s.db.commit()
    d=SyncDaemon(c, s, type("Client", (), {})())

    d.scan_once()

    assert starts == [0]
    assert (c.state_dir/'source-schema-v2.complete').exists()
    assert (c.state_dir/'initial-backfill.complete').exists()
    s.close()


def test_fresh_install_without_initial_backfill_seeds_eof(tmp_path, monkeypatch):
    from sync.discovery import Source

    c=cfg(tmp_path); c.initial_backfill=False
    p=tmp_path/'memory.md'; p.write_text('existing history', encoding='utf-8')
    source=Source('memory:~/memory.md','persistent',p,c.device_id)
    monkeypatch.setattr('sync.daemon.discover_sources', lambda _config: [source])
    s=Store(config=c); d=SyncDaemon(c, s, type("Client", (), {})())

    assert d.scan_once() == 0

    assert s.pending_count() == 0
    assert s.cursor(source.source_key)["offset"] == p.stat().st_size
    assert (c.state_dir/'initial-backfill.complete').exists()
    assert (c.state_dir/'source-schema-v2.complete').exists()
    s.close()


def test_empty_remote_ack_is_not_durable(tmp_path, monkeypatch):
    class EmptyResponse:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b""

    c = cfg(tmp_path)
    monkeypatch.setenv("FUNES_API_TOKEN", "redacted-test-token")
    monkeypatch.setattr("sync.client.request.urlopen", lambda *args, **kwargs: EmptyResponse())
    monkeypatch.setattr("sync.client.time.sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="remote ingest failed after retries"):
        SyncClient(c).ingest([{"raw_text": "must not be acknowledged"}])


def test_keychain_token_is_stored_and_verified(monkeypatch):
    monkeypatch.setenv("FUNES_API_TOKEN", "redacted-test-token")
    monkeypatch.setenv("USER", "tester")
    calls = []

    class Result:
        returncode = 0
        stdout = "redacted-test-token\n"

    def run(args, **kwargs):
        calls.append(args)
        return Result()

    monkeypatch.setattr("sync.launchd.shutil.which", lambda name: "/usr/bin/security")
    monkeypatch.setattr("sync.launchd.subprocess.run", run)
    assert persist_keychain_token()
    assert calls[0][1:3] == ["add-generic-password", "-U"]
    assert calls[1][1:3] == ["find-generic-password", "-a"]


def test_one_shot_backfill_drains_all_pending(tmp_path):
    class Client:
        def ingest(self, records):
            return {"accepted": len(records)}

        def health(self):
            return False

    c=cfg(tmp_path)
    c.auto_discover=False
    s=Store(config=c)
    p=tmp_path/'x.jsonl'; p.write_text('{"role":"user","text":"one"}\n')
    from sync.discovery import Source
    chunks=parse_file(Source('pi:~/x','pi',p,'d'))
    s.upsert_chunks(chunks)
    # Add a second record so one flush pass cannot accidentally prove the loop.
    p.write_text('{"role":"user","text":"one"}\n{"role":"assistant","text":"two"}\n')
    s.upsert_chunks(parse_file(Source('pi:~/x','pi',p,'d')))
    assert s.stats()['pending'] == 2
    SyncDaemon(c, s, Client()).run(once=True)
    assert s.stats()['pending'] == 0
    s.close()


def test_shutdown_interrupts_one_shot_backfill(tmp_path):
    class StoreWithBacklog:
        db = None

        def __init__(self):
            self.remaining = 3

        def pending(self, _limit):
            if not self.remaining:
                return []
            return [{"record_id": str(self.remaining), "payload": '{"raw_text":"one"}', "attempts": 0}]

        def ack(self, _record_ids):
            self.remaining -= 1

        def pending_count(self):
            return self.remaining

    class Client:
        def __init__(self):
            self.calls = 0

        def ingest(self, records):
            self.calls += 1
            daemon.running = False
            return {"accepted": len(records)}

    c = cfg(tmp_path)
    c.auto_discover = False
    store = StoreWithBacklog()
    client = Client()
    daemon = SyncDaemon(c, store, client)
    daemon.scan_once = lambda: 0
    daemon._stop_watcher = lambda: None

    daemon.run(once=True)

    assert client.calls == 1
    assert store.remaining == 2


def test_flush_batch_respects_serialized_byte_limit(tmp_path):
    payloads = [
        json.dumps({"raw_text": "a" * 20}),
        json.dumps({"raw_text": "b" * 20}),
        json.dumps({"raw_text": "c" * 20}),
    ]

    class FakeStore:
        def __init__(self):
            self.acked = []

        def pending(self, limit):
            return [
                {"record_id": str(index), "payload": payload, "attempts": 0}
                for index, payload in enumerate(payloads[:limit])
            ]

        def ack(self, record_ids):
            self.acked.extend(record_ids)

        def fail(self, *_args):
            raise AssertionError("successful upload must not fail queue rows")

    class Client:
        def __init__(self):
            self.records = []

        def ingest(self, records):
            self.records = records
            return {"accepted": len(records)}

    c = cfg(tmp_path)
    c.batch_size = 3
    c.max_batch_bytes = len(payloads[0].encode("utf-8")) + 1
    store = FakeStore()
    client = Client()

    assert SyncDaemon(c, store, client).flush_once() == 1
    assert len(client.records) == 1
    assert store.acked == ["0"]


def test_filesystem_event_interrupts_remote_flush_burst(tmp_path):
    class StoreWithBacklog:
        db = None

        def pending(self, _limit):
            return [{"record_id": "one", "payload": '{"raw_text":"one"}', "attempts": 0}]

        def ack(self, _record_ids):
            return None

        def pending_count(self):
            return 1

    class Client:
        def __init__(self):
            self.calls = 0

        def ingest(self, records):
            self.calls += 1
            daemon._wake.set()
            if self.calls == 2:
                daemon.running = False
            return {"accepted": len(records)}

    c = cfg(tmp_path)
    c.auto_discover = False
    client = Client()
    daemon = SyncDaemon(c, StoreWithBacklog(), client)
    daemon.scan_once = lambda: 0
    daemon._start_watcher = lambda: None
    daemon._stop_watcher = lambda: None

    daemon.run()

    assert client.calls == 2


def test_shutdown_interrupts_remote_flush_burst_without_wake(tmp_path):
    class StoreWithBacklog:
        db = None

        def __init__(self):
            self.has_pending = True

        def pending(self, _limit):
            if not self.has_pending:
                return []
            return [{"record_id": "one", "payload": '{"raw_text":"one"}', "attempts": 0}]

        def ack(self, _record_ids):
            self.has_pending = False

        def pending_count(self):
            return int(self.has_pending)

    class Client:
        def __init__(self):
            self.calls = 0
            self.health_calls = 0
            self.snapshot_calls = 0

        def ingest(self, records):
            self.calls += 1
            daemon.running = False
            return {"accepted": len(records)}

        def health(self):
            self.health_calls += 1
            return True

        def sync_snapshot(self):
            self.snapshot_calls += 1

    c = cfg(tmp_path)
    c.auto_discover = False
    client = Client()
    daemon = SyncDaemon(c, StoreWithBacklog(), client)
    daemon.scan_once = lambda: 0
    daemon._start_watcher = lambda: None
    daemon._stop_watcher = lambda: None

    daemon.run()

    assert client.calls == 1
    assert client.health_calls == 0
    assert client.snapshot_calls == 0


def test_remote_failure_keeps_pending_for_recovery(tmp_path):
    class OfflineClient:
        def ingest(self, records):
            raise RuntimeError("durable commit unavailable")

    c = cfg(tmp_path)
    c.auto_discover = False
    s = Store(config=c)
    p = tmp_path / "x.jsonl"
    p.write_text('{"role":"user","text":"must survive outage"}\n')
    from sync.discovery import Source
    chunk = parse_file(Source("pi:~/x", "pi", p, "d"))[0]
    s.upsert_chunks([chunk])
    daemon = SyncDaemon(c, s, OfflineClient())
    assert daemon.flush_once() == 0
    assert s.pending_count() == 1
    s.close()
