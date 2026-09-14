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


def test_daemon_caches_discovery_until_invalidated(tmp_path, monkeypatch):
    from sync.discovery import Source

    c = cfg(tmp_path)
    memory = tmp_path / "memory.md"
    memory.write_text("stable memory", encoding="utf-8")
    source = Source("memory:~/memory.md", "persistent", memory, c.device_id)
    calls = []

    def discover(_config):
        calls.append(True)
        return [source]

    monkeypatch.setattr("sync.daemon.discover_sources", discover)
    store = Store(config=c)
    daemon = SyncDaemon(c, store, type("Client", (), {})())

    daemon.scan_once()
    daemon.scan_once()
    assert len(calls) == 1

    daemon._invalidate_source_cache()
    daemon.scan_once()
    assert len(calls) == 2
    store.close()


def test_directory_create_invalidates_discovery_cache_and_wakes(tmp_path):
    c = cfg(tmp_path)
    store = Store(config=c)
    daemon = SyncDaemon(c, store, type("Client", (), {})())
    daemon._source_cache = ()
    daemon._source_cache_at = time.monotonic()
    event = type(
        "Event",
        (),
        {
            "src_path": str(tmp_path / ".codex/sessions/imported"),
            "dest_path": "",
            "event_type": "created",
            "is_directory": True,
        },
    )()

    daemon._handle_filesystem_event(event)

    assert daemon._source_cache is None
    assert daemon._wake.is_set()
    store.close()


def test_cross_boundary_directory_move_invalidates_discovery_cache(tmp_path):
    c = cfg(tmp_path)
    store = Store(config=c)
    daemon = SyncDaemon(c, store, type("Client", (), {})())
    daemon._source_cache = ()
    daemon._source_cache_at = time.monotonic()
    event = type(
        "Event",
        (),
        {
            "src_path": str(tmp_path / "logs/imported"),
            "dest_path": str(tmp_path / ".codex/sessions/imported"),
            "event_type": "moved",
            "is_directory": True,
        },
    )()

    daemon._handle_filesystem_event(event)

    assert daemon._source_cache is None
    assert daemon._wake.is_set()
    store.close()


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


def test_schema_epoch_reprocesses_unchanged_zero_record_source_at_eof(
    tmp_path, monkeypatch
):
    from sync.discovery import Source

    c=cfg(tmp_path)
    p=tmp_path/'claude.jsonl'
    p.write_text(
        json.dumps({"type":"user","message":{"role":"user","content":"previously skipped Claude turn"}})+"\n",
        encoding='utf-8',
    )
    source=Source('claude:~/.claude/projects/agent.jsonl','claude',p,c.device_id)
    monkeypatch.setattr('sync.daemon.discover_sources', lambda _config: [source])
    starts=[]
    real_parse=parse_file

    def tracked_parse(item, start=0):
        starts.append(start)
        return real_parse(item, start)

    monkeypatch.setattr('sync.daemon.parse_file', tracked_parse)
    s=Store(config=c)
    stat=p.stat()
    s.register_source(source,stat)
    s.set_cursor(source.source_key,stat.st_size,stat.st_ino,stat.st_size)
    (c.state_dir/'initial-backfill.complete').write_text('legacy', encoding='utf-8')
    (c.state_dir/'source-schema-v2.complete').write_text('legacy', encoding='utf-8')
    d=SyncDaemon(c,s,type("Client",(),{})())

    assert d.scan_once() == 1
    assert starts == [0]
    assert s.stats()["records"] == 1
    assert (c.state_dir/'source-schema-v2.complete').exists()
    s.close()


def test_automation_identity_migration_reprocesses_only_once(tmp_path,monkeypatch):
    from sync.discovery import Source

    c=cfg(tmp_path)
    p=tmp_path/'.codex/automations/job/automation.toml'
    p.parent.mkdir(parents=True)
    p.write_text('name = "job"\n',encoding='utf-8')
    source=Source('codex_memory:~/.codex/automations/job/automation.toml','codex_memory',p,c.device_id)
    monkeypatch.setattr('sync.daemon.discover_sources',lambda _config:[source])
    calls=[]
    real_parse=parse_file

    def tracked_parse(item,start=0):
        calls.append(start)
        return real_parse(item,start)

    monkeypatch.setattr('sync.daemon.parse_file',tracked_parse)
    s=Store(config=c)
    stat=p.stat()
    s.register_source(source,stat)
    s.upsert_chunks(real_parse(source))
    s.ack([chunk.record_id for chunk in real_parse(source)])
    s.set_cursor(source.source_key,stat.st_size,stat.st_ino,stat.st_size)
    (c.state_dir/'initial-backfill.complete').write_text('legacy',encoding='utf-8')
    (c.state_dir/'source-schema-v2.complete').write_text('legacy',encoding='utf-8')
    daemon=SyncDaemon(c,s,type("Client",(),{})())

    daemon.scan_once()
    daemon.scan_once()

    assert calls == [0]
    assert daemon._automation_identity_marker.exists()
    s.close()


def test_zero_record_migration_does_not_replay_converged_sources_forever(
    tmp_path,monkeypatch
):
    from sync.discovery import Source

    c=cfg(tmp_path)
    first_path=tmp_path/'first.jsonl'
    second_path=tmp_path/'second.jsonl'
    content=json.dumps({
        "type":"user",
        "sessionId":"shared-session",
        "uuid":"shared-message",
        "message":{"role":"user","content":"same copied turn"},
    })+"\n"
    first_path.write_text(content,encoding='utf-8')
    second_path.write_text(content,encoding='utf-8')
    sources=[
        Source('claude:first','claude',first_path,c.device_id),
        Source('claude:second','claude',second_path,c.device_id),
    ]
    monkeypatch.setattr('sync.daemon.discover_sources',lambda _config:sources)
    calls=[]
    real_parse=parse_file

    def tracked_parse(item,start=0):
        calls.append(item.source_key)
        return real_parse(item,start)

    monkeypatch.setattr('sync.daemon.parse_file',tracked_parse)
    s=Store(config=c)
    for source in sources:
        stat=source.path.stat()
        s.register_source(source,stat)
        s.set_cursor(source.source_key,stat.st_size,stat.st_ino,stat.st_size)
    (c.state_dir/'initial-backfill.complete').write_text('legacy',encoding='utf-8')
    (c.state_dir/'source-schema-v2.complete').write_text('legacy',encoding='utf-8')
    daemon=SyncDaemon(c,s,type("Client",(),{})())

    daemon.scan_once()
    first_calls=list(calls)
    calls.clear()
    daemon.scan_once()

    assert first_calls == ['claude:first','claude:second']
    assert calls == []
    assert daemon._zero_record_marker.exists()
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


def test_appendable_cursor_waits_at_last_complete_newline(tmp_path, monkeypatch):
    from sync.discovery import Source

    c = cfg(tmp_path)
    p = tmp_path / "rollout.jsonl"
    header = json.dumps(
        {"type": "session_meta", "payload": {"id": "cursor-session"}}
    ) + "\n"
    first = json.dumps(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "first",
                "role": "user",
                "content": "first complete turn",
            },
        }
    ) + "\n"
    partial = json.dumps(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "second",
                "role": "assistant",
                "content": "等待换行",
            },
        },
        ensure_ascii=False,
    )
    p.write_text(header + first + partial, encoding="utf-8")
    source = Source("codex:~/sessions/cursor.jsonl", "codex", p, c.device_id)
    monkeypatch.setattr("sync.daemon.discover_sources", lambda _config: [source])
    s = Store(config=c)
    daemon = SyncDaemon(c, s, type("Client", (), {})())

    assert daemon.scan_once() == 1

    cursor = s.cursor(source.source_key)
    complete_offset = len((header + first).encode("utf-8"))
    assert cursor["offset"] == complete_offset
    assert cursor["size"] == p.stat().st_size

    with p.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    assert daemon.scan_once() == 1
    assert s.stats()["records"] == 2
    s.close()


def test_no_backfill_seed_preserves_unterminated_append(tmp_path, monkeypatch):
    from sync.discovery import Source

    c = cfg(tmp_path)
    c.initial_backfill = False
    p = tmp_path / "seed-rollout.jsonl"
    header = json.dumps(
        {"type": "session_meta", "payload": {"id": "seed-session"}}
    ) + "\n"
    historical = json.dumps(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "historical",
                "role": "user",
                "content": "must remain skipped",
            },
        }
    ) + "\n"
    partial = json.dumps(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "after-install",
                "role": "user",
                "content": "安装时尚未换行",
            },
        },
        ensure_ascii=False,
    )
    p.write_text(header + historical + partial, encoding="utf-8")
    source = Source("codex:~/sessions/seed.jsonl", "codex", p, c.device_id)
    monkeypatch.setattr("sync.daemon.discover_sources", lambda _config: [source])
    starts = []
    real_parse = parse_file

    def tracked_parse(item, start=0):
        starts.append(start)
        return real_parse(item, start)

    monkeypatch.setattr("sync.daemon.parse_file", tracked_parse)
    s = Store(config=c)
    daemon = SyncDaemon(c, s, type("Client", (), {})())

    assert daemon.scan_once() == 0

    cursor = s.cursor(source.source_key)
    complete_offset = len((header + historical).encode("utf-8"))
    assert cursor["offset"] == complete_offset
    assert cursor["size"] == p.stat().st_size

    with p.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    assert daemon.scan_once() == 1
    assert starts == [complete_offset]
    assert s.stats()["records"] == 1
    s.close()


def test_legacy_mid_line_cursor_forces_full_reconcile(tmp_path, monkeypatch):
    from sync.discovery import Source

    c = cfg(tmp_path)
    p = tmp_path / "legacy-rollout.jsonl"
    header = json.dumps(
        {"type": "session_meta", "payload": {"id": "legacy-session"}}
    ) + "\n"
    stale_line = json.dumps(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "legacy-fragment",
                "role": "assistant",
                "content": "stale fragment",
            },
        }
    ) + "\n"
    p.write_text(header + stale_line, encoding="utf-8")
    source = Source("codex:~/sessions/legacy.jsonl", "codex", p, c.device_id)
    stale = parse_file(source)[0]
    current_line = json.dumps(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "current",
                "role": "assistant",
                "content": "completed after upgrade",
            },
        }
    )
    p.write_text(header + current_line, encoding="utf-8")
    before_append = p.stat()
    legacy_offset = len(header.encode("utf-8")) + len(current_line.encode("utf-8")) // 2
    monkeypatch.setattr("sync.daemon.discover_sources", lambda _config: [source])
    starts = []
    real_parse = parse_file

    def tracked_parse(item, start=0):
        starts.append(start)
        return real_parse(item, start)

    monkeypatch.setattr("sync.daemon.parse_file", tracked_parse)
    s = Store(config=c)
    s.register_source(source, before_append)
    s.upsert_chunks([stale])
    s.set_cursor(
        source.source_key,
        legacy_offset,
        before_append.st_ino,
        before_append.st_size,
    )
    for marker in (
        "initial-backfill.complete",
        "source-schema-v2.complete",
        "zero-record-repair-v1.complete",
    ):
        (c.state_dir / marker).write_text("legacy", encoding="utf-8")
    daemon = SyncDaemon(c, s, type("Client", (), {})())

    with p.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    assert daemon.scan_once() == 1
    assert starts == [0]
    assert s.get(stale.record_id)["source_missing"] is True
    cursor = s.cursor(source.source_key)
    assert cursor["offset"] == p.stat().st_size
    assert cursor["size"] == p.stat().st_size
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


def test_remote_source_reconciliation_resumes_and_only_queues_missing(tmp_path):
    class Client:
        def __init__(self):
            self.calls=[]

        def missing_source_identities(self,identities):
            self.calls.append(list(identities))
            return [identity for identity in identities if identity in {"b","d"}]

    c=cfg(tmp_path)
    s=Store(config=c)
    for record_id in ("d","a","c","b"):
        from sync.parsers import Chunk
        s.upsert_chunks(
            [Chunk(record_id,"source","codex","session.jsonl","session",0,"user",record_id,record_id)]
        )
    s.ack(["a","b","c","d"])
    client=Client()
    daemon=SyncDaemon(c,s,client)

    first=daemon.reconcile_remote_sources(1)

    assert first == {"complete":False,"checked":4,"queued":2}
    assert client.calls == [["a","b","c","d"]]
    assert s.meta_value(daemon._remote_source_cursor_key) == "d"
    assert s.pending_count() == 2
    assert not daemon._remote_source_marker.exists()

    resumed=SyncDaemon(c,s,client).reconcile_remote_sources(1)
    assert resumed == {"complete":False,"checked":0,"queued":0}
    s.ack(["b","d"])
    completed=SyncDaemon(c,s,client).reconcile_remote_sources(1)
    assert completed == {"complete":True,"checked":0,"queued":0}
    assert daemon._remote_source_marker.exists()
    assert s.meta_value(daemon._remote_source_cursor_key) == ""

    forced=SyncDaemon(c,s,client).reconcile_remote_sources(1,force=True)
    assert forced == {"complete":False,"checked":4,"queued":2}

    other=cfg(tmp_path)
    other.remote_url="https://other-memory.example"
    other_daemon=SyncDaemon(other,s,client)
    assert other_daemon._remote_source_marker != daemon._remote_source_marker
    assert other_daemon._remote_source_cursor_key != daemon._remote_source_cursor_key
    assert other_daemon.reconcile_remote_sources(1)["checked"] == 4
    s.close()


def test_remote_source_reconciliation_stops_between_batches(tmp_path):
    class InventoryStore:
        def __init__(self):
            self.cursor=""

        def meta_value(self,_key):
            return self.cursor

        def set_meta(self,_key,value):
            self.cursor=value

        def record_ids_after(self,after,_limit):
            return [] if after == "second" else (["first"] if not after else ["second"])

        def enqueue_records(self,_identities):
            return 0

        def pending_count(self):
            return 0

    class Client:
        def __init__(self):
            self.calls=0

        def missing_source_identities(self,_identities):
            self.calls+=1
            daemon.running=False
            return []

    store=InventoryStore()
    client=Client()
    daemon=SyncDaemon(cfg(tmp_path),store,client)
    daemon.running=True

    result=daemon.reconcile_remote_sources(3,interruptible=True)

    assert result == {
        "complete":False,
        "checked":1,
        "queued":0,
        "interrupted":True,
    }
    assert client.calls == 1


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


def test_idle_daemon_does_not_call_operator_snapshot_without_new_work(tmp_path):
    class IdleStore:
        db = None

        def pending(self, _limit):
            return []

        def pending_count(self):
            return 0

    class Client:
        def __init__(self):
            self.health_calls = 0
            self.snapshot_calls = 0

        def health(self):
            self.health_calls += 1
            return True

        def sync_snapshot(self):
            self.snapshot_calls += 1

    class Wake:
        def __init__(self):
            self.waits = 0

        def is_set(self):
            return False

        def wait(self, _timeout):
            self.waits += 1
            if self.waits == 2:
                daemon.running = False

        def clear(self):
            return None

        def set(self):
            return None

    c = cfg(tmp_path)
    c.auto_discover = False
    client = Client()
    daemon = SyncDaemon(c, IdleStore(), client)
    daemon._wake = Wake()
    daemon.scan_once = lambda: 0
    daemon._start_watcher = lambda: None
    daemon._stop_watcher = lambda: None

    daemon.run()

    assert daemon._wake.waits == 2
    assert client.health_calls == 0
    assert client.snapshot_calls == 0


def test_successful_ingest_is_not_followed_by_full_snapshot(tmp_path):
    class StoreWithOneRecord:
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
            self.ingest_calls = 0
            self.health_calls = 0
            self.snapshot_calls = 0

        def ingest(self, records):
            self.ingest_calls += 1
            return {"accepted": len(records)}

        def health(self):
            self.health_calls += 1
            return True

        def sync_snapshot(self):
            self.snapshot_calls += 1

    class Wake:
        def is_set(self):
            return False

        def wait(self, _timeout):
            daemon.running = False

        def clear(self):
            return None

        def set(self):
            return None

    c = cfg(tmp_path)
    c.auto_discover = False
    client = Client()
    daemon = SyncDaemon(c, StoreWithOneRecord(), client)
    daemon._wake = Wake()
    daemon.scan_once = lambda: 0
    daemon._start_watcher = lambda: None
    daemon._stop_watcher = lambda: None

    daemon.run()

    assert client.ingest_calls == 1
    assert client.health_calls == 0
    assert client.snapshot_calls == 0


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
