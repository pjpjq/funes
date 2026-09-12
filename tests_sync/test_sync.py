import json, os, tempfile, time
from pathlib import Path
from sync.config import Config
from sync.discovery import discover_sources
from sync.parsers import parse_file
from sync.store import Store
from sync.daemon import SyncDaemon
from sync.launchd import render_plist

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
    src=Source('pi:~/x','pi',p,'d'); chunks=parse_file(src); assert s.upsert_chunks(chunks)==1; assert s.upsert_chunks(chunks)==1; assert s.stats()['pending']==1; s.ack([chunks[0].record_id]); assert s.stats()['pending']==0

def test_launchagent(monkeypatch):
    monkeypatch.setenv("FUNES_API_TOKEN", "redacted-test-token")
    d=render_plist('/usr/bin/python3'); assert d['Label']=='com.funes.sync'; assert d['RunAtLoad']
    assert "FUNES_API_TOKEN" not in d["EnvironmentVariables"]


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
