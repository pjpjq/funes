import json
from pathlib import Path

from sync.config import Config
from sync.discovery import Source, discover_sources
from sync.parsers import parse_file
from sync.store import Store


def test_codex_call_and_result_have_distinct_stable_ids(tmp_path):
    path = tmp_path / "rollout.jsonl"
    rows = [
        {"type": "session_meta", "payload": {"id": "session-1", "cwd": "/repo"}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:00Z", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "CPA 第二轮为什么丢上下文？"}]}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:01Z", "payload": {"type": "function_call", "name": "exec_command", "call_id": "call-1", "arguments": "{\"cmd\":\"echo ok\"}"}},
        {"type": "response_item", "timestamp": "2026-01-01T00:00:02Z", "payload": {"type": "function_call_output", "call_id": "call-1", "output": "ok"}},
    ]
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    source = Source("codex:~/.codex/sessions/rollout.jsonl", "codex", path, "dev-a")
    chunks = parse_file(source)
    assert len(chunks) == 3
    assert chunks[0].session_id == "session-1"
    assert chunks[1].record_id != chunks[2].record_id
    assert {c.content_type for c in chunks} == {"user_message", "tool_call", "tool_result"}
    assert all(c.as_dict()["source_agent"] == "codex" for c in chunks)


def test_same_session_converges_across_devices(tmp_path):
    path = tmp_path / "rollout.jsonl"
    row = {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "same"}]}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    a = parse_file(Source("codex:~/a", "codex", path, "dev-a"))[0]
    b = parse_file(Source("codex:~/b", "codex", path, "dev-b"))[0]
    assert a.record_id == b.record_id


def test_memory_update_is_one_record_and_queue_is_idempotent(tmp_path):
    path = tmp_path / "MEMORY.md"
    path.write_text("# decision\n保留 raw_text\n", encoding="utf-8")
    source = Source("codex_memory:~/.codex/memories/MEMORY.md", "codex_memory", path, "dev")
    cfg = Config(tmp_path, tmp_path / "state", tmp_path / "config.toml")
    store = Store(config=cfg)
    first = parse_file(source)
    assert store.upsert_chunks(first) == 1
    assert store.upsert_chunks(first) == 1
    path.write_text("# decision\n更新 retrieval_text\n", encoding="utf-8")
    second = parse_file(source)
    assert store.upsert_chunks(second) == 1
    assert store.stats()["records"] == 1
    assert store.stats()["pending"] == 1
    store.close()


def test_discovery_excludes_auth_and_finds_archived(tmp_path):
    codex = tmp_path / ".codex"
    (codex / "archived_sessions").mkdir(parents=True)
    (codex / "archived_sessions" / "old.jsonl").write_text("{}\n", encoding="utf-8")
    (codex / "auth.json").write_text("{\"token\":\"secret\"}\n", encoding="utf-8")
    cfg = Config(tmp_path, tmp_path / "state", tmp_path / "config.toml")
    found = discover_sources(cfg)
    paths = {s.path for s in found}
    assert codex / "archived_sessions" / "old.jsonl" in paths
    assert codex / "auth.json" not in paths
