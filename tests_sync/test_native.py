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


def test_native_id_wins_when_session_envelope_is_missing(tmp_path):
    path = tmp_path / "pi-native-no-session.jsonl"
    path.write_text(
        json.dumps({"type": "message", "id": "native-1", "message": {"role": "user", "content": "before"}}) + "\n",
        encoding="utf-8",
    )
    source = Source("pi:~/sessions/native.jsonl", "pi", path, "dev-a")
    before = parse_file(source)[0]
    path.write_text(
        json.dumps({"type": "message", "id": "native-1", "message": {"role": "user", "content": "after"}}) + "\n",
        encoding="utf-8",
    )
    after = parse_file(source)[0]
    assert before.record_id == after.record_id


def test_pi_append_keeps_native_session_identity(tmp_path):
    path = tmp_path / "pi-session.jsonl"
    path.write_text(
        json.dumps({"type": "session", "id": "pi-real-session", "cwd": "/repo"})
        + "\n"
        + json.dumps({"type": "message", "id": "m1", "message": {"role": "user", "content": "first"}})
        + "\n",
        encoding="utf-8",
    )
    source = Source("pi:~/.pi/agent/sessions/pi-session.jsonl", "pi", path, "dev-a")
    first = parse_file(source)
    offset = path.stat().st_size
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "message", "id": "m2", "message": {"role": "assistant", "content": "second"}}) + "\n")
    appended = parse_file(source, offset)
    assert first[0].session_id == "pi-real-session"
    assert appended[0].session_id == "pi-real-session"
    assert first[0].record_id != appended[0].record_id


def test_codex_fallback_append_matches_full_rescan_without_native_id(tmp_path):
    path = tmp_path / "codex-fallback.jsonl"
    path.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "fallback-session"}}) + "\n"
        + json.dumps({"type": "response_item", "timestamp": "2026-01-01T00:00:00Z", "payload": {"type": "message", "role": "user", "content": "first"}}) + "\n",
        encoding="utf-8",
    )
    source = Source("codex:~/sessions/fallback.jsonl", "codex", path, "dev-a")
    offset = path.stat().st_size
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "response_item", "timestamp": "2026-01-01T00:00:01Z", "payload": {"type": "message", "role": "assistant", "content": "second"}}) + "\n")
    appended = parse_file(source, offset)
    full = parse_file(source)
    assert len(appended) == 1
    assert appended[0].record_id == full[1].record_id
    assert len({chunk.record_id for chunk in full}) == len(full)


def test_pi_fallback_append_matches_full_rescan_without_native_id(tmp_path):
    path = tmp_path / "pi-fallback.jsonl"
    path.write_text(
        json.dumps({"type": "session", "id": "pi-fallback-session"}) + "\n"
        + json.dumps({"type": "message", "timestamp": "2026-01-01T00:00:00Z", "message": {"role": "user", "content": "first"}}) + "\n",
        encoding="utf-8",
    )
    source = Source("pi:~/sessions/fallback.jsonl", "pi", path, "dev-a")
    offset = path.stat().st_size
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "message", "timestamp": "2026-01-01T00:00:01Z", "message": {"role": "assistant", "content": "second"}}) + "\n")
    appended = parse_file(source, offset)
    full = parse_file(source)
    assert len(appended) == 1
    assert appended[0].record_id == full[1].record_id
    assert len({chunk.record_id for chunk in full}) == len(full)


def test_memory_converges_across_absolute_repo_paths(tmp_path):
    first_path = tmp_path / "checkout-a" / "MEMORY.md"
    second_path = tmp_path / "checkout-b" / "MEMORY.md"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    content = "# shared decision\n保留跨设备稳定 ID\n"
    first_path.write_text(content, encoding="utf-8")
    second_path.write_text(content, encoding="utf-8")
    a = parse_file(Source("codex_memory:/Users/a/repo/MEMORY.md", "codex_memory", first_path, "dev-a", "/Users/a/repo"))[0]
    b = parse_file(Source("codex_memory:/Volumes/work/repo/MEMORY.md", "codex_memory", second_path, "dev-b", "/Volumes/work/repo"))[0]
    assert a.record_id == b.record_id


def test_global_memory_identity_keeps_home_relative_directory(tmp_path):
    first_path = tmp_path / "first" / "automation.toml"
    second_path = tmp_path / "second" / "automation.toml"
    replica_path = tmp_path / "replica" / "automation.toml"
    for path in (first_path, second_path, replica_path):
        path.parent.mkdir()
        path.write_text('name = "same heading"\n', encoding="utf-8")
    first = parse_file(Source("codex_memory:~/.codex/automations/first/automation.toml", "codex_memory", first_path, "dev-a"))[0]
    second = parse_file(Source("codex_memory:~/.codex/automations/second/automation.toml", "codex_memory", second_path, "dev-a"))[0]
    replica = parse_file(Source("codex_memory:~/.codex/automations/first/automation.toml", "codex_memory", replica_path, "dev-b"))[0]

    assert first.record_id != second.record_id
    assert first.record_id == replica.record_id


def test_external_codex_home_automation_identity_uses_logical_anchor(tmp_path):
    first_path = tmp_path / "disk-a" / "automations" / "first" / "automation.toml"
    second_path = tmp_path / "disk-a" / "automations" / "second" / "automation.toml"
    replica_path = tmp_path / "disk-b" / "automations" / "first" / "automation.toml"
    for path in (first_path, second_path, replica_path):
        path.parent.mkdir(parents=True)
        path.write_text('name = "same heading"\n', encoding="utf-8")
    first = parse_file(Source(f"codex_memory:{first_path}", "codex_memory", first_path, "dev-a"))[0]
    second = parse_file(Source(f"codex_memory:{second_path}", "codex_memory", second_path, "dev-a"))[0]
    replica = parse_file(Source(f"codex_memory:{replica_path}", "codex_memory", replica_path, "dev-b"))[0]

    assert first.record_id != second.record_id
    assert first.record_id == replica.record_id


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


def test_unheaded_persistent_memory_update_keeps_identity_and_increments_version(tmp_path):
    path = tmp_path / "memory.md"
    path.write_text("plain memory before edit\n", encoding="utf-8")
    source = Source("memory:persistent", "persistent", path, "dev")
    cfg = Config(tmp_path, tmp_path / "state", tmp_path / "config.toml")
    store = Store(config=cfg)
    try:
        before = parse_file(source)
        assert len(before) == 1
        assert before[0].message_id.endswith(":__preamble__")
        assert store.upsert_chunks(before) == 1

        path.write_text("plain memory after edit\n", encoding="utf-8")
        after = parse_file(source)
        assert len(after) == 1
        assert after[0].record_id == before[0].record_id
        assert store.upsert_chunks(after) == 1

        row = store.db.execute(
            "SELECT version FROM records WHERE record_id=?", (after[0].record_id,)
        ).fetchone()
        assert row["version"] == 2
        assert store.stats()["records"] == 1
    finally:
        store.close()


def test_large_unheaded_persistent_memory_parts_are_unique_and_stable(tmp_path):
    path = tmp_path / "memory.md"
    path.write_text("a" * 6100, encoding="utf-8")
    source = Source("memory:persistent", "persistent", path, "dev")

    before = parse_file(source)
    assert len(before) == 2
    assert len({chunk.record_id for chunk in before}) == 2
    assert all(len(chunk.raw_text) <= 6000 for chunk in before)

    path.write_text("b" + "a" * 6099, encoding="utf-8")
    after = parse_file(source)
    assert [chunk.record_id for chunk in after] == [chunk.record_id for chunk in before]


def test_repeated_memory_headings_are_unique_and_stable_across_updates(tmp_path):
    path = tmp_path / "memory.md"
    path.write_text("# decision\nfirst\n# decision\nsecond\n", encoding="utf-8")
    source = Source("memory:persistent", "persistent", path, "dev")
    cfg = Config(tmp_path, tmp_path / "state", tmp_path / "config.toml")
    store = Store(config=cfg)
    try:
        before = parse_file(source)
        assert len(before) == 2
        assert len({chunk.record_id for chunk in before}) == 2
        assert before[0].message_id.endswith(":# decision")
        assert before[1].message_id.endswith(":# decision:occurrence:1")
        assert store.upsert_chunks(before) == 2

        path.write_text("# decision\nupdated first\n# decision\nupdated second\n", encoding="utf-8")
        after = parse_file(source)
        assert [chunk.record_id for chunk in after] == [chunk.record_id for chunk in before]
        assert store.upsert_chunks(after) == 2

        versions = store.db.execute("SELECT version FROM records ORDER BY record_id").fetchall()
        assert [row["version"] for row in versions] == [2, 2]
        assert store.stats()["records"] == 2
    finally:
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
