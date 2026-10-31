from __future__ import annotations

import json
from pathlib import Path
from sync.config import Config
from sync.discovery import Source, discover_sources
from sync.parsers import parse_file


def test_discovers_claude_history_jsonl_in_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(parents=True)
    history_file = claude_dir / "history.jsonl"
    history_file.write_text('{"sessionId":"abc","display":"test"}' + chr(10), encoding="utf-8")

    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    sources = discover_sources(cfg)
    matches = [s for s in sources if s.path == history_file or s.source_key == "claude:~/.claude/history.jsonl"]

    assert len(matches) == 1
    src = matches[0]
    assert src.kind == "claude"
    assert src.source_agent == "claude_code"
    assert src.source_type == "session"
    assert src.source_key == "claude:~/.claude/history.jsonl"
    assert src.project == ""


def test_discovers_claude_history_jsonl_in_custom_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    custom_claude = tmp_path / "custom_claude_dir"
    custom_claude.mkdir(parents=True)
    history_file = custom_claude / "history.jsonl"
    history_file.write_text('{"sessionId":"xyz","display":"custom"}' + chr(10), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom_claude))

    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    sources = discover_sources(cfg)
    matches = [s for s in sources if s.path == history_file]

    assert len(matches) == 1
    src = matches[0]
    assert src.kind == "claude"
    assert src.source_agent == "claude_code"
    assert src.source_type == "session"
    assert src.source_key == f"claude:~/{history_file.resolve().relative_to(tmp_path.resolve())}"


def test_claude_history_jsonl_deduplication(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    real_claude = tmp_path / ".claude"
    real_claude.mkdir(parents=True)
    history_file = real_claude / "history.jsonl"
    history_file.write_text('{"sessionId":"dedup"}' + chr(10), encoding="utf-8")

    symlink_claude = tmp_path / "claude_symlink"
    symlink_claude.symlink_to(real_claude)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(symlink_claude))

    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    sources = discover_sources(cfg)
    matches = [s for s in sources if s.source_key == "claude:~/.claude/history.jsonl"]

    assert len(matches) == 1


def test_discovers_both_custom_and_home_claude_history_if_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    home_claude = tmp_path / ".claude"
    home_claude.mkdir(parents=True)
    (home_claude / "history.jsonl").write_text('{"sessionId":"home"}' + chr(10), encoding="utf-8")

    custom_claude = tmp_path / "custom_claude"
    custom_claude.mkdir(parents=True)
    (custom_claude / "history.jsonl").write_text('{"sessionId":"custom"}' + chr(10), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom_claude))

    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    sources = discover_sources(cfg)
    keys = {s.source_key for s in sources if "history.jsonl" in str(s.path)}

    assert keys == {
        "claude:~/.claude/history.jsonl",
        "claude:~/custom_claude/history.jsonl",
    }


def test_parse_claude_history_raw_chinese_and_sessions(tmp_path):
    history_file = tmp_path / "history.jsonl"
    records = [
        {
            "display": "你好，请帮我分析一下并发调度逻辑",
            "sessionId": "sess-alpha-001",
            "project": "/Users/test/workspace/alpha",
            "timestamp": 1711800000000,
            "pastedContents": {"file.txt": "sensitive content"},
        },
        {
            "display": "继续检查错误边界处理",
            "sessionId": "sess-beta-002",
            "project": "/Users/test/workspace/beta",
            "timestamp": 1711803600000,
        },
    ]
    history_file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    source = Source(
        source_key="claude:~/.claude/history.jsonl",
        kind="claude",
        path=history_file,
        device_id="dev-1",
        project="",
    )
    chunks = parse_file(source)
    assert len(chunks) == 2

    assert chunks[0].text == "你好，请帮我分析一下并发调度逻辑"
    assert chunks[0].raw_text == "你好，请帮我分析一下并发调度逻辑"
    assert chunks[0].session_id == "sess-alpha-001"
    assert chunks[0].project == "/Users/test/workspace/alpha"
    assert chunks[0].role == "user"
    assert chunks[0].content_type == "user_message"
    assert chunks[0].timestamp == "2024-03-30T12:00:00+00:00"
    assert "pastedContents" not in chunks[0].metadata

    assert chunks[1].text == "继续检查错误边界处理"
    assert chunks[1].session_id == "sess-beta-002"
    assert chunks[1].project == "/Users/test/workspace/beta"
    assert chunks[1].timestamp == "2024-03-30T13:00:00+00:00"


def test_parse_claude_history_append_identity_stability(tmp_path):
    history_file = tmp_path / "history.jsonl"
    line1 = json.dumps({"display": "第一条指令", "sessionId": "sess-1", "timestamp": 1711800000000}) + "\n"
    line2 = json.dumps({"display": "第二条指令追加", "sessionId": "sess-1", "timestamp": 1711800060000}) + "\n"
    history_file.write_text(line1 + line2, encoding="utf-8")

    source = Source(
        source_key="claude:~/.claude/history.jsonl",
        kind="claude",
        path=history_file,
        device_id="dev-1",
        project="",
    )
    full_chunks = parse_file(source, start=0)
    assert len(full_chunks) == 2

    offset_line2 = len(line1.encode("utf-8"))
    append_chunks = parse_file(source, start=offset_line2)
    assert len(append_chunks) == 1

    # Identity must be strictly identical across full scan and append scan
    assert append_chunks[0].record_id == full_chunks[1].record_id
    assert append_chunks[0].text == full_chunks[1].text
    assert append_chunks[0].session_id == full_chunks[1].session_id


def test_parse_claude_history_cross_path_deduplication(tmp_path):
    path_a = tmp_path / "path_a" / "history.jsonl"
    path_b = tmp_path / "path_b" / "history.jsonl"
    path_a.parent.mkdir()
    path_b.parent.mkdir()
    content = json.dumps({"display": "相同内容跨路径", "sessionId": "sess-shared", "timestamp": 1711800000000}) + "\n"
    path_a.write_text(content, encoding="utf-8")
    path_b.write_text(content, encoding="utf-8")

    source_a = Source(
        source_key="claude:~/.claude/history.jsonl",
        kind="claude",
        path=path_a,
        device_id="dev-1",
        project="",
    )
    source_b = Source(
        source_key="claude:~/custom/history.jsonl",
        kind="claude",
        path=path_b,
        device_id="dev-1",
        project="",
    )
    chunks_a = parse_file(source_a)
    chunks_b = parse_file(source_b)
    assert len(chunks_a) == 1
    assert len(chunks_b) == 1
    assert chunks_a[0].record_id == chunks_b[0].record_id


def test_parse_claude_native_transcript_preserved(tmp_path):
    transcript_file = tmp_path / "session.jsonl"
    lines = [
        json.dumps({"type": "user", "message": {"role": "user", "content": "用户问题"}, "sessionId": "sess-trans"}),
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "助手回复"}, "sessionId": "sess-trans"}),
        json.dumps({"type": "summary", "summary": "会话摘要", "sessionId": "sess-trans"}),
    ]
    transcript_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    source = Source(
        source_key="claude:transcript",
        kind="claude",
        path=transcript_file,
        device_id="dev-1",
        project="",
    )
    chunks = parse_file(source)
    assert len(chunks) == 3
    assert [c.role for c in chunks] == ["user", "assistant", "summary"]
    assert [c.content_type for c in chunks] == ["user_message", "assistant_message", "summary"]
    assert chunks[0].text == "用户问题"
    assert chunks[1].text == "助手回复"
    assert chunks[2].text == "会话摘要"


def test_parse_claude_history_skips_non_dict_and_invalid_display(tmp_path):
    history_file = tmp_path / "history.jsonl"
    lines = [
        "12345",
        '"a standalone string"',
        "[1, 2, 3]",
        "null",
        json.dumps({"sessionId": "s1"}),  # missing display
        json.dumps({"display": None, "sessionId": "s2"}),
        json.dumps({"display": True, "sessionId": "s3"}),
        json.dumps({"display": False, "sessionId": "s4"}),
        json.dumps({"display": {"nested": "obj"}, "sessionId": "s5"}),
        json.dumps({"display": "   \n\t  ", "sessionId": "s6"}),  # whitespace only
        json.dumps({"display": "有效输入", "sessionId": "s7"}),
    ]
    history_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    source = Source(
        source_key="claude:~/.claude/history.jsonl",
        kind="claude",
        path=history_file,
        device_id="dev-1",
        project="",
    )
    chunks = parse_file(source)
    assert len(chunks) == 1
    assert chunks[0].text == "有效输入"
    assert chunks[0].session_id == "s7"


def test_parse_claude_history_timestamp_variants(tmp_path):
    history_file = tmp_path / "history.jsonl"
    lines = [
        json.dumps({"display": "毫秒数字", "sessionId": "s-ts", "timestamp": 1711800000000}),
        json.dumps({"display": "毫秒字符串", "sessionId": "s-ts", "timestamp": "1711800000000"}),
        json.dumps({"display": "秒级数字", "sessionId": "s-ts", "timestamp": 1711800000}),
        json.dumps({"display": "秒级字符串", "sessionId": "s-ts", "timestamp": "1711800000"}),
    ]
    history_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    source = Source(
        source_key="claude:~/.claude/history.jsonl",
        kind="claude",
        path=history_file,
        device_id="dev-1",
        project="",
    )
    chunks = parse_file(source)
    assert len(chunks) == 4
    # Numeric ms and string ms must produce identical UTC ISO timestamp
    assert chunks[0].timestamp == chunks[1].timestamp == "2024-03-30T12:00:00+00:00"
    # Numeric sec and string sec must produce identical UTC ISO timestamp
    assert chunks[2].timestamp == chunks[3].timestamp == "2024-03-30T12:00:00+00:00"
