from __future__ import annotations

import sync.discovery as discovery
from sync.config import Config
from sync.discovery import discover_sources


def test_discovers_bounded_project_memories_with_source_metadata(tmp_path, monkeypatch):
    repo = tmp_path / "code" / "project"
    (repo / ".git").mkdir(parents=True)
    for name in ("AGENTS.md", "CLAUDE.md", "MEMORY.md"):
        (repo / name).write_text(name, encoding="utf-8")
    ignored = repo / "node_modules" / "dependency"
    ignored.mkdir(parents=True)
    (ignored / "AGENTS.md").write_text("ignore", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "memory.md").write_text("outside", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FUNES_PROJECT_ROOTS", str(outside))

    cfg = Config(tmp_path, tmp_path / ".state", tmp_path / "config.toml")
    sources = discover_sources(cfg)
    by_path = {source.path: source for source in sources}

    assert by_path[repo / "AGENTS.md"].source_agent == "codex"
    assert by_path[repo / "AGENTS.md"].source_type == "agents_md"
    assert by_path[repo / "CLAUDE.md"].source_agent == "claude_code"
    assert by_path[repo / "CLAUDE.md"].source_type == "project_instruction"
    assert by_path[repo / "MEMORY.md"].source_agent == "shared"
    assert by_path[repo / "MEMORY.md"].source_type == "memory"
    assert by_path[repo / "MEMORY.md"].project == str(repo)
    assert outside / "memory.md" in by_path
    assert ignored / "AGENTS.md" not in by_path


def test_project_memory_walk_has_a_directory_budget(tmp_path, monkeypatch):
    visited = []

    def fake_walk(_root):
        for index in range(100):
            visited.append(index)
            yield str(tmp_path / str(index)), [], []

    monkeypatch.setattr(discovery.os, "walk", fake_walk)
    assert list(discovery._project_memory_files(tmp_path, max_directories=10)) == []
    assert len(visited) == 11
