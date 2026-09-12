#!/usr/bin/env python3
"""Small, deterministic retrieval benchmark for the unified service.

This intentionally uses the service's ASCII-token scoring path as a repeatable
offline proxy.  It does not pretend to be a BGE embedding score when no
translation provider/model is configured; the report records that boundary.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.*:/-]*", re.I)


@dataclass(frozen=True)
class Memory:
    ident: str
    raw: str
    shadow: str
    source_agent: str
    source_type: str


@dataclass(frozen=True)
class Query:
    text: str
    shadow: str
    expected: str


def _tokens(text: str) -> set[str]:
    return {x.lower() for x in TOKEN_RE.findall(text)}


def _score(query: str, text: str) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    return len(q & _tokens(text)) / len(q)


def _memories() -> list[Memory]:
    rows = [
        ("codex-cpa", "我把 CPA 部署在 Northflank 上，第二轮 previous_response_id 会导致上下文丢失。", "CPA deployed on Northflank loses conversation context on the second turn when previous_response_id is used.", "codex", "session"),
        ("pi-tailscale", "MacBook Pro 通过 Tailscale 连接办公室 Mac mini 时出现高延迟。", "Tailscale connection from MacBook Pro to office Mac mini has high latency.", "pi", "session"),
        ("codex-gemini", "默认 Gemini 使用 High reasoning effort。", "Default Gemini configuration uses High reasoning effort.", "codex", "memory"),
        ("codex-chatcmpl", "Codex 调用 CPA 时 previous_response_id 返回 chatcmpl-* 导致 second turn context 丢失。", "Codex CPA call returns chatcmpl-* for previous_response_id and loses second turn context.", "codex", "session"),
        ("claude-launchd", "Claude Code 的 LaunchAgent 必须使用 KeepAlive，并从 Keychain 读取 token。", "Claude Code LaunchAgent needs KeepAlive and reads token from macOS Keychain.", "claude_code", "memory"),
        ("codex-hf", "Hugging Face Space 重启后用私有 dataset 恢复 raw_text，embedding 可以重建。", "Hugging Face Space restart restores raw_text from a private dataset; embeddings are rebuildable.", "codex", "session"),
        ("pi-surge", "Pi 里记录过 Surge 与 Tailscale userspace 共存，不能改坏现有代理。", "Pi notes Surge and Tailscale userspace coexistence; preserve the existing proxy.", "pi", "memory"),
        ("codex-d1", "D1 social_signal 查询必须先 SHOW COLUMNS，不能猜 schema。", "D1 social_signal queries must inspect schema with SHOW COLUMNS before SQL.", "codex", "session"),
        ("claude-bm25", "Claude 记忆检索默认保留 BM25，并把 tool_result 降权。", "Claude memory retrieval keeps BM25 and downweights tool_result content.", "claude_code", "memory"),
        ("pi-offline", "Pi 离线时把上传放入 pending queue，恢复网络后指数退避重试。", "Pi offline uploads stay in a pending queue and retry with exponential backoff after recovery.", "pi", "session"),
        ("codex-rollback", "部署必须带 health probe 和 automatic rollback，不能只看 CI 绿色。", "Deployment requires a health probe and automatic rollback, not only a green CI check.", "codex", "session"),
        ("claude-raw", "原始 session 永远是 source of truth，retrieval_text 只能是 derived shadow。", "The original session is always the source of truth; retrieval_text is only a derived shadow.", "claude_code", "memory"),
        ("pi-cross", "Pi 查询 Codex 历史时保留 source_agent=codex metadata。", "Pi can query Codex history while retaining source_agent=codex metadata.", "pi", "session"),
        ("codex-device", "多台 Mac 使用 privacy-safe device_id，不能上传 MAC address。", "Multiple Macs use a privacy-safe device_id and never upload a MAC address.", "codex", "memory"),
        ("claude-secret", "Claude 项目扫描要排除 auth.json、.env 和 credentials。", "Claude project discovery excludes auth.json, .env, and credentials files.", "claude_code", "project_instruction"),
    ]
    # Add 45 realistic distractors so rank metrics are not a toy single-document test.
    topics = [
        ("latency", "延迟排查要同时看 TCP handshake、日志时间戳和 metrics。", "latency diagnosis checks TCP handshake, log timestamps, and metrics."),
        ("memory", "固定 memory 部署先定位 resident-memory 来源，不能直接调大限制。", "fixed-memory deployment identifies resident memory sources before changing limits."),
        ("release", "发布前必须 read-back remote SHA 和服务状态。", "release verification reads back the remote SHA and service status."),
        ("logs", "Grafana Loki 查询按 UTC 日志和本地 UTC+8 对齐。", "Grafana Loki searches align UTC logs with local UTC+8."),
        ("config", "配置文件不保存 API key，只保存 token_env。", "configuration files store token_env, never an API key."),
        ("watcher", "文件 watcher 只处理 create、modify、rename、append。", "the filesystem watcher handles create, modify, rename, and append."),
        ("dedupe", "稳定 source_identity 让重复扫描不会增加 memory 数量。", "stable source_identity prevents repeated scans from increasing memory count."),
        ("reindex", "切换 BGE-M3 前从 raw_text 重建 retrieval index。", "switching to BGE-M3 rebuilds the retrieval index from raw_text."),
        ("mcp", "远程 MCP 不可用时使用本地 stdio bridge。", "a local stdio bridge is used when remote MCP is unavailable."),
    ]
    for i in range(5):
        for key, raw, shadow in topics:
            if len(rows) >= 60:
                break
            rows.append((f"distractor-{key}-{i}", raw, shadow + f" variant {i}", "pi" if i % 2 else "codex", "session"))
    return [Memory(*row) for row in rows[:60]]


def _queries(memories: list[Memory]) -> list[Query]:
    by_id = {m.ident: m for m in memories}
    rows = [
        ("CPA 第二轮为什么丢上下文？", "CPA Northflank previous_response_id second turn context loss", "codex-cpa"),
        ("之前 Pi 里讨论过的 Tailscale 延迟", "Pi Tailscale MacBook Pro office Mac mini high latency", "pi-tailscale"),
        ("Gemini 默认 reasoning 配置是什么？", "Gemini default High reasoning effort configuration", "codex-gemini"),
        ("chatcmpl-* 和 previous_response_id 的关系", "Codex CPA chatcmpl-* previous_response_id context", "codex-chatcmpl"),
        ("Claude 的后台任务如何保持运行？", "Claude Code LaunchAgent KeepAlive Keychain token", "claude-launchd"),
        ("HF Space 重启后如何恢复记忆？", "Hugging Face Space restart private dataset raw_text rebuild embeddings", "codex-hf"),
        ("Pi 和 Surge 共存要注意什么？", "Pi Surge Tailscale userspace coexistence preserve proxy", "pi-surge"),
        ("social_signal 查询前为什么要看字段？", "D1 social_signal SHOW COLUMNS schema SQL", "codex-d1"),
        ("tool_result 在检索中应该怎样处理？", "Claude BM25 downweight tool_result retrieval", "claude-bm25"),
        ("断网时 Pi 的记忆会丢吗？", "Pi offline pending queue exponential backoff retry", "pi-offline"),
        ("部署失败怎样自动恢复？", "deployment health probe automatic rollback CI", "codex-rollback"),
        ("raw_text 和 retrieval_text 谁是真源？", "original session source of truth derived retrieval_text shadow", "claude-raw"),
        ("Pi 能不能搜 Codex？", "Pi query Codex source_agent metadata", "pi-cross"),
        ("多台 Mac 如何避免泄露硬件标识？", "privacy-safe device_id no MAC address multiple Macs", "codex-device"),
        ("Claude 扫描时哪些文件不能上传？", "Claude discovery excludes auth.json .env credentials", "claude-secret"),
        ("watcher 监听哪些变化？", "filesystem watcher create modify rename append", "distractor-watcher-0"),
        ("重复扫描如何去重？", "stable source_identity repeated scans dedupe", "distractor-dedupe-0"),
        ("换 embedding 模型如何重建？", "BGE-M3 rebuild retrieval index raw_text", "distractor-reindex-0"),
        ("远程 MCP 挂了怎么办？", "remote MCP local stdio bridge unavailable", "distractor-mcp-0"),
        ("日志时区怎么对齐？", "Grafana Loki UTC UTC+8 log timestamps", "distractor-logs-0"),
    ]
    return [Query(text, shadow, expected) for text, shadow, expected in rows if expected in by_id]


def evaluate(memories: list[Memory], queries: list[Query], shadow: bool) -> dict[str, float]:
    ranked = []
    for query in queries:
        q = query.shadow if shadow else query.text
        scored = sorted(((
            _score(q, m.shadow if shadow else m.raw),
            # Stable tie-break keeps this benchmark deterministic while still
            # making the raw Chinese/English mismatch visible.
            m.ident,
        ) for m in memories), reverse=True)
        ranked.append([ident for _, ident in scored])
    out = {}
    for k in (1, 3, 5):
        out[f"recall@{k}"] = sum(q.expected in hits[:k] for q, hits in zip(queries, ranked)) / len(queries)
    return out


def main() -> None:
    memories = _memories()
    queries = _queries(memories)
    result = {
        "memories": len(memories),
        "queries": len(queries),
        "method": "ASCII-token overlap proxy for the deployed FTS path; not a fabricated BGE score",
        "before_raw_chinese_to_bge_en_proxy": evaluate(memories, queries, shadow=False),
        "after_english_retrieval_shadow_proxy": evaluate(memories, queries, shadow=True),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = Path(__file__).resolve().parents[1] / "docs" / "chinese-retrieval-results.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
