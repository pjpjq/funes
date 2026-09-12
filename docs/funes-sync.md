# funes-sync：本地多 Agent 记忆同步

`sync/` 是唯一的同步实现；`scripts/funes-sync/funes-sync` 只是兼容入口。它发现 Codex（含
active/archived）、Claude Code、pi 的会话，以及 Codex/Claude/pi 和仓库中的 `AGENTS.md`/`MEMORY.md`，
将每个新文件转换为带稳定 source id 的记录。生产远端路径应由原生 `funes index` →
`funes push` 完成；这会保留 Lance vector/BM25/rerank/recency/neighbors 和 TruffleHog gate。
兼容 HTTP 服务只用于迁移/测试，不替代 Funes 检索引擎。

## 快速使用

```sh
python3 -m sync sources
python3 -m sync backfill                         # 首次扫描并尽量排空队列
python3 -m sync status
FUNES_BIN="$HOME/.local/bin/funes" FUNES_MEMORY=org/funes-memory \
  python3 -m sync run
```

`~/.local/share/funes-sync`（可由 `FUNES_STATE_DIR` 覆盖）保存 SQLite cursor、source identity、
content hash 和 pending queue。mtime/hash 未改变时不会重写源文件；源文件消失只标记
`source_missing`，不删除远端历史。

## 命令

- `backfill`：发现、解析、去重，并在远端返回 `durable=true` 后才删除 pending。
- `run`：按 `FUNES_SYNC_INTERVAL`（默认 300 秒）循环；`drain` 只排空既有队列。
- `status`、`sources`、`doctor`、`logs [--follow]`：只读诊断。
- `install`：原子写入 `~/Library/LaunchAgents/com.funes.sync.plist`；遇到非本工具 plist 会拒绝覆盖，`--force` 仅允许覆盖同 label。
- `start`、`stop`、`restart`：macOS `launchctl bootstrap/bootout`；Linux 上安全返回错误，不启动后台进程。

离线 push 失败不会丢数据：pending 队列留在 SQLite，只有远端 durable ACK 才清空。原生
`funes push` 的 CAS、恢复失败保护和 TruffleHog gate 是远端发布的最终边界；LaunchAgent
通过 macOS Keychain 读取 token，不把 token 写入 plist。

## 测试

```sh
python3 -m unittest discover -s scripts/funes-sync/tests -p 'test_*.py'
```

测试覆盖稳定去重、离线队列重试及中文原文保留。
