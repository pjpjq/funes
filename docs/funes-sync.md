# funes-sync：本地多 Agent 记忆同步

`sync/` 是唯一的同步实现；`scripts/funes-sync/funes-sync` 只是兼容入口。它发现 Codex（含
active/archived）、Claude Code、pi 的会话，以及 Codex/Claude/pi 和仓库中的 `AGENTS.md`/`MEMORY.md`，
将每个新文件转换为带稳定 source id 的记录。生产远端路径应由原生 `funes index` →
`funes push` 完成；这会保留 Lance vector/BM25/rerank/recency/neighbors 和 TruffleHog gate。
兼容 HTTP 服务只用于迁移/测试，不替代 Funes 检索引擎。

当 `FUNES_MEMORY_ONLY=1` 用于避免轻量 HTTP daemon 重复运行 native index 时，它仍会把
session 原文和 source metadata 写入加密 sidecar；历史 session 的派生索引则由
`deploy/funes-sync/native-backfill.sh` 调用官方原生 parser/index/push。它完成首次回填后不会退出，
而是按 `FUNES_NATIVE_BACKFILL_RECONCILE_INTERVAL`（默认 300 秒）继续扫描 Codex、Pi、Claude
和 Hermes 的新建/追加 session；`--yes` 会一次 drain 完整 tier backlog，避免每 60 秒重建
text/vector index；这样 memory-only daemon 不会留下未来 session 的接管空档。macOS `lockf`
内核锁会在最后一个进程描述符关闭时自动释放，不依赖 PID 或删除 lock 文件，Mac/进程重启不会
因 stale lock 永久停摆。
原生 auto-discovery 之外还显式复用官方 parser 扫描 Codex `archived_sessions`/`subagents`、
Pi legacy/custom session roots 和 Claude `history`，不会把这些目录改用兼容 parser。
原生 push 成功后，LaunchAgent 会通过受保护的 `/warm` 通知让 HF Space 在后台刷新读取
worker；通知默认按 `FUNES_NATIVE_WARM_MIN_INTERVAL=300` 节流，避免历史回填期间重复加载
embedding/index。通知失败不影响本地已完成的 durable push。

## 快速使用

```sh
python3 -m sync sources
python3 -m sync backfill                         # 首次扫描并尽量排空队列
python3 -m sync reconcile                        # 强制核对远端 source identity 并仅补缺失项
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
  每批同时受 `FUNES_SYNC_BATCH` 条数和 `FUNES_SYNC_MAX_BATCH_BYTES`（默认 16 MiB）限制。
- `reconcile`：分页调用受保护的 `/sources/check`，用持久 cursor 断点核对远端；只把缺失
  identity 重新入队。首次运行由 daemon 自动执行一次，切换远端会按 URL 隐私指纹重新执行；
  显式命令会强制重查同一 URL，适合远端重建恢复。
- `status`、`sources`、`doctor`、`logs [--follow]`：只读诊断。
- `install`：原子写入 `~/Library/LaunchAgents/com.funes.sync.plist`；遇到非本工具 plist 会拒绝覆盖，`--force` 仅允许覆盖同 label。
- `start`、`stop`、`restart`：macOS `launchctl bootstrap/bootout`；Linux 上安全返回错误，不启动后台进程。

离线 push 失败不会丢数据：pending 队列留在 SQLite，只有远端 Hub commit 完成并由 operation
poll 返回 `durable=true` 后才清空。`202` 只表示处理中，永远不会触发本地 ACK。原生
`funes push` 的 CAS、恢复失败保护和 TruffleHog gate 是远端发布的最终边界；LaunchAgent
通过 macOS Keychain 读取 token，不把 token 写入 plist。

## 测试

```sh
python3 -m unittest discover -s scripts/funes-sync/tests -p 'test_*.py'
```

测试覆盖稳定去重、离线队列重试及中文原文保留。
