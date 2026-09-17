# funes-sync：本地多 Agent 记忆同步

`sync/` 是唯一的同步实现；`scripts/funes-sync/funes-sync` 只是兼容入口。它发现 Codex（含
active/archived）、Claude Code、pi 的会话，以及 Codex/Claude/pi 和仓库中的 `AGENTS.md`/`MEMORY.md`，
将每个新文件转换为带稳定 source id 的记录。默认生产路径把原文和 source metadata 通过
authenticated HTTP 持久化到加密 sidecar；HF Space 的 canonical reconciler 再从该 source of
truth 增量构建可重建的 Lance vector/BM25 索引。检索仍由原生 Funes 引擎完成。

macOS 默认只安装轻量 `com.funes.sync` LaunchAgent。它在 `FUNES_MEMORY_ONLY=1` 模式下仍会
发现、解析、去重、watch 和上传 Codex/Pi/Claude session 与 memory 文件；Space 负责 derived
index，因此 Mac 不需要持续重建同一份本地向量索引。只有没有 canonical reconciler 的 legacy
远端才设置 `FUNES_NATIVE_PRIMARY=true`，额外安装 `com.funes.native-backfill` 并运行
`deploy/funes-sync/native-backfill.sh`。从 native 模式切回默认模式后再次运行 `funes sync install`
会停止并移除旧 helper plist，不会让它在重启后复活。

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
- `install`：原子写入 `~/Library/LaunchAgents/com.funes.sync.plist`；遇到非本工具 plist 会拒绝覆盖，`--force` 仅允许覆盖同 label。可从 `config.toml` 的 `[sync]` 或既有 plist 环境变量持久继承 `remote_timeout`（`FUNES_REMOTE_TIMEOUT`）与 `remote_transient_retries`（`FUNES_REMOTE_TRANSIENT_RETRIES`），token 等敏感凭据绝不落盘。
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
