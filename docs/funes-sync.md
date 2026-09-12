# funes-sync：本地多 Agent 记忆同步

`scripts/funes-sync/funes-sync` 是一个只写 `~/.funes` 的 stdlib Python 桥接层。它发现 Codex（含
active/archived）、Claude Code、pi 的会话，以及 Codex/Claude/pi 和仓库中的 `AGENTS.md`/`MEMORY.md`，
将每个新文件转换为带稳定 source id 的 Claude-compatible JSONL，再调用现有 `funes index`。
原始 UTF-8 内容同时保存在生成记录的 `funes_sync.raw` 和 assistant message 中；不会修改 agent 配置。

## 快速使用

```sh
scripts/funes-sync/funes-sync sources
scripts/funes-sync/funes-sync once                 # 一次扫描、索引
scripts/funes-sync/funes-sync status
FUNES_BIN="$HOME/.local/bin/funes" FUNES_MEMORY=org/funes-memory \
  scripts/funes-sync/funes-sync daemon
```

`FUNES_HOME`（默认 `~/.funes`）下的 `sources/` 是合成 JSONL，`sync-state.json` 记录 mtime/size/hash、
soft-missing 标记和远端 pending 队列，`funes-sync.log` 是运行日志。mtime 和 hash 均未改变时不会重写
源文件；源文件消失只标记 `missing`，不删除历史 JSONL。

## 命令

- `once`/`sync`：发现、转换、调用 `FUNES_BIN index <sources> --harness claude --yes`；配置 `FUNES_MEMORY` 后再调用 `funes push`。
- `daemon`：按 `FUNES_SYNC_INTERVAL`（默认 300 秒）循环。
- `status`、`sources`、`doctor`、`logs [--follow]`：只读诊断。
- `install`：原子写入 `~/Library/LaunchAgents/com.funes.sync.plist`；遇到非本工具 plist 会拒绝覆盖，`--force` 仅允许覆盖同 label。
- `start`、`stop`、`restart`：macOS `launchctl bootstrap/bootout`；Linux 上安全返回错误，不启动后台进程。

离线 push 失败不会丢数据：pending 队列留在状态文件，下一次成功的 index/push 后才清空。网络和 token
由现有 funes CLI 管理；同步层不保存 token。

## 测试

```sh
python3 -m unittest discover -s scripts/funes-sync/tests -p 'test_*.py'
```

测试覆盖稳定去重、离线队列重试及中文原文保留。
