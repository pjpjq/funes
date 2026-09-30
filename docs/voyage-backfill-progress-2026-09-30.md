# Voyage 回填进展与验收边界（2026-09-30）

## 当前结论

**生产服务能够恢复查询就绪，但回填没有取得新进展；本轮不是性能验收通过。**
本文的状态采样时间为 **2026-09-30 15:30:39–15:30:44 UTC / 北京时间 23:30:39–23:30:44**。
9 月 25 日的吞吐观测是历史记录，不能作为当前速率或 ETA。

| 项目 | 本轮证据 |
| --- | --- |
| Space 控制面 | `RUNNING`，`cpu-basic`，1 replica |
| 生产 revision | `720090670ef7ccbd12cd3eec71026d6e13cbbdd4`，未部署本地 `22ca099` 指标变更 |
| `/health` | 15:27 UTC 返回 `200` |
| 就绪状态 | 15:27 UTC `/ready=503 native_warm_warming`、`/sync/status=503`；15:30 UTC `/ready/search=200`、`/sync/status=200` |
| 既有 MCP recall | 本轮首次调用返回 `503`；不把早期成功 recall 算成本轮通过 |
| 原始来源记录 | `4,194,602` |
| eligible / indexed / pending | `1,818,757 / 545,223 / 1,273,423` |
| held / invalid | `111 / 0` |
| checkpoint / 全量完成 | `checkpoint_current=true`，`complete=false`，`cutover_ready=false`，`optimize=pending` |
| 最近成功回填时间 | `2026-09-25T10:07:30.330722Z`，距采样约 **125 小时** |
| 最近周期 | `attempted=64, indexed=0, held=0, durable=true`，耗时 `15,929 ms` |
| reconciler | `thread_alive=true`，`phase=sleeping`，下一轮等待 `300 s` |

`1,818,757 - 545,223 - 111 = 1,273,423`，计数内部一致；这不代表
pending 已处理完，也不证明每条 pending 的原因。`indexed` 计数变化本身不能证明向量丢失。

## 已定位的监控缺口；尚未定位的底层错误

生产 `space/server.py` 的失败路径可以将 native ingest 失败转换为逐条 `retry`
（如 `native_exit`、超时、报告解析失败或 stale），随后成功持久化这些失败状态。
`_canonical_reconcile_background()` 却只按 `durable` 清空 `last_error`
和失败计数，因而可能同时出现 `indexed=0`、`durable=true`、`last_error=null`。
PostgreSQL 状态快照又返回 `failures=null`、`failure_counts_status=deferred`。

因此，**线程存活、PG 状态写入成功及 `last_error=null` 均不能证明 embedding 正常**。
监控误报路径已由代码确认，但底层 native ingest 失败的实际类别/原因尚未取得；
不能据此断言 Voyage 额度耗尽、provider 限流、网络故障或迁移损坏。

## 已有实现与本轮回归

- `0067742` / `e628184`：内容不变、仅 `source_version` 变化时保留有效 native index 状态，并补充回归。
- `c6258e8`：纠正旧 key 的限额判断；保留现有 key，但没有证据确认其账单等级或精确 RPM/TPM。
- `22ca099`：增加 allowlist 指标（ingest/Voyage 分阶段耗时、HTTP 状态、请求/token、pacer/backoff），不输出 raw 文本或凭据；透传有界 profile concurrency。
- `22ca099`：增加 `canonical_ab.py` 三档控制器和测试；**控制器尚未接入真实执行调度**，有单元测试不等于已运行生产 A/B。

以下命令在本机、代码提交 `22ca099` 上于 **2026-09-30 重新执行**：

```text
PYTHONPATH=. pytest -q tests_space/test_canonical_ab.py tests_space/test_server.py tests_space/test_deployment_entrypoint.py
214 passed in 43.79s; exit 0

PYTHONPATH=. pytest -q tests_space/test_server.py service/tests/test_service.py -k 'source_version_only_change_skips_native_reconcile or source_version_change_preserves or source_version_change_with_new_raw'
5 passed, 316 deselected in 0.08s; exit 0

git diff --check HEAD~2..HEAD
exit 0
```

以上仅证明定向 Python 回归。未在当前代码上重新完成 Rust `fmt` / `clippy` /
`test`、ONNX clippy 或真实 provider 吞吐验收。旧报告中的 Rust 通过记录不能替代这些验证。

## Alice 与持久化边界

2026-09-30 本轮通过本机 `ssh alice` 直连新机器，实测 **16 CPU、32,049 MiB RAM、
根卷约 287 GiB 可用**。旧 `/tmp/funes-ab-22ca099`、`/root/work/funes`、
`/root/.cargo/bin` 均不存在；不能把旧编译任务描述为仍在运行或已经成功。

Alice 只用于可重建的临时重任务；用户确认其每 24 小时销毁。本轮未在 Alice 启动
新的长编译、回填或 embedding。后续构建/测试日志和校验结果须及时带回本机或 GitHub，
不得只留在 Alice。

## 未完成项与下一步

1. 先读取正确来源库或生产运行日志中的脱敏错误类别，解释为何 `attempted>0` 而 `indexed=0`；不能用重复重启代替诊断。
2. 修复/验证失败可观测性，并在新 Alice 上恢复可复现的 Rust 测试和构建；保存持久证据。
3. 将 A/B 控制器接入隔离执行，真实测量 `baseline_c2_r64`、`c4_r64`、`c4_r128` 各 4 周期，记录吞吐、请求/token、429/5xx、阶段耗时。
4. 只有回归与小规模 A/B 通过后，才决定剩余回填配置和可信 ETA；当前没有“最优配置”或“吞吐已提升”的结论。
5. 全量完成后再验收 queue、跨 Agent recall、中文 benchmark 和恢复能力；不把历史通过结果当成本次完整交付。

本轮未修改 HF 配置或 Secret、未主动重启 Space、未重建 PG/Lance、未清空 checkpoint，
也未提交新的批量 Voyage 回填请求。状态检查和已有服务自身活动不等于暂停后台任务。
