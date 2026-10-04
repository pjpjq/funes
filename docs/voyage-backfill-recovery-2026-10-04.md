# Voyage 回填恢复与验收边界（2026-10-04）

## 本轮结论

**远程 Lance open 的超时已解除，生产连续两个周期成功索引；真实查询稳定性和全量回填尚未通过。**
以下时间均为 UTC；北京时间加 8 小时。计数是生产采样值，不是估算。

本轮继续运行现有生产任务，没有重建/重新导入 PG、清空 Lance/checkpoint、轮换 Secret
或再次重启 Space。生产继续使用 `voyage-4-lite`、1024 维、schema 2。

## 发布身份与最小修复

| 项目 | 实际值 |
| --- | --- |
| 源码修复 | `f64efc1dd3e787e61dc4df6b78f87437c0de3c73` |
| 发布资产提交 | `be256128add230476ec08a9e2ec5b9fa07ddcfd4` |
| HF Space revision | `8f34235a11d6156b3bba73d36e3273abbc518bf7` |
| 解压二进制 SHA256 | `755e66f5ecc88b1bde6b1ddf5d628a847cf34d8bc9174bb49db3336279770d97` |
| CI / Release | runs `37171337191` / `37171337192` 均 success，real-embedder integration 已 success |

源码让 GET/HEAD/range 优先直接访问 deterministic shard key；只有 NotFound 才兼容旧
flat 路径。缺失的 sharded version hint 不再回退到陈旧 flat hint。生产旧仓库仍然
缺少 sharded hint，因此仅有源码修复还不足以避免首次枚举全部版本。

03:02:01，对 `bolikoto/funes-memory-voyage` 完成一次受 `parent_commit` CAS 保护的
hint 补齐：

```text
parent: ad635fb2bc81cda5040e486619da1416296e1b01
commit: 3b17946deac9ede2919453a9224a46096eb4c603
path:   __funes_shards__/v1/a1/chunks.lance/_versions/latest_version_hint.json
value:  {"version":12755}
```

只新增上述一个文件；逐对象对比确认原有 **40,791** 个对象的 blob ID、大小、LFS
元数据不变。版本由 **12,755** 个 V2 manifest 的实际文件名解码确认；没有修改
manifest、已有向量、来源文本、PG 状态或 checkpoint。该补齐操作没有调用 Voyage。

03:21:50 的只读 GET 返回 `200`，hint 已由正常回填自动更新为 `12757`，对应
repo commit `ea95d912a64cc3a3ff08ca7ba913f8427693ecc1`；无需逐周期手工补 hint。

## 生产恢复：失败与两个连续成功周期

| 周期结束 UTC | attempted / indexed | 周期 ms | remote_open ms | revision_lookup ms | Voyage 请求 / 输入 / tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 03:04:49，前一次尝试 | 32 / 0 | 1,804,017 | 1,381,113.18 | 161,604.78 | 0 / 0 / 0 |
| 03:14:16，补 hint 后首次成功 | 32 / 32 | 267,180 | 243.03 | 166,553.86 | 24 / 188 / 155,881 |
| 03:16:13，第二次成功 | 32 / 32 | 116,130 | 293.58 | 16,293.80 | 24 / 187 / 155,561 |

失败周期报告 `TimeoutExpired:32`，且 Voyage 为零请求，证明这次失败发生在 embedding
之前；不是该周期的 provider 429。两个成功周期所有 Voyage 请求均为 HTTP 200，
backoff 为零，失败计数清零，`durable=true`，checkpoint 保持 current。

首次成功周期 embedding 为 `61,867.81 ms`；第二次为 `64,713.51 ms`、vector reuse
为 `25,667.12 ms`、Lance write commit 为 `2,240.52 ms`。pacer wait 是并发请求累计值
（第二次 `117,557.28 ms`），**不可把它与 wall time 直接相加**。输入计数是 chunk
embedding 输入，不是 source document 行数。

03:07:57 开始的 native warm 于 03:12:41 完成，用时 **284 秒**；随后 `/ready=200`、
active worker alive。后续一次替换 warm 于 03:15:14–03:16:06 完成，用时 52 秒。
这不是容器重启实验，不能推广为所有冷启动固定耗时。

## 实际计数与仍未完成的门槛

| 采样 UTC | documents | eligible | indexed | pending | held | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 03:10:42 | 4,363,095 | 1,921,985 | 644,518 | 1,277,302 | 165 | 0 |
| 03:17:30 | 4,363,326 | 1,922,095 | 644,582 | 1,277,348 | 165 | 0 |
| 03:20:13 | 4,363,547 | 1,922,199 | 644,582 | 1,277,452 | 165 | 0 |
| 03:32:06 | 4,363,945 | 1,922,329 | 644,653 | 1,277,511 | 165 | 0 |

首次两轮 `indexed` 累计增长 **64**，03:32 采样相对恢复前累计增长 **135**；同期持续新增来源，所以不能要求 pending 在每个采样点
都下降。各采样均为 `checkpoint_current=true`、`complete=false`、`cutover_ready=false`。
最近成功周期粗速率为 `32 / 116.130 ≈ 0.276 documents/s`；这是两个恢复周期中的一个，
不是完成性能 A/B 后的稳定吞吐，不能用于承诺全量 ETA。

## 查询、鉴权和本机同步验收

- 生产控制面为 `RUNNING`，hardware `cpu-basic`；本轮 `/health` 均为 200。
- 错误 application token 返回 401；正确 token 的 canonical `/get`（明确不存在的
  `source_identity`）返回 404 `not_found`，耗时 1.867 秒。这验证鉴权与路径，不冒充
  成功读取真实记忆。
- 根 agent 的真实 `funes-remote recall` 返回 HTTP 503。后续观测到 read worker
  `last_failure.category=timeout`，并自动进入 replacement warm；03:20 的 `/ready`
  再次为 503。不能把 `/ready` 曾返回 200 算作真实 recall 或跨 Agent 通过。
- 本机只读检查：1,789,520 records；pending/pending uploads/failed uploads 均为 0；
  Codex 3,413、Pi 9、Claude 280、memory files 139 已 discovered/parsed/synced；
  `com.funes.sync` running，PID 89966，历史 backfill 标记均 true。
- Codex stdio MCP initialize/tools-list 已通过；Pi extension 与 Claude MCP 配置存在。
  这不证明真实 Codex→Pi 或 Pi→Codex recall。尚无 OOM/RSS/负载证据，不能把 worker
  timeout 的底层原因直接归为内存不足。

### 查询预算核对（代码证据，不是超时根因结论）

`sync/mcp_bridge.py` 与 Pi extension 的默认 recall 单次请求为 8 秒、总预算为 18 秒；
Space `VOYAGE_NATIVE_TIMEOUT` 为 12 秒、`VOYAGE_HTTP_TIMEOUT` 为 14 秒。客户端存在
提前放弃的预算错配，但 HTTP 断开没有被映射为 native cancellation，不能用客户端
8 秒预算解释服务端自身的 12 秒 timeout。`NativeMcpWorker._call()` 收到 timeout 后主动
终止子进程是已确认机制；底层慢阶段仍需诊断。

新 hint 在每次 captured commit 中自动生成，且与新 manifest 归入最后 CAS activation
chunk，确保数据对象全部上传完成后再更新版本；真实 hint 自动增长已验证此闭环。

## 下一步，保持现有数据与运行任务

1. 定位真实 read timeout 的阶段；在查询可靠前不重复启动大规模中文 benchmark。
2. 核对每轮 32 documents / 24 Voyage requests 的字符/token/请求约束；先小规模 A/B，
   再选择配置，不凭恢复后的单轮数据宣称最优吞吐。
3. 保留现有 checkpoint 继续回填，完成后再做 optimize/cutover 与双向跨 Agent 验收。
4. 最终验收仍要求 pending/invalid 为零、complete/cutover-ready、raw Chinese 正确返回。
   不把 held 的 secret-gated 记录强行上传。

脱敏机器证据见 `docs/evidence/voyage-backfill-recovery-2026-10-04.json`。临时原始状态
采样位于 `/tmp/funes-status-*Z.json`；这里仅保留 allowlisted 计数、阶段耗时和状态，
不保存 credential、Authorization 或完整原始会话。
