# Voyage 回填恢复与验收边界（2026-10-04）

## 本轮结论（15:08:29 UTC 最新快照）

**当前仍未完成全量回填；生产 worker 正在一次 warm replacement，`/ready=503` 是 `native_warm_warming`，不是本轮新增错误。** HF revision `0a8ee555234349aaf747b215fb77348a0eb9deaf` 与预期一致；不重启、不重建、不清理 Lance/PG/checkpoint。

1. **当前生产状态**：Space `RUNNING`，`/health=200`；15:08:29 UTC `/ready=503`、`/sync/status=503`，warm started `15:01:50Z`、`finished_at=null`、`refresh_pending=true`。source store 已 ready；native warm 尚未完成。
2. **回填仍有进展**：documents `4,402,146`；eligible `1,948,205`（随后只读接口读到 `1,948,212`）；indexed `659,072`；pending `1,288,947`（随后只读接口读到 `1,288,954`）；held `186`；invalid `0`；checkpoint current；complete/cutover_ready 均 false。最新成功周期 `37/37`、`80,857 ms`，durable=true。
3. **Voyage 实测**：最新周期 24 次 HTTP 200、无 retry；embedding `70,555 ms`。本轮 A/B 已完成 12/12，268 条 durable 写入、192 请求，429/5xx/retry 均为 0。短测最快为 `c4_r64`（周期均值约 `1.661 rows/s`，仅 cycle wall time）；`c4_r128` 因单周期约 `153k–159k` tokens 超过生产本地 `120k TPM/60s` pacer，约 `61s` embedding，不能采用。
4. **限流结论与边界**：慢因是批次 token 预算超过进程内 120k TPM 滑窗，不是已证实的 Voyage 429/5xx。`c4_r64` 的短测跨 native process 重置 pacer，不能直接外推长期稳定吞吐；当前生产变量未改、Secret 未轮换、Space 未主动重启。
5. **完成门槛**：仍需 `pending=0`、`invalid=0`、`complete=true`、`cutover_ready=true`、最终 optimize/native ready。当前 backlog 仍会随新 eligible 记录增长，因此没有可诚实承诺的有限 ETA。

本节是当前快照；下方历史时间线保留先前部署、维护、查询和失败证据，不覆盖当前事实。

## 发布身份与版本演进

| 项目 | 08:xx 已核验发布身份（历史） | 03:xx 历史恢复版本 |
| --- | --- | --- |
| 源码提交 | `49390fe`（阶段性维护索引） | `f64efc1dd3e787e61dc4df6b78f87437c0de3c73` / `be256128add230476ec08a9e2ec5b9fa07ddcfd4` |
| HF Space revision | `4d5a40f70ca658901e4543559bf49a1c588356f9` | `8f34235a11d6156b3bba73d36e3273abbc518bf7` |
| 解压二进制 SHA256 | - | `755e66f5ecc88b1bde6b1ddf5d628a847cf34d8bc9174bb49db3336279770d97` |
| CI / Release | 连续集成通过 | runs `37171337191` / `37171337192` 均 success |

### 新修复发布与部署后门禁（13:26–13:51 UTC）

13:12 文档更新时，bitmap / snapshot warm guard / canonical A/B runner **当时尚未部署**；之后已完成原子发布。`/tmp/funes-bitmap-atomic-deploy-20261004-result.json` 的时间为 `2026-10-04T13:26:03.880466+00:00`、readback_verified=true；13:51:19 控制面/runtime SHA 已匹配新 revision。**文件部署成功不是服务验收成功**：ready 503 与 active maintenance 仍阻断最终结论。

| 验证对象 | 当前结果 | 证据与边界 |
| --- | --- | --- |
| 源码 / 资产 / 当前 HEAD | 发布源码 `2a96c9864211e86f7d1f055f4a29497e19a7d1dd`；资产 `3ba7c9620fc0199968d2b0d68ea818f97b8dbf32`；当前 HEAD `06bbc7da9e7109f7aa3223ca69748ba47db7017b` | 发布身份取自原子部署文件；HEAD 已本地读回，CI 全 success 来自协调代理；三者分别记录 |
| HF revision / 保留范围 | `0a8ee555234349aaf747b215fb77348a0eb9deaf`，原子 readback 成功，13:51 runtime SHA 匹配 | preserves_environment=true，secrets_changed=false，source_store_changed=false，indexes_cleared=false；不展示环境或 Secret 值 |
| health / ready / sync/status | 200 / 503 / 503 | stage RUNNING 不等于 ready；warm 为 warming，maintenance active，部署服务门禁未通过 |
| Linux 二进制 / zstd 包 | 二进制 SHA256 `11183d5d4418b4d81bddff53e92859301d681158f7d17a57a27065ff87a02512`；zstd SHA256 `aa504f22a41c18ab1b0e1d230d0921a009bc2e8fbd876f3440262a0f48b7ba87`，66,158,972 bytes | 发布产物与之前本地核验一致；Trixie glibc 2.41 version/help 通过为历史构建验证，不冒充运行中二进制直接读取 |
| CI / deployment entrypoint 回归 | HEAD `06bbc7d` CI 全 success；7 passed / 0.34 秒 | 协调代理证据；并非生产 ready、过滤性能或 A/B 已通过 |

### 构建与测试历史（13:12 更新时）

| 验证对象 | 结果 | 证据与边界 |
| --- | --- | --- |
| Python 完整依赖套件 | 907 passed / 39 skipped，118.12 秒 | `/tmp/funes-full-python-20261004.log`；4 warnings、16 subtests passed；运行后又做了 snapshot warm 时序修复 |
| 修复后的 focused 套件 | 324 passed / 1 skipped，58.01 秒 | `/tmp/funes-final-focused-20261004.log`；不冒充修复后的全套复跑 |
| Rust 测试与 clippy | 335 passed；普通 / ONNX clippy 通过 | 协调代理提供的 Alice 日志结果，非本机重跑 |
| 当时部署状态 | 13:12 尚未部署 | 历史时间线保留；13:26 原子部署后已被新发布证据更新，不能继续作为当前状态 |

源码让 GET/HEAD/range 优先直接访问 deterministic shard key；只有 NotFound 才兼容旧 flat 路径。缺失的 sharded version hint 不再回退到陈旧 flat hint。生产旧仓库仍然缺少 sharded hint，因此仅有源码修复还不足以避免首次枚举全部版本。

03:02:01，对 `bolikoto/funes-memory-voyage` 完成一次受 `parent_commit` CAS 保护的 hint 补齐：

```text
parent: ad635fb2bc81cda5040e486619da1416296e1b01
commit: 3b17946deac9ede2919453a9224a46096eb4c603
path:   __funes_shards__/v1/a1/chunks.lance/_versions/latest_version_hint.json
value:  {"version":12755}
```

只新增上述一个文件；逐对象对比确认原有 **40,791** 个对象的 blob ID、大小、LFS 元数据不变。版本由 **12,755** 个 V2 manifest 的实际文件名解码确认；没有修改 manifest、已有向量、来源文本、PG 状态或 checkpoint。该补齐操作没有调用 Voyage。

03:21:50 的只读 GET 返回 `200`，hint 已由正常回填自动更新为 `12757`，对应 repo commit `ea95d912a64cc3a3ff08ca7ba913f8427693ecc1`；无需逐周期手工补 hint。

## 生产恢复：03:xx 历史失败与连续成功周期

| 周期结束 UTC | attempted / indexed | 周期 ms | remote_open ms | revision_lookup ms | Voyage 请求 / 输入 / tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 03:04:49，前一次尝试 | 32 / 0 | 1,804,017 | 1,381,113.18 | 161,604.78 | 0 / 0 / 0 |
| 03:14:16，补 hint 后首次成功 | 32 / 32 | 267,180 | 243.03 | 166,553.86 | 24 / 188 / 155,881 |
| 03:16:13，第二次成功 | 32 / 32 | 116,130 | 293.58 | 16,293.80 | 24 / 187 / 155,561 |

失败周期报告 `TimeoutExpired:32`，且 Voyage 为零请求，证明这次失败发生在 embedding 之前；不是该周期的 provider 429。两个成功周期所有 Voyage 请求均为 HTTP 200，backoff 为零，失败计数清零，`durable=true`，checkpoint 保持 current。

首次成功周期 embedding 为 `61,867.81 ms`；第二次为 `64,713.51 ms`、vector reuse 为 `25,667.12 ms`、Lance write commit 为 `2,240.52 ms`。pacer wait 是并发请求累计值（第二次 `117,557.28 ms`），**不可把它与 wall time 直接相加**。输入计数是 chunk embedding 输入，不是 source document 行数。

03:07:57 开始的 native warm 于 03:12:41 完成，用时 **284 秒**；随后 `/ready=200`、active worker alive。后续一次替换 warm 于 03:15:14–03:16:06 完成，用时 52 秒。这不是容器重启实验，不能推广为所有冷启动固定耗时。

## 08:xx 索引维护与 Manifest 覆盖改善

08:28:19 至 08:36:24 UTC，在 Space `4d5a40f70ca658901e4543559bf49a1c588356f9` 上执行阶段性维护，运行耗时 **485,860 ms**（约 8.1 分钟），结果为 `maintenance_success`，无错误（`has_error=false`）。

- **Manifest 元数据**：
  - 仓库：`bolikoto/funes-memory-voyage`
  - 提交 Revision：`91fa1cd946dcf6edf53467c47e9b104737e8c416`
  - Manifest 版本：`12866`（对应 `chunks.lance/_versions/18446744073709538749.manifest`）
  - 文件大小：1,620,103 bytes
  - SHA256：`8f7c05f8a27765cf6287bda286bba31e0974fefa1c25c51e1397d6fd1f56c9b2`
- **4 项索引联合覆盖对比（基线 v12849 vs 维护后 v12866）**：
  基线 live fragments 为 11,313，physical rows 为 1,574,367；维护后 live fragments 为 11,326，physical rows 为 1,577,005。

| 索引名称 | 维护前覆盖分片 (行数) | 维护前未覆盖分片 (行数) | 维护后覆盖分片 (行数) | 维护后未覆盖分片 (行数) | 未覆盖物理行比例变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `text_idx` | 7,540 (1,232,225) | 3,773 (342,142) | 11,325 (1,576,810) | 1 (195) | 21.73% → 0.012% |
| `vector_idx` | 7,540 (1,232,225) | 3,773 (342,142) | 11,325 (1,576,810) | 1 (195) | 21.73% → 0.012% |
| `source_identity_idx` | 7,540 (1,232,225) | 3,773 (342,142) | 11,325 (1,576,810) | 1 (195) | 21.73% → 0.012% |
| `source_agent_idx` | 7,540 (1,232,225) | 3,773 (342,142) | 11,325 (1,576,810) | 1 (195) | 21.73% → 0.012% |

- **分片与行数净变化**：已覆盖分片净增 +3,785，已覆盖物理行数净增 +344,585；未覆盖分片减少 -3,772（降至 1），未覆盖物理行数减少 -341,947（从 342,142 降至 195，减少 99.94%）。
- **物理行概念澄清**：未覆盖物理行数（342,142 → 195）严格对应 Lance `chunks.lance` 的底层物理分块记录数，**绝非**业务层 source document 计数。
- **标量索引说明**：Lance 当前**不存在标量 `source_type` 索引**（`scalar_source_type_index_present = false`）。
- **验证边界**：仅基于 Fragment bitmap 联合与物理行数比对，不包含删除行调整、查询性能与质量测试、向量重算、重构索引、远程写入或 checkpoint 变动。

## 实际计数采样与本地同步状态

### 生产采样历史与最新观测

| 采样 UTC | documents | eligible | indexed | pending | held | invalid | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 03:10:42 | 4,363,095 | 1,921,985 | 644,518 | 1,277,302 | 165 | 0 | 恢复初期基线 |
| 03:17:30 | 4,363,326 | 1,922,095 | 644,582 | 1,277,348 | 165 | 0 | 连续成功第 2 周期 |
| 03:20:13 | 4,363,547 | 1,922,199 | 644,582 | 1,277,452 | 165 | 0 | 调和器持续运行 |
| 03:32:06 | 4,363,945 | 1,922,329 | 644,653 | 1,277,511 | 165 | 0 | 累计增长 135 docs |
| 08:57:01 | 4,377,681 | 1,930,496 | 649,360 | 1,280,965 | 171 | 0 | 08:xx 历史稳定采样 |
| 13:12（分钟精度） | 4,394,131 | 1,942,124 | 656,928 | 1,285,013 | 183 | 0 | 部署前历史采样 |
| 13:51:19 | 4,397,670 | 1,945,305 | 657,345 | 1,287,777 | 183 | 0 | 新 revision 已部署；warm/maintenance 未完成 |

在历史 08:57:01 采样中：
- `indexed` 达到 649,360，相较 03:32 增长 4,707。
- `pending` 为 1,280,965，`held` 为 171，`invalid` 为 0。
- `checkpoint_current = true`，`complete = false`，`cutover_ready = false`。
- 调和器状态活跃（`active = true`，phase `native_ingest`），单轮用时 72,484 ms，本轮 attempted 33 / indexed 33，durable=true，Voyage 请求 24 次（全部 HTTP 200，输入 188，消耗 154,063 tokens）。
- 控制面 `/health = 200`，`/ready = 200`，预热状态为 ready（08:54:11 至 08:54:16Z，耗时 5 秒）。

### 13:12 UTC 历史采样（部署前）

`indexed=656,928`，较 08:57 增加 7,568；但新 eligible 记录仍在进入，因此不能由 indexed 增长推断 pending 已清零。当前 `pending=1,285,013`，`held=183`、`invalid=0`，checkpoint current，回填与 cutover 均未完成。

最近一轮 35/35、72,888 ms，24 次 Voyage HTTP 200、154,507 tokens；embedding `62,126.63 ms`，pacer 累计 `117,422.96 ms`，backoff `0`。pacer 是并发请求累计等待，**不可与周期 wall time 相加**。维护累计 9 成功 / 0 失败，最近于 12:56:04 完成、耗时 7,309 ms。

### 部署后生产快照与观测边界

- **13:35:42.316218 UTC runtime detail**：health 200、ready 503、sync/status 503；source store ready=true、native worker 未 configured/alive，warm 开始于 13:27:41、refresh_pending=true。`/admin/canonical-ab/status` HTTP 200，phase idle，完成 0/12 cycles；这是该时点状态，不是已执行的 A/B 结果。
- **13:51:19.374346 UTC live summary（最新）**：runtime SHA 与 `0a8ee555234349aaf747b215fb77348a0eb9deaf` 匹配；warm 开始于 `2026-10-04T13:49:07Z`、finished_at=null、refresh_pending=false；phase `index_maintenance`，maintenance 于 `2026-10-04T13:38:23.265915+00:00` 开始、仍 active，successes/failures=0/0、last_finished_at=null。计数重置后的本进程 0/0 不推翻部署前累计 9/0 的历史维护结果。
- **首次观测周期 metrics**：remote_open `487.16 ms`，revision_lookup `631677.86 ms`，secret_scan `3006.62 ms`；Voyage 0 requests / 0 inputs / 0 tokens / 0 backoff。last_result、last_finished_at、last_duration_ms 均为空，不能计作成功周期。
- **验收边界**：checkpoint_current=true，但 complete=false、cutover_ready=false、optimize=pending；仍有 `1,287,777` pending。后续 live、过滤查询、Gemini 需求 C 与吞吐 A/B 结果由协调代理另行追加。

### 本机只读同步状态（13:40 UTC；不等于远程回填）

`/tmp/funes-bitmap-local-status-20261004.json` 无独立采样 utc 字段；最后成功同步为 `2026-10-04T13:40:21.936312+00:00`，文件 mtime 为 `2026-10-04T13:41:16.308838+00:00`（mtime 不是采样时间）。records `1,803,851`、sources `4,189`（active `3,861`），pending / pending uploads / failed uploads 均 0；discovered / parsed / synced 同为 Codex 3,435、Pi 9、Claude 280、memory 137。

initial_backfill_complete=true，source_schema_complete=true，zero_record_repair_complete=true，automation_identity_complete=true，remote_source_reconciliation.complete=true；**remote_ready=false**。这些证明本机来源同步/上传队列完成，不证明远程 Lance 全量向量回填或服务 ready。

### 本机只读同步状态（08:46:44 UTC）

- 数据源总数：4,182（活跃 3,856）。
- 本地记录总数：1,798,613；pending / pending uploads / failed uploads 均为 0。
- 来源会话发现与同步闭环（discovered / parsed / synced 均为 100%）：
  - Codex sessions：3,428
  - Pi sessions：9
  - Claude sessions：280
  - Memory files：139
- 按 Agent 统计记录数：`codex` 1,781,077；`claude_code` 17,501；`pi` 34；`shared` 1。
- 守护进程状态：`com.funes.sync` 已安装并加载运行（`/Users/pwd/Library/LaunchAgents/com.funes.sync.plist`）；`com.funes.native-backfill` 未安装且未加载。

## 真实 MCP 探针历史（08:xx；保留失败时间线）

以下保留 08:xx 当时的失败结果，不表示它们都仍是当前阻断。Pi 的后续成功见本节末；Gemini 需求 C 与生产过滤修复仍待验收。

### 1. 已通过项

| 探测类型 | 耗时 | 状态 | 验证结果 |
| --- | ---: | --- | --- |
| Codex CPA 普通查询 | 5,814 ms | PASSED | 成功返回真实中文原文，内容匹配验证通过 |
| Codex → Pi Tailscale 查询 | 1,698 ms | PASSED | 跨 Agent 检索命中真实中文上下文，验证通过 |
| 混合标识符查询 | 3,124 ms | PASSED | 混合多源标识符召回，中文原文验证通过 |
| `/get` 已知身份读取 | - | PASSED | 端点 HTTP 200，精确字面量匹配，无 shadow 数据 |

### 2. 失败与未通过项

1. **Gemini 无过滤查询失败**：
   - 耗时 2,571 ms，返回结果的 Top 3 中缺失 High 优先级关键记录，**需求 C 未达成**（`top3_missing_high_requirement_c_failed`）。
2. **Gemini `source_type=memory` 过滤查询超时失败（HTTP 503）**：
   - 08:48:50Z 触发 `read_vector timeout`，08:48:58Z worker 进程崩溃退出并返回 HTTP 503。
   - 随后系统触发 native warm 自动拉起替换 worker 并恢复健康。
   - **当时状态：过滤优化推进中，尚未修复。** 之后 bitmap/guard 修改已获测试证据，并于 13:26 原子部署；截至 13:51 ready 仍为 503，尚未进行部署后真实过滤验收，仍禁止把生产查询记为 fixed；当时底层确认 Lance 缺乏标量 `source_type` 索引。
3. **Pi 真实 CLI 探针失败（HTTP 401）**：
   - 执行耗时 8.377 秒，因模型提供商上游鉴权报错 HTTP 401 失败，产生 0 个 tool 事件。
   - 探针已对鉴权与凭据实施严格脱敏保护，未泄露 Secret。
   - **当时双向跨 Agent recall 未通过**（历史证据 `bidirectional_recall_passed = false`）；09:37 Pi 重试成功，不能继续把此 401 写成当前阻断。

### 3. 后续 Pi→Codex 真实 CLI 成功（09:37 UTC）

脱敏文件 `/tmp/pi-daili-cross-recall-attempt2-20261004.json` 记录开始时间 `2026-10-04T09:37:06.832282+00:00`，用时 57.442 秒、退出码 0、`passed=true`，确实观察到指定的 `funes_recall` 调用与成功 tool 结果（3 条记录）。记录 `98101`、`91597`、`95854` 均来自 Codex session/user_message，原始中文与 `CPA` / `previous_response_id` 锚点验证为 true；不保存原始正文。

单独 `/tmp/pi-daili-provider-test-20261004.json` 为 HTTP 200、有效 message、12.42 秒。Pi 探针使用临时 session，内置工具与自动 recall 禁用；设置与 extension 未改变。它证明该次 Pi→Codex 真实链路成功，不证明所有模型提供商均正常，也不替代 Gemini 需求 C、过滤性能或全量回填的最终验收。

### 查询预算核对（代码证据，不是超时根因结论）

`sync/mcp_bridge.py` 与 Pi extension 的默认 recall 单次请求为 8 秒、总预算为 18 秒；Space `VOYAGE_NATIVE_TIMEOUT` 为 12 秒、`VOYAGE_HTTP_TIMEOUT` 为 14 秒。客户端存在提前放弃的预算错配，但 HTTP 断开没有被映射为 native cancellation，不能用客户端 8 秒预算解释服务端自身的 12 秒 timeout。`NativeMcpWorker._call()` 收到 timeout 后主动终止子进程是已确认机制；底层慢阶段仍需诊断。

新 hint 在每次 captured commit 中自动生成，且与新 manifest 归入最后 CAS activation chunk，确保数据对象全部上传完成后再更新版本；真实 hint 自动增长已验证此闭环。

## 下一步，保持现有数据与运行任务

1. **先等本次 warm/maintenance 结束并读回 ready**：新 revision 已部署且 SHA 匹配；当前 ready 503、maintenance active，服务门禁未通过，不再重复部署。
2. **门禁通过后验收真实查询**：复测 Gemini `source_type=memory` 与需求 C，保留中文原文/来源身份；Pi 历史 401 不再作为当前待修项。状态 4 仍未通过。
3. **再执行受控吞吐 A/B**：13:35 A/B 为 idle、0/12；不在维护期间把普通周期 metrics 当作 A/B 测量结果。
4. **持续观察全量回填**：最新 `1,287,777` pending；状态 3 需等 complete/cutover_ready=true 后再终验，本机 queue=0 不替代此门槛。
5. **保留数据与 checkpoint**：不清空 Lance、不重建 PG、不强行上传 held；历史失败、旧采样、构建与部署证据并存。

脱敏机器证据见 `docs/evidence/voyage-backfill-recovery-2026-10-04.json`。临时原始状态采样位于 `/tmp/funes-status-*Z.json`；这里仅保留 allowlisted 计数、阶段耗时和状态，不保存 credential、Authorization 或完整原始会话。
