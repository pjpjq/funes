# Northflank PostgreSQL 全量导入演练

**2026-09-23，北京时间；基线 COPY 与全列校验已完成，最终增量捕获进行中，尚未切换生产。**

## 范围和不可越过的验收边界

- 目标是共享 addon 中的独立数据库 `funes_source`；源为只读冻结
  SQLite，3,590,437 条 memories。源文件、HF、Lance 和现有向量均保留。
- COPY、全列摘要比对、checkpoint 提交是独立阶段。进度行数仅表示
  已提交 COPY，不表示全文校验、补建索引或生产切换已完成。
- `migration_ready` 必须保持 false；没有停写和 tail catchup 确认，
  不运行 `--finalize --tail-confirmed`。
- 不禁用 fsync、同步提交、WAL 或主键/唯一约束来追求吞吐。
- 用户先要求固定免费规格，随后于本轮明确授权实际尝试一次临时
  **1 vCPU / 2 GiB /副本**。只改计算 plan，三副本与每副本 40 GiB
  存储不变；演练完成后需恢复 `nf-compute-20`，不扩大其他执行范围。

源 SHA256：

```text
8f72d49271e326f30acb489b921c686ac3b4c4186cd991d251475205c4467003
```

## 02:59 基线完成；最终增量尚未封存

2026-09-23 02:59:53，北京时间，原 importer 正常退出（exit 0），
迁移台账为 `phase=verified`，四表实际全列读回摘要与固定源一致：

| 表 | 已验证行数 |
|---|---:|
| memories | 3,590,437 |
| translation_cache | 63 |
| reindex_controls | 0 |
| sync_state | 1 |

此时 `migration_ready=false`。PG database 大小为
17,971,592,215 字节（约 16.74 GiB），尚不包含全部延后构建索引，
不能当作最终容量。该次主从检查的 replica lag 为 0；不是后续时刻保证。

用户随后授权导入完成后切换 HF。三台来源机器的 connector 已停止，
冻结清单的 pending 均为 0；队列清零不等于服务器全部版本已持久化。
2026-09-23 03:52 发起的生产只读 `/sync/status` 返回
`source_store.documents=3,678,253`，耗时 165.847 秒；HF 仍使用 SQLite。

最后增量捕获必须保留现存源数据。已定位旧运行时的 full `/sync`
返回 `remote_history_not_restored`：进程内覆盖版本和整个源仓库 HEAD
比较，外部非 source 提交也可能使其失配。普通追加提交不会修复这一
覆盖证明；不能以重启、伪造时间戳 replay 或清空 Hub 活跃清单绕过。
当前仅重建固定 revision 的后缀，未宣称覆盖全部 live source metadata。

新增离线工具 `build_source_postgres_tail_snapshot.py` 与
`apply_source_postgres_tail.py` 分别支持固定版本重建和原子 COPY 差量应用，
保留基线与旧源。未经完整源边界核验，不设置 `migration_ready=true`。
原始 SQLite、HF 配置、Lance 与 Voyage embedding 均未因本阶段重建。

## 已执行的批量路径调整

1. COPY 上限从 10,000 行 / 64 MiB 调整到 **100,000 行 / 256 MiB**；
   达到任一上限即提交，因此实际批次常小于 100,000 行。
2. 跨网全列校验的 named cursor `itersize` 从 128 调到 **4096**，减少
   网络往返；摘要算法、源哈希和断点校验均未取消。
3. 七个非唯一 B-tree 延后构建，保留主键和 source identity 唯一约束。

不要为再次调批量而频繁重启 importer。恢复前会重新验证完整已提交
prefix；本次在 609,556 行断点恢复，读回约 3.3 GB 后才继续写入。
修改源文件不会热更新已运行的 Python 进程。

## 真实进度和吞吐，不是提速承诺

| 北京时间，2026-09-23 | 已提交 memories | 冻结源总数 |
|---|---:|---:|
| 01:29:49 | 1,034,875 | 3,590,437 |
| 01:35:25 | 1,313,718 | 3,590,437 |
| 01:40:14 | 1,478,901 | 3,590,437 |
| 01:43:41 | 1,593,841 | 3,590,437 |
| 01:54:48 | 1,930,054 | 3,590,437 |
| 02:04:55 | 2,178,749 | 3,590,437 |

代表性连续批次：

| 已提交行区间 | 秒 | 行/秒 |
|---|---:|---:|
| 676,054 → 752,248 | 61.68 | 1,235 |
| 752,248 → 824,994 | 57.90 | 1,256 |
| 824,994 → 896,035 | 142.59 | 498 |
| 1,449,258 → 1,478,901 | 64.69 | 458 |

不同阶段行宽和数据库负载不同，不能只按最快批次外推，也不能宣称
已获得固定的 2–4 倍提速。最终 ETA 需用最近多个已提交批次重算，
另加全量摘要验证和已有轻量索引构建时间。01:43:41 前八个批次按
总行数/总耗时加权为 711.8 行/秒，当时剩余 COPY 外推 46.8 分钟；
这是当时速率下的估计，不是全流程完成期限。

01:54:48 已提交 **53.76%**；最近八批加权降至 **515.95 行/秒**，
剩余 COPY 按该窗口外推 **53.6 分钟**。这也说明不能沿用早期较快
批次的 ETA；仍不包括全量摘要与补建索引。

## 瓶颈诊断：数据库侧，不是 Alice CPU 不够

Northflank addon API 和 plan API 实测：

| 项目 | 实测 |
|---|---|
| deployment plan | `nf-compute-20` |
| 每副本资源 | 0.2 vCPU / 512 MiB |
| 副本/存储 | 3 副本，各 40 GiB，NVMe |
| PostgreSQL | 16.14 |
| shared_buffers | 108 MiB |
| max_wal_size / checkpoint_timeout | 1 GiB / 300 秒 |

01:25、01:30、01:35 的 Northflank CPU 指标中，一个副本为
97.21%、92.23%、96.68%；三个副本的 memory 指标接近 100%。
平台 memory 指标可能包含文件缓存，不能当成相同大小的进程 RSS，
也不能单凭该值断言 OOM。Alice importer 实测 RSS 约 378 MiB，
CPU 累计均值约 8%–11%，不是 Alice CPU 满载。

40 次迁移连接 wait-event 采样中：33 次 `IO/DataFileRead`、
2 次 `IO/DataFileWrite`、5 次 active/no wait。01:35 时唯一索引
155.2 MiB，已超过 108 MiB shared_buffers。**索引缓存不足导致反复
读页是有依据的解释，但不是已隔离证明的唯一原因**；PG CPU 上限、
TOAST 压缩和平台 I/O 也可能共同限制吞吐。

不能用增加 Alice 写入进程数解决数据库资源上限；当前演练路径也
持有排他锁，未经设计的并行 COPY 会竞争。`maintenance_work_mem`
主要影响后续建索引，不是当前 COPY 的提速旋钮。只有证明 requested
checkpoints 在快速增长后，才考虑无重启增加 `max_wal_size`，而且
它并不能直接消除 `DataFileRead`。上述诊断阶段未改共享 PG 的参数或规格。

后续只读采样发现：01:38:31 → 01:55:15 的约 16 分 44 秒内，
`checkpoints_req` 从 **16 增至 25**，`checkpoints_timed` 保持 2627，
`stats_reset` 相同。这是集群级计数，不能单独归因于 Funes，也不能
仅凭此证明每次都是 WAL 阈值触发。该增量提供了进一步检查
`max_wal_size` 的依据，不是修改成功或已经提速的证据。

供决策的 API 价格快照：`nf-compute-100-2` 是每副本 1 vCPU / 2 GiB，
每小时 $0.033；当前每副本每小时 $0.008。三个副本临时升配的计算
资源价差为 **3 × (0.033 − 0.008) = $0.075/小时**，不包括原有存储、
网络、税费等。共享 addon 升配可能触发滚动切换和重连；该价格诊断
阶段尚未执行，后续获批尝试见下一节。

用户随后明确固定免费规格。只读读取官方 `/v1/swagger-json` 与 addon
状态：公开定义没有 dry-run 参数或额度预校验接口，不能假定付费规格
PATCH 一定被拒绝。因此没有发送升配 PATCH，也没有声称观察到拒绝。
数据库内只读权限核对同时确认当前角色对 `max_wal_size` 没有
`ALTER SYSTEM` 权限，也没有 `pg_reload_conf()` 执行权限；未改 WAL
参数。继续沿用已有 COPY 优化，不因试探控制面而中断导入。

## 02:18 获批实际升配尝试

用户随后明确要求“你可以升配试试”，因此只提交一次以下最小请求：

```http
PATCH /v1/projects/new-api/addons/daili-postgres
Content-Type: application/json

{"billing":{"deploymentPlan":"nf-compute-100-2"}}
```

- 2026-09-23 02:18:02 北京时间发起；真实响应为 **HTTP 200**，不是
  dry-run，也不是额度拒绝。此前没有发送 PATCH，不能混淆两次阶段。
- 02:18:13 控制面已记入 reconcile；GET 回读 `status=scaling`、
  `planId=nf-compute-100-2`、`replicas=3`、`storageSize=40960 MiB`、
  `storageClass=nvme`。请求被接受不等于滚动切换已完成。
- 变更前最后检查 importer 仍为 PID 111659、active/running；
  02:16:44 已提交 2,371,038 / 3,590,437 条 memories。
- 等待控制面恢复 running，并检查 importer 进度；不主动重启导入，
  不修改 HF、Lance 或源 SQLite。若连接因切换退出，仅按已批准的
  `--resume --defer-ready` 路径恢复，已提交 checkpoint 不清零。
- 临时规格实际是否产生新增费用以平台计费为准；上述 API 标价差
  为三副本合计 $0.075/小时。演练完成后恢复原 plan 的动作尚未执行。

02:22:33 GET 确认 `status=running`、pendingActions 为空，新 plan 与
三副本/40 GiB 均保持。只读主库探针确认 `pg_is_in_recovery=false`，
`shared_buffers` 从 108 MiB 增至 **492 MiB**；主库启动时间为
02:19:25。这证明不只是提交请求被接受，滚动变更已经生效。

切换期间 importer 于 02:19:21 提交到 **2,404,898** 行；02:20:20
返回 exit 1（未确认 migration readiness）。原 unit 不自动重启。
最后已提交 checkpoint 必须保留；仅在新主库就绪并确认 marker 后，
启动同一 `--resume --defer-ready` driver，恢复前仍校验已提交 prefix。

恢复前只读 marker 实测：`phase=copying`，memories 的 `rows` 与
`last_rowid` 均为 **2,404,898**，`complete=false`；冻结源 SHA256、
25,935,802,368 字节大小与四表数量匹配原快照，verification 与
final_verification 均为空。COPY 与 marker 在同一事务提交；中断批次
不会把已提交断点推到未提交数据之后。

02:24:19 同一 unit 已重新启动，PID **118067**、active/running，日志
进入 `starting_resume` / `hashing_source`，参数仍为 100k/256 MiB，
保留 `--resume --defer-ready`。独立 monitor unit 同时 active/running。
恢复后的 marker 探针触及 importer 排他锁，按 2 秒 lock timeout 返回
`55P03`；不据此误判 PG 失联或强行解除锁。当前尚无恢复后的新批次
吞吐样本，不能把 CPU 上限 5 倍或内存 4 倍直接当作实际导入提速倍数。

## 监控、identifier 和最终容量验收

`monitor_source_postgres_rehearsal.py` 只记录计数、尺寸、WAL、
COPY tuples/bytes 和迁移连接等待事件，不输出 query 或原文。
COPY 期间部分元数据读取可能因排他锁返回 `55P03`；独立监控分项
必须继续，不能误报整个 PG 失联。replica 查询失败也不抹掉 primary
结果。先完成 multi-host target 选择，再将监控会话设置成只读。

PG 的 identifier 路径是 **native/Lance 找 candidate source identities →
PG source_identity B-tree 回填**，不是对 `search_identifiers` 执行
`LIKE '%...%'`。`benchmark_postgres_identifier_lookup.py` 检验这条
PG hydration 路径，不能冒充 PG 内容 token 搜索或真实 semantic recall。
每次先非执行 EXPLAIN；拒绝 Seq Scan/错误索引，再执行 ANALYZE 和
完整 fetch，分别报告 server/client 延迟和返回 identity 集合。

本次已从冻结 SQLite 的前 8,001 行中选出五类各 100 个真实 source
identities，并在只读副本发起中途检查。副本确认处于 recovery；首个
非执行 EXPLAIN 被 `memories` 的 `AccessExclusiveLock` 阻断后立即
停止。**ANALYZE=0、fetch=0**：没有延迟样本，也没有宣称索引命中、
身份在 PG 中存在或无 Seq Scan 已通过。待 COPY 完成释放锁后重跑。
演练机的病例与报告只落在受限目录，未提交原文、私有 URL 或凭据。

尚待全量导入后的验收：

1. 四表完整逻辑摘要与冻结 SQLite 匹配，源 SHA256 不变。
2. 不改 readiness，只补建已经审核的七个非唯一 B-tree。
3. 以最终九个 memories B-tree 的 schema 测 heap、TOAST、index、total；
   当前仅两个索引的中间容量不能与 17.087903 GiB 样本外推直接比较。
4. 完整规模的主库 canonical hydration EXPLAIN 与 server p95；不把
   replica/prefix 中途结果当作 3,590,437 行最终验收。

## 本轮代码回归证据

- Alice disposable PostgreSQL 测试库运行 `service/tests tests_space
  tests_scripts tests_sync`：**658 passed、17 skipped、51 subtests passed，
  81.84 秒，退出码 0**。确认 pytest 退出后残留进程数为 0。
- 上述 17 项跳过：1 项需要 native binary/HF cache、2 项需要 macOS
  `lockf`、12 项因无 Node、2 项只适用于 Linux secret-file fallback。
  这些是平台/依赖边界，不把跳过项计为通过。
- Alice 此轮没有同步新增 identifier benchmark 的两个文件；本机单独
  覆盖 benchmark、monitor 和 PG native routing：**94 passed、1 skipped，
  2.42 秒**。两组有重叠，不能相加为总通过数。
- 全部测试未对 Northflank 运行可改 schema 的 fixture，未下载模型，
  未重新 embedding、重启 HF 或切换生产。`git diff --check` 通过。

40 GiB/副本减去 17.087903 GiB 样本外推的名义余量是
**22.912097 GiB**，不是 2.912097 GiB。实际可用空间还需扣除其他
数据库、WAL、临时文件等。最终报告以真库全量测量为准。
