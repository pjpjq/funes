# Voyage canonical backfill：并发验收记录（2026-09-25）

## 结论

- 生产继续使用既有用户指定的付费 Voyage key；本次没有创建、轮换或输出 key。
- Hugging Face Space `bolikoto/funes-memory` 的 `FUNES_VOYAGE_CONCURRENCY` 已从 `2` 调到 `4`，其余回填参数保持不变。
- 并发 `4` 已稳定运行，但当前证据**不能证明它比并发 `2` 有显著吞吐收益**；不应把 2→4 写成已完成的性能提升。
- 已 indexed 记录没有重复调用 Voyage；回填从现有 checkpoint 继续。
- 全量尚未完成：最新只读样本为 `512,021 indexed / 1,208,652 pending`，`invalid=0`、`held=110`、`checkpoint_current=true`。

## 生产配置（2026-09-25）

| 参数 | 值 |
| --- | ---: |
| `FUNES_VOYAGE_CONCURRENCY` | `4` |
| `FUNES_VOYAGE_MAX_REQUEST_TOKENS` | `9000` |
| `FUNES_VOYAGE_TOKENS_PER_MINUTE` | `120000` |
| `FUNES_VOYAGE_MIN_REQUEST_INTERVAL` | `0` |
| `FUNES_CANONICAL_INDEX_BATCH` | `128` |
| `FUNES_CANONICAL_INDEX_REQUEST_ROWS` | `64` |
| `FUNES_CANONICAL_INDEX_MAX_CHARS` | `96000` |
| `FUNES_CANONICAL_INDEX_MIN_REQUEST_INTERVAL` | `1` |

镜像中的安全默认值仍为并发 `2`；生产环境变量覆盖为 `4`，没有为变量覆盖单独重建镜像。

## 可比生产观测

以下是回填 reconciler 的只读状态采样。每个成功周期都报告 `attempted=64, indexed=64, held=0, durable=true`。

| UTC | 并发 | indexed | pending | 最近周期耗时 |
| --- | ---: | ---: | ---: | ---: |
| 06:38:01 | 2 | 505,493 | 1,214,660 | 15,306 ms |
| 06:43:08 | 4 | 506,645 | 1,213,593 | 15,292 ms |
| 06:48:27 | 4 | 507,861 | 1,212,493 | 15,545 ms |
| 06:49:14 | 4 | 508,053 | 1,212,311 | 15,188 ms |
| 06:55:59 | 4 | 509,461 | 1,210,916 | 15,809 ms |
| 07:01:09 | 4 | 510,613 | 1,209,878 | 16,176 ms |
| 07:07:48 | 4 | 512,021 | 1,208,652 | 15,738 ms |

从 06:55:59 到 07:07:48，净完成 `2,560` 条，耗时约 `709 s`，约 **3.61 docs/s**。从 06:38:01 到 07:07:48 的长窗口约 **3.65 docs/s**。周期耗时主要包含候选状态重校验、密钥过滤、Lance 写入/版本持久化和状态回写，不等于 Voyage HTTP 单独耗时。

## 批次数与并发边界

`src/inference/voyage.rs` 会把文档按 token budget 切成多个请求，并使用有界线程作用域并发；实际并发为：

```text
min(FUNES_VOYAGE_CONCURRENCY, batches.len())
```

当前 64 行样本的 dry-run 结果是 3 个请求批次 `[10, 15, 39]`，所以即使配置为 `4`，该样本最多同时运行 3 个 Voyage 请求。`/private/tmp/funes_voyage_ab_profiles.json` 标记这些 profile 为 `executed=false`；它们不能当作真实生产 A/B 结果。

因此本次生产数据只能支持：并发 `4` 可运行、无状态错误、没有明显高于并发 `2` 的已证实收益。若要继续优化，应先在不切生产的测试路径加入分阶段计时（Voyage 请求、过滤、Lance 写入、状态回写）和真实请求批次数统计，再改变 `REQUEST_ROWS` 或 `MAX_CHARS`。

## 稳定性边界

- `/health`（无鉴权）：`200`
- `/ready`（带鉴权）：`200`
- `/ready/search`（带鉴权）：`200`
- `/ready`（无鉴权）：`401`
- `thread_alive=true`、`last_error=null`、`consecutive_failures=0`
- 观测窗口内状态端点未出现 HTTP `429/5xx`；这只能说明端点和 reconciler 观测未发现错误，不能证明 Voyage provider 内部绝无重试。
- `hub_cache.state=running`，checkpoint 持续前进；没有清空 Lance、HF cache、PG 或 checkpoint。

## 状态修复与回归测试

已包含在当前分支的修复：

- `0067742`：仅 `source_version` 变化且 raw/content 未变化时，保留有效 native index 状态。
- `e628184`：PostgreSQL 对应回归覆盖。

本轮已通过的定向验证：

```text
pytest -q tests_space/test_deployment_entrypoint.py                  4 passed
cargo test --lib inference::voyage::tests                         25 passed
PYTHONPATH=. pytest -q tests_space/test_server.py service/tests/test_service.py \
  -k 'source_version_only_change_skips_native_reconcile or source_version_change_preserves or source_version_change_with_new_raw'
                                                                     5 passed, 312 deselected
```

## 成本与完成门槛

- 本轮没有对已 indexed 记录重新 embedding；只继续处理当前 pending checkpoint。
- 生产回填仍会按 pending 原文消耗 Voyage token，这是必要的新 embedding 成本；并发配置本身不改变单条记录的 embedding 结果。
- 当前完成门槛仍是：`pending=0`、`invalid=0`、`complete=true`、`cutover_ready=true`，并通过最终 native ready、跨 Agent recall 和恢复测试。
- 以最近约 `3.6 docs/s` 且不计新增 eligible 记录估算，剩余 `1,208,652` 条约需 **3.8–4.0 天**。这是运行速率估算，不是承诺；应以连续观测重新计算。

## 证据文件

- 生产采样（不含 secrets/raw sessions）：`/Users/pwd/.local/share/funes-memory-sync-v2/audits/voyage-concurrency4-observations-20260925.jsonl`
- 批次 profile（dry-run，未执行）：`/private/tmp/funes_voyage_ab_profiles.json`
- 当前分支：`feat/unified-agent-memory`
- 当前代码提交：`c29d03e`
- 当前 Space revision：`720090670ef7ccbd12cd3eec71026d6e13cbbdd4`
