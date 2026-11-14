# 两个 Space，复用现有索引

这是部署契约，不代表生产已切换或 PR 已达到合并门槛。

## 拓扑与数据不变量

| 服务 | 角色 | 职责 | 持久权限 |
| --- | --- | --- | --- |
| 原 `bolikoto/funes-memory` | `query` | 中文检索、过滤检索、跨 Agent recall | 现有 PG 只读账号、现有 HF dataset 只读 token |
| 新 Writer Space（创建前核对名称） | `writer` | ingest、backfill、maintenance、最终 optimize | 同一 PG 的写账号、同一 HF dataset 的写 token |

两个 Space 使用同一个经过验证的源版本和原生二进制。不同进程运行在不同
Space 容器，不共享内存上限；这不等于任意一个进程都已通过内存验收。

必须继续使用 **现有 PG 数据库/schema、`bolikoto/funes-memory-voyage`、原有
Voyage profile、checkpoint、native generation 和 source identity**。
禁止创建空库代替源库、运行迁移/重建命令、清 checkpoint、批量重算 embedding。
本地 HF 下载缓存可以冷启动，它不是持久索引，也不需要重新 embedding。

## 配置和权限

两个 Space 都显式设置 `FUNES_SERVICE_ROLE` 和 `FUNES_POSTGRES_DSN`；未设置角色
仍是兼容模式 `combined`，不能把它误当成隔离部署。使用相应 `.env.example` 作为
**变量名清单**，将凭据分别写入 HF Secrets，不要提交实际值或复制整个 shell 环境。

发布前必须将预编译 Dockerfile 的 source revision、binary URL、archive SHA256 和
binary SHA256 一起固定到本次验收的构建产物，不能沿用旧部署镜像的默认二进制。
先在隔离副本验证 `funes query-snapshot`、严格 query 角色和 worker 热换；CLI
`--version` 能运行不代表新协议可用。两个 Space 均需读回相同的二进制校验和。

Query 的 PG 账号只能读取已存在 schema 中的业务表；不要授予建表、DML 或迁移权限。
即便暂时用了写账号，应用层也必须禁写，但这不能替代数据库权限边界。
Query 的 HF token 只读现有数据集；禁止上传 Hub cache checkpoint。
Writer 的租约连接必须连接同一 PG 数据库且保留 session，不能经过 transaction
pooler。两者保留原有 embedding provider/model/dimensions/schema 配置。
Voyage API key 仍用于新查询向量或未完成的 backfill，不重新计算已索引文档。

读取端的 native child 使用固定 HF commit SHA 和 Lance version。后台只在 Lance
版本变化后尝试一个候选 reader，验证成功后才替换旧 reader；上传分片但尚未激活的
HF commit 不应触发切换。候选失败保留旧 reader，首次没有有效快照时不 ready。

## 地址切换

- MCP/recall/search/get 客户端继续使用原 Query Space 地址。
- ingest、异步 ingest operation 轮询及写管理接口改用 Writer Space 地址。
- 分别核对同步客户端和 MCP bridge 的配置；不要全局替换共用的
  `FUNES_REMOTE_URL`，导致读请求跑到 Writer 或写请求跑到 Query。
- Query 不代理写请求；迁移期间同步客户端保留原队列和 source identity，以原来的
  幂等协议重试，不能删除队列或把失败响应当成已持久化。

同步客户端的地址优先级为进程 `FUNES_REMOTE_URL`、共享 TOML 的 `[remote].url`、
最后才是 `[sync].remote_url`。因此仅同时填写 TOML 的读写两段不能完成分流：
同步 LaunchAgent 必须单独覆盖 Writer URL，MCP/Pi 保持 Query URL（或使用独立
`FUNES_CONFIG`）。读回实际 LaunchAgent 环境，而不只检查 TOML。保留原 state
目录、device identity 和队列；新 URL 触发的 presence 检查不是索引丢失。

## 受控上线顺序

1. 保存原 Space revision、变量/secret **名称**、PG checkpoint 和 HF index revision
   的脱敏读回；确认两个服务使用同一数据/profile。准备只读凭据，先用私有副本验证。
2. 使用验收通过的二进制创建暂停状态的 Writer Space，配置同一数据集和数据库。
   不得在原 combined writer 仍运行时启动新 Writer：旧版本不参加新租约协议。
3. 暂停同步入口并等待旧写请求结束，受控停止旧 writer/maintenance 后将原 Space
   切为 Query。确认旧写进程退出且 Query 可用，再启动新 Writer 并核对唯一租约。
4. 将同步写地址切到 Writer，恢复原队列；确认 indexed 继续增长、pending 净下降、
   无 HF400，并同时验收真实中文、Codex/Pi/Gemini 过滤和跨 Agent recall 60 分钟。
5. 读回两个 Space 的 revision/role/ready、单写者、checkpoint 和最终 optimize 状态。
   `pending=0`、`invalid=0`、`complete=true`、`cutover_ready=true`、最终 optimize
   成功且检索验收通过前，不 merge。

PG session advisory lock 只阻止遵守协议的 writer，并不是 HF 存储层 fencing。
失联期间已有 HF 请求可能仍在执行；保留 HF parent-CAS，并在切换时确认旧 writer
退出。租约丢失须停止新写任务并回收 native 写进程，不能自动反复抢锁继续写。

## 回滚

查询候选失败首先保留已验证的旧 reader，不回滚索引或 checkpoint。需要回滚服务时，
先停新 Writer 并确认无写进程/在途提交，再恢复已验证的旧服务版本和原写地址。
不得同时启动旧 combined 与新 Writer；不得恢复陈旧数据库快照覆盖新增进度。

## 验收证据

保存测试 commit、二进制校验和、Python/原生测试结果、真实双 writer 竞争和失联测试、
Query 禁写和旧 reader 保留测试，以及两个独立容器的 RSS/cgroup 峰值。
CI 通过、单次 recall 命中或 `oom_kill=0` 都不能单独代替完整生产验收。
