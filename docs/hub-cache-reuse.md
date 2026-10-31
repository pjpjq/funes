# HF Space 的跨容器 Hub 缓存复用

## 存储边界

- PostgreSQL：原始记忆、来源身份和 canonical checkpoint，不存大块索引缓存。
- 原有私有 Dataset：已提交的 Lance 数据、向量与索引，仍是派生索引的持久副本。
- 私有 Bucket：Hub 已下载的不可变文件缓存；不是第二个 Lance writer。
- Space 本地 POSIX 目录：锁、临时下载与 snapshot 软链接；每次启动可由 manifest 恢复。

不能只把 `HF_HOME` 指到 Bucket：HF 挂载的软链接不跨卸载持久化。
这里把 blobs 保存为普通 Bucket 对象，manifest 记录缓存链接；启动只读取 manifest
并建立本地链接，不全量复制大文件，也不逐个远端 stat 全缓存。
缺失或不可读的缓存仍由既有 hf-hub/FetchStore 回源；原文、checkpoint 不受影响。

## 配置

```text
HF_HOME=/data/huggingface
HF_HUB_CACHE=/data/huggingface/hub
FUNES_HUB_CACHE_BUCKET=<owner/private-bucket>
FUNES_HUB_CACHE_MOUNT=/mnt/funes-hub-cache
FUNES_HUB_CACHE_INTERVAL=300
FUNES_HUB_CACHE_MAX_BYTES=34359738368
```

Space 挂载 Bucket 为只读；后台通过已有 `HF_TOKEN` 的官方 Bucket API 上传缓存。
Bucket 必须私有，token 仅放 Space Secret。缓存代码不保存或上传 token、refs、locks。
未配置 Bucket 时保持原路径行为；达到容量上限只停止扩充缓存，不阻断检索/写入。

```python
from huggingface_hub import HfApi, Volume
api = HfApi()  # 从环境读取现有 Secret
api.set_space_volumes(
    '<owner/space>',
    volumes=[Volume(type='bucket', source='<owner/private-bucket>',
                    mount_path='/mnt/funes-hub-cache', read_only=True)],
)
```

注意：`set_space_volumes` 替换整个挂载列表；部署必须保留已有卷。
缓存的每个 namespace 只允许一个 checkpoint writer，不能多副本并行覆盖 manifest。
先成功发布 blobs，后发布 manifest；退出前未落盘的缓存只影响性能，不影响持久数据。

## 复用不等于零启动成本

恢复的是文件缓存，不是进程内 Lance 句柄或操作系统页缓存。
Bucket 上的字节仍需按实际访问读取，首次读和新的文件仍会产生 I/O。
这不触发全库 embedding、不改变 profile 或 generation，也不替代尚未完成的
canonical 索引追赶。缓存保存上限不是预分配空间。

## 2026-09-24 容量盘点

Canonical Dataset revision: `06deeb41dbfcb102eb0c19769ef84ba945bc0792`。
下表是该 revision 的**全部仓库文件**，包含旧版本，不能当作活跃缓存大小。

| 类别 | 文件数 | 字节 | GiB |
|---|---:|---:|---:|
| data | 7,546 | 6,296,196,242 | 5.864 |
| indices（含旧版本） | 4,010 | 16,894,492,427 | 15.734 |
| manifests（含旧版本） | 9,081 | 4,150,811,023 | 3.866 |
| transactions/deletions/其他 | 9,129 | 4,121,337 | 0.004 |
| 合计 | 29,766 | 27,345,621,029 | 25.468 |

活跃 Lance version 9080 的实际引用集合如下；预填只覆盖这一集合，不搬历史垃圾。

| 类别 | 文件引用数 | 字节 |
|---|---:|---:|
| 数据分片 | 7,544 | 6,296,143,080 |
| 活跃索引 | 37 | 420,572,830 |
| manifest / version hint | 2 | 1,074,334 |
| 删除标记 / 事务 | 19 | 14,662 |
| 合计 | 7,602 | 6,717,804,906 |

即 **6.256 GiB**，映射到 **7,601 个唯一 etag 对象**（两个引用共用一个对象）。
其中 4,811 个 Xet 文件以服务端 copy 预填；2,791 个小 Git 文件经过 Git SHA1 校验后上传。
这些是固定 revision 的活跃文件，不代表随后新增文件，也不代表完整历史容量。

实际 Bucket 用量与恢复耗时必须由部署验收补录，不能拿仓库总量替代。
