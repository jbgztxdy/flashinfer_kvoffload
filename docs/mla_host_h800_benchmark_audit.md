# H800 比较口径核查（2026-10-29）

用户要求按部署时开启 HiSparse 的原生路径比较，不替 HiSparse 加优化。核查结果：**现有结果不能直接证明“完整 HiSparse 方案不如完整 Memcpy＋LRU 方案”，但也没有证据据此否定我们 kernel 的 H800 性能瓶颈。** 原始数据保留，需按实际计时范围解读。

## HiSparse 默认路径

当前集群 SGLang HEAD 为 `faafa9810212b2c390e66eb6d61cc3a2445b993d`。开启 `enable_hisparse` 后，生产 coordinator 的 `_run_swap_in_kernel` 调用 `load_cache_to_device_buffer_mla`，执行当前 query 的缓存命中检查、LRU 和未命中搬运，再计算 attention。

该路径**没有显式把多个 MTP query 的 top-k 合并后去重搬运**。但它有跨调用的 LRU：前一个 query 读入的页，如果后续还在缓存里，会命中而不再读远端。不能把“没有跨 MTP 批量去重”说成“每次重复引用都重读”。

源码另有 `load_cache_to_device_buffer_spec_mla`（2–4 步、最多 8192 个 top-k occurrence），但对当前 `python/sglang/srt` 检索无该接口及 `HiSparseSpecState` 的引用；生产 coordinator 仍调用单步接口。**有独立 kernel API 不等于开启 HiSparse 就会使用它。** 不替 HiSparse 接入该接口，不改其 LRU 或取数策略。MTP8/top-k2048 也超出该独立接口的限制。

## 旧测试实际测量内容

四份 HiSparse 对照 JSON 均记录同一 benchmark SHA，精确对应本仓库 `67d13373:temp/h800/bench_rounds.py`；kernel 为 `dbbc02c6` 的 H800 版本。

| 内容 | HiSparse | Memcpy＋LRU |
|---|---|---|
| 命中检查和 LRU | 原生 GPU kernel，计时 | CPU 预先算好，不计时 |
| PCIe gather | 原生 swap-in 读 Host，计时 | CPU gather 到 pinned 内存，不计时；只计 H2D |
| NVLink gather | 原生 swap-in 读 peer HBM，计时 | GPU index_select 读 peer HBM，计时 |
| MTP8 | 单步接口串行调用 8 次 swap-in＋attention | 合并 MTP8 的页集合，搬运未命中，再一次 attention |
| 完整数值校验 | 每一步采样 query 行与参考比较 | 不校验，只有搬运字节和 attention 的计时替身 |

Memcpy baseline 只将本步未命中写到 compact 区域的前部，attention 却按本步全部 unique 页的 remapping 读该区域，没有实现真实缓存命中槽位和完整数据维护。它可作为预先知道命中/未命中的 GPU 工作量参考，不能当成能产出正确输出的完整 LRU 实现。NVLink 的 GPU gather 确实计时，但 CPU 预计算和缓存维护缺项仍然存在。

HiSparse 的计时范围还包含测试适配器的 top-k/seq_len 拷贝、最新 token gather 和 query contiguous；我们 kernel 的索引拷贝在 start event 之前。现有总时间不能分离原生 swap-in、attention 与适配开销。MTP8 用单步接口串行扩展的结果应按此标注，不泛化为全部原生 speculative 执行模式。

## NVLink 是否真实使用

`700540` 环境记录 GPU0↔GPU1 为 **NV8**。VMM 分配属性记录远端为 `location_type=1, location_id=1`，本地/staging 为 GPU0；两种方案使用同一远端分配。DMA/SM gather 探针约 177/167 GB/s。因此未发现把远端错放本地、或把 Host 路径标作 NVLink 的证据。

原始稳态时间，单位 µs：

| MTP / 请求 | 我们 | HiSparse 单步接口串行适配 | Memcpy GPU 工作量参考 |
|---|---:|---:|---:|
| 8 / 1 | 83.17 | 828.99 | 45.89 |
| 8 / 16 | 610.00 | 1436.66 | 430.50 |
| 8 / 128 | 4581.63 | 4367.06 | 3450.27 |
| 1 / 256 | 793.85 | 907.57 | 938.32 |

MTP8 的 HiSparse 调用 8 次 attention；单请求每次 FlashMLA 的纯计算测得约 62 µs，而合并 8 行的 FA2 约 33 µs。因此差距不能简单归因为 HiSparse 搬运更慢，也不能从这些数字推断完整部署吞吐。

## 证据边界与后续

旧对照 MTP8 仅 4 步、稳态取后 2 步、5 pass；不足以代表长 assistant decode。旧脚本只有固定 GPU sleep，没有测量/验证 CPU 入队是否在 guard 结束之前，因此没有证据排除计时中的 CPU 间隙，**也没有证据证明这些短测试已受污染**。第三轮新版计时已加入分段入队与 acceptance 检查，但不能倒推旧数据已验证。

旧 JSON 未记录 SGLang/sgl_kernel 的源码 SHA，当前外部源码核查不是历史外部字节的完整复原。两份核查 JSON 位于 `temp/h800/audit_20261029/`，记录所有 19 个旧配置、原始文件 SHA、计时范围和当前 SGLang 源码摘录。

后续保持 HiSparse 默认原生路径，优先拆分 swap-in/attention/适配时间、固定外部源码版本，并补齐正确的 Memcpy 缓存数据维护及数学参考验证。当前排队作业及其源码快照不改；本次审计没有产生新的 H800 性能数据。
