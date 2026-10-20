# 远端 KV 稀疏 MLA 算子：A100 全量 batch 测试（PCIe / NVLink，MTP8 / MTP1，单步与跨 decode）

代码 `dbbc02c6`（融合 LRU 的最终 kernel）。与 HiSparse 的对比需要 FlashMLA（SM90+），在 H800 上做。本文只摘录部分 batch 点，所有结果见下面的索引。

## 结果文件索引

| 内容 | 每种方法的全部测量值 | 比值表（÷下限 / ÷朴素较快者） | 原始数据 |
|---|---|---|---|
| 单步，PCIe，MTP8（25 个 batch 点 × 7 个远端占比） | [full_tables.md#sweep-host-mtp8](../temp/a100_full/full_tables.md#sweep-host-mtp8) | [tables.md](../temp/a100_full/tables.md) | [sweep_host_mtp8.json](../temp/a100_full/sweep_host_mtp8.json) |
| 单步，NVLink，MTP8 | [full_tables.md#sweep-peer-mtp8](../temp/a100_full/full_tables.md#sweep-peer-mtp8) | [tables.md](../temp/a100_full/tables.md) | [sweep_peer_mtp8.json](../temp/a100_full/sweep_peer_mtp8.json) |
| 单步，PCIe，MTP1（14 个 batch 点） | [full_tables.md#sweep-host-mtp1](../temp/a100_full/full_tables.md#sweep-host-mtp1) | [tables.md](../temp/a100_full/tables.md) | [sweep_host_mtp1.json](../temp/a100_full/sweep_host_mtp1.json) |
| 单步，NVLink，MTP1（12 个 batch 点） | [full_tables.md#sweep-peer-mtp1](../temp/a100_full/full_tables.md#sweep-peer-mtp1) | [tables.md](../temp/a100_full/tables.md) | [sweep_peer_mtp1.json](../temp/a100_full/sweep_peer_mtp1.json) |
| 跨 decode + LRU，PCIe，MTP8（14 个 batch 点，逐步） | [full_tables.md#rounds-host-mtp8](../temp/a100_full/full_tables.md#rounds-host-mtp8) | [tables.md](../temp/a100_full/tables.md) | [rounds_host_mtp8.json](../temp/a100_full/rounds_host_mtp8.json) |
| 跨 decode + LRU，NVLink，MTP8 | [full_tables.md#rounds-peer-mtp8](../temp/a100_full/full_tables.md#rounds-peer-mtp8) | [tables.md](../temp/a100_full/tables.md) | [rounds_peer_mtp8.json](../temp/a100_full/rounds_peer_mtp8.json) |
| 跨 decode + LRU，PCIe，MTP1（10 个 batch 点） | [full_tables.md#rounds-host-mtp1](../temp/a100_full/full_tables.md#rounds-host-mtp1) | [tables.md](../temp/a100_full/tables.md) | [rounds_host_mtp1.json](../temp/a100_full/rounds_host_mtp1.json) |
| 跨 decode + LRU，NVLink，MTP1（9 个 batch 点） | [full_tables.md#rounds-peer-mtp1](../temp/a100_full/full_tables.md#rounds-peer-mtp1) | [tables.md](../temp/a100_full/tables.md) | [rounds_peer_mtp1.json](../temp/a100_full/rounds_peer_mtp1.json) |
| LRU 容量 2048 槽，PCIe / NVLink | [PCIe](../temp/a100_full/full_tables.md#rounds-host-mtp8-c2048) · [NVLink](../temp/a100_full/full_tables.md#rounds-peer-mtp8-c2048) | [tables.md](../temp/a100_full/tables.md) | [PCIe](../temp/a100_full/rounds_host_mtp8_c2048.json) · [NVLink](../temp/a100_full/rounds_peer_mtp8_c2048.json) |
| LRU 容量 8192 槽，PCIe / NVLink | [PCIe](../temp/a100_full/full_tables.md#rounds-host-mtp8-c8192) · [NVLink](../temp/a100_full/full_tables.md#rounds-peer-mtp8-c8192) | [tables.md](../temp/a100_full/tables.md) | [PCIe](../temp/a100_full/rounds_host_mtp8_c8192.json) · [NVLink](../temp/a100_full/rounds_peer_mtp8_c8192.json) |

- [`full_tables.md`](../temp/a100_full/full_tables.md)：每种方法的每个测量值。单步给出中位数和 p10～p90，并附实际远端占比、去重率、去重远端字节；跨 decode 给出每一步每种方法的耗时、我们每步的未命中页数，以及不开缓存时的去重远端页数。
- [`tables.md`](../temp/a100_full/tables.md)：每个格子相对理论下限和相对朴素较快者的比值，以及远端占比 > 0 时输给朴素方法的格子清单。
- JSON 里还有每个样本的原始耗时（`samples_us`）、配置、GPU 状态、各内存段的实际驻留位置和源码哈希。
- 生成与复现：[`run.sh`](../temp/a100_full/run.sh)（全部测量）、[`full_tables.py`](../temp/a100_full/full_tables.py)、[`tables.py`](../temp/a100_full/tables.py)；运行日志 `temp/a100_full/run.log`（未纳入 git）。
- 开发过程中的测量、方案取舍和失败尝试见 [`mla_host_a100_lru.md`](mla_host_a100_lru.md)。

## 测试设置

| 项目 | 设置 |
|---|---|
| GPU | A100-SXM4-40GB（GPU0）。远端：PCIe 读 NUMA0 pinned DRAM（实测 25～26 GB/s），或 NV12 NVLink 读 GPU1 HBM（实测 260～263 GB/s） |
| KV / top-k | BF16，576 维（1152 B/token），top-k 2048，64 头；回放实测 DSA 代理 top-k，最近 4096 位置始终在本地 |
| batch | MTP8：1～128 请求（8～1024 行）25 个点；MTP1：1～512 请求（NVLink 到 256，受 GPU1 显存限制）12～14 个点 |
| 单步 | 每点 7 个远端占比（0%～30%），8 轮 × 2 组索引，中位数 |
| 跨 decode | 32 个连续 token（MTP8 为 4 步、MTP1 为 32 步），除最新 token 外全部 KV 在远端，取后一半步的平均；每请求 4096 槽，另测 2048、8192 |
| ours | PCIe：staged 模式（fetch CTA 去重搬运 + 写缓存）；NVLink：原地读模式（attention CTA 写回缓存） |
| 朴素方法 | Memcpy（PCIe：CPU gather 到 pinned 内存不计时 + H2D；NVLink：GPU gather 计时）、Direct（原生 kernel 原地读）。跨 decode 时两者都配同样的逐步 LRU（`memcpy_lru`、`direct_lru`），LRU 记账在 CPU 上且**不计时**，对它们偏宽松 |
| 下限 | 计算下限 = 所有选中行都在本地时的原生 attention；传输下限 = 去重远端字节 ÷ 实测 gather 带宽；取两者较大值 |

## 结论

1. **跨 decode 稳态（目标场景，开 LRU）：**
   - **PCIe：** MTP8 从 1 到 128 请求的 14 个点，全部快于带 LRU 的朴素方法（为较快者的 0.74～0.92 倍）；≥ 24 请求离下限 1.01～1.08×，128 请求 1.01×。MTP1 ≥ 32 请求时快 2%～13%，128～512 请求贴着下限（0.95～1.00×）；MTP1 1～16 请求与 Memcpy+LRU 持平或慢 3%～9%。开缓存比不开快 2.0～5.6 倍。
   - **NVLink：** MTP8 在 8、16、32～128 请求时快于朴素方法（0.82～0.89 倍），12、24 请求持平或慢 11%，≤ 6 请求慢 12%～54%；MTP1 ≥ 64 请求快 17%～23%，更小时慢 7%～49%。离计算下限 1.16～1.38×（大 batch）；原地读模式要重复读 MTP 行共享的页，见 `docs/mla_host_a100_lru.md` 第 7 节。
2. **单步不开缓存：**
   - **PCIe，MTP8：** 175 个格子中，远端占比 > 0 的只有 1 请求、5% 一格输给 Direct；其余是朴素较快者的 0.61～0.94 倍。≥ 28 请求、远端占比 10%～30% 时离下限 0.98～1.13×（5% 时前置开销占比大，最高 1.38×）。
   - **PCIe，MTP1：** 没有可去重的页，staging 多一次中转，基本都慢于 Direct（大 batch 慢 2%～5%，小 batch 更多）。这种场景的收益来自跨 decode 缓存，单步应该直接用原生 kernel。
   - **NVLink：** 原生 Direct 已经只比计算下限慢 1.00～1.15×；表中的 staged 模式在几乎所有格子都更慢。推荐配置（原地读、不开缓存）在 wrapper 里直接走原生 kernel，所以单步 NVLink 上我们的结果就等于 Direct。
3. **LRU 容量：** PCIe 上耗时基本正比于未命中数，8192 槽比 4096 槽再快 1.4～1.8 倍，且始终快于朴素方法；NVLink 上 2048 槽时未命中太多，原地读重复读远端，慢于朴素方法（1.3～1.5×），4096 槽以上恢复领先（128 请求：0.82～0.92 倍）。

## 跨 decode 稳态（开 LRU，每请求 4096 槽，µs）

### PCIe，MTP8

| 请求 | 行 | ours | 不开缓存 | memcpy_lru | direct_lru | 计算下限 | 传输下限 | ours ÷ 下限 | ours ÷ 朴素较快者 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 130 | 273 | 154 | 246 | 50 | 60 | 2.15 | 0.84 |
| 2 | 16 | 266 | 566 | 358 | 713 | 79 | 183 | 1.45 | 0.74 |
| 4 | 32 | 379 | 934 | 449 | 794 | 112 | 259 | 1.46 | 0.84 |
| 8 | 64 | 986 | 2043 | 1195 | 2644 | 281 | 792 | 1.24 | 0.82 |
| 16 | 128 | 1772 | 3748 | 2134 | 5185 | 505 | 1584 | 1.12 | 0.83 |
| 32 | 256 | 3236 | 7039 | 3887 | 10352 | 754 | 3101 | 1.04 | 0.83 |
| 64 | 512 | 6762 | 14412 | 7965 | 21744 | 1248 | 6636 | 1.02 | 0.85 |
| 128 | 1024 | 13695 | 29126 | 16379 | 45078 | 2487 | 13606 | 1.01 | 0.84 |

### NVLink，MTP8

| 请求 | 行 | ours | 不开缓存 | memcpy_lru | direct_lru | 计算下限 | ours ÷ 计算下限 | ours ÷ 朴素较快者 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 98 | 101 | 63 | 68 | 51 | 1.93 | 1.54 |
| 4 | 32 | 188 | 324 | 146 | 150 | 112 | 1.68 | 1.29 |
| 8 | 64 | 325 | 673 | 376 | 394 | 280 | 1.16 | 0.86 |
| 16 | 128 | 581 | 1270 | 679 | 727 | 502 | 1.16 | 0.86 |
| 32 | 256 | 936 | 2385 | 1088 | 1206 | 753 | 1.24 | 0.86 |
| 64 | 512 | 1715 | 4753 | 1945 | 2371 | 1247 | 1.38 | 0.88 |
| 128 | 1024 | 3223 | 9474 | 3909 | 4838 | 2488 | 1.30 | 0.82 |

### PCIe，MTP1

| 请求 | ours | 不开缓存 | memcpy_lru | direct_lru | ours ÷ 下限 | ours ÷ 朴素较快者 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 58 | 121 | 53 | 57 | 1.76 | 1.09 |
| 16 | 343 | 1488 | 334 | 518 | 1.28 | 1.03 |
| 64 | 1177 | 5733 | 1302 | 2118 | 1.06 | 0.90 |
| 128 | 2321 | 11359 | 2675 | 4327 | 1.00 | 0.87 |
| 512 | 8040 | 44994 | 8916 | 15441 | 0.95 | 0.90 |

### NVLink，MTP1

| 请求 | ours | 不开缓存 | memcpy_lru | direct_lru | ours ÷ 计算下限 | ours ÷ 朴素较快者 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 39 | 40 | 43 | 1.82 | 1.49 |
| 16 | 125 | 175 | 114 | 117 | 1.54 | 1.09 |
| 64 | 314 | 672 | 401 | 402 | 1.10 | 0.78 |
| 128 | 567 | 1270 | 735 | 749 | 1.13 | 0.77 |
| 256 | 964 | 2392 | 1156 | 1170 | 1.27 | 0.83 |

上表只摘了部分 batch 点。全部 batch 点见 [比值表](../temp/a100_full/tables.md)；每一步、每种方法的耗时见 [全量结果](../temp/a100_full/full_tables.md#rounds-host-mtp8)（各层级、各 MTP 分节）。

## 单步不开缓存（摘录，µs，括号内为 ÷下限 / ÷朴素较快者）

### PCIe，MTP8

| 请求 | 行 | 0% | 10% | 20% | 30% |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 47 (1.06/1.06) | 78 (1.75/0.88) | 118 (1.70/0.93) | 142 (1.43/0.92) |
| 4 | 32 | 131 (1.17/1.17) | 191 (1.57/0.76) | 321 (1.32/0.90) | 434 (1.24/0.93) |
| 16 | 128 | 442 (0.86/0.87) | 605 (1.20/0.63) | 1069 (1.13/0.75) | 1493 (1.08/0.82) |
| 32 | 256 | 783 (0.97/1.02) | 1109 (1.13/0.64) | 2028 (1.06/0.77) | 2894 (1.03/0.83) |
| 64 | 512 | 1447 (1.06/1.05) | 2253 (1.06/0.68) | 4112 (1.00/0.80) | 5556 (1.00/0.84) |
| 128 | 1024 | 2758 (0.99/1.06) | 4261 (1.04/0.66) | 8119 (1.01/0.79) | 11418 (1.00/0.84) |

全部 25 个 batch 点见 [比值表](../temp/a100_full/tables.md)，各方法耗时见 [全量结果](../temp/a100_full/full_tables.md#sweep-host-mtp8)。NVLink 和 MTP1 的单步：[NVLink MTP8](../temp/a100_full/full_tables.md#sweep-peer-mtp8)、[PCIe MTP1](../temp/a100_full/full_tables.md#sweep-host-mtp1)、[NVLink MTP1](../temp/a100_full/full_tables.md#sweep-peer-mtp1)；解读见上面结论第 2 条。

## LRU 容量（MTP8，ours µs（每步未命中页）/ 朴素较快者 µs）

全部 batch 点见 [比值表](../temp/a100_full/tables.md)；逐步数据见 [2048 槽 PCIe](../temp/a100_full/full_tables.md#rounds-host-mtp8-c2048)、[8192 槽 PCIe](../temp/a100_full/full_tables.md#rounds-host-mtp8-c8192)、[2048 槽 NVLink](../temp/a100_full/full_tables.md#rounds-peer-mtp8-c2048)、[8192 槽 NVLink](../temp/a100_full/full_tables.md#rounds-peer-mtp8-c8192)。

| 层级 | 请求 | 2048 槽 | 4096 槽 | 8192 槽 |
|---|---:|---:|---:|---:|
| PCIe | 16 | 2679 (56.7k) / 3176 | 1772 (36.2k) / 2134 | 1012 (19.1k) / 1325 |
| PCIe | 128 | 20772 (475k) / 23730 | 13695 (311k) / 16379 | 7492 (166k) / 9423 |
| NVLink | 16 | 864 / 777 | 581 / 679 | 567 / 595 |
| NVLink | 128 | 6096 / 4654 | 3223 / 3909 | 2913 / 3163 |

## 局限

- A100 数据；H800（PCIe Gen5、132 SM）需要复测，作业已排队。
- 小 batch：每个池的 LRU 步（20～30 µs）和前置阶段在关键路径上；PCIe 上仍然领先，NVLink 和 PCIe MTP1 的小 batch 慢于朴素方法。按约定暂不处理。
- 朴素方法的 LRU 记账不计时；真正带设备端 LRU 的对比对象是 HiSparse。
- top-k 是 DSA 代理（DeepSeek-V2-Lite attention），不是 GLM-5.2 indexer 的真实 trace；跨 decode 只有 32 个连续 token。
