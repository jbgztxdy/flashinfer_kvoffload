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
| ours | 同一个 kernel（`BatchMLAHostBatchKernel`），按运行时开关走不同分支：单步测试 PCIe 和 NVLink 都用 staged 模式；跨 decode 测试 PCIe 用 staged + LRU，NVLink 用原地读 + LRU。详见下一节 |
| 朴素方法 | Memcpy（PCIe：CPU gather 到 pinned 内存不计时 + H2D；NVLink：GPU gather 计时）、Direct（原生 kernel 原地读）。跨 decode 时两者都配同样的逐步 LRU（`memcpy_lru`、`direct_lru`），LRU 记账在 CPU 上且**不计时**，对它们偏宽松 |
| 下限 | 取 max(计算下限, 传输下限)。计算下限 = 所有选中行都在本地时测得的最快 attention（单步：原生全本地与 0% 远端时两种 kernel 取最快；跨 decode：原生全本地）；传输下限 = 去重远端字节 ÷ 链路实测峰值 max(DMA, SM gather)。两项都取实测最好值，保证不会被任何实现低于；计算下限有约 1%～4% 的测量波动 |

## 各比较对象的实现

**我们的 PCIe 和 NVLink 是同一个 kernel，不是两个 kernel。** 代码和编译产物只有一份（`include/flashinfer/attention/mla_host_batch.cuh`），由两个运行时开关选择分支：`host_stage`（staged / 原地读）和是否调用了 `set_host_cache`（开 / 关 LRU）。两种远端层的唯一区别是远端那段虚拟地址映射到哪里：PCIe 测试映射到 pinned Host DRAM，NVLink 测试映射到 GPU1 的 HBM（`temp/h800/tiers.py`）。kernel 本身不知道远端是什么，只用页号 ≥ `host_page_start` 判断一页在远端。

各测试里实际用的配置：

| 测试 | PCIe 远端 | NVLink 远端 |
|---|---|---|
| 单步（`sweep_*`），列名 ours | staged，无缓存，8 个 fetch CTA | **同样是 staged**，无缓存，8 个 fetch CTA |
| 跨 decode，列名 ours | staged + LRU（`ours_stage`） | 原地读 + LRU（`ours_inplace`） |
| 跨 decode，列名「不开缓存」 | staged，无缓存（与单步 ours 相同） | 原地读、无缓存：wrapper 直接走原生 kernel，**等同于 Direct** |

所以单步 NVLink 表里的 ours 是 staged 模式，这不是推荐配置。推荐配置（原地读、无缓存）就是原生 Direct，因此单步 NVLink 上我们的推荐配置结果等于 Direct 那一列。

所有方法共享的条件：一次 decode 步里每个 query 行（MTP8 时每请求 8 行）各有 2048 个选中页（page_size 1，一页就是一个 token 的 576 维 KV），一部分在本地 HBM，一部分在远端；计时只包括 GPU 上的执行。

### 一个 kernel 的公共骨架

一次 cooperative launch，所有 CTA 常驻，内部按阶段推进：

1. **prepare**：把每行的条目分成本地和远端两类。远端页在整个 batch 内用原子 CAS 去重，第一个 CAS 成功的行「认领」这一页。行被切成若干段分给多个 CTA 并行处理。
2. **grid barrier**，然后 **resolve**：按行顺序给认领的页编号（staging 槽号），并据此把条目改写成指向 staging 或缓存的编码。
3. **attention items**：沿用原生 MLA 的 tile 循环（QK、online softmax、PV），每个 item 是（行，64 个 head，一段 KV）。大 batch 一行一个 item；小 batch 把行拆成多段，最后再做一次 grid barrier，按 LSE 合并各段结果。
4. 条目的编码：本地页是原页号；`-(r+1)` 表示读 kernel 自管的 HBM buffer 第 r 行。这块 buffer 前面是 LRU 缓存槽，后面是 staging 槽，所以 tile 加载只需一个基址。

### 无 LRU：我们的 staged 模式（单步 PCIe / NVLink、跨 decode PCIe「不开缓存」）

- **去重**：同一个远端页在整个 batch 里只取一次。MTP8 时 8 个 token 选中的页约一半重叠，去重率约 0.41。
- **取数**：grid 末尾的 8 个 fetch CTA 按 staging 槽号顺序，用 `cp.async` 把远端页读进 shared memory（双缓冲），再写到 HBM staging，每写完一块（16 或 64 行）发布一个就绪标志。读完后，这些 CTA 也转去做 attention item。
- **重叠**：attention item 先算本地 tile，只在要用远端 tile 之前等对应块的标志。所以远端传输和本地部分的计算是并行的。
- **适用**：PCIe（约 27 GB/s）。瓶颈是带宽，去重直接减少要搬的字节数。在 NVLink 上，8 个 fetch CTA 最多只能读约 130～170 GB/s（单个 SM 约 21 GB/s 上限），所以这个模式在 NVLink 上慢于 Direct。

### 无 LRU：我们的原地读模式（跨 decode NVLink「不开缓存」）

不需要 prepare 也不需要 staging：每个条目直接按原页号由 attention 读远端。这和原生 kernel 完全一样，所以 wrapper 在这种配置下直接调原生 kernel，结果等同于 Direct。

### Memcpy（单步）

1. 对本步所有远端引用做去重（`torch.unique`），得到去重后的远端页列表，并把条目改写成指向一段本地 staging 区。
2. PCIe：CPU 用 `index_select` 把这些页从 pinned Host 内存收集到一块连续的 pinned buffer（**不计时**），然后一次 H2D DMA 拷到 HBM staging 区。NVLink：GPU 上 `index_select` 从 GPU1 收集到 GPU0 的 staging 区（计时）。
3. 原生 MLA kernel 在全本地的条目上做 attention。
4. 传输和计算串行，但只搬去重后的字节。

### Direct（单步）

原生 MLA kernel（`BatchMLAPagedAttentionWrapper.run`，不传 `host_page_start`）直接用原始条目。远端页通过统一地址映射原地读取，**每个引用各读一次**，不去重；传输和计算在 tile 流水线里自然重叠。

### 计算下限（`attn` / `floor_fa2`）

原生 MLA kernel，条目改写成全部指向本地 HBM（与 Memcpy 相同的改写），不做任何拷贝。只用于计时（staging 区内容是什么不重要），表示「没有任何远端访问」时的耗时。

### 有 LRU 的四种方法（跨 decode `rounds_*`）

设定：每个请求一个 LRU 池（默认 4096 槽），跨 decode 步保留。LRU 语义四种方法相同：本步被引用的页刷新最近使用时间；未命中的页按「最久未用、且本步没有被引用」的顺序淘汰槽位；池满且没有可淘汰的槽时，这一页就不缓存。缓存是远端页的只读副本，淘汰不需要回写。

**ours，staged + LRU（PCIe，`ours_stage`）**
1. prepare：每个远端条目先查全局表 `resident[page]`。命中就改写成缓存槽编码，并写入本 launch 的时间戳；未命中的才去重、进 staging。
2. LRU item（每池一个，排在所有 attention item 之前，由 worker CTA 执行）：读入池内全部时间戳，用 radix select 选出最旧的若干槽（数量 = 本池本步的未命中页数），并更新 `resident`、`slot_page`、`victim_of`。
3. fetch CTA 照常取数、写 staging、发布就绪标志，然后从 shared memory 里的同一份数据把页再写进它的 victim 槽。如果这一块的 victim 还没选好，就记下来，循环结束后再从 staging 拷到缓存（HBM 到 HBM），这样 PCIe 读取不会被 LRU 卡住。
4. 远端每页只读一次，命中的页完全不走远端。

**ours，原地读 + LRU（NVLink，`ours_inplace`）**
1. prepare：命中同上；未命中的页由认领的那个条目打上标记位，并记录它是该行第几个认领。
2. 第一次 grid barrier 后，每个池的 LRU 由一个 CTA 选出 victim、完成绑定，并把 victim 写进 `entry_victim[条目]`。
3. attention 加载 tile 时，用 `cp.async` 把这一 tile 各行的 victim 一起预取进 shared memory。tile 落地后，认领行直接把 shared memory 里的这一行写到 victim 槽。没有 fetch CTA，108 个 SM 都参与读远端。
4. 代价：只去重「写缓存」这一步。同一页被同一步的多个 MTP 行引用时，每个引用各读一次远端（MTP8 平均 2.1 次）。NVLink 带宽够用，所以这比占用 SM 做 fetch 更快。

**Memcpy + LRU（`memcpy_lru`）**
1. LRU 记账在 CPU 上模拟（与上面同样的逐步 LRU），提前算好每一步的未命中页，**不计时**。
2. 每步只把未命中页拷到本地（PCIe：CPU gather 不计时 + H2D；NVLink：GPU gather 计时），然后原生 kernel 在全本地的条目上做 attention。
3. 只用于计时：attention 读的是包含本步全部远端页的紧凑区域（相当于命中页已经在缓存里），不检查结果正确性。

**Direct + LRU（`direct_lru`）**
1. LRU 同样在 CPU 上提前算好，不计时。
2. 原生 kernel 中，命中的条目读本地页（读任意本地页，只用于计时），未命中的条目原地读远端。
3. 随后用 GPU gather 把未命中页拷进本地，模拟填充缓存（计时）。

**说明**：两种朴素方法没有设备端的缓存管理，它们的命中检测和淘汰开销没有计入，这对它们有利。真正在设备上做 LRU 的对比对象是 SGLang HiSparse（swap-in kernel + FlashMLA），要在 H800 上测。

## 结论

1. **跨 decode 稳态（目标场景，开 LRU）：**
   - **PCIe：** MTP8 从 1 到 128 请求的 14 个点，全部快于带 LRU 的朴素方法（为较快者的 0.74～0.92 倍）；≥ 24 请求离下限 1.03～1.11×，128 请求 1.03×。MTP1 ≥ 32 请求时快 2%～13%，128～512 请求离下限 1.04～1.10×；MTP1 1～16 请求与 Memcpy+LRU 持平或慢 3%～9%。开缓存比不开快 2.0～5.6 倍。
   - **NVLink：** MTP8 在 8、16、32～128 请求时快于朴素方法（0.82～0.89 倍），12、24 请求持平或慢 11%，≤ 6 请求慢 12%～54%；MTP1 ≥ 64 请求快 17%～23%，更小时慢 7%～49%。离计算下限 1.16～1.38×（大 batch）；原地读模式要重复读 MTP 行共享的页，见 `docs/mla_host_a100_lru.md` 第 7 节。
2. **单步不开缓存：**
   - **PCIe，MTP8：** 175 个格子中，远端占比 > 0 的只有 1 请求、5% 一格输给 Direct；其余是朴素较快者的 0.61～0.94 倍。≥ 28 请求、远端占比 10%～30% 时离下限 1.00～1.16×（5% 时前置开销占比大，最高 1.44×）。
   - **PCIe，MTP1：** 没有可去重的页，staging 多一次中转，基本都慢于 Direct（大 batch 慢 2%～5%，小 batch 更多）。这种场景的收益来自跨 decode 缓存，单步应该直接用原生 kernel。
   - **NVLink：** 原生 Direct 已经只比理论下限慢 1.0～1.3×；表中的 staged 模式在几乎所有格子都更慢。推荐配置（原地读、不开缓存）在 wrapper 里直接走原生 kernel，所以单步 NVLink 上我们的结果就等于 Direct。
3. **LRU 容量：** PCIe 上耗时基本正比于未命中数，8192 槽比 4096 槽再快 1.4～1.8 倍，且始终快于朴素方法；NVLink 上 2048 槽时未命中太多，原地读重复读远端，慢于朴素方法（1.3～1.5×），4096 槽以上恢复领先（128 请求：0.82～0.92 倍）。

## 跨 decode 稳态（开 LRU，每请求 4096 槽，µs）

### PCIe，MTP8

| 请求 | 行 | ours | 不开缓存 | memcpy_lru | direct_lru | 计算下限 | 传输下限 | ours ÷ 下限 | ours ÷ 朴素较快者 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 130 | 273 | 154 | 246 | 50 | 59 | 2.21 | 0.84 |
| 2 | 16 | 266 | 566 | 358 | 713 | 79 | 178 | 1.49 | 0.74 |
| 4 | 32 | 379 | 934 | 449 | 794 | 112 | 253 | 1.50 | 0.84 |
| 8 | 64 | 986 | 2043 | 1195 | 2644 | 281 | 771 | 1.28 | 0.82 |
| 16 | 128 | 1772 | 3748 | 2134 | 5185 | 505 | 1542 | 1.15 | 0.83 |
| 32 | 256 | 3236 | 7039 | 3887 | 10352 | 754 | 3019 | 1.07 | 0.83 |
| 64 | 512 | 6762 | 14412 | 7965 | 21744 | 1248 | 6461 | 1.05 | 0.85 |
| 128 | 1024 | 13695 | 29126 | 16379 | 45078 | 2487 | 13247 | 1.03 | 0.84 |

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
| 16 | 343 | 1488 | 334 | 518 | 1.41 | 1.03 |
| 64 | 1177 | 5733 | 1302 | 2118 | 1.16 | 0.90 |
| 128 | 2321 | 11359 | 2675 | 4327 | 1.10 | 0.87 |
| 512 | 8040 | 44994 | 8916 | 15441 | 1.04 | 0.90 |

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
| 1 | 8 | 47 (1.06/1.06) | 78 (1.75/0.88) | 118 (1.74/0.93) | 142 (1.46/0.92) |
| 4 | 32 | 131 (1.17/1.17) | 191 (1.61/0.76) | 321 (1.36/0.90) | 434 (1.28/0.93) |
| 16 | 128 | 442 (1.00/0.87) | 605 (1.31/0.63) | 1069 (1.16/0.75) | 1493 (1.11/0.82) |
| 32 | 256 | 783 (1.04/1.02) | 1109 (1.16/0.64) | 2028 (1.09/0.77) | 2894 (1.06/0.83) |
| 64 | 512 | 1447 (1.15/1.05) | 2253 (1.09/0.68) | 4112 (1.03/0.80) | 5556 (1.02/0.84) |
| 128 | 1024 | 2758 (1.11/1.06) | 4261 (1.07/0.66) | 8119 (1.04/0.79) | 11418 (1.03/0.84) |

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
