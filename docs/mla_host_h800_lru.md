# 远端 KV 稀疏 MLA 算子：H800 实测（PCIe / NVLink，含与 SGLang HiSparse 的单层对比）

状态（2026-10-26）：全部作业已完成；调优作业在排队。

| 作业 | 内容 | 状态 |
|---|---|---|
| 697235 fi-validate | 单元测试 + 小规模冒烟测试（含 HiSparse） | **已完成**（2026-10-21 16:38，1 卡，6 分钟） |
| 700539 fi-rounds-host | PCIe：跨 decode + LRU，ours / HiSparse / Memcpy+LRU / Direct+LRU / 两种下限 | 已完成（10-24） |
| 700540 fi-rounds-peer | NVLink：同上 | 已完成（10-26） |
| 706158 fi-full-host | PCIe 全量 batch 扫描（单步 + 跨 decode） | 已完成（10-26） |
| 706159 fi-full-peer | NVLink 全量 batch 扫描 | 已完成（10-26） |
| 1004362、1004363 | H800 调优：fetch CTA 数、staged vs 原地读 | 排队 |

各比较对象的实现方式见 [`mla_host_a100_full.md`](mla_host_a100_full.md)「各比较对象的实现」一节；A100 上的全量结果也在那里。

## 1. 验证作业（697235）

### 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA H800（g40，132 SM，80 GB），PCIe Gen5 x16，NUMA0 |
| 软件 | torch 2.13.0+cu130，CUDA 13.0；SGLang HiSparse 与 sgl_kernel FlashMLA 来自 `rtx/yjw` 已有环境 |
| 代码 | `dbbc02c6`（与 A100 全量测试相同的 kernel；工作区仅 `decode.cuh` 多一个与本 kernel 无关的行尾空格） |
| 远端带宽（PCIe，Host pinned DRAM） | DMA 55.3 GB/s，SM 随机 gather 35.4 GB/s（A100：27.0 / 26.3） |

### 正确性

- `tests/attention/test_mla_host_batch.py`：**71 passed**（117.9 s）。
- 所有方法（含 HiSparse）对参考实现的最大绝对误差在 0.0019～0.0041 之间（BF16，容差 1e-2）。
- HiSparse 的 `load_cache_to_device_buffer_mla` 和 `sgl_kernel.flash_mla.flash_mla_sparse_fwd` 都能正常调用。

### 单步、不开缓存（PCIe，MTP8，µs）

| 请求 | 行 | 远端占比 | 计算下限 | Memcpy | Direct | ours（staged） | ours（原地读） |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0% | 28.7 | 28.8 | 28.5 | 28.1 | 28.6 |
| 1 | 8 | 30% | 28.4 | **87.2** | 172.3 | 109.4 | 173.0 |
| 16 | 128 | 0% | 156.0 | 156.8 | 155.9 | 204.5 | 156.0 |
| 16 | 128 | 30% | 157.0 | **779.4** | 2451.3 | 1111.5 | 2450.8 |

这两个 30% 格子里 ours 慢于 Memcpy，这和 A100 上的结论（ours 领先）不一致。可能的原因：Memcpy 的 CPU gather 不计时，而 H800 上它的 H2D 是连续 DMA（55 GB/s），kernel 的随机 gather 只有 35 GB/s，差 1.6 倍；A100 上两者几乎相同（27 vs 26 GB/s）。16 请求、0% 一格比原生慢 31%，也要用全量数据确认。待全量结果出来后分析（可能要在 H800 上调 fetch CTA 数或改用 TMA 取数）。

### 跨 decode、开 LRU（PCIe，稳态 µs，每请求 4096 槽）

| MTP | 请求 | 步数 | ours staged | ours 原地读 | 不开缓存 | HiSparse | 下限：FA2 | 下限：FlashMLA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8 | **42.1** | 48.2 | 85.3 | 106.5 | 22.8 | 62.0 |
| 1 | 4 | 8 | 87.8 | **78.5** | 305.0 | 142.7 | 29.5 | 63.1 |
| 8 | 2 | 4 | **205.6** | 325.4 | 392.3 | 1124.6 | 34.8 | 524.3* |

\* 当时的 FlashMLA 下限是按 MTP 行逐个调用的，已改为对所有行一次调用（`67d13373`），正式作业用新版本。HiSparse 本身的测法不变：每个 draft token 依次做一次 swap-in 和一次 FlashMLA，这就是它实际的运行方式（不支持多行 verify，也不做跨行去重）。

这组小规模数据里，ours 比 HiSparse 快 1.6～5.5 倍。拆开看有三部分来源，等正式数据出来后分别量化：
1. **MTP 跨行去重、一次 launch 处理所有行**：HiSparse 每个 token 单独 swap-in。
2. **取数与计算重叠**：HiSparse 先 swap-in 再算，两步串行。
3. **attention kernel 本身**：小 batch 时 FlashMLA 不拆 KV，1～4 行时纯计算 62 µs，FA2 为 23～30 µs。

注意：规模很小（1～4 请求，2 次 pass），只用于冒烟，不作结论。

### 文件

- 原始数据：[`sweep_host_tiny.json`](../temp/h800/results/fi-validate-697235/sweep_host_tiny.json)、[`rounds_host_tiny.json`](../temp/h800/results/fi-validate-697235/rounds_host_tiny.json)、[`rounds_host_mtp8_tiny.json`](../temp/h800/results/fi-validate-697235/rounds_host_mtp8_tiny.json)
- 环境：[`env.txt`](../temp/h800/results/fi-validate-697235/env.txt)；完整日志：[`fi-validate-697235.out`](../temp/h800/results/fi-validate-697235/fi-validate-697235.out)
- 作业脚本：[`validate.sh`](../temp/h800/validate.sh)、[`job.sbatch`](../temp/h800/job.sbatch)

## 2. 跨 decode + LRU，与 HiSparse 对比（700539、700540，2026-10-24～26）

代码 `dbbc02c6`，默认配置（PCIe staged + 8 个 fetch CTA，NVLink 原地读），每请求 4096 槽，除最新 token 外 KV 全在远端，后一半步平均，µs。H800 实测远端带宽：PCIe DMA 55.4 GB/s、SM gather 31～34 GB/s；NVLink（GPU0↔GPU1）DMA 177 GB/s、SM gather 167 GB/s。

| 远端 | MTP | 请求 | 本文 | HiSparse | Memcpy+LRU | Direct+LRU | HiSparse ÷ 本文 | 本文 ÷ 朴素较快者 | 本文搬运速率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PCIe | 8 | 1 | 134 | 910 | 113 | 309 | **6.78** | 1.19 | 11.8 GB/s |
| PCIe | 8 | 4 | 444 | 1249 | 352 | 1048 | **2.82** | 1.26 | 15.3 |
| PCIe | 8 | 16 | 1459 | 2438 | 961 | 3808 | **1.67** | 1.52 | 28.6 |
| PCIe | 8 | 64 | 5838 | 7373 | 4247 | 16623 | **1.26** | 1.37 | 29.9 |
| PCIe | 8 | 128 | 11849 | 14553 | 8495 | 34541 | **1.23** | 1.39 | 30.2 |
| PCIe | 1 | 1 | 43 | 111 | 36 | 37 | **2.57** | 1.20 | 5.6 |
| PCIe | 1 | 8 | 140 | 192 | 103 | 195 | **1.37** | 1.35 | 24.1 |
| PCIe | 1 | 64 | 1012 | 904 | 656 | 1565 | 0.89 | 1.54 | 27.1 |
| PCIe | 1 | 256 | 3507 | 3189 | 2595 | 5818 | 0.91 | 1.35 | 29.2 |
| PCIe | 1 | 512 | 7104 | 6395 | 4810 | 11813 | 0.90 | 1.48 | 29.4 |
| NVLink | 8 | 1 | 83 | 829 | 46 | 61 | **9.97** | 1.81 | |
| NVLink | 8 | 16 | 610 | 1437 | 430 | 861 | **2.36** | 1.42 | |
| NVLink | 8 | 64 | 2305 | 2597 | 1703 | 3592 | **1.13** | 1.35 | |
| NVLink | 8 | 128 | 4582 | 4367 | 3450 | 7408 | 0.95 | 1.33 | |
| NVLink | 1 | 8 | 77 | 151 | 57 | 67 | **1.95** | 1.36 | |
| NVLink | 1 | 64 | 249 | 314 | 266 | 361 | **1.26** | **0.94** | |
| NVLink | 1 | 256 | 794 | 908 | 938 | 1280 | **1.14** | **0.85** | |

**与 A100 的结论不同，H800 上默认配置没有达到目标：**
- **对 HiSparse**：MTP8 全部 batch 和 MTP1 小 batch 更快（1.13～10×），但 PCIe MTP1 ≥ 64 请求慢 9%～11%，NVLink MTP8 128 请求慢 5%。
- **对 Memcpy+LRU**：除 NVLink MTP1 ≥ 64 请求外都更慢（1.19～1.81×）。注意 PCIe 的 Memcpy 其 CPU gather 不计时，只计 55 GB/s 的连续 DMA，这个基线偏宽松；但 NVLink 的 Memcpy 是 GPU gather 并计时的，在这里输是真实的。
- **单步不开缓存**（706158、706159）：PCIe MTP8 远端 > 0 的 120 格中输 101 格（本文 ÷ 朴素较快者 0.74～1.44）；NVLink MTP8 输 104 格（0.84～2.32）。

**原因（初步）**：
1. **PCIe 搬运带宽只到约 30 GB/s**（大 batch），而链路 DMA 为 55 GB/s、HiSparse 的 swap-in（每请求一个 CTA，几百个 CTA 并发）约 33 GB/s。8 个 fetch CTA 是按 A100（27 GB/s）调的，在 PCIe Gen5 上在途读请求不够。离参考下限（按 55 GB/s）1.8～1.9×。
2. **attention 本身比 FlashMLA 慢一倍**：全本地时本文所用的 FA2 MLA 在 128 请求 MTP8 需要 1.2 ms，FlashMLA sparse（SM90 专用）只要 0.57 ms。本文 kernel 的 attention 部分目前仍是 SM80 写法，没有用到 Hopper 的 wgmma/TMA；这在计算占比高的 NVLink 场景最明显。
3. **NVLink 原地读的重复读更贵**：H800 的 NVLink（177 GB/s）相对其算力更慢，MTP8 每个未命中页平均读 2.1 次，128 请求时实际读入速率约 165 GB/s，已接近链路上限。

**正在排队的调优作业**（1004362、1004363）：PCIe staged 的 fetch CTA 数（8/16/32/64）、PCIe 原地读；NVLink 上 staged（去重）配 16/32/64 个 fetch CTA 与原地读对比。

数据：`temp/h800/results/fi-rounds-{host-700539,peer-700540}/`，汇总 `temp/h800/results/summary.md`。

## 3. 全量 batch 扫描（706158、706159）

单步：PCIe / NVLink × MTP8（20 个 batch）/ MTP1（10 个 batch）× 7 个远端占比；跨 decode：MTP8 12 个 batch、MTP1 10 个 batch，含本文不开缓存、Memcpy+LRU、Direct+LRU、计算下限。全部表格见 `temp/h800/results/summary.md`，原始数据在 `temp/h800/results/fi-full-{host-706158,peer-706159}/`。跨 decode 的结论与第 2 节一致（例如 PCIe MTP8 128 请求：本文 12001 µs，Memcpy+LRU 8697，Direct+LRU 34587，不开缓存 23452；NVLink MTP1 512 请求：本文 1527，Memcpy+LRU 1855，Direct+LRU 2559）。
