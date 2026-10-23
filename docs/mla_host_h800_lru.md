# 远端 KV 稀疏 MLA 算子：H800 实测（PCIe / NVLink，含与 SGLang HiSparse 的单层对比）

状态（2026-10-23）：H800（g40）上目前只完成了验证作业。正式测试在排队，结果出来后补进本文：

| 作业 | 内容 | 状态 |
|---|---|---|
| 697235 fi-validate | 单元测试 + 小规模冒烟测试（含 HiSparse） | **已完成**（2026-10-21 16:38，1 卡，6 分钟） |
| 700539 fi-rounds-host | PCIe：跨 decode + LRU，ours / HiSparse / Memcpy+LRU / Direct+LRU / 两种下限 | 排队 |
| 700540 fi-rounds-peer | NVLink：同上 | 排队 |
| 706158 fi-full-host | PCIe 全量 batch 扫描（单步 + 跨 decode） | 排在 700539 后 |
| 706159 fi-full-peer | NVLink 全量 batch 扫描 | 排在 700540 后 |

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

## 2. 跨 decode + LRU，与 HiSparse 对比（待 700539 / 700540）

## 3. 全量 batch 扫描（待 706158 / 706159）
