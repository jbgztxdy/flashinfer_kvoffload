# 批量 MTP 稀疏注意力：Memcpy / Direct 基准

## 结论

把 batch 扩大到 1～128 个请求、每个请求 MTP 8 个 token（最多 1024 个 query 行）后，两条基准路径都基本被 PCIe 带宽限制：

- **只要有 Host 命中，Direct 就比 Memcpy 快**。5% 命中时 Direct 耗时约为 Memcpy 的 0.57～0.69；命中率越高差距越小，1024 行、30% 时只差约 2%（26.83 vs 27.43 ms），两者都在跑满链路。
- **相对全本地的放大随 batch 增大而增大**。8 行时，20% 命中 Direct 是全本地的 3.6×；1024 行时是 6.3×。全本地 attention 随 batch 增长较慢，Host 读取量则严格按 query 行数线性增长。
- **单 batch 已有的结论不能直接沿用**。单个 query 时，20% 命中的 Host 数据约 0.45 MiB；1024 行时为 461 MiB，按约 27 GB/s 需要约 18 ms，而全本地 attention 只有约 2.8 ms。纯靠 kernel 内重叠无法隐藏这部分传输。
- **CPU gather 本身与 H2D 相当**。1024 行、30% 时 CPU gather 约 23.8 ms（单独计时，未计入 Memcpy GPU 时间），实际 Memcpy 流程还需要流水化这一步。

按全本地耗时的 20% 作为可接受的额外开销估算，PCIe 能覆盖的 Host 数据量约为 `0.2 × T_local × 27 GB/s`：8 行时约 0.24 MB（约 1.3% 命中），1024 行时约 15 MB（约 0.6% 命中）。在本测试的独立随机 top-k 假设下，只靠调度优化很难把 5% 以上的命中率压到目标以内；需要减少实际经过 PCIe 的字节数，例如利用 MTP token 之间 top-k 的重叠去重，或把热点页留在 HBM。

## 测试设置

| 项目 | 设置 |
|---|---|
| GPU | GPU 3：A100-PCIE-40GB，Host NUMA 1，CPU 32-63,96-127。GPU 0～2 被其他 vLLM 服务占用 |
| 形状 | 64 head，512 latent + 64 RoPE，BF16，page size 1，top-k=2048，sm_scale=1/16，原生 FA2 MLA kernel |
| batch | 请求数 1/2/4/8/16/32/64/128，每个请求 MTP 8 个 token，即 8～1024 个 query 行 |
| query 组织 | 每个 MTP token 是一个 qo_len=1 的独立行，有自己的 top-k，模拟 DSA 每个 token 单独选择 |
| KV | 每个请求 56K 上下文，一半在 HBM、一半在 pinned Host DRAM（CUDA VMM 映射到同一地址范围）。总计 734 万页，Host 约 4.2 GB |
| top-k 生成 | 每行在所属请求的本地 / Host 区域中独立随机选取，随机位置替换为 Host 页；同一请求 8 个 token 之间没有刻意制造重叠 |
| 命中率 | 0%、5%、10%、15%、20%、25%、30%，指每行 2048 个选中页中来自 Host 的比例 |
| 计时 | CUDA Graph 内重复调用，每个 graph 约 10 ms（最多 100 次），GPU event 计时；20 轮、4 组索引，取中位数 |

两条路径：

- **Memcpy**：整个 batch 的 Host 页先去重，CPU gather 到 pinned buffer，在 graph 内执行 H2D DMA 后运行原算子。GPU 时间包括 DMA 和 attention；CPU gather 单独记录。去重后的行数约为引用数的 93%～99%。
- **Direct**：原算子直接读取映射的 Host 内存，不去重；同一页被多行引用时会重复经过 PCIe，部分可能命中 L2。

所有 case 在前 4 轮按 FP32 参考做抽样行检查，最大绝对误差 0.007。p90 / 中位数基本在 1.07 以内。

0% 时两条路径运行同一个原算子，理论上耗时相同；但 256～1024 行时两者相差 5%～11%（例如 512 行时 1.29 vs 1.44 ms）。两条路径使用不同的 workspace 和 KV 地址映射，差异可能来自 HBM / L2 布局，具体原因未查明。因此大 batch 下 5% 以内的 Direct / Memcpy 差异不宜过度解读。

## 结果

单位为 µs，超过 1000 µs 的写作 ms。

### Direct GPU 耗时

| 请求数 | Query 行 | 0% | 5% | 10% | 15% | 20% | 25% | 30% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 44.6 | 55.7 | 91.1 | 125.7 | 160.8 | 195.3 | 230.2 |
| 2 | 16 | 77.0 | 91.5 | 162.0 | 231.9 | 301.5 | 371.3 | 440.3 |
| 4 | 32 | 117.3 | 172.1 | 311.9 | 449.3 | 588.1 | 726.3 | 865.2 |
| 8 | 64 | 284.1 | 393.7 | 652.2 | 907.3 | 1.18 ms | 1.45 ms | 1.72 ms |
| 16 | 128 | 511.8 | 744.6 | 1.22 ms | 1.71 ms | 2.27 ms | 2.83 ms | 3.38 ms |
| 32 | 256 | 772.1 | 1.23 ms | 2.28 ms | 3.39 ms | 4.51 ms | 5.62 ms | 6.74 ms |
| 64 | 512 | 1.29 ms | 2.28 ms | 4.53 ms | 6.75 ms | 8.99 ms | 11.22 ms | 13.44 ms |
| 128 | 1024 | 2.83 ms | 4.55 ms | 9.03 ms | 13.46 ms | 17.94 ms | 22.39 ms | 26.83 ms |

### Memcpy GPU 耗时（H2D + attention）

| 请求数 | Query 行 | 0% | 5% | 10% | 15% | 20% | 25% | 30% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 44.5 | 91.5 | 126.1 | 158.9 | 191.6 | 222.5 | 253.0 |
| 2 | 16 | 77.5 | 160.7 | 230.1 | 293.6 | 358.6 | 423.1 | 482.1 |
| 4 | 32 | 119.3 | 268.3 | 409.1 | 539.6 | 663.9 | 789.9 | 909.6 |
| 8 | 64 | 285.5 | 580.5 | 848.6 | 1.11 ms | 1.37 ms | 1.61 ms | 1.86 ms |
| 16 | 128 | 514.8 | 1.09 ms | 1.63 ms | 2.15 ms | 2.66 ms | 3.16 ms | 3.64 ms |
| 32 | 256 | 815.7 | 1.89 ms | 2.97 ms | 4.02 ms | 5.05 ms | 6.04 ms | 7.01 ms |
| 64 | 512 | 1.44 ms | 3.52 ms | 5.66 ms | 7.75 ms | 9.82 ms | 11.81 ms | 13.73 ms |
| 128 | 1024 | 2.70 ms | 6.96 ms | 11.30 ms | 15.47 ms | 19.58 ms | 23.56 ms | 27.43 ms |

### Direct 相对全本地（同 batch 的 0% Direct）

| 请求数 | Query 行 | 5% | 10% | 15% | 20% | 25% | 30% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 1.25× | 2.04× | 2.82× | 3.61× | 4.38× | 5.16× |
| 8 | 64 | 1.39× | 2.30× | 3.19× | 4.14× | 5.10× | 6.07× |
| 32 | 256 | 1.59× | 2.95× | 4.39× | 5.84× | 7.28× | 8.73× |
| 128 | 1024 | 1.61× | 3.19× | 4.76× | 6.34× | 7.91× | 9.48× |

### 等效 PCIe 带宽

Direct 按 `Host 引用字节 / (T - T_0%)` 计算，Memcpy 按 `去重后字节 / (T - T_0%)` 计算：

| Query 行 | Direct 10% | Direct 30% | Memcpy 10% | Memcpy 30% |
|---:|---:|---:|---:|---:|
| 8 | 40.6 GB/s | 30.5 GB/s | 22.7 GB/s | 25.2 GB/s |
| 64 | 41.1 GB/s | 31.5 GB/s | 26.2 GB/s | 26.7 GB/s |
| 1024 | 39.0 GB/s | 30.2 GB/s | 27.4 GB/s | 27.2 GB/s |

Direct 低命中率时超过链路上限，是因为部分读取与本地计算重叠，扣除 0% 时间后被低估；30% 时约 30 GB/s，接近 PCIe 4.0 ×16 的约 31.5 GB/s。Memcpy 的 DMA 约 27 GB/s。

完整表格（含相对全本地的 Memcpy、Direct / Memcpy、CPU gather）见 [random_mtp8.md](../temp/host_attention_batch/random_mtp8.md)，逐项数据见 [random_mtp8.csv](../temp/host_attention_batch/random_mtp8.csv) 与 [random_mtp8.json](../temp/host_attention_batch/random_mtp8.json)。

## 页号上限：32 位偏移溢出（已修复）

原 MLA kernel 在 `load_kv` 中按 `indices[q] * ckv_stride_page` 计算偏移，两者都是 32 位，乘积在加到指针之前就按 32 位回绕。576 维 KV 下，页号超过约 745.6 万（2³² / 576，BF16 下单层缓存约 8 GiB）就会读到错误的行，kernel 不报错。

- 首次按每请求 128K 上下文测试时，128 个请求共 1677 万页，输出明显错误（最大误差 0.48），日志见 [failed_context128k_uint32_overflow.log](../temp/host_attention_batch/failed_context128k_uint32_overflow.log)。因此上面的主表使用每请求 56K。
- [check_mla_uint32_offset.py](../temp/check_mla_uint32_offset.py) 用 800 万页的缓存确认：修复前，页号超过上限时输出与「按回绕地址计算」的参考一致（误差 0.002），与正确结果相差 0.64；修复后反过来，与正确结果的误差为 0.0012。
- 修复方式：相乘前把页号转成 `int64_t`。涉及 `mla.cuh` 的 `load_kv`、`run_host_fetch_warps`、`load_private_host_segment`，以及 `mla_hopper.cuh` 的 `prefetch_offset`（Hopper 路径把结果存进 `int64_t`，但乘法本身仍是 32 位，同样会溢出；未在 Hopper 上运行验证）。
- [check_mla_host_large_pages.py](../temp/check_mla_host_large_pages.py) 在 1677 万页的 HBM / Host 混合缓存上，原算子与融合 Host 算子在 0%～100% 命中下误差均不超过 0.0016。
- 同样的写法也存在于 `persistent.cuh` 的 `prefetch_offest`（`BatchAttention`）。[check_persistent_uint32_offset.py](../temp/check_persistent_uint32_offset.py) 在 479 万页（8 KV head × 128 维，上限 419 万页）下复现：修复前，页号超过上限时误差为 0.198；修复后为 0.0013。`decode.cuh` / `prefill.cuh` 通过 `page.cuh` 中参数为 `size_t` 的 `get_elem_offset*` 计算偏移，不受影响。
- 回归：随机抽样 `test_batch_attention.py`、`test_deepseek_mla.py`（fa2）和 `test_mla_page.py` 共 419 项，376 项通过，43 项为测试本身的跳过条件（causal 下 qo_len > kv_len），日志见 [offset64_pytest.log](../temp/host_attention_batch/offset64_pytest.log)。
- 寄存器用量变化在 ±1 以内，融合 Host kernel 仍为 242 个、零栈。56K 主表复测（[random_mtp8_offset64.md](../temp/host_attention_batch/random_mtp8_offset64.md)）56 项耗时比值中位数 0.999；单 batch 融合 kernel 与修复前 `.so` 配对测试，1M 上下文 10%～30% 命中差异在 0.2% 以内（[single_offset64_paired.json](../temp/host_attention_batch/single_offset64_paired.json)）。

修复后按每请求 128K 重测（[random_mtp8_ctx128k.md](../temp/host_attention_batch/random_mtp8_ctx128k.md)），全部 56 项正确性检查通过（最大误差 0.0062）。32 行及以上的 Direct 耗时与 56K 一致，差异在 1%～4% 以内；Memcpy 在大 batch 下高约 1%～4%。8～16 行的结果波动明显（例如 8 行 0% 的 p10 / 中位数为 45 / 62 µs），复测 8 行时回到 45 µs，16 行仍偏高，原因未查明，小 batch 以 56K 表为准。

## 局限与下一步

- 本轮 top-k 按每个 token 独立随机生成，是 Host 流量的偏悲观情形。实际 MTP 相邻 token 的 top-k 往往大量重叠；需要测量真实重叠率，并评估按请求去重读取的收益。
- GPU 3 是 PCIe 版 A100，与之前 SXM 版的单 batch 结果不能直接比较绝对值；本文均与同卡、同 batch 的 0% 结果比较。GPU 3 与正在运行 vLLM 的 GPU 2 在同一个 PCIe switch 下，本轮未观察到明显波动，但未排除共享链路的影响。
- 现有 `BatchMLAPrivateHostKernel` 只支持单个 query，本轮未包含优化路径。
- 只测试了随机布局；SSD 延迟模拟与 top-k / 元数据融合尚未开始。

## 复现

```bash
export PYTHONPATH=.
export FLASHINFER_WORKSPACE_BASE="$PWD/temp/jit_host_batch"
export FLASHINFER_CUDA_ARCH_LIST=8.0
export TMPDIR="$PWD/temp"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_JOBS=8

/mnt/data/mhy-vllm-venv/bin/python temp/bench_glm52_host_batch.py \
  --device 3 --cpu-affinity 32-63,96-127 \
  --output temp/host_attention_batch/random_mtp8.json
python temp/summarize_glm52_host_batch.py temp/host_attention_batch/random_mtp8.json
```

全量运行约 4 分钟。已有输出文件时脚本会拒绝覆盖。
