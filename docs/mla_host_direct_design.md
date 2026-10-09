# HBM / Host DRAM 混合 KV 的 MLA attention 设计

## 1. 目标与当前结果

本设计用于 GLM 5.2 的 DSA top-k=2048 稀疏 decode 实验：KV cache 分布在 GPU HBM 和 pinned Host DRAM，GPU 直接读取选中的 Host KV，并尽量把 PCIe 拉取与本地 attention 计算重叠。测试对象是单层 attention；模型执行、DSA 索引器与 top-k 计算均未计入时间。

当前实现位于 `codex/mla-host-overlap` 分支，算子版本为 `da4ede5f`。入口仍是 `BatchMLAPagedAttentionWrapper.run(..., host_page_start=...)`，默认调用 `BatchMLAPrivateHostKernel<32>`。完整测量和验证结果来自上一轮已保存的数据；本次更新表格、figure 和设计说明。

在随机命中布局下，全本地原始算子约 27.4 µs。Host 命中率 10% / 15% 时，本实现约 27.7 µs；20% 时约 30.7～31.0 µs；25% 时约 34.9～35.1 µs；30% 时约 40.1～40.7 µs。25% 和 30% 相比上一版 V44 优化算子分别缩短约 10% 和 7%，但相对全本地的吞吐仍下降约 22% 和 32%，尚未达到允许下降 20% 的目标。

## 2. 输入范围与三条数据路径

本实现是针对当前实验形状的专用路径：FA2 backend、一个 noncausal decode query、64 个 query heads、512 维 latent KV、64 维 RoPE、page size=1、2048 个选中页。实验的 `sm_scale=1/16`。输入使用匹配的 BF16 或 FP16，末维连续且满足 16-byte 对齐要求，GPU 至少有 108 个 SM，并满足动态 shared memory 容量要求。实测设备为 GPU 1：A100-SXM4-40GB。

| 对比对象 | 数据路径 | 计时范围 |
|---|---|---|
| Memcpy | CPU gather 到 pinned buffer → 串行 H2D DMA → 原始 attention | GPU 时间包括每轮 DMA、依赖交接和 attention；CPU gather 单列 |
| Direct | 原始 `BatchMLAPagedAttentionKernel` 直接读取混合地址空间中的 HBM / Host KV | attention GPU 时间 |
| Segmented | 本地计算 CTA 读取 HBM；Host 计算 CTA 直接把 Host KV 分段读入自己的 shared memory；同一 kernel 合并结果 | 一个融合 attention kernel 的 GPU 时间 |

历史 V44 路径也保留：producer warp groups 先把 Host KV 读入内部 HBM scratch，Host 计算 CTA 等待 KV-ready 标志，再从 scratch 读入 shared memory。它与本地 attention 在同一 kernel 中并行，是最近一次优化的配对参照。可通过编译宏 `FLASHINFER_MLA_HOST_STAGED_BASELINE` 选择，也可加载保留的 `baseline_v44.so`。更早的 V35 表格和图已归档。

## 3. 混合地址空间与 GPU 直访

最终实验采用 CUDA Driver VMM，而不是把 CPU tensor 转成 CUDA tensor：

1. `cuMemAddressReserve` 预留一段连续虚拟地址。
2. `cuMemCreate` 分别创建 GPU HBM 和 Host NUMA 物理分配。
3. `cuMemMap` 把前半段映射到 GPU HBM，后半段映射到 GPU 所在 NUMA 节点的 Host DRAM。
4. `cuMemSetAccess` 授予 GPU 访问两段内存的权限，并使 CPU 能访问 Host 段。
5. 使用 `cuMemGetAllocationPropertiesFromHandle` 验证实际物理位置；不能仅凭 PyTorch tensor 的 CUDA device 标签判断数据位于 HBM。

这里 `CU_MEM_ALLOCATION_TYPE_PINNED` 表示物理页不能迁移，实际所属位置由 allocation 的 `location` 决定。测试代码检查 Host NUMA VMM 支持以及映射粒度。普通 mapped pinned memory 直访也有独立探测代码；最终混合 cache 的连续指针和物理位置由上述 VMM 路径提供。

每个 token 的布局为 512 个 latent 元素加 64 个 RoPE 元素，BF16 / FP16 下共 `(512+64)×2=1152 B`。`host_page_start` 是物理页索引的分界：小于它的页属于 HBM，大于等于它的页属于 Host。实验始终各分配一半 cache 容量；“Host 命中率”描述的是选中的 2048 行中有多少来自 Host，不是整个 cache 的容量比例。

128K、256K、512K、1M 表示总 KV 上下文长度。算子只读取 top-k 对应行，因此 PCIe 有效数据量由 Host 命中数决定，不随总上下文线性增加。更大的 cache 仍可能影响地址转换和访存局部性。

## 4. CTA 分工与索引分块

kernel 采用 cooperative launch，固定启动 108 个 CTA。每个 CTA 有 256 个线程，分成两个 128-thread warp group。

| CTA 编号 | 工作 |
|---|---|
| 0～15 | 16 个本地 attention 分块 |
| 16～47 | 32 个 Host attention 分块 |
| 48～79 | 32 个有效 merge CTA；每个 warp group 合并一个 head，共 64 个 head |
| 80～107 | 更新 epoch 后返回；当前固定布局中不参与结果合并 |

`prepare_host_split` 扫描 2048 个选中页，使用 warp ballot、popcount 和 warp 前缀计数形成本地 / Host 的独立序号。某一 tier 内的第 `rank` 行被分配到 `rank % chunks`，在该 CTA 的紧凑索引数组中位于 `rank / chunks`。这使分块长度至多相差一行，同时避免在算子外提前重排索引。

32 个 Host 分块是一个具体的 tile 边界优化：

| Host 命中率 | Host 行数 | 每个 Host 分块最多行数 | 计算路径 |
|---|---:|---:|---|
| 20% | 410 | 13 | 16-key narrow tile |
| 25% | 512 | 16 | 16-key narrow tile，恰好不需要扩大 tile |
| 30% | 614 | 20 | 单个 32-key tile |
| 40% | 819 | 26 | 单个 32-key tile |
| 100% | 2048 | 64 | 常规多个 32-key tile |

上一版只有 28 个 Host 分块，25% 时每块最多 19 行，已经进入 32-key 路径。32 个分块使 25% 回到 narrow 路径，同时减少了旧版 KV 中转和就绪同步。

## 5. 分段读取与 QK 重叠

对于最多 32 行的 Host 分块，计算 CTA 用 `cp.async` 从映射的 Host 地址直接读取到自身 shared memory。Query 从 GPU 内存读取；Host KV 分成三个组，随后逐段累加 QK：

| 提交顺序 | 数据 | 等待点及其后的计算 |
|---|---|---|
| 0 | Query：latent 与 RoPE | 与下一组一起等待完成 |
| 1 | KV latent 前 256 维 | `wait_group<2>` + CTA barrier → 计算前半段 QK |
| 2 | KV latent 后 256 维 | `wait_group<1>` + CTA barrier → 累加后半段 QK |
| 3 | KV RoPE 64 维 | `wait_group<0>` + CTA barrier → 累加 RoPE QK |

四组已提交后，前半段到达即可开始计算，此时后续 Host 读取仍可在途。`wait_group` 保证本线程要求的异步组完成，CTA barrier 保证跨线程使用 shared memory 时数据已就绪。无效行使用 zero fill，并在 logits 阶段 mask，避免未初始化数据参与运算。

16-key narrow 路径由 warp group 0 完成 QK 和 attention，PV 分两次计算各 256 个输出维度；32-key 路径在两个 warp group 间分摊 QK 的 key 维度，随后进行必要的 softmax / PV 协作。超过 32 行的 Host 分块使用常规多 tile 路径；当前的“按特征分段 QK 重叠”专门用于 single-tile 分支。

本地 CTA 与 Host CTA 同时推进。Host 分段读取和 QK 的重叠发生在同一计算 CTA 内。默认路径没有独立 KV staging kernel，也没有 KV 的 H2D memcpy；内部 HBM 仍用于保存各分块的 attention 输出。

## 6. 结果发布与稳定合并

每个计算分块输出归一化的部分结果 `O_i` 和 base-2 的 `LSE_i`。计算完成后，CTA 执行 `__threadfence()` 和同步，再用 `st.release.gpu.global.u64` 发布当前 epoch 的结果就绪标志。

merge warp group 先轮询自己所需的分块，再用 acquire load 建立对 partial output / LSE 的读取顺序。每轮运行都更新 epoch；不能把前一轮 graph replay 的 ready 标志当作本轮数据已完成。Query 与 output 共用部分 shared storage，其复用也保留了需要的 warp 同步。

对一组分块，合并使用稳定的 log-sum-exp 权重：

```text
m = max(LSE_i)
w_i = exp2(LSE_i - m)
O_group = sum(w_i * O_i) / sum(w_i)
LSE_group = m + log2(sum(w_i))
```

本地与 Host 分别完成组内合并，再按同样的规则合并两个 tier。空分块的 LSE 为负无穷、权重为零。若调用者要求以自然对数返回 LSE，最终乘以 `ln(2)`。这种分区合并保持了 softmax 的归一化关系；BF16 / FP16 的 partial output 和概率转换仍存在数值误差，需要按已验证的容差使用。

Python 保留了 5.25 MiB 的 scratch 容量，以兼容历史路径；当前 48 个计算分块的 partial output 实际占 3 MiB。另有 32 KiB state buffer，包含 epoch、ready 标志和 partial LSE。已编译的 BF16 / FP16 kernel 均为每线程 242 个寄存器，未报告 stack 或 local spill。

CUDA Graph capture 前需调用一次完成 JIT 和内部 buffer 分配。相同 wrapper 的调用必须串行，避免共享 scratch 和 epoch state 被并发覆盖。Host 内容更新在 GPU 读取完成后执行；正确性测试通过显式同步验证了这个场景。

## 7. PCIe 预算与尚未隐藏的尾部

当前机器的 PCIe 4.0 ×16 单向编码上限约 31.51 GB/s，大块映射 Host 读取实测约 27.03 GB/s。计算和输入数据见 [bandwidth_budget.json](../temp/host_attention_pcie20/bandwidth_budget.json) 与[带宽原始记录](../temp/mapped_host_read_bandwidth.csv)。下面的估算只计算有效 KV 数据；实际还存在请求开销、启动和稀疏访问成本。

| Host 命中率 | 数据量 | 按 27.03 GB/s 估算 | 稀疏读取并写入 HBM 的基准 |
|---|---:|---:|---:|
| 25% | 576 KiB | 21.82 µs | 25.41 µs |
| 30% | 690.75 KiB | 26.17 µs | 29.76 µs |

按历史全本地参考 27.46 µs 计算，延迟增加 20% 的预算为 32.96 µs，吞吐下降 20% 的预算为 34.33 µs。这是两个不同的判定标准。带宽本身没有排除达标的可能，但本地时间包含 HBM 读取、同步和结果合并，不能全部当成可覆盖 CPU 拉取的计算时间。

完整 logits 依赖全部 latent 与 RoPE 数据；softmax、PV、partial output 发布和最终合并仍有 Host 数据到达后的依赖尾部。分段 QK 只能提前其中一部分计算。上述读取并写入 HBM 的基准还包含目标写入，属于实现参考，不是本直访算子的物理下限；当前未单独给出各阶段的可靠稳态耗时。

最新随机命中的配对结果如下，单位为 µs：

| 上下文 | V44 25% | 当前 25% | V44 30% | 当前 30% |
|---|---:|---:|---:|---:|
| 128K | 38.87 | 35.07 | 43.63 | 40.67 |
| 256K | 38.87 | 35.03 | 43.13 | 40.20 |
| 512K | 38.85 | 35.01 | 43.17 | 40.22 |
| 1M | 38.80 | 34.89 | 43.13 | 40.06 |

完整的头部 / 尾部布局、Memcpy / Direct baseline 和 0%～40% 每 5% 一档的结果见[主报告](../temp/host_attention_comparison/report.md)。当前 25% 和 30% 在上述两种 20% 标准下均未达标；20% 命中时也还有约 12%～13% 的延迟增加，尚未实现完全隐藏 Host 拉取。

## 8. 测量和正确性证据

- 4 个上下文档位，随机 / 头部 / 尾部布局，0%～40% 每 5% 一档，另有 60% / 80% / 100%。布局描述 Host 行在 top-k 列表中的位置；各 tier 内的物理页仍随机采样。
- 每个 case 对各方法使用相同 Query、KV 和选中索引；40 轮、8 组索引、每个 CUDA Graph 内 100 次串行 GPU cycle。方法和命中率执行顺序随机化；表格为中位数，图中误差线为 p10～p90。
- Memcpy 每个 GPU cycle 都执行一次真实 H2D DMA。CPU packing、top-k、索引更新、规划和初始化排除在 GPU 时间之外；CPU packing 和串行 wall time 在[分项表](../temp/host_attention_comparison/details.md)中单列。
- 完整比较包含 576 项独立 FP32 输出检查；最大绝对误差 0.003749。另有 372 项 BF16 / FP16、LSE base-2 / base-e、graph replay、Host 内容修改和分块边界检查，最大输出绝对误差 0.004857；native API 回归及无效输入检查也通过。
- racecheck 和 synccheck 均无错误。预热后 Nsight Compute 测得 25% / 30% 的 sysmem fill 分别为 589824 / 707328 B，等于全部选中 Host 行的数据量；这些 profiling duration 不用于性能表。
- 初次完整测试中，外部 GPU 1 负载使部分全本地参考上浮到 32～34 µs。受影响的随机 1M 和头部 128K / 256K 已整组重测；初始数据、重测数据和替换来源全部保留，单次 40 轮中未裁剪样本。

详细证据见 [verification.json](../temp/host_attention_pcie20/verification.json)。普通寄存器读取、扩大 Host 分块、提前 merge 预取、重复 QK 和 warp 按行读取等试验也保存在[优化分析](../temp/host_attention_pcie20/analysis.md)及 Git 历史中。当前选择的是收益较稳定的分段 `cp.async` 版本。

## 9. 代码入口与复现

| 文件 | 入口及作用 |
|---|---|
| [mla/_core.py](../flashinfer/mla/_core.py) | `run` 的 `host_page_start` 校验、内部 buffer 和 graph capture 前预热 |
| [jit/attention/modules.py](../flashinfer/jit/attention/modules.py) | `_host_overlap_v2` JIT 命名 |
| [attention/mla.cuh](../include/flashinfer/attention/mla.cuh) | `prepare_host_split`、`load_private_host_segment`、`compute_host_split`、`BatchMLAPrivateHostKernel`、`merge_private_host_group` |
| [batch_mla_run.cu](../csrc/batch_mla_run.cu) | FFI 参数连接 |
| [混合内存构造](../temp/bench_glm52_sparse_decode_hbm_host_direct.py) | `MixedHBMHost`：VMM 和物理位置验证 |
| [比较测试](../temp/bench_glm52_host_comparison.py) | 三条主路径、V44 frozen module、CPU packing 与 GPU 计时 |
| [表格与图导出](../temp/plot_glm52_host_comparison.py) | 从已保存的 `after_*.json` 更新主报告、CSV 和 figures |

在仓库根目录更新展示，使用现有完整数据即可：

```bash
MPLCONFIGDIR=temp/matplotlib python temp/plot_glm52_host_comparison.py
```

重新测量时选择新的输出目录；已有 raw JSON 会拒绝覆盖。以下命令使用本机既有 Python / CUDA 环境：

```bash
export PYTHONPATH=.
export FLASHINFER_WORKSPACE_BASE="$PWD/temp/jit_host_pcie_final"
export FLASHINFER_CUDA_ARCH_LIST=8.0
export TMPDIR="$PWD/temp"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MAX_JOBS=8

for layout in random head tail; do
  /mnt/data/mhy-vllm-venv/bin/python temp/bench_glm52_host_comparison.py \
    --host-kernel auto --layout "$layout" --rounds 40 --index-sets 8 \
    --baseline-module temp/host_attention_pcie20/baseline_v44.so \
    --output "temp/host_attention_new/after_${layout}.json"
done

MPLCONFIGDIR=temp/matplotlib python temp/plot_glm52_host_comparison.py \
  --input-dir temp/host_attention_new
```

本设计的下一步需要进一步缩短读取启动、Host 依赖计算和结果合并的尾部，并检查固定 CTA 布局的利用率。当前证据支持上述单层、固定形状的结果；完整 GLM 推理、其他 batch / head / page 配置及其他 GPU 的性能需要另外验证。
