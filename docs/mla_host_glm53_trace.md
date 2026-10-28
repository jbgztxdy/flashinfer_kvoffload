# GLM-5.3 真实 top-k trace：新语料验证与回放接入（2026-10-28）

正式目标是用 64K 上下文、1024 连续位置的真实 top-k 和 index score，比较融合 LRU 的远端 KV sparse MLA 与朴素方法、HiSparse，并逐步报告缓存命中率和两个下限。**64K 单篇采集已完成，完整数据检查通过但全位置 loss gate 失败；四篇带 assistant 窗口的采集已排队。下面的 8K A100 数据仅为接入检查。**

## 语料与验证

模型是集群的 `zai-org--GLM-5.3` FP8 权重，采集使用逐层反量化到 BF16 的脚本。语料来自 `nebius/SWE-rebench-openhands-trajectories`（CC-BY-4.0），只取 resolved 轨迹，每篇一条，原生 GLM 模板渲染。四篇均截至 65536 token，原长为 122916 / 121172 / 119659 / 118550，构建作业 1040732 成功。

四篇开头 5675 token 都是 system，因此交接中原建议的 4K 检查不能覆盖 assistant。验证作业 **1040816** 改用原语料文档 1 的 8192 token，2 张 A800，运行 19 分 15 秒，全部 78 层完成。保存的 top-k 形状为 `[21,2,256,2048]`，21 个 indexer 层及 78 层复用映射完整。

| 检查 | 结果 |
|---|---:|
| 第 0 层分块/稠密 top-k 重合率 mean / min | 1.0000 / 1.0000 |
| 第 0 层记录分数与稠密参考的最大相对误差 | 5.21e-07 |
| 全部保存行：因果范围、无重复、分数排序、有效数量、补位、统计一致性 | 通过 |
| absorbed / expanded attention 最大相对误差 | 4.27e-03 |

完整 next-token loss 按**目标 token**的角色统计，`loss[r]` 对应 `role[r+1]`：

| 角色 | token 数 | 平均 loss |
|---|---:|---:|
| system | 5674 | 4.224 |
| user | 1564 | 2.843 |
| assistant | 793 | **0.827** |
| tool | 160 | 4.367 |

严格检查作业 **1040870 返回失败**：除首块外仍有一块 `[1536,1792)` loss 为 6.3025，超过 6 的阈值。该块的 256 个目标 token 全是 system，内容为原始工具描述中转义的 JSON 文本。数据检查及 assistant loss 均通过。复核后继续已授权的单篇 64K 采集，保留严格失败记录和原因，未改语料或阈值。不能把这次结果描述为全部数值判据通过。

原始检查、角色诊断和复核决定分别在 [`swe_check_20261028.json`](../temp/glm_trace/results/swe_check_20261028.json)、[`swe_check_with_roles_20261028.json`](../temp/glm_trace/results/swe_check_with_roles_20261028.json)、[`swe_check_decision_20261028.json`](../temp/glm_trace/results/swe_check_decision_20261028.json)，完整 GPU 日志在 [`glm-swe-check-1040816.out`](../temp/glm_trace/results/glm-swe-check-1040816.out)。

## A100 回放接入检查

直接读取 `doc0.pt` 的 selections 和 group_starts，现有 `tiers.py` 无需转换。复现使用 `temp/h800/bench_rounds.py --captures temp/glm_trace/validation8k/doc0.pt --context 8192`，PCIe MTP1 为 256 步、1/4 请求；NVLink MTP8 为 32 步、1/4/16 请求；均 3 次冷启动、4096 槽/请求、8 个 fetch CTA。查询和 KV 是随机 BF16，真实的是 GLM top-k，未宣称为完整模型 decode。

本文 staged 和原地读在每步都对采样行校验，最大绝对误差 ≤ 0.006066。朴素方法按原有脚本仅用于计时，CPU LRU 不计时，PCIe CPU gather 不计时；未改变基线口径。下表为后一半步平均，µs，命中率按远端**去重页**计。

| 链路 | MTP | 请求 | 本文模式 | 本文 µs | Memcpy+LRU | Direct+LRU | 去重页命中率 | 下限 a | 下限 b |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| PCIe | 1 | 1 | staged | 50.35 | 44.66 | 45.67 | 98.46% | 31.74 | 31.74 |
| PCIe | 1 | 4 | staged | 54.78 | 61.82 | 59.94 | 99.05% | 46.08 | 46.08 |
| NVLink | 8 | 1 | 原地读 | 77.31 | 56.32 | 58.56 | 91.51% | 43.82 | 43.82 |
| NVLink | 8 | 4 | staged | 155.84 | 118.66 | 121.66 | 95.51% | 108.26 | 108.26 |
| NVLink | 8 | 16 | 原地读 | 496.19 | 520.96 | 524.16 | 94.35% | 435.48 | 435.48 |

下限 a = max(纯本地计算, 本步去重后未命中字节/实测链路带宽)，下限 b = 纯本地 attention 的最好实测值。NVLink 补测 0% 远端的 CUDA graph attention 候选，包含原生和本文无缓存路径；其计时方式与逐步回放不同，用于得到最好实测计算下限。未截断比值。全部方法、逐步未命中、命中率、耗时、两下限和比值见 [`a100_8k_smoke.md`](../temp/glm_trace/results/a100_8k_smoke.md)，原始 JSON 和样本在同目录。

这些窗口包括 system/user/tool 位置，且上下文仅 8K；不能据其高命中率外推正式 64K assistant decode。单请求和 NVLink 4 请求仍输给朴素方法，不能隐去。

## 64K 单篇采集与检查

**1040875 已完成**：8×A800、g08、17 分 59 秒（GPU 计算 952 秒），峰值每卡 8.7 GiB。保存 21 个 indexer 层、16×1024 连续位置、top-k 2048 的 selections/scores，原始 `doc0.pt` 为 5.3 GiB。CPU 作业 **1079470** 验证全部保存行的形状、因果范围、无重复、分数排序和统计，以及 78 层复用映射，数据检查全部通过。

| 目标角色 | token 数 | 平均 loss |
|---|---:|---:|
| system | 5674 | 4.134 |
| user | 954 | 3.173 |
| assistant | 7635 | **0.550** |
| tool | 51272 | **9.842** |

**严格 gate 返回 1，未全部达标。** 154 个首块之后的高 loss 块有 1 个纯 system、138 个纯 tool（包含最后不足 256 token 的块），其余为混合角色，没有 assistant 占多数的高块。assistant 平均 loss 通过；不能用它掩盖工具输出明显高于预期。继续已授权的四篇采集以获得完整 assistant 窗口，保留失败报告、不更改阈值和语料，不再重复排查已验证的采集实现。证据和复核决定为 [`swe64k_doc0_inspection_20261028.json`](../temp/glm_trace/results/swe64k_doc0_inspection_20261028.json)、[`swe64k_doc0_decision_20261028.json`](../temp/glm_trace/results/swe64k_doc0_decision_20261028.json)。

原始 trace SHA-256 为 `169b10b9a3ca58af1a56e601aa52485832fd7dc182bf5756cd3841925cb4fd9d`。`08_export_replay.py` 仅导出未改动的全部 top-k 和窗口/角色/层出处，scores 保留在原件。CPU 导出 **1079523** 成功，2.7 GiB 的本机副本为 `temp/glm_trace/replay64k/doc0.pt`，SHA-256 `15b21ce91eeeb40b48d6d77db708f912a40a4259ffcef301d3ee681a41690d72`。binary 不纳入 git，导出凭据在 [`swe64k_doc0_replay_export_20261028.json`](../temp/glm_trace/results/swe64k_doc0_replay_export_20261028.json)。`run_64k_replay.py` 在本机串行覆盖真实 PCIe/NVLink 及 50/100 GB/s＋20 µs，1024 连续位置、7 pass，并保存实际选用窗口和角色计数；第 0 篇没有完整 assistant 窗口，结果须标为混合角色回放。

## A100：64K、1024 连续位置的混合角色回放

已完成 6 份计时 JSON、15 个 batch 配置，每配置 7 次冷启动，LRU 4096 槽/请求，固定 f16/h64。MTP1 为 1024 步，MTP8 为 128 步；请求数 1/4/16 分别对应 MTP8 的 8/32/128 行。两种本文模式共 14592 次 launch 的逐步采样行校验通过，最大绝对误差 0.004352。kernel 为第二轮优化 `614655f1` 的版本，完整 header SHA-256、输入出处和全部 pass 数据见 [`manifest.json`](../temp/glm_trace/results/a100_64k_doc0/manifest.json)。

以下为后一半步的逐步中位数平均，µs。此处比较方法间耗时，**不作为相对旧 kernel 的优化幅度**；旧/新代码的严格对照仍见两份 A100 调优报告。

| 链路 | MTP | 请求 | 本文模式 | 本文 | Memcpy+LRU | Direct+LRU | 下限 a | 下限 b | 本文÷a | 本文÷b |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 真实 NVLink | 1 | 1 | staged | 63.84 | 46.75 | 46.07 | 31.74 | 31.74 | 2.01 | 2.01 |
| 真实 NVLink | 1 | 4 | staged | 57.25 | 56.89 | 60.28 | 46.08 | 46.08 | 1.24 | 1.24 |
| 真实 NVLink | 8 | 16 | 原地读 | 543.15 | 615.74 | 616.03 | 494.59 | 494.59 | 1.10 | 1.10 |
| 真实 PCIe | 1 | 4 | staged | 75.74 | 96.28 | 93.62 | 47.19 | 46.08 | 1.61 | 1.64 |
| 100 GB/s＋20 µs | 1 | 1 | staged | 85.08 | 69.58 | 116.35 | 31.74 | 31.74 | 2.68 | 2.68 |
| 100 GB/s＋20 µs | 8 | 1 | staged | 150.05 | 133.63 | 250.64 | 50.94 | 47.10 | 2.95 | 3.19 |
| 100 GB/s＋20 µs | 8 | 4 | staged | 231.30 | 204.34 | 558.24 | 108.54 | 108.54 | 2.13 | 2.13 |
| 100 GB/s＋20 µs | 8 | 16 | staged | 685.25 | 867.33 | 2445.07 | 494.59 | 494.59 | 1.39 | 1.39 |
| 50 GB/s＋20 µs | 8 | 4 | staged | 233.65 | 248.85 | 625.86 | 114.80 | 108.54 | 2.04 | 2.15 |
| 50 GB/s＋20 µs | 8 | 16 | staged | 722.05 | 1122.06 | 2801.26 | 522.30 | 494.59 | 1.38 | 1.46 |

下限 b 采用相同 trace/批次的**正常构建**全 HBM attention 最快样本；a = max(b, 逐步未命中字节/链路带宽)。模拟场景使用设置的 50/100 GB/s，不能误用物理 NVLink 的 DMA 测量值；固定访问延迟未计入 a，所以 a 是乐观下限。所有比值直接计算，未截断。全部模式、配置、逐步命中率、两下限及误差在 [`results.md`](../temp/glm_trace/results/a100_64k_doc0/results.md)。

100 GB/s＋20 µs 的 MTP8 16 请求，staged 比 Memcpy+LRU 快约 21%，但仍是纯 device 下限的 1.39 倍；小 batch 单/四请求仍有输给 Memcpy 的场景。真实 NVLink 的 16 请求原地读距下限约 10%，这不表示高延迟场景也能达到同样比例。混合窗口的稳态去重页命中率：MTP1 单请求约 64%、四请求约 92%；MTP8 的 1/4/16 请求约 42%/73%/67%。不同请求数选中的真实层/窗口不同，不能将曲线差异全归因于 batch。朴素基线仍只计 GPU，PCIe CPU gather 和 CPU LRU 未计时、数值不校验；HiSparse 仍待 H800。

## 正在排队及后续

- **1079524**：a01，8 张 A800，90 分钟，全部四篇 64K，`--window-plan results/swe_windows_64k.json`，16×1024 窗口，完整 loss，输出 `traces/swe64k_full_20261028`。18:23 为 Priority 排队。
- **1079527**：CPU 4 核 / 16G，依赖 `afterok:1079524`，逐篇执行原严格检查；某篇失败也继续检查其余篇，保留各自 JSON 和整体失败退出码。
- 正式回放记录每步缓存命中率、本文/纯本地比值和两下限；包括 A100，以及 H800 的 HiSparse 对比。
- 角色过滤要保留真实位置和时间顺序。四篇中只有文档 1 有 3 段、文档 3 有 1 段 assistant 连续长度 ≥1024；不能删去中间 tool/user 位置后仍假称 1024 连续 token。
- **后续全量的窗口计划已准备**：原来的 16 个均匀窗口在四篇中都没有完整覆盖一段 1024 连续 assistant token。`07_plan_windows.py` 将最近的窗口移至文档 1 的 35184 / 41887 / 48296、文档 3 的 50621，每篇仍为 16×1024，其他窗口保留，文档 0/2 不变。完整位置及角色计数在 [`swe_windows_64k.json`](../temp/glm_trace/results/swe_windows_64k.json)。已用真实语料检查四段均全为 assistant、窗口有序/无重叠/在文档后半段，默认 64K/8K/4K 窗口与此前逐个一致，错误的语料或 span 会拒绝。
- 采集脚本新增可选 `--window-plan results/swe_windows_64k.json`，已同步并由 1079524 使用。1040875 未传该参数，保留原均匀窗口。模型计算路径未改，输出增加采集脚本/窗口助手的源码哈希及计划哈希。新窗口路径通过 CPU 检查，GPU 全流程须随正式采集验证。
- 持续常规调优已获用户授权，保留单 kernel 融合；H800 仍待三个硬件实验结果，拆 kernel 的架构取舍仍需先询问。

作业、源码哈希和依赖变更见 [`swe_pipeline_20261028.json`](../temp/glm_trace/results/swe_pipeline_20261028.json)，集群数据根目录为 `/WORK/PUBLIC/luyouy_work/rtx/yjw/test/glm_trace`。语料 SHA-256 为 `f82182b02107b998bf87a95fa855890a919fd9cb6f3438a4e8f73b5fbf67bbaa`，8K trace SHA-256 为 `742c9a61dc6db424a9c7fca5a38fc25c573fbf702ee59afb27f4b1ed0ea906fe`。二进制 trace 保存在集群，接入检查的本地副本在 `temp/glm_trace/validation8k/doc0.pt`，未纳入 git。
