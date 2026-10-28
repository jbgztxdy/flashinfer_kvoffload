# GLM-5.3 真实 top-k trace：新语料验证与回放接入（2026-10-28）

正式目标是用 64K 上下文、1024 连续位置的真实 top-k 和 index score，比较融合 LRU 的远端 KV sparse MLA 与朴素方法、HiSparse，并逐步报告缓存命中率和两个下限。**目前新语料和 8K 验证已完成，64K 单篇采集排队；下面的 A100 数据只证明回放接入可用，不能替代正式性能结果。**

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

## 正在排队及后续

- **1040875**：a01，8 张 A800，2 小时，第 0 篇 64K，16×1024 记录窗口，保存 selections、scores、row_stats、角色及完整 loss。最初依赖严格检查，复核 system 异常后解除原依赖，没有重复提交。10/28 11:43 为 Priority 排队，暂估 10/31 09:19。
- 单篇完成后检查保存数据、角色 loss 和耗时，再采集其余文档；真实 64K/1024 trace 尚未得到，不能标记完成。
- 正式回放记录每步缓存命中率、本文/纯本地比值和两下限；包括 A100，以及 H800 的 HiSparse 对比。
- 角色过滤要保留真实位置和时间顺序。四篇中只有文档 1 有 3 段、文档 3 有 1 段 assistant 连续长度 ≥1024；不能删去中间 tool/user 位置后仍假称 1024 连续 token。
- **后续全量的窗口计划已准备**：原来的 16 个均匀窗口在四篇中都没有完整覆盖一段 1024 连续 assistant token。`07_plan_windows.py` 将最近的窗口移至文档 1 的 35184 / 41887 / 48296、文档 3 的 50621，每篇仍为 16×1024，其他窗口保留，文档 0/2 不变。完整位置及角色计数在 [`swe_windows_64k.json`](../temp/glm_trace/results/swe_windows_64k.json)。已用真实语料检查四段均全为 assistant、窗口有序/无重叠/在文档后半段，默认 64K/8K/4K 窗口与此前逐个一致，错误的语料或 span 会拒绝。
- 采集脚本新增可选 `--window-plan results/swe_windows_64k.json`；已同步集群。作业 1040875 未传该选项，继续原均匀窗口，用于单篇计时；**全量采集尚未提交**，应等单篇耗时和数据核查后带该参数。模型计算路径未改，输出增加采集脚本/窗口助手的源码哈希及计划哈希。新窗口路径目前通过 CPU 检查，GPU 全流程须随正式采集验证。
- H800 优化顺序仍待用户确认，单 kernel 融合约束保留。

作业、源码哈希和依赖变更见 [`swe_pipeline_20261028.json`](../temp/glm_trace/results/swe_pipeline_20261028.json)，集群数据根目录为 `/WORK/PUBLIC/luyouy_work/rtx/yjw/test/glm_trace`。语料 SHA-256 为 `f82182b02107b998bf87a95fa855890a919fd9cb6f3438a4e8f73b5fbf67bbaa`，8K trace SHA-256 为 `742c9a61dc6db424a9c7fca5a38fc25c573fbf702ee59afb27f4b1ed0ea906fe`。二进制 trace 保存在集群，接入检查的本地副本在 `temp/glm_trace/validation8k/doc0.pt`，未纳入 git。
