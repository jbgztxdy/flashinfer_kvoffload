# 交接：远端 KV 稀疏 MLA 算子项目（2026-10-27）

给下一个 agent：这里只写状态和指针，细节请读引用的文件。

## 1. 目标（用户原话的要点）

用户的长期 goal（逐项）：
1. 把算子和项目迁移到集群 `kaituo1000_login21` 的 `/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer`。**已完成。**
2. 适配 H 卡（H800），把跨 decode 的 LRU 缓存融合进同一个算子（不拆成两个 kernel），性能逼近理论极限。**算子已完成；H800 上还没达到目标（见 §3.1）。**
3. 在 PCIe 和 NVLink（用 GPU 间 NVLink 模拟）下分别测试，batch 尽量多（MTP8，1～1024 行）。**A100 全量已完成；H800 全量已跑但默认配置表现差，需调优后重测。**
4. 与 SGLang HiSparse 做单层 decode 性能对比（不跑端到端），目的是**体现本文 kernel 相对 HiSparse 的优势**（HiSparse 按它实际的方式运行，不要替它优化）。**H800 已有数据，部分场景输。**

后来新增的任务：
5. 在 A100 上模拟 50、100 GB/s 带宽和更高访问延迟（≤ 20 µs），对比本文 kernel、两种朴素方法、纯本地 HBM。**已完成**，见 `docs/mla_host_link_emulation.md`。
6. 用 GLM-5.3 的真实 indexer trace（语料为拼接的 claudecode-trace）跑多步，记录 LRU 命中率变化，以及与纯本地 HBM 的性能差距（尽量接近纯本地）。**trace 采集进行中（§3.2）**，采集完成后在 A100 上测。

## 2. 已完成的工作（读这些文件）

| 内容 | 位置 |
|---|---|
| 汇总报告（带图，面向汇报） | `docs/mla_host_lru_report.md`（另有 `docs/mla_host_lru_report_external.md`，未跟踪，非本 agent 所写） |
| A100 全量 batch 测试（PCIe/NVLink × MTP8/MTP1，单步/跨 decode，方法定义） | `docs/mla_host_a100_full.md`，数据 `temp/a100_full/` |
| 开发过程、方案取舍、失败尝试 | `docs/mla_host_a100_lru.md` |
| H800 结果（验证、与 HiSparse 对比、全量扫描） | `docs/mla_host_h800_lru.md`，数据 `temp/h800/results/` |
| 链路带宽/延迟模拟（A100） | `docs/mla_host_link_emulation.md`，数据 `temp/sim_link/`，模拟器 `include/flashinfer/attention/far_link_emu.cuh`（只在 `-DFLASHINFER_FAR_EMU` 下编入，正常构建 SASS 与原来逐字节相同） |
| kernel | `include/flashinfer/attention/mla_host_batch.cuh`，launcher `csrc/batch_mla_run.cu`，Python `flashinfer/mla/_core.py`（`host_stage`、`set_host_cache` 等），测试 `tests/attention/test_mla_host_batch.py`（71 项） |
| 测试脚本 | `temp/h800/bench_rounds.py`（跨 decode + LRU）、`temp/h800/bench_sweep.py`（单步）、`temp/h800/tiers.py`（VMM 远端映射、trace 回放），两者支持 `--emu` |

kernel 设计要点：同一个 cooperative kernel 两种模式，PCIe 用 staged（fetch CTA 全 batch 去重搬运，同时写入 LRU），NVLink 用原地读（attention CTA 直接读远端，并把未命中页写回缓存）；UVA + VMM 统一地址；LRU 是每请求一个池加全局 `resident` 表，用 radix select 选 victim。

## 3. 未完成的任务

### 3.1 H800 调优并胜过 HiSparse（目标 2、3、4）
- 问题（`docs/mla_host_h800_lru.md` §2）：默认配置（8 个 fetch CTA）下，PCIe 搬运只有约 30 GB/s（DMA 为 55 GB/s）；FA2 MLA 的 attention 比 FlashMLA 慢约一倍；HiSparse 在 PCIe MTP1 ≥ 64 请求、NVLink MTP8 128 请求时更快；大部分场景输给 Memcpy+LRU。
- 已排队的调优作业：`1004362`（PCIe：fetch CTA 8/16/32/64，加原地读）、`1004363`（NVLink：staged 16/32/64 对比原地读），脚本 `temp/h800/tune_{host,peer}.sh`，结果会写入集群上的 `temp/h800/results/fi-tune-*`。调度器估计 11/02 开始。
- 参考：A100 链路模拟显示，链路低于 NVLink 速度时 16 个 fetch CTA 比 8 个快约 1.8 倍。
- 之后的工作：根据调优结果选参数重测 HiSparse 对比；若 attention 仍是瓶颈，考虑 Hopper 路径（wgmma/TMA）或接 FlashMLA（这需要和用户讨论）。

### 3.2 GLM-5.3 trace 采集（目标 6）
- 模型：`/WORK/PUBLIC/luyouy_work/rtx/models/zai-org--GLM-5.3`（FP8，141 个分片，已校验；tokenizer 与 5.2 完全相同）。
- 语料：`rtx/yjw/test/glm_trace/corpus_64k.pt`（4 篇 × 64K token，由 claudecode-trace 的 138 个会话拼接而成），生成脚本为 `01_build_corpus.py`。
- 采集：`rtx/yjw/test/glm_trace/02_prefill_trace.py`（本地副本 `temp/glm_trace/`），在 A800 上用 BF16 逐层 prefill（SGLang 的 DSA 需要 SM90，A800 跑不了），输出每个 indexer 层的 top-k。作业脚本 `03_trace.sbatch`（默认 1 卡；用户允许最多 8 卡）。
- **验证作业 `1031711`**（1 张 A800，4 层，4K token，`--check`）在排队，日志在 `rtx/yjw/test/glm_trace/logs/`。检查项：分块与不分块 indexer 的 top-k 一致；absorbed 与展开形式的 attention 误差；全 78 层跑完时的 next-token loss（应为 1～2，约 11.9 表示实现有误）。
- **用户最新要求**：不需要和 SGLang 对比，但要保证实现正确；**trace 要记录完整信息，最终要的是 index score 和 top-k**。当前脚本只保存了 top-k，**需要补上 index score**（top-k 位置对应的分数；如果需要，也可以保存每行完整分数的统计量），并保存足够的元数据（层号、文档、位置、组起点、会话来源等）。
- 采集完成后：把 trace 转成 `temp/h800/tiers.py` 的格式（`selections [layers, groups, span, topk]`、`group_starts`），在 A100 上做多步回放（span 1024），记录每步命中率和本文 ÷ 纯本地。
- 注意：只有 21 个层跑 indexer（第 0、1、2 层，以及之后每隔 4 层一个），其余层复用上一层的 top-k。

### 3.3 待办，需要提醒用户
- **4 层共用一次 prepare 和 LRU**：共享 top-k 的层可以复用去重和 LRU 结果，能去掉小 batch 上每层约 20～30 µs 的前置开销。用户要求在合适的时候提醒他。

## 4. 环境与规则

- **集群**：`ssh kaituo1000_login21`。登录节点没有 GPU，所有 GPU 工作都通过 Slurm。H800 在 partition `h01`（g40、g44），每个作业最多 4 卡，尽量用 1～2 卡，时限要短；作业模板 `temp/h800/job.sbatch`（不绑定节点）。A800 在 partition `a01`，可用于 trace 采集（最多 8 卡）。所有文件放在 `/WORK/PUBLIC/luyouy_work` 下。远端的项目说明在 `rtx/MEMORY.md`。
- **本机 A100**：GPU0 是近端；GPU1 经 NV12 NVLink 作为远端；Host 内存在 NUMA0。常用环境：`CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_a100_final FLASHINFER_CUDA_ARCH_LIST=8.0 CUDA_VISIBLE_DEVICES=0,1 numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python ...`；链路模拟构建另加 `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_FAR_EMU`，并使用独立的 `FLASHINFER_WORKSPACE_BASE=temp/jit_emu`。
- **用户偏好**：
  - 结果表都要同时给出两个下限：(a) max(纯本地计算, 去重后未命中字节 ÷ 链路带宽)，(b) 纯本地计算。下限要用实测最好值，保证比值 ≥ 1。
  - 数据都保存到文件并提交。
  - 报告要自包含，并突出本文 kernel 的优势，但不能隐瞒输的场景。
  - 和用户讨论方案时，先给推荐方案再执行。
  - 尽量节省 token：长时间等待时用静默的后台轮询脚本，不要用频繁的 cron。
  - 模拟、实验代码要简洁，不影响正常构建，后期会删除。
- **工作区里非本 agent 的未提交改动**：`docs/mla_host_lru_report.md`、`temp/report_lru/plot.py` 与 png、`include/flashinfer/attention/decode.cuh`（一个行尾空格），以及未跟踪的 `docs/mla_host_lru_report_external.md`、`docs/mla_host_lru_report.pdf` 等。不要回滚这些改动；提交时只 `git add` 自己改的文件。
