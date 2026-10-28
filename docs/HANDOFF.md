# 交接：远端 KV 稀疏 MLA 算子项目（2026-10-28）

给下一个 agent：这份文档记录项目状态和文件指针，细节请读引用的文件。第 3 节（正在跑的作业）和第 4 节（下一步）最要紧。

## 1. 项目目标

做一个 FlashInfer 里的 **远端 KV 稀疏 MLA decode 算子**，面向 DeepSeek/GLM 的 DSA：每个 query 只读 indexer 选出的 top-k 2048 个 KV 行，每行 576 个 bf16（1152 字节）。KV 放在远端，远端是 Host DRAM（走 PCIe）或另一张 GPU 的 HBM（走 NVLink）。**跨 decode 的 LRU 缓存融合在同一个 cooperative kernel 里**，不拆成两个 kernel。

**用户的核心目的**：这个带 LRU 的 DSA kernel 本身的性能，要明显优于两种朴素方法和 SGLang HiSparse，并且逼近理论下限。

用户提出的目标，以及各自的状态：
1. 把算子和项目迁移到集群 `kaituo1000_login21`，路径 `/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer`。**已完成。**
2. 适配 H 卡（H800），LRU 融合进同一个算子，性能逼近理论极限。**算子已完成；H800 上还没达标**（§4.1）。
3. 分别在 PCIe 和 NVLink（用 GPU 间 NVLink 模拟）下测试，batch 尽量多（MTP8 和 MTP1，1～1024 行）。**A100 全量已完成；H800 全量已跑，但默认配置表现差。**
4. 与 SGLang HiSparse 做单层 decode 对比，不跑端到端，目的是体现本文 kernel 的优势。HiSparse 要按它实际的方式运行，不要替它优化。**H800 已有数据，部分场景输。**
5. 在 A100 上模拟 50 和 100 GB/s 的带宽，以及更高的访问延迟（≤ 20 µs）。**已完成**，见 `docs/mla_host_link_emulation.md`。
6. 用 GLM-5.3 的真实 trace 跑多步，记录 LRU 命中率，以及与纯本地 HBM 的差距。**进行中**（§4.2）。
   - 用户 10/28 明确：trace 的用途是测本文 LRU DSA kernel 与其他方式的对比。所以需要一份**能记录每个 decode step 的 top-k** 的 trace，最重要的是 index score 和 top-k。
   - 实现必须正确，但不需要和 SGLang 对比。
   - 数据集可以换成更合适的，可以自己从 hf-mirror 或 modelscope 下载。

## 2. 已完成工作的文档和数据

| 内容 | 位置 |
|---|---|
| 汇总报告（带图，面向汇报） | `docs/mla_host_lru_report.md`（工作区有非本 agent 的改动，见 §6） |
| A100 全量测试和方法定义 | `docs/mla_host_a100_full.md`，数据在 `temp/a100_full/` |
| 开发过程、方案取舍、失败的尝试 | `docs/mla_host_a100_lru.md` |
| H800 结果（验证、HiSparse 对比、全量） | `docs/mla_host_h800_lru.md`，数据在 `temp/h800/results/`，汇总在 `temp/h800/results/summary.md` |
| 链路带宽和延迟模拟（A100） | `docs/mla_host_link_emulation.md`，数据在 `temp/sim_link/`，模拟器在 `include/flashinfer/attention/far_link_emu.cuh`（只在 `-DFLASHINFER_FAR_EMU` 下编入） |
| A100 100 GB/s 高延迟与小 batch 新调优（10/28） | `docs/mla_host_a100_100gb_tuning.md`；对照数据/源码哈希/未采用补丁在 `temp/sim_link/tune_100/`，75 项测试通过 |
| A100 第二轮：LRU 更新重叠与小 batch 分块 | `docs/mla_host_a100_lru_tuning.md`；正向/反向和原推荐参数对照在 `temp/sim_link/tune_lru/` |
| kernel | `include/flashinfer/attention/mla_host_batch.cuh`（约 1700 行；主体是 `BatchMLAHostBatchKernel`，在约 1505 行） |
| launcher | `csrc/batch_mla_run.cu` |
| Python 接口 | `flashinfer/mla/_core.py`（`host_stage`、`host_fetch_ctas`、`set_host_cache`、`_run_host_batch`） |
| 测试 | `tests/attention/test_mla_host_batch.py`（71 项） |
| 跨 decode 测试脚本（含 HiSparse 和各下限） | `temp/h800/bench_rounds.py` |
| GLM-5.3 新语料验证、真实 top-k 回放接入及未完成项 | `docs/mla_host_glm53_trace.md`；8K 接入结果 `temp/glm_trace/results/a100_8k_smoke.md`（尚非正式 64K/1024 结果） |
| 单步测试脚本 | `temp/h800/bench_sweep.py` |
| VMM 远端映射 | `temp/h800/tiers.py` |
| H800 作业模板 | `temp/h800/job.sbatch`，环境 `temp/h800/env_job.sh` |
| 早期的 fetch 带宽微基准 | `temp/fetch_bw/fetch_bw.cu` |

**kernel 设计要点**
- 两种模式共用一个 cooperative kernel：
  - **PCIe 用 staged**：fetch CTA 对全 batch 去重后搬运，同时写入 LRU。
  - **NVLink 用原地读**：attention CTA 直接读远端，并把未命中的页写回缓存。
- 地址用 UVA + VMM 统一。
- LRU 是每个请求一个池，加一张全局 `resident` 表；victim 用 radix select 选。
- 阶段顺序：prepare → grid.sync → resolve/LRU → grid.sync → fetch + attention items → 可选 merge。

## 3. 集群上正在跑和排队的作业（2026-10-28 中午）

查看命令：`ssh kaituo1000_login21 'bash -lc "squeue -u luyouy"'`。登录节点要用 `bash -lc` 才能找到 Slurm 命令。

| 作业 | 分区 | 内容 | 状态 | 结果在哪里 |
|---|---|---|---|---|
| **1004362** `fi-tune-host` | h01，1 张 H800，40 分钟 | PCIe 调优：staged 的 fetch CTA 取 8/16/32/64，另测原地读和 HiSparse。脚本 `temp/h800/tune_host.sh` | PENDING，调度器估计 11/02 02:34 开始 | 集群 `rtx/yjw/flashinfer/temp/h800/results/fi-tune-host-1004362/`，日志在 `temp/h800/logs/` |
| **1004363** `fi-tune-peer` | h01，2 张 H800，30 分钟 | NVLink 调优：staged（16/32/64 个 fetch CTA，去重）对比原地读。脚本 `temp/h800/tune_peer.sh` | PENDING，估计 11/02 | 同上，目录 `fi-tune-peer-1004363/` |
| **1040680** `fi-bulkbw` | h01，1 张 H800，10 分钟 | **关键实验**：Hopper 的 `cp.async.bulk`（1-D TMA）读 Host 随机行，能否快于 SM cp.async（约 33 GB/s），接近 DMA 的 55 GB/s。同时校验拷贝内容是否正确（`bulk copy check: 0 bad vectors`）。源码 `temp/fetch_bw/bulk_bw.cu` | PENDING，估计 11/02 | 同上，目录 `fi-bulkbw-1040680/host.txt` |
| **1040732** `glm-corpus` | a01，只用 CPU，1 小时 | 用新数据集构建 4 × 64K 的语料 `corpus_swe.pt` | **COMPLETED，0:0，12 分 25 秒** | 集群 `rtx/yjw/test/glm_trace/corpus_swe.pt`，日志 `rtx/yjw/test/glm_trace/logs/glm-corpus-1040732.out` |
| **1040816** `glm-swe-check` | a01，2 张 A800，90 分钟 | 新语料 loss + top-k 验证：原语料第 2 篇，8K，`--groups 2 --span 256 --loss-all --check` | **COMPLETED，0:0，19 分 15 秒** | 集群 `rtx/yjw/test/glm_trace/traces/swe_check_20261028/`，日志 `logs/glm-swe-check-1040816.out` |
| **1040870** `glm-swe-inspect` | a01，只用 CPU，4 核 / 8G / 10 分钟 | `05_inspect_trace.py` 检查保存的全部 top-k/分数，并按目标 token 角色统计完整 loss | **FAILED，1:0：仅一块 system 文本 loss 6.30 超阈值；数据/assistant 检查通过** | 集群 `rtx/yjw/test/glm_trace/results/swe_check_20261028.json`，日志 `logs/glm-swe-inspect-1040870.out` |
| **1040875** `glm-swe-64k1` | a01，8 张 A800，2 小时 | 第 1 篇 64K，16×1024 记录窗口，保留完整 loss；先测单篇耗时 | **PENDING，Priority，暂估 10/31 09:19**；已核查 system 异常并于 11:39 解除依赖，见 §4.2 | 集群 `rtx/yjw/test/glm_trace/traces/swe64k_doc0_20261028/`，日志 `logs/glm-swe-64k1-1040875.out` |

- 1040816 已完成全部 78 层；第 0 层 top-k 重合率 mean/min 均 1.0000，记录分数最大相对误差 5.21e-07，排序/有效数/补位检查均通过；absorbed vs expanded attention 最大相对误差 4.27e-03。最终数据和角色 loss 见 §4.2。H800 新调优顺序仍待用户确认。
- h01 的两个节点 g40、g44，16 张 H800 一直被别人占满，作业只能排队，没有办法加速。重新提交也一样，不要重复提交。
- 上一次语料作业 1040688 立即失败，退出码 127。原因是用 `--wrap` 时节点上的 sh 没有执行 `source`。已改为用 bash 写的 `04_corpus.sbatch` 重新提交。
- 10/28 11:07 再查：三个 H800 作业仍为 Priority 排队，预计 11/02 02:34:29，无结果目录；没有重复提交。
- 静默轮询脚本：`temp/glm_trace/watch_jobs.sh`，每 600 秒查一次，查询失败继续重试。10/28 11:19 已在集群 nohup 启动，PID `4160646`，观察三个 H800 作业和 `1040816`；日志和 PID 文件在 `rtx/yjw/test/glm_trace/logs/watch_20261028.{log,pid}`。继续工作前先用 `ps -p 4160646 -o pid,etime,args` 核查进程，避免重复启动。
- 新 trace 链的静默轮询 PID `65160`，观察 `1040870,1040875`，日志/PID 在同目录 `watch_trace_20261028.{log,pid}`；10/28 11:42 已用 `ps` 核实存活。
- 两地 `mla_host_batch.cuh` 的哈希不同：已逐行比对，差异均为本机的 `FAR_EMU_*` 模拟钩子，集群版本不含这些钩子。后续实卡优化仍以集群代码为准，不能直接用本机文件覆盖。

## 4. 未完成的任务和下一步

### 4.1 H800 kernel 性能：最优先，这是用户的核心目的

**10/28 后续用户授权持续调优，目标是逐步逼近理论下限。无需对同范围的常规优化反复确认；保留单 kernel 融合，涉及拆 kernel 的架构取舍仍需询问。** 已做第二轮：staged LRU 映射绑定放到 fetch、空 miss 池快速返回、LRU 双计数前缀合并、小 batch HBM 分块预留 Host 并行空间。相对 d92e7e83，同参数反向顺序重测：100GB/s无附加延迟 MTP8 1/2请求快10%–11%，MTP1四请求快9%；GLM8K四请求快约12%。完整结果和局限在 `docs/mla_host_a100_lru_tuning.md`。保持实测下限和所有更慢的格子，不把不同参数的收益混作单个代码改动；H800未验证。

**10/28 用户后续明确认可在本机 A100 优化 100 GB/s＋高延迟，以及单独优化小 batch，已执行，不再等待这两项确认。** 保留单 kernel 融合 LRU。新结果见 `docs/mla_host_a100_100gb_tuning.md`：kernel 调整 fetch CTA>8 时的搬运块切换阈值；100 GB/s＋20 µs，MTP8 单请求 f16 同参数 125.95→101.89 µs（−19.1%），128 请求结合 f24 调参 5609.98→4444.16 µs（−20.8%）；MTP1 四请求结合 f32 为 101.18→79.36 µs（−21.6%）。这是相对旧版固定 f16/h64 基线，不能称为相对旧版所有最优配置。75 项完整测试通过，最终 7-pass 对照每步采样行 max abs 0.006007。模拟器未修改。更深流水、单 warp 扫描、缩 grid 的候选未采用，补丁与数据保留。H800 原排队作业源码未改，本轮 H800 性能尚未验证；涉及拆 kernel 的架构决策仍需用户确认。

**现状**：H800 默认配置下，大部分场景输给 Memcpy+LRU；PCIe MTP1 ≥ 64 请求和 NVLink MTP8 128 请求输给 HiSparse。数据在 `docs/mla_host_h800_lru.md` §2。

**10/28 对已有数据做的时间分解**（各 mode 取后一半 step 的中位数，单位 µs）：

| 场景 | 本文 | 每步未命中量 | 按 SM gather 实测速率算的传输时间 | FA2 下限 → FlashMLA 下限 | HiSparse | Memcpy+LRU |
|---|---:|---:|---:|---:|---:|---:|
| PCIe MTP8，128 请求 | 11849 | 311K 页（358 MB） | 10690（33.5 GB/s） | 1209 → 573 | 14553 | 8495 |
| PCIe MTP1，512 请求 | 7104 | 181K 页 | 6726（31 GB/s） | 626 → 446 | 6395 | 4810 |
| NVLink MTP8，128 请求（原地读） | 4582 | 312K 页 | 2148（167 GB/s） | 1202 → 568 | 4367 | 3450 |
| NVLink MTP1，256 请求 | 794 | 89K 页 | 612 | 321 → 232 | 908 | 938 |

**结论：主要问题在传输，不在 attention。**
- **PCIe**：本文受 SM 读 Host 的上限（约 33 GB/s）限制。Memcpy+LRU 用 DMA，速率 55 GB/s，而且**它的 CPU gather 不计时**，这个基线偏宽松。即使换成 FlashMLA，PCIe 大 batch 也最多省约 5%。
- **NVLink**：原地读时每个未命中页平均被读约 2.1 次，link 接近饱和。改成 staged 去重，理论上能省 2 ms 以上，远大于 FlashMLA 能省的 0.6 ms。

用户先批准了「方案 A：attention 改接 FlashMLA sparse decode + TMA 带宽测试 + 4 层共用 prepare」。看到上面的分解后，我向用户提出了新顺序，用户随后要求交接，**还没有明确回复同意新顺序**。新顺序如下，下一个 agent 应先和用户确认：

1. **PCIe 传输路径**：看 1040680 的结果。
   - 如果 bulk copy 接近 50 GB/s，就把 fetch CTA 的搬运改用 `cp.async.bulk` 加 mbarrier。位置在 `fetch_host_batch_pages`，约 623 行；现在是 cp.async 16 B 加分块的 ready flag。这样仍在单 kernel 内。
   - 如果 bulk copy 也只有约 33 GB/s，说明这台 H800 上只有拷贝引擎能达到 DMA 速率。需要让用户选：(a) 接受这个上限，改变对比的表述；(b) 把搬运交给 CPU gather + DMA，这就不再是单 kernel。**不要替用户做这个决定。**
2. **NVLink 改用 staged 去重**：
   - 先在本机 A100 上用链路模拟器把带宽限到约 167 GB/s（H800 NVLink 的 SM gather 实测值）。方法见 `docs/mla_host_link_emulation.md`，用 `--emu` 参数。
   - 比较 staged 16/32/64 个 fetch CTA 和原地读，提前定好配置。1004363 会在 H800 上确认。
3. **让 Memcpy+LRU 基线公平**：给 CPU gather 计时。在 `temp/h800/bench_rounds.py` 的 `load()` 里，`torch.index_select(far_cpu, ...)` 目前不计时。报告里同时给出计时和不计时两种结果，并如实说明。
4. **FlashMLA attention**：仍值得做，主要帮 NVLink 去重之后的场景和小 batch，但属于第二位。
   - 方案 A 的定义：去重、搬运、LRU 保留在本文 kernel，attention 改用 FlashMLA 的 `flash_mla_with_kvcache(indices=...)` sparse decode。
   - 集群上的 sgl_kernel：`rtx/.venv/lib/python3.12/site-packages/sgl_kernel/flash_mla.py`。接口有 `get_mla_metadata(cache_seqlens, num_q_tokens_per_head_k, num_heads_k, num_heads_q, is_fp8_kvcache, topk)`、`flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v, tile_scheduler_metadata, num_splits, softmax_scale, indices=...)`、`flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v)`。
   - 要点：本文 kernel 的 `remap` 编码规则见 `resolve_host_batch_row`，约 786 行。非负值是原 KV 页；负值 -1-row 是 staging 或缓存槽。要接 FlashMLA，需要输出一个统一的 KV 视图加 indices，比如把 staging 和缓存放到和 KV 同一 VA 段之后的连续区域。
   - 这样「单 kernel 全融合」的说法会弱化，报告里要如实写。
5. **4 层共用一次 prepare 和 LRU**（用户要求在合适的时候提醒他）。共享同一份 top-k 的层可以复用去重和 LRU 的结果，小 batch 每层能省约 20～30 µs。比如 PCIe MTP8 1 请求，本文 134 µs，而 FA2 下限只有 34 µs。
6. 选好参数后在 H800 上重测 HiSparse 对比和全量 batch，并更新 `docs/mla_host_h800_lru.md` 和报告 §6。

### 4.2 GLM-5.3 trace 采集

**模型**：`/WORK/PUBLIC/luyouy_work/rtx/models/zai-org--GLM-5.3`，FP8，141 个分片，已校验。只有 21 个层跑 indexer：第 0、1、2 层，以及之后每隔 4 层一个（6、10、…、74）。其余层复用上一层的 top-k。

**采集脚本**：`temp/glm_trace/02_prefill_trace.py`，集群副本在 `rtx/yjw/test/glm_trace/`，两边一致。在 A800 上用 BF16 逐层 prefill。SGLang 的 DSA 只支持 SM90，A800 跑不了。

脚本在这个 session 里的改动：
- **修复了一个致命 bug**：原来的记录代码缩进在 `--check` 分支下，正式采集会只记录第 0 层，最后在 `torch.stack([])` 处崩溃。已修复，commit `b9f0f02d`。
- **输出的 `doc<i>.pt` 包含**：
  - `selections [indexer 层, groups, span, 2048]` int32：文档内位置，按分数降序，-1 补位；
  - `scores`：形状同上，fp32 index score，补位处为 -inf；
  - `row_stats [.., 5]`：n_valid、sum、sumsq、max、topk_sum；
  - 可选 `full_rows`（`--full-rows`，每组最后一个位置的整行分数）；
  - 元数据：`positions`、`group_starts`、`layers`、`source_layer`（78 层各自用哪个 indexer 层）、`members`、`meta`、`role`（每个位置的角色：system/user/assistant/tool）、`loss`、`config`。
- **诊断参数**：
  - `--check`：分块与不分块对比，分数对比稠密参考，absorbed 与展开形式对比；
  - `--dense`：全因果 attention；
  - `--loss-all`：每个位置的 loss 和 top-5 预测，写到 `<out>/loss.pt`；
  - `--ref-layers`：逐层与 transformers 5.12.1 的 `GlmMoeDsaDecoderLayer` 对比，需要单卡加 `--dense`；
  - `--pos-offset`、`--indexer-interleave`。
- **未提交**：`role`、`meta` 两个输出字段（工作区改动，见 §6）。

**正确性诊断的结论：采集实现是正确的，不要再查实现。**

| 检查 | 结果 |
|---|---|
| 分块 vs 不分块 top-k | 重合率 1.0000 |
| 记录的分数 vs 稠密计算 | 最大相对误差 4.76e-07 |
| 78 层逐层对比 transformers | 整体相对误差 1～4%，bf16 加 MoE 路由翻转的正常量级 |
| 稀疏 vs 稠密 attention | loss 曲线几乎重合 |
| 同一段重复两遍 | 第二遍 loss 0.05～0.09 |
| 位置偏移 0 / 1792 / 16384 | loss 不变 |

- indexer RoPE 用交错排布（config `indexer_rope_interleave: True`），和 SGLang 一致，见 `deepseek_v2.py:1828`。transformers 写死为非交错，不必理会。
- 唯一的问题在**旧语料格式**。旧语料 `corpus_64k.pt` 由 `01_build_corpus.py` 生成：
  - 工具 JSON 被直接塞进 system，没有用 GLM 原生的 `tools=` 模板；
  - 工具调用写成 `<tool_call>{json}`，不是原生格式；
  - 多个会话拼接在一篇文档里。
- 这导致某些段落 loss 6～10。同一段文本单独输入时 loss 约 2。**旧语料作废**，原始的 claudecode-trace 也不在这台机器和集群上。

**新数据集**：`nebius/SWE-rebench-openhands-trajectories`，CC-BY-4.0，从 hf-mirror 下载到集群 `rtx/data/swe-rebench-oh/`。
- `trajectories.parquet` 共 2079503354 字节，sha256 `14048dd1…efa1`，与上游 x-linked-etag 一致。另有 `tools.json` 和 README。
- 内容是 OpenHands 修 GitHub issue 的多轮 agent 轨迹，平均 64 轮，使用 OpenAI 格式的 tool_calls。
- 新的语料构建脚本 `temp/glm_trace/01b_build_corpus_swe.py`（**未提交**）：
  - 每篇文档只用一条轨迹，不拼接；只取 resolved 的轨迹，取最长的几条截断到 64K；
  - 用 GLM-5.3 原生模板渲染（`apply_chat_template(msgs, tools=tools)`）；
  - 记录每个位置的角色。
  - 已在合成对话上验证：格式正确，前缀一致，只有一个 `[gMASK]`。
- 作业脚本 `temp/glm_trace/04_corpus.sbatch`（**未提交**）。

**后续步骤**：
1. **语料构建和格式检查已完成（10/28）**：4 篇原长 122916 / 121172 / 119659 / 118550，均截到 65536；单条轨迹/文档，角色字段齐全。文档 0 仅有一个 `[gMASK]`，50 个原生 `<tool_call>`、49 个 `<tool_response>`；已检查 user / assistant / tool 的渲染片段。CPU 检查记录见 `temp/glm_trace/results/corpus_swe_inspection_20261028.json`（集群原件为 `corpus_swe_inspection.json`）。
2. **验证作业 1040816 已完成**：`NPROC=2 sbatch --job-name=glm-swe-check --time=01:30:00 --gres=gpu:a800:2 03_trace.sbatch --corpus corpus_swe_check.pt --docs 1 --tokens 8192 --groups 2 --span 256 --loss-all --check --out traces/swe_check_20261028`。
   - `corpus_swe_check.pt` 保留原语料第 2 篇（索引 1）及其角色、元数据，用于本次短验证，不替换正式的 4 篇语料。
   - **不要使用原建议的 4K 前缀**：4 篇的前 5675 token 全为 system，4K 验证没有 assistant；所选 8K 前缀含 793 个 assistant token（第 1 篇只有 121 个）。
   - 判据：按 `role` 分开统计 loss，assistant token 应 ≤ 约 1.5；除第一块外，没有 ≥ 约 6 的 256 位置块；tool 输出可以在 3～5。
   - 注意：现在的 `loss` 字段只统计记录窗口，请看 `--loss-all` 的全位置统计。
   - `loss.pt` 中 `loss[r]` 预测 `tokens[r+1]`，按角色统计时要用目标 token 的 `role[r+1]`，不能直接用 `role[r]`。
   - CPU 检查脚本 `temp/glm_trace/05_inspect_trace.py` 和 `05_inspect.sbatch`：输出 JSON，只有数据完整性、assistant mean ≤ 1.5、首块之后所有 256-token 块 mean < 6 均通过才返回 0。
   - **结果**：全部 21 层的 `[21,2,256,2048]` selections/scores、78 层复用映射、每行因果范围/无重复/分数排序/有效数/统计一致性通过。按目标角色的 mean loss：system 4.224，user 2.843，**assistant 0.827（793 token）**，tool 4.367。唯一超阈值块 `[1536,1792)` mean 6.3025，全是 system 工具描述中的转义 JSON 文本。严格 gate 返回失败，未把它写成完全达标。
   - **复核决定（11:39）**：assistant/tool 达到预期，唯一高块已定位为原始 system 工具描述，格式和采集实现无异常；按已授权的 trace 流程继续单篇 64K，保留严格失败与复核原因，不改语料或阈值。已用 `scontrol update JobId=1040875 Dependency=""` 解锁原作业，未重提交。证据：`temp/glm_trace/results/swe_check_20261028.json`、`swe_check_with_roles_20261028.json`、`swe_check_decision_20261028.json`、`glm-swe-check-1040816.out`（集群原件同名/对应日志）。
3. **正式采集**：先跑 1 篇 64K 测时间。indexer 部分与长度平方成正比；4K、2 卡约 17 分钟，64K 要外推，可能需要 8 卡和较长时限。然后跑全部 4 篇。
   - 单篇计时作业 1040875 已按原授权流程排队；最初依赖 CPU gate 1040870，现已按上面的复核解除依赖。`NPROC=8`、`--docs 1 --tokens 65536 --groups 16 --span 1024 --loss-all`，2 小时时限。源码哈希、原依赖和当前状态在 `temp/glm_trace/results/swe_pipeline_20261028.json`。
   - 记录窗口：现在是在文档后半段均匀取 `--groups 16` 组、每组 `--span 1024` 个连续位置。
   - 可以考虑只取 assistant 段作为 decode 位置，因为真实 decode 只生成 assistant token；或者保留全部位置，回放时用 `role` 过滤。
   - **窗口覆盖准备已完成**：原均匀 16×1024 窗口没有一个全为 assistant；已生成 `temp/glm_trace/results/swe_windows_64k.json`。文档 1 将最近的 3 个窗口起点移至 35184 / 41887 / 48296，文档 3 移至 50621，共覆盖 4 段完整的 1024 连续 assistant token；其余窗口保持，文档 0/2 不变。没有拼接或跳过中间 token。
   - `02_prefill_trace.py` 新增可选 `--window-plan`，复用 CPU 模块 `trace_windows.py`，核对语料 SHA-256、文档 members、长度、窗口顺序/范围/不重叠；输出新增 `source_sha256` / `window_plan_sha256`。已同步集群，默认窗口经 CPU 检查与原 64K/8K/4K 版本完全一致。1040875 未传参数，仍为原默认；后续全量待单篇核查后加 `--window-plan results/swe_windows_64k.json`。新可选路径尚待 GPU 全流程验证。
4. 把 trace 转成 `temp/h800/tiers.py` 的格式（`selections [layers, groups, span, topk]`、`group_starts`），在本机 A100 上做多步回放（span 1024）。
   - **8K/256 位置接入检查已完成**：实际无需转换，`tiers.py` 可直接读取 GLM 的保存格式。A100 PCIe MTP1（1/4 请求，256 步）和 NVLink MTP8（1/4/16 请求，32 步）每步采样行校验通过，max abs ≤0.006066；完整逐步耗时/未命中/去重页命中率/两下限及比值在 `temp/glm_trace/results/a100_8k_smoke.md`，复现 `bash temp/glm_trace/run_8k_smoke.sh`。这是短上下文接入检查，包含 system/user/tool，不能替代 64K/1024 正式结果。
   - 8K MTP8 的 16 请求原地读测得 496.2 µs；原生纯本地初测最好样本 496.6 µs，出现比值略低于 1，已另测 0% 远端的 attention（原生 + 本文无缓存）CUDA graph，最好为 435.5 µs；报告取该最好实测值并保留补测原始数据，没有截断比值。
   - 记录每步命中率，以及本文 ÷ 纯本地，还有两个下限（见 §5）。
   - 真实 trace 的未命中量会改变传输和计算的比例，这也会影响 §4.1 的结论。最后在 H800 上也跑一遍。

## 5. 环境和规则

- **集群**：`ssh kaituo1000_login21`，即 luyouy@166.111.143.25，Slurm 登录节点，没有 GPU。
  - 所有 GPU 工作都通过 Slurm。
  - **H800** 在 partition `h01`（g40、g44，各 8 张）：每个作业**最多 4 卡，尽量 1～2 卡，时限要短**。不要提交 8 卡作业。
  - **A800** 在 partition `a01`（27 个节点）：用于 trace 采集，**最多 8 卡**。
  - 所有文件放在 `/WORK/PUBLIC/luyouy_work`（即 `~/WORK`）下；`/home/fit` 已超配额。
  - 远端约定见 `rtx/AGENTS.md`，项目说明在 `rtx/MEMORY.md`。
  - Python 环境：`source rtx/env.sh`，用 `rtx/.venv`（transformers 5.12.1、torch cu130）。
  - 网络：hf-mirror 能访问，huggingface.co 不能。
  - ssh 后台命令要加 `-n`。用 pkill 时注意别匹配到自己的命令，上次就因此误杀并导致重复下载。
- **本机**：路径 `/home/rtx/projects/flashinfer`（= `/mnt/data/rtx/projects/flashinfer`），10/28 接续检查时实际分支为 `codex/mla-host-overlap`。
  - GPU：GPU0、GPU1 是 A100-SXM4-40GB，两者之间是 NV12；GPU3 是 A100-PCIE。
  - GPU0 作近端，GPU1 经 NVLink 作远端；Host 内存在 NUMA0。
  - 常用环境：`CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_a100_final FLASHINFER_CUDA_ARCH_LIST=8.0 CUDA_VISIBLE_DEVICES=0,1 numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python ...`。
  - 链路模拟构建另加 `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_FAR_EMU`，并用 `FLASHINFER_WORKSPACE_BASE=temp/jit_emu`。
  - 本机 nvcc：`/usr/local/cuda-13.1/bin/nvcc`，可以编 sm_90a，但不能运行。
- **同步代码**：本机改动后用 scp 同步到集群对应路径。先传到 `.new` 再 mv，避免正在运行的作业读到写了一半的文件。kernel 代码以集群上 `rtx/yjw/flashinfer` 的 git 为准（作业会记录 commit 和 dirty）。

**用户偏好**：
- 结果表要同时给出两个下限：(a) max(纯本地计算, 去重后未命中字节 ÷ 链路带宽)；(b) 纯本地计算。下限用实测最好值，保证比值 ≥ 1。
- 数据保存到文件并提交。
- 报告要自包含，突出本文 kernel 的优势，但**不能隐瞒输的场景**。
- 与 HiSparse 对比时，按它实际的方式运行（每请求 swap-in 加 FlashMLA，没有跨行去重）。不要说它被低估，也不要替它优化。
- **讨论方案时先给推荐方案，用户确认后再执行**；涉及表述或架构取舍的决定（比如放弃单 kernel）必须问用户。
- 节省 token：长时间等待时用**静默的后台轮询脚本**（`while squeue ...; do sleep 600; done; cat 日志`，用 run_in_background），不要用频繁的 cron。
- 模拟和实验代码要简洁，不影响正常构建，后期会删除。
- 用中文和用户交流。

## 6. 工作区状态（提交时只 `git add` 自己改的文件）

- **非本 agent 的改动，不要回滚、不要提交**：`docs/mla_host_lru_report.md`、`temp/report_lru/plot.py` 及两张 png、`include/flashinfer/attention/decode.cuh`（一个行尾空格）；未跟踪文件有 `docs/mla_host_lru_report_external.md`、各种 pdf、`package*.json`、`temp/hb_regs/`、`temp/*.nsys-rep` 等。
- **本 agent 最后一批改动已提交**（commit 见 git log）：
  - `temp/glm_trace/02_prefill_trace.py`（新增 `role`、`meta` 输出）；
  - `temp/glm_trace/01b_build_corpus_swe.py`；
  - `temp/glm_trace/04_corpus.sbatch`；
  - 本文档 `docs/HANDOFF.md`。
- `temp/glm_trace/ref/` 下的参考文件（transformers 建模代码、config）是诊断用的，未跟踪，可以不提交。
- 本 session 的 commit：`b9f0f02d`（index score 和 bug 修复）、`c9d07220`、`461d14ba`、`a78d1d9c`（诊断参数）、`250304dc`（bulk 带宽微基准）。
