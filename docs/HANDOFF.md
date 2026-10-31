# 交接：远端 KV sparse MLA kernel 持续优化

更新于 **2026-10-31 19:50保留检查点，Asia/Shanghai**。本文是当前入口；旧交接完整保存在 [历史记录](HANDOFF_history_before_20261031.md)。冲突时优先读当前用户要求、本文和实际进程/数据。

## 1. 任务目标与授权

持续优化 FlashInfer 的远端 KV sparse MLA decode **单 cooperative kernel**，融合跨query去重、跨步LRU、远端读取、attention及结果合并。在指定带宽/延迟下显著优于正确的先搬后算Memcpy，尽量接近逐步`max(纯device计算, 必需远端搬运)`的完美重叠参考；低远端流量时尽量接近纯device。

用户最新要求：**“还能继续优化吗？让我们的kernel在各种场景下都表现优秀。”** 已授权同范围的实现、实验、修复和回归，不需反复确认。此前完成交接后，用户已要求继续 active goal；本轮已经恢复独立候选实现与验证。

覆盖小/中/大batch、MTP1/8、有/无LRU、50/100 GB/s、附加延迟0/2/5/10/20 µs、物理PCIe/NVLink。允许按形状和路径选择不同调度，但必须区分代码、调参和路由收益，避免某格变快而其他场景回退。保持单kernel、数学计算和缓存载荷/状态正确。拆kernel或明显改变目标的架构决策需用户确认。

**不优化HiSparse**；如需对比，按原生部署接口运行，不替它接入多步去重。当前主任务是我们kernel对Memcpy和纯本地计算的比较，不模拟整个模型其他部分。

## 2. 当前运行状态与新一轮进度

### 2.1 最新续接：中等 batch 派发修正已保留，继续大 batch 主目标

生产头现为 `balanced_cached_sm80.cuh` 的 **52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5**。原81891229…225b冻结在本轮 `base.cuh`，上轮整行成果提交 `5157ff83` 和检查点 `0ff2fa66` 保持不变。本轮保留记录见 [独立报告](mla_host_a100_balanced_dispatch_20261031.md)及 `balanced_retention.json`，旧 `promote_medium.py` 继续禁用。

最终完成 SXM 120 格受限链路、4 格正常构建真实NVLink/PCIe，以及 GPU3 80 格正常构建独立HBM/PCIe。共102配对，改变路径的82个快1.29%–28.78%；224行恢复原分片的20配对最大负差0.31%，均在1%原路径测量门槛内。未隐藏负差。SXM正常构建MTP8/32请求、完整1024位置×5pass：NVLink1047.27→901.58µs（13.91%），PCIe2887.49→2790.52µs（3.36%）。

`retain_balanced.py --install`已实际执行并通过完整源码、覆盖、测试、编译等价、原矩阵SHA门槛；它冻结当时两份审计和其他证据，随后逐字节安装已测试候选。生产未新增GPU数学。最终候选实际通过105项正常构建测试，包含2/4×SM Graph/padding/映射/实际cache载荷；模拟9个BF16和正常18个BF16/FP16 kernel指令文本及资源均与base相同。

规则只读rows/max_kv_len/SM-grid/fetch_ctas/partial_capacity五个标量，用原tail planner、固定tile读写成本和现有前后两端循环估计波次。≥4×SM沿用原整行，2–4×SM且CTA_TILE_KV=32才比较；108SM/fetch16/K2048下216行整行，224–248行原分片，256/320/384行整行。单项thread-local有界记忆覆盖全部键，360检查通过；CPU微测cold47.7µs/warm0.002µs，不是端到端Python CPU时延。不用链路/命中/计时逐格挑选。

统一threshold2/threshold2_sm80仍已拒绝：224行正常构建、无LRU低远端比例慢3.2%–4.4%。它的110格LRU、330格无LRU、18控制、6确认及80正常构建均完整保留，不能冒称最终修正版重跑了全部原范围。`defer_cache`四链路慢29%–35%，warp_ready/row_ready也已拒绝，不重复无新机制筛选。

本机前一轮runner已全部结束：SXM91049 exit0（120格），SXM41792 exit0（4格正常），GPU3原43216末检因外部PID435854进入而exit1，补齐缺项的31760 exit0。失败未完成raw保存在 `gpu3/rejected/`，追加日志和state未覆盖；重启只跳过完成格。**没有活动GPU runner需要继续poll或重复启动。**

GPU3是A100-PCIE/NUMA1，同卡独立HBM不是NVLink，物理PCIe和Memcpy pinned源都绑NUMA1；参考在本GPU正常构建重测，PCIe约27.04GB/s、HBM约689GB/s。SXM0/1是NUMA0实际NV12，绝对时间不拼接。其他用户任务随时可能进入，继续资源预检/末检、共享锁，不终止其他人进程。

**大batch 100GB/s约32%–34%差距未改善，整体goal仍active。** 下一步读真实fetch/attention/LRU代码，在独立include/JIT实现新机制，优先研究fetch的cache scatter占用和attention计算尾部；从attention共享KV tile写回缓存是尚未实现的可选方向，不能视为收益。单kernel和精确LRU/载荷要求不变。

### 2.2 已完成阶段诊断（不重复启动）

目录：`temp/glm_trace/tune_general_20261031/`。`base.cuh`为上轮已验证生产kernel，SHA256：

```text
81891229df298b90219410b5a4df1e6c20dc4cb1ee9c0233bd17ad383b29225b
```

已完成3格插桩诊断：MTP8、128请求、完整1024位置、staged、LRU4096槽/请求、fetch24、split off、每格1次冷启动pass；128步/格。输出采样max abs≤0.00195504，前后无其他GPU0/1进程。**不是新优化收益或正式多pass结果；未改生产kernel。**

| 链路 | 插桩kernel均值 µs | fetch开始 µs | 最晚fetch完成 µs | 最晚attention完成 µs | 最晚LRU项完成 µs |
|---|---:|---:|---:|---:|---:|
| 100 GB/s＋20 µs | 3492.11 | 190.81 | 3064.77 | 3479.67 | 242.93 |
| 50 GB/s＋20 µs | 5549.53 | 183.78 | 5409.24 | 5537.79 | 233.67 |
| 物理NVLink | 3361.10 | 196.28 | 2393.22 | 3348.56 | 249.83 |

相对每步最早kernel进入时间；表中完成时间取每步各CTA最大值后平均。fetch/LRU/attention重叠，不能相加当开销分摊。整行路径不再做尾部merge。准备/解析到fetch开始约0.19 ms；100GB下fetch后约0.41 ms的attention尾部，NVLink约0.96 ms。提示查计算效率、fetch占用SM和工作分配，不能断言差距全来自LRU。

原件与入口：

- `profile.log`：三格均`complete`；原exec session20309已结束，不需再等。
- `raw/profile_staged_f24_i64_soff_lru_{g100_l20000,g50_l20000,nvlink}_mtp8_b128.json`：每步事件、miss/命中、source SHA、flat时间戳。
- `phase_summary_handoff.json`：交接时CPU解析128步×108CTA后的逐步/平均阶段数据。
- `handoff_state_20261031.json`：源码/原件SHA、进程与队列快照。
- `profile.cuh`与独立`include_profile/`、`jit_profile/`：诊断不覆盖生产。
- `bench.py`从上轮完整trace脚本复制，新增`--positions`、`--route staged/inplace`、`--fractions`；已运行上述完整staged诊断及256位置的LRU筛选；inplace/fractions等新增组合尚未全面验证。

旧主汇报所说“最终路径阶段计时未取得”已被本次诊断补上；主汇报§7.6和本轮Markdown已同步。`audit_general.py`独立重解raw的128步×108CTA；审计现有数据不等于全目标完成。

### 2.3 脚本注意事项

`bench_glm52_host_batch`的传递导入会重设源码路径，独立实验需在它导入**后**再次设置`jit_env.FLASHINFER_INCLUDE_DIR`；新bench已处理。更改头文件要独立JIT，核实执行header SHA，避免旧缓存。

插桩动态扩展Python内部state，按128步、最多128CTA保存时间戳，**不能原样用于MTP1、不同步数或>128CTA**；先检查容量/布局。新脚本`--positions`短窗口只适合筛选，最终必须完整1024位置、多pass冷启动确认。

## 3. 已完成实验与当前基线

### 3.1 原5304格矩阵保持原件

- 数据：`temp/glm_trace/kernel_report_20261029/`；[完整报告](mla_host_glm53_kernel_matrix_20261029.md)。
- 无LRU4641格＋trace/LRU663格，另有312个纯device参考格；审计完成。
- 矩阵SHA：`e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99`。
- 固定batched staged、fetch16/item64/split on，SMALL_PREPARE关闭。全部A100实测，**不是H800**，不要改写或混入新结果。

### 3.2 上轮整行优化已验证并保全提交

提交`5157ff83`；数据`temp/glm_trace/tune_100gb_20261030/`；[优化报告](mla_host_a100_100gb_optimization_20261030.md)；[主汇报](mla_host_glm53_report.md)§7.6已列新结果。主图1–6仍是原固定矩阵，新实现单列。

生产头新增`WholeRowHostBatchParams`：整行直接输出，省尾部partial/merge；编译期排除原地读写回和零远端模拟检查。**真实fetch处的带宽/延迟模拟及staging就绪等待保留，模拟器没改。**

派发条件：`stage=True`、`host_split=False`、64 heads、query行数≥4×SM数、staging≥rows×max_kv_len。A100-SXM4为432行阈值；100GB大形状推荐fetch24/split off，小形状fetch16/on。这不是自动检测链路速度；不满足条件走通用路径。

完整trace、staged＋LRU4096、100GB＋20µs、5pass冷启动全步均值：

| MTP | 请求 | 旧默认 µs | 旧同参数 µs | 新实现 µs | 较旧默认降低 | 高于重叠参考 | Memcpy加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 2885.27 | 2521.29 | 2193.83 | 24.0% | 31.8% | 1.625× |
| 8 | 64 | 2462.30 | 2180.34 | 1833.68 | 25.5% | 34.1% | 1.756× |
| 8 | 128 | 4388.65 | 4160.36 | 3560.53 | 18.9% | 32.3% | 1.770× |

旧默认fetch16/split on，旧同参数/新实现fetch24/split off；代码本身下降13.0%/15.9%/14.4%。50GB＋20µs同三形状距参考7.3%/9.9%/6.7%。**100GB距参考10%以内的目标仍未达成，不要当作全目标完成。**

`summary.json/audit.json complete=true`：577审计格、944064次计时launch、59760有效队列；最终100GB trace195格（39形状×5延迟）、其他链路33格、最终无LRU84格、5pass新实现12格，另有旧控制和候选。103项pytest通过，含2个整行Graph/padding/缓存载荷测试。六个通用BF16 kernel的反汇编指令文本及资源未变，见`compiler_summary.json`和gzip SASS。

未采用候选`fastpipe4`、`tile64`、`queue`、`local_loop`、`fetch128`、`regpipe3`等保留，不要从头重复。邻近fetch20=3650.93µs、28=3524.58µs（3pass）；24=3560.53µs（5pass）/3533.37µs（3pass最终矩阵）。差异小且测量轮次不同，**不能证明24全局最优，也不能说28必然更慢**；旧交接部分表述过强，以raw为准。

### 3.3 路由与小batch已有证据

- [staged/原地读对照](mla_host_glm53_route_comparison.md)，数据`temp/glm_trace/route_compare_20261030/`：同trace/LRU4096，MTP8、16/64/128请求、3pass；所测100/50GB均staged更快，物理NVLink原地读更快。128请求原地读miss引用约509MB/步，staged unique约260MB，重复约1.96；两模式均有LRU。
- 小batch宏`FLASHINFER_HOST_BATCH_SMALL_PREPARE`默认关闭；[准备优化记录](mla_host_a100_prepare_tuning.md)、`temp/sim_link/tune_prepare/`。只在GPU3 A100-PCIE验证一些收益，不直接拼SXM绝对时间；需独立JIT、同源码开/关对照。
- [H800比较审计](mla_host_h800_benchmark_audit.md)：旧HiSparse、Memcpy/Direct计时/载荷校验有口径限制，不能用旧表下完整方案胜负结论。不替HiSparse接新多步接口。

## 4. 本机环境、计时与正确性

- 仓库`/mnt/data/rtx/projects/flashinfer`，同`/home/rtx/projects/flashinfer`；分支`codex/mla-host-overlap`，整行优化提交=`5157ff83`，当前HEAD请用git核实。
- Python`/mnt/data/mhy-vllm-venv/bin/python`；上轮raw为PyTorch2.11.0+cu130。
- CUDA_HOME=`/usr/local/cuda-13.1`，nvcc同目录，架构8.0。
- GPU0/1 A100-SXM4-40GB、真实NV12；0计算、1远端，Host pinned绑NUMA0；GPU3 A100-PCIE，GPU2其他服务。
- 共享`temp/glm_trace/kernel_report_20261029/gpu01.lock`串行自有实验；运行前看`resources.snapshot()`，前后`require_exclusive()`，不抢占其他进程。

未插桩新目录基线复现示例（**本轮尚未运行**）：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
CUDA_HOME=/usr/local/cuda-13.1 FLASHINFER_CUDA_ARCH_LIST=8.0 \
PYTHONPATH=. TMPDIR=/mnt/data/rtx/projects/flashinfer/temp \
FLASHINFER_WORKSPACE_BASE=/mnt/data/rtx/projects/flashinfer/temp/glm_trace/tune_general_20261031/jit_base \
FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_FAR_EMU TUNE_VARIANT=base \
numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python -u \
temp/glm_trace/tune_general_20261031/bench.py \
--kind lru --mtp 8 --requests 128 --settings g100_l20000 \
--passes 3 --fetch-ctas 24 --host-split off
```

新候选复制include到`include_<variant>`，配独立`jit_<variant>`，记录实际header SHA；`compile.py`可在`CUDA_VISIBLE_DEVICES=''`下CPU预编译。正常构建去掉EMU宏；完整测试参考`temp/glm_trace/tune_100gb_20261030/test_final.py`，JIT源码路径需`temp/pytest_shim/jitpaths.py`或等价设置。

测量规则：

1. 固定seed BF16、64 heads、KV512＋64、1152B/行、topk2048/page1。完整trace每请求1024位置，MTP1/8分别1024/128步；3pass冷启动全步算术平均，关键改进同轮/反向5pass确认，不取最好样本代替均值。
2. CUDA event配guard或16步CUDA Graph，CPU发射间隙不计入；输入上传、数学校验、miss诊断放事件外。Memcpy含真实搬运、必要cache scatter和完整attention，CPU规划/packing不计时。
3. 两个参考同时列：纯device，以及`mean_t(max(mean_pass(C_t), mean_pass(unique_miss_t)×1152/(1000×BW)))`；**不是max(mean(C),mean(T))**。公式/probe见`temp/glm_trace/briefing_report_20261030/references.py`。
4. max是乐观重叠参考，不含附加延迟/共享资源竞争；纯device是指定attention实现的参考，不是绝对数学下界。比值偶尔<1需保留解释，不能改时间/裁剪来保证≥1。
5. 明标LRU容量、trace/快照、staged/原地读、形状、远端比例、带宽/延迟、参数和SHA；保留更慢格子、失败和源码快照，不替换原矩阵。
6. 检查缓存映射/实际KV载荷、padding、冷启动、Graph/跨launch状态；生产变动后跑合适回归。当前完整套件`tests/attention/test_mla_host_batch.py`为105项。

Trace由GLM-5.3在A800逐层BF16 prefill、SWE-rebench OpenHands语料原生模板teacher forcing采集。正式回放4个连续assistant窗口×21层=84组、1024位置；>84请求复用capture但远端/LRU地址独立，>128 near物理KV按128请求alias以适配40GB。Q/KV随机，真实的是访问局部性；MTP8非真实草稿/接受率。采集已完成，不重采。1079524=`COMPLETED 0:0`，1079527=`FAILED 1:0`是全角色严格gate失败；assistant窗口检查通过，不能说全语料gate全过。详见[trace记录](mla_host_glm53_trace.md)及历史交接。

## 5. 集群正在排队的任务（本次重新查询）

登录`ssh -n kaituo1000_login21`（luyouy@166.111.143.25），用`bash -lc`。原快照`temp/glm_trace/tune_general_20261031/h800_queue_20261031.txt`，最新再查结果为`h800_queue_20261031_1925.txt`，先读实际结果；此前五任务仍Priority/Dependency。

| JobID | 名称 | GPU | 状态 | 内容/依赖 |
|---:|---|---:|---|---|
| 1004362 | fi-tune-host | H800×1 | PENDING/Priority | 原基线PCIe调参，40分钟 |
| 1004363 | fi-tune-peer | H800×2 | PENDING/Priority | 原基线NVLink staged/原地读，30分钟 |
| 1040680 | fi-bulkbw | H800×1 | PENDING/Priority | cp.async.bulk随机Host搬运带宽/正确性，10分钟 |
| 1089379 | fi-tuned-assist-host | H800×1 | PENDING/Dependency | afterany:1004362:1004363:1040680；旧第三轮快照pytest＋assistant PCIe，20分钟 |
| 1089380 | fi-tuned-assist-peer | H800×2 | PENDING/Dependency | afterok:1089379；同快照assistant NVLink，20分钟 |

前三个调度估计11/02 02:34:29开始，**不保证**。没有新H800/A800任务在运行，无本轮H800性能结果，不重复提交、不改这些任务的源码。

- 根`/WORK/PUBLIC/luyouy_work/rtx/`；远端操作先读`AGENTS.md`、`MEMORY.md`。所有GPU任务经Slurm；输出/cache/temp都在`/WORK`树，不写`/tmp`或超配额`/home/fit`。先`source rtx/env.sh`，环境`rtx/.venv`。
- H800单作业最多4卡，尽量1～2卡短任务；A800采集最多8卡。分区权限看远端规则/实时状态，不绕到无权限分区。
- 原生产`rtx/yjw/flashinfer`；旧第三轮独立树`rtx/yjw/flashinfer_tuned_20261029`、快照`2b782ed3`。**不含本机最新整行优化和后续全部修复**。旧workspace改变涉及csrc/Python，不能只移植header。
- 日志`rtx/yjw/flashinfer/temp/h800/logs/<jobname>-<jobid>.out`；原结果`temp/h800/results/fi-tune-{host,peer}-<jobid>/`、`fi-bulkbw-1040680/host.txt`。assistant日志位置也在原生产树logs，工作目录却是独立树，以scontrol为准。
- 查询watcher仍活：PID4160646（原三任务＋旧检查）、1174145（1089379/1089380），都是`watch_jobs.sh`，只查询、不占GPU。不要再启同一watcher。
- 原矩阵归档`rtx/yjw/test/glm_trace/kernel_matrix_final_20261030/`，见`ARCHIVE_MANIFEST.json`；本机新优化未同步为排队代码。
- 远端没有`rg`，用Python/grep；不要因工具差异误判队列。

## 6. 工作区与提交：必须保留未提交成果

**上轮生产header、测试、完整实验目录及优化报告已提交`5157ff83`（665个明确自有文件）。** 当前仍继续共享dirty工作区，不丢弃未提交内容。

本轮自己的新增/修改包括`docs/HANDOFF.md`、`docs/mla_host_glm53_report.md`、`docs/mla_host_a100_readiness_20261031.md`和`temp/glm_trace/tune_general_20261031/`；交接历史与goal prompt也属本任务。第三项的include/JIT输入已冻结，不覆盖其源码。`restore_include.py`可重建缺失的独立include，但拒绝覆盖不同的已有目录。

其他用户改动，不回滚、不提交：`docs/mla_host_lru_report.md`、`include/flashinfer/attention/decode.cuh`行尾空白、`temp/report_lru/plot.py`与两张PNG、各PDF、`package*.json`、大量其他untracked。只add明确自己的路径，不`git add .`。

如新agent换机器/新checkout，需复制dirty文件和实验目录，不能只传分支。`include_*/jit_*`为可重建且ignore的实验缓存；`.cuh`候选、source_snapshots、raw JSON必须保留。用户要Markdown自己导PDF，不新增PDF/HTML，图用本地相对链接。

## 7. 建议接续顺序

1. 复查源码SHA、dirty状态、GPU进程、Slurm；读已完成阶段诊断，不重复启动。上轮整行成果已提交并重新审计，不重做其整轮实验；先检查§2.1现有第三项runner。
2. 以新目录`base.cuh`和上轮正式确认作基线。重点定位100GB下fetch/attention争用和尾部；物理NVLink可比较原地读，小batch另查prepare/同步。
3. 独立include/JIT实现少量有依据候选，短窗口筛选后完整trace、同参数新旧/反向多pass确认；不要重跑原矩阵、重复已失败候选或重采trace。
4. 保留候选需覆盖大小batch、MTP1/8、50/100GB及高低延迟、有/无LRU、PCIe/NVLink；检查默认未走分支的代码生成/性能，防编译器引入回退。不要按测试结果暗中挑最优。
5. 正确性/缓存载荷/Graph回归通过后，保存每步/每pass、命中率、两参考、源码SHA、guard/失败，更新Markdown和审计，提交本任务自己的成果。新诊断和正式收益分开表述。
6. H800任务完成后先读硬件结果、修正比较口径，再隔离迁移当前优化；不覆盖原排队代码、不重复提交。A100模拟与H800实测分开。

目标仍是持续逼近下限；完成一轮参数扫描不等于全任务完成。已有30%差距不能写成“已接近极限”，也不要为了汇报隐藏慢格。回退需修复或拒绝候选并记录依据。

下一agent的goal文字见 [NEXT_AGENT_GOAL.md](NEXT_AGENT_GOAL.md)。
