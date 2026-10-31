# 交接：远端 KV sparse MLA kernel 持续优化

更新于 **2026-11-01 01:09 CTA协作取数与单stage控制检查点，Asia/Shanghai**。本文是当前入口；旧交接完整保存在 [历史记录](HANDOFF_history_before_20261031.md)。冲突时优先读当前用户要求、本文和实际进程/数据。

## 1. 任务目标与授权

持续优化 FlashInfer 的远端 KV sparse MLA decode **单 cooperative kernel**，融合跨query去重、跨步LRU、远端读取、attention及结果合并。在指定带宽/延迟下显著优于正确的先搬后算Memcpy，尽量接近逐步`max(纯device计算, 必需远端搬运)`的完美重叠参考；低远端流量时尽量接近纯device。

用户最新要求：**“还能继续优化吗？让我们的kernel在各种场景下都表现优秀。”** 已授权同范围的实现、实验、修复和回归，不需反复确认。此前完成交接后，用户已要求继续 active goal；本轮已经恢复独立候选实现与验证。

覆盖小/中/大batch、MTP1/8、有/无LRU、50/100 GB/s、附加延迟0/2/5/10/20 µs、物理PCIe/NVLink。允许按形状和路径选择不同调度，但必须区分代码、调参和路由收益，避免某格变快而其他场景回退。保持单kernel、数学计算和缓存载荷/状态正确。拆kernel或明显改变目标的架构决策需用户确认。

**不优化HiSparse**；如需对比，按原生部署接口运行，不替它接入多步去重。当前主任务是我们kernel对Memcpy和纯本地计算的比较，不模拟整个模型其他部分。

## 2. 当前运行状态与新一轮进度

### 2.0 最新生产：共享页号已保留；本轮三项新候选未采用

**当前生产header是cached_ids，SHA=`b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a`，不再是52919。下文§2.1–2.6保留各阶段历史，冲突时以本节及实际文件为准。** 单kernel、数学、精确LRU/真实cache载荷、模拟器和原5304矩阵不变；只在已有whole CTA32且启用LRU的路径共享块页号，未改路径原9EMU/18normal kernel指令与资源完全相同。基线52919保存在W/base.cuh及其他独立实验树。

**最新C=`temp/glm_trace/tune_cofetch_20261101/`已完成并拒绝。** 原型af610…bf6f将attention减为1stage，腾出2×16行service buffer，所有CTA计算中/等待时推进取数，所有pool先做LRU再grid barrier；独立控制020b…0b41只减attention stage、用padding保留原64行fetch双缓冲。两版均105正常+2长队列/全cache载荷测试（共214项），34EMU/68normal原库内实例全等价，实际单kernel/108CTA/256线程/1CTA驻留已核实。cofetch静态38272B+动态111104B、REG244EMU/242normal；control静态2304+动态147968、REG240，均stack/local0。GPU3同参256×3pass的100GB0/20µs原型慢38.940%/53.030%，control慢11.794%/12.284%，50GB慢15.761%/0.470%。9格/6配对、1728计时launch/108segment、两参考完整。原control preflight外部counter失败及仅补缺失候选的重试保留；全部C runner已terminal。见[报告](mla_host_a100_cofetch_20261101.md)。

后继HC=`temp/glm_trace/tune_hybrid_cofetch_20261101/`已实现，SHA9168…8259：保留原2stage attention和专用64行双缓冲主通道，worker用额外2×4行缓冲辅助；原gap flag只在prepare/compaction完成后复用为16行job cursor，原last-CTA退出复位。16行ready flag须等worker四个subchunk全部落地，dedicated每次领四个16行job。保持数学/LRU/模拟器/单kernel。**这仍是候选，不是生产改进。** 正常编译+105测试handle86861、EMU编译41882（已exit0）、长队列测试68190、EMU代码生成31554；先核实原handle/continuation_state，不重复。必须107正确性、34/68等价、实际Graph派发与同参性能后才能下结论。

生产提交为 **0f82e9de**。本轮P=`temp/glm_trace/tune_qk_pair_20261031/`实现不同KV tile共享Q fragment，确认历史regpipe3只改fetch buffer而未覆盖该机制；第一版75ea…d6f9f正常105测试通过、34EMU/68normal原实例相同、新REG255/STACK0，但GPU3同参256×3pass的100GB慢8.1%–8.5%。独立提前页号版adc6…f155fb同样105通过、原实例相同，新STACK8；三链路耗时降低−0.130%/+0.115%/+0.067%，没有稳定收益。两版均实际Graph确认单个新PairedQK kernel、108CTA/256线程、147968B dynamic shared、1CTA/SM。12格/6配对、2304计时launch、两参考完整，均拒绝。[报告](mla_host_a100_qk_pair_20261101.md)。首次base因外部counter_payload进入而preflight失败，原日志/state及仅重试base保留；SXM始终等待外部GPU0/1进程，取消自己的未启动runner，无SXM性能结果。

F=`temp/glm_trace/tune_fetch_warp_20261101/`的新warp_rows仅将大cached whole-row fetch读/staging写/cache写按每warp一行布局，复用72向量的地址，保留64行双缓冲/数学/LRU/模拟器。源码a156…404e；105正常测试、34EMU/68normal原实例等价、实际单新WarpRows kernel均通过；新REG244/STACK0，目标静态指令7888→7744。GPU3同参256×3pass，100GB0/20µs慢0.764%/0.030%，50GB20µs慢0.088%，无采用信号，6格/3配对与两参考完整。[报告](mla_host_a100_fetch_warp_20261101.md)。三候选均不扩展完整窗口/正常物理链路/全形状矩阵；原负差和全部源快照保留。

P/F所有CPU/GPU runner已terminal；P/continuation_state.json与F/continuation_state.json记录全部handle，不重启旧runner。新增restore_includes.py会核验既有树或从冻结host/math恢复缺失树，不覆盖既有include。此次不是blocked：已实现三个独立调度候选并完成315项正常测试、完整原实例等价和9个短窗口配对，但没有新增生产改进，历史100GB约32%–34%主差距仍未解决，goal继续active。

W补充矩阵896格/448配对的测量与两参考审计均完整，cached_planned_complete=true、pending_references=0。96个改变的LRU配对原矩阵耗时降低未加权均值2.028%、最大11.603%，最初四个大形状100GB负差仍保留；完整5pass前后夹测后最差−0.357%。未改路径R40PCIe0.3原−2.849%在夹测中+0.004%，其余原路径由代码身份和已有夹测/实际不同函数的同分配证据核对，原负差未删。保留标准明确不承诺每次独立内存分配都更快。

SXM正常构建、MTP8/R128、full1024×5pass、f24/off：真实NVLink3349.277→3229.662µs（3.571%），PCIe9937.678→9957.666µs（慢0.201%）。100GB完整确认仅约0.3%收益，不能声称解决历史32%–34%差距。整体goal仍active。

`retain_cached_ids.py --install`已实际通过门槛并逐字节安装已测b6原件；W/cached_ids_retention.json及retention_snapshot冻结完整审计/覆盖/编译/测试/慢格理由。生产root/include、独立jit_retained_normal、正常BF16/FP16 CPU编译及105项测试再次通过，记录tests_production_gpu3.json（13.16s）。测试source5845…保持不变。

上一轮本机GPU/CPU调度器均已terminal：W原矩阵92282、完整确认86595、慢格夹测82401、补小参考28183、生产编译37851和生产测试28695全部exit0；R的四确认和三版诊断也均exit0；H/Q/U拒绝候选无活动任务。**不要再poll或重复启动这些handle。** 最终audit83501和coverage/report已成功。41741初次将生产测试记录当候选记录而缺compile_exit_code，失败日志audit_production_record_schema_failure.log保留，已显式区分两种记录并复核通过。旧审计98052因单请求floor缺mathSHA而失败的日志保留；新同硬件normal小参考已补齐，没有覆盖原floor。

H800五个旧Slurm作业11/01 00:45:46仍Priority/Dependency，两个watcher活，保存C/h800_queue.txt；没有覆盖排队源或重复提交。P/F/C全部本机runner结束，HC新候选状态见上文和实际进程。下一步在b6上继续新机制推进大100GB目标；不要重复ready/prebind/scatter/四缓冲/tail/半head双驻留/重复QK/配对QK/warp按行取数/本轮单stage全CTA取数或纯参数扫描。已完成等待/CTA预算诊断支持关注fetch占据的SM，但不能将全部差距单因归因。HC保留2stage并只让worker辅助，与C已拒绝的单stage全CTA搬运分开核实；勿改模拟器/参考以制造收益。

### 2.1 最新续接：中等 batch 派发修正已保留，继续大 batch 主目标

生产头现为 `balanced_cached_sm80.cuh` 的 **52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5**。原81891229…225b冻结在本轮 `base.cuh`，上轮整行成果提交 `5157ff83` 和检查点 `0ff2fa66` 保持不变。本轮保留记录见 [独立报告](mla_host_a100_balanced_dispatch_20261031.md)及 `balanced_retention.json`，旧 `promote_medium.py` 继续禁用。

最终完成 SXM 120 格受限链路、4 格正常构建真实NVLink/PCIe，以及 GPU3 80 格正常构建独立HBM/PCIe。共102配对，改变路径的82个快1.29%–28.78%；224行恢复原分片的20配对最大负差0.31%，均在1%原路径测量门槛内。未隐藏负差。SXM正常构建MTP8/32请求、完整1024位置×5pass：NVLink1047.27→901.58µs（13.91%），PCIe2887.49→2790.52µs（3.36%）。

`retain_balanced.py --install`已实际执行并通过完整源码、覆盖、测试、编译等价、原矩阵SHA门槛；它冻结当时两份审计和其他证据，随后逐字节安装已测试候选。生产未新增GPU数学。最终候选实际通过105项正常构建测试，包含2/4×SM Graph/padding/映射/实际cache载荷；模拟9个BF16和正常18个BF16/FP16 kernel指令文本及资源均与base相同。

规则只读rows/max_kv_len/SM-grid/fetch_ctas/partial_capacity五个标量，用原tail planner、固定tile读写成本和现有前后两端循环估计波次。≥4×SM沿用原整行，2–4×SM且CTA_TILE_KV=32才比较；108SM/fetch16/K2048下216行整行，224–248行原分片，256/320/384行整行。单项thread-local有界记忆覆盖全部键，360检查通过；CPU微测cold47.7µs/warm0.002µs，不是端到端Python CPU时延。不用链路/命中/计时逐格挑选。

统一threshold2/threshold2_sm80仍已拒绝：224行正常构建、无LRU低远端比例慢3.2%–4.4%。它的110格LRU、330格无LRU、18控制、6确认及80正常构建均完整保留，不能冒称最终修正版重跑了全部原范围。`defer_cache`四链路慢29%–35%，warp_ready/row_ready也已拒绝，不重复无新机制筛选。

本机前一轮runner已全部结束：SXM91049 exit0（120格），SXM41792 exit0（4格正常），GPU3原43216末检因外部PID435854进入而exit1，补齐缺项的31760 exit0。失败未完成raw保存在 `gpu3/rejected/`，追加日志和state未覆盖；重启只跳过完成格。**上述中等batch验证没有活动runner；新大batch实验见§2.2。**

GPU3是A100-PCIE/NUMA1，同卡独立HBM不是NVLink，物理PCIe和Memcpy pinned源都绑NUMA1；参考在本GPU正常构建重测，PCIe约27.04GB/s、HBM约689GB/s。SXM0/1是NUMA0实际NV12，绝对时间不拼接。其他用户任务随时可能进入，继续资源预检/末检、共享锁，不终止其他人进程。

**大batch 100GB/s约32%–34%差距未改善，整体goal仍active。** 下一步读真实fetch/attention/LRU代码，在独立include/JIT实现新机制，优先研究fetch的cache scatter占用和attention计算尾部；从attention共享KV tile写回缓存已在下一轮实现并测出回退，详见§2.2，不再重复。单kernel和精确LRU/载荷要求不变。

### 2.2 新大batch实验：共享tile缓存写回未保留，整行四缓冲待性能结果

新目录 `temp/glm_trace/tune_scatter_20261031/`（下称Q），基线为已保留的52919…ff2c5，生产未再改变。当前新实验及失败记录见[fetch资源报告](mla_host_a100_fetch_resources_20261031.md)和Q的`audit.json`/`decisions.json`/`continuation_state.json`。

- `attention_cache`从attention共享KV tile写LRU缓存，以原子victim认领去重；105正常测试通过，但GPU3 fetch24同参数100GB两格慢约12%、50GB慢1.2%，拒绝。
- `owner_cache`保留prepare CAS所有者避免新原子，105测试通过；静态审查发现临时bit30可能与大源页号冲突，被`owner_complement`取代，原始源冻结。第三个短筛选格因外部PID486390进入而末检失败，未纳入有效结果，没有完整配对声明。
- `owner_complement`改为临时补码标记，恢复完整源ID，只在容量受限的负staging索引使用所有者位；105测试通过。fetch24/16/8同参对照已完整完成：24慢约11%–13%；16在100GB+20µs快1.5%但0µs慢9%，50GB略慢；8虽比同参base快11%–22%，仍不及现有fetch24推荐。当前不采用缓存写回迁移。
- `whole_pipe4`保留fetch内即时cache fill，仅整行CTA_TILE_KV=32使用32行/四缓冲。**历史fastpipe4源码没有WholeRowHostBatchParams，没有测过这一组合**，因此这不是重复其原筛选。六个通用及另两个整行配置指令/资源相同；目标CTA32由7784降为7664条、244寄存器、stack/local0。105项正常测试已通过。最终源SHA=`e0ec1012ed7b29237fbaeca2e7985f15d0247dc06ede40903283134373112bb7`。

`whole_pipe4`的fetch24/16/12/8同参screen均已完成，83402及75786明确exit0；默认24近持平或略慢，减少fetch也无替代原推荐配置的收益，当前不采用。正常BF16/FP16编译审计16个未改实例相同，2目标变化，全部stack/local0。

接着独立实现`tail_wave2`：前部保持整行直接输出，仅最后grid行拆为两个现有partial item并沿用merge，复用原数学、仍单cooperative kernel。仅在已走whole、CTA_TILE_KV=32、≥4SM且partial容量足够时启用，其他派发不变；SHA=`10e8a73039b037bd1500a04b7a1217b3ccc20dbf247783982ea841517bb1c31c`。105项正常测试通过，但三链路同参短筛选全部慢0.87%–1.04%，拒绝。原九个模拟kernel指令/资源不变，新增三个tail实例stack/local0。

**全部GPU runner已结束，不要再poll或重复启动**：74331（tail正常105）、40736（EMU编译）、51442（tail同参筛选）、84718（pipe4正常105）、34123、25115、75786、83402都已exit0。CPU正常tail代码生成审计27774也已exit0：18个原BF16/FP16实例指令/资源相同，6个新增tail实例stack/local0。没有活动runner。全部五个新候选均不采用。Q与P所有GPU3任务共用`P/gpu3_correctness.lock`，未终止其他用户进程。GPU0/1本轮持续被其他用户占用，所以没有本轮SXM/NVLink性能新结论。

下一步继续定位大batch计算/fetch争用；不要再无新机制重复缓存写回迁移、四缓冲整行或最后一波二分片。当前GPU3同卡独立HBM模拟不能冒称真实NVLink或SXM收益。**大batch主目标仍未完成，生产保持ce341a02。**

### 2.3 最新：共享页号原版被正常PCIe回退拒绝，缓存限定修正版验证中

新目录 `temp/glm_trace/tune_wait_20261031/`（W）。生产仍为ce341a02/52919…ff2c5。前轮Q任务全terminal，不重做。新报告：[CTA预算及fetch元数据](mla_host_a100_wait_budget_20261031.md)。W/audit.json只含插桩，W/performance_audit.json含性能；不要互相覆盖。

新CTA预算诊断已完成（不是重做旧阶段表）：SXM0/1、MTP8/R128、LRU4096、完整1024位置、fetch24/off、1插桩pass，100GB20µs/50GB20µs/NVLink。128epoch×108CTA，每步合计1024attention/128LRU工作项。累计等待调用/attention时间为5.90%/45.21%/3.31%，100GB每个fetcher只做1.31个attention item。计时含元数据和timer，插桩影响调度，不能作精确开销分摊；它支持关注fetch吞吐/保留SM。

本轮四候选：

- `prebind`提前处理已选victim的元数据，让它与cp.async在途重叠，105正常测试通过；fetch24基本±0.7%，16在100GB20µs快5%但仍不如推荐24且NVLink负差，20也无足够替代收益。不采用。
- `direct_cache`让获准页直接shared→victim，其余shared→staging，attention在原flag acquire后查victim。105测试通过，但100GB0/20µs慢约1%，不采用。
- **`fetch_ids`原版拒绝**：共享页号供向量取数/清理/缓存绑定复用。105测试通过；大batch完整5pass曾100GB20µs3524.77→3476.42µs（1.37%），正常NVLink3322.15→3224.73µs（2.93%）。但GPU3正常构建无LRU、MTP8/R32、PCIe远端0.1初测慢18.5%，反向5pass仍慢11.24%（830.57→923.90µs），前后base830.57/830.98稳定。0.3初测8%负差未复现，全部原件保留。不能只报大batch收益。
- **当前`cached_ids`**：共享页号只给有LRU的whole CTA32新实例，全部原generic/whole实例shared_fetch_ids=false。模拟9个和正常BF16/FP16的18个原kernel指令/资源完全相同；仅新增1/2个缓存实例，全部stack/local0。105项正常GPU3测试通过。原PCIe0.1格复核835.17µs（较830.57慢0.55%），无LRU确实使用原kernel。SHA=`b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a`。**尚未采用。**

原版矩阵保留partial状态：GPU3 68/88、SXM limited320/640、SXM normal0/136（后者一次旧自有CUDA context退出尾声触发预检，无GPU计时）。发现回退后，仅SIGSTOP自有CPU父调度器，等各当前benchmark子进程自然结束，再SIGTERM/SIGCONT父进程；没有给GPU子进程或其他用户发信号。记录`rejected_matrix_stop.json`。原计划/源码/全部慢格不覆盖，不再启动原版`validate_ids.py`。

当前缓存限定版使用新cohort和`validate_cached_ids.py`，完整计划覆盖8形状、MTP1/8、LRU1024位置和无LRU四快照×3比例、50/100GB/s五延迟及正常NVLink/PCIe、小batch控制。GPU3独立正常88格；SXM640模拟格和136正常格分别审计。参考同硬件，未修改原5304矩阵。

旧SXM handles **27470/27738/75965/59866均已exit1**，不要再poll。完整确认和正常矩阵预检遇到前一自有CUDA context退出尾声；模拟矩阵后段遇到外部PID643266/643267进入。所有失败raw/日志保留，未纳入计时结论。现在`bench.py`在拿到gpu01锁后等待进程/显存/利用率空闲再打开CUDA；没有忽略外部进程。

**GPU3的88正常格和新增大形状32格均完整**，37750/17179已exit0，不重启。SXM模拟640/640及正常136/136格均已测量；正常132/136格已带参考审计，4个MTP1/R1控制格等待补参考。GPU3大形状R128 LRU的HBM3423.26→3374.46µs（1.43%），PCIe9933.76→9960.84（慢0.27%）；R64两链路约慢0.24%–0.43%。GPU3没有测MTP1/R512同卡HBM（40GB限制），该形状留SXM双GPU计划。

对所有原路径超过1%的负差做前后5pass夹测，`controls_audit.json`：R32无LRU PCIe和R28 LRU HBM的负差没有复现；MTP1/R1 PCIe仍为base前52.0822、候选54.0782、base后52.0358µs（慢3.879%）。另在同进程载入两独立二进制，共用同一query/indices、Host KV/cache/workspace交替5+5次完整冷启动，base53.389000、cached_ids53.383400µs（降低0.01049%）。10次均校验输出和最终4096行实际cache载荷。此支持分配/环境因素解释，但因果未确定，原负差不删、不放宽门槛。逐步miss因同一步stamp下并行入槽顺序略有变化，不声称所有pass完全相同。

W原恢复handle92282、父658154/子739126已全部exit0，limited640及native136都测完，GPU3另120格也完整，共896性能格。**不要再poll/restart92282。** 下文旧阶段数据保留；最新审计需区分测量和参考门槛。

W完整SXM 5pass：100GB0/20µs快0.383%/0.288%、50GB20µs快0.216%、EMU构建NVLink快3.059%；正常NVLink3349.28→3229.66µs（3.571%），PCIe9937.68→9957.67（慢0.201%）。原3pass的大形状100GB负差约1.1%–1.75%和原路径0远端负差最高约2.22%仍保留，没有用5pass覆盖旧格。正常矩阵也已有慢格，待完整后分类与夹测。

94802、58067、92776/60443、94033、7575（新夹测）、62149（同分配诊断）均exit0。`continuation_state.json`列真实新handle。cached_planned_complete仍false，只有完整新原件、正常物理链路和全场景通过才能保留。当前大batch历史32%–34%差距未被已保留代码改善，整体goal active。

### 2.4 32-head双驻留：拒绝，不重跑

独立H目录`temp/glm_trace/tune_heads_20261031/`，报告[32-head CTA](mla_host_a100_half_heads_20261031.md)。host头SHA=`d8dc5371b4e28b15764bf331a39eb579678190c91f22902e048b52627b35038e`，私有math头SHA=`e5968e7605bb657de30e5600c4a8fc09b977a3e2e7163f64e46718de519289a3`。两32-head项、128线程、1stage/32KV、32行双缓冲，无head合并，生产未改。

105正常测试通过；Graph实查1kernel、grid(1,216,1)、block(32,2,2)、dynamic shared73984B、实际occupancy2，确实走新路径。新增kernel寄存器241、stack/local0；EMU32和正常64个原库内实例（含纯device）指令/资源相同，新增唯一符号1/2个、各在两个库内出现。原计数审计漏算第二库导致断言失败，日志保留，修正后成功。审计显式要求64个正常原实例等价才沿用同硬件参考；没有跳过math头SHA检查。旧128CTA插桩器从未用于H。

GPU3受限链路、MTP8/R128、LRU4096、256位置×3pass，fetch24/32/48共18格/9配对完成。24的100GB0/20µs慢7.96%/30.11%；32在0µs快3.47%但20µs慢4.31%；48同参快6.47%/7.10%，但3668µs仍慢于同轮推荐base f24的3553/3580µs。50GB快0.36%–1.54%，无足够替代收益。候选拒绝，不扩大完整窗口/native/全矩阵。全部H GPU runner已terminal0：96997、79051、61333、82666、16552、57912、29770；代码生成78165/34929也exit0。

### 2.5 重复QK与当前缩短展开候选

独立`temp/glm_trace/tune_qk_20261031/`，`qk_local.cuh` SHA=`030da403e7d11d4bb2544684fad08676d5ffa1d9c6d571a8fcb00fcba5e8fe18`。只在原大形状whole、CTA_KV32且≥4SM行时用原生`KernelTraits<false,2,false,512,64,64,32,...>`。两个warp group各算完整QK，PV仍分特征，用既有false-QK_SHARD数学省m_wg/P交换。不改mla.cuh、缓存、搬运、模拟器、参考、单kernel架构。

**此全展开候选已拒绝**：105正常测试通过；EMU32和正常64个原库内kernel指令/资源相同；新增kernel REG255、STACK192B、LOCAL0。GPU3受限链路、MTP8/R128、LRU4096、256位置×3pass、f24同参3配对，100GB0/20µs慢22.77%/22.39%，50GB20µs慢1.57%。[报告](mla_host_a100_qk_local_20261031.md)。新栈被明确记录，初始无栈断言失败保留，后续资源审计报告false，未伪称零spill。所有handles29098、44081、41125、59185、82166 terminal0；13953初始spill gate exit1。首次CLI遗漏--gpu的exit2日志也保留。

后续U目录`temp/glm_trace/tune_qk_loop_20261031/`，`qk_unroll4`：host SHA仍030da403，私有math SHA=`1e3e5731ed0df5c98508b58210c0e2b979daf5d8f6a53613f23a40522e229aea`。只对false-QK_SHARD/CTA32的outer feature loop设展开4，其余实例显式完整原展开因子，须验证原64个正常库内kernel等价才可沿用参考。试图减少活跃片段/192B栈，不修改原失败源。

此缩短展开候选也已拒绝。105正常测试通过，EMU32和正常64个原实例指令/资源完全相同；新实例REG244、STACK0、LOCAL0。GPU3同参256位置×3pass，100GB仍慢约22%，50GB也慢；[报告](mla_host_a100_qk_loop_20261031.md)。消除栈没有使这个机制取得收益，不再无新机制扫描unroll2/8。全部原handles10603、54470、14315、61075、90680已terminal0，不重启。U/continuation_state.json已更新。

### 2.6 新R候选：保留缓存限定派发，恢复运行期LRU判断

`temp/glm_trace/tune_cached_runtime_20261031/`，`runtime_ids.cuh` SHA=`85ef148314998a7973c615e1f3a3853ea2be027ce17bdc2ccfbc162495948a93`。与W/cached_ids相比，只把host_batch_caching的constexpr true恢复为原`params.pool_capacity != 0`。派发仍只在有LRU的whole CTA32启用共享页号，无LRU保持原kernel。没有改数学/cache策略/模拟器/参考。观察到W/cached_ids新目标比原fetch_ids目标少8条EMU指令，normal目标也有变化；大batch负差不能仅由此归因，需要本候选测量。

R正常105测试通过；原32EMU/64normal库内kernel指令/资源不变，新缓存目标REG244/STACK0/LOCAL0，且确实与最初fetch_ids缓存目标逐条指令/资源相同（EMU/BF16/FP16），无LRU仍用原kernel。`predecessor_comparison.json`明确这是代码身份，不代表性能/因果证明。

R四组完整确认已全部结束：26独立格/13配对（6短格、20完整1024×5pass格）。SXM100GB0/20µs慢0.653%/0.122%，正常NVLink快1.841%、PCIe慢0.085%；GPU3 full100GB0µs慢0.458%、20µs快0.378%、50GB快0.308%，正常独立HBM快1.395%、PCIe慢0.290%。没有稳定的100GB优势，不采用，不能说R已跑完整跨形状矩阵。[报告](mla_host_a100_cached_runtime_20261031.md)。原handles45270、68557、66889、18639、98493、91439、44565、24263、96705均terminal0，勿重启。

新增R/allocation_probe：SXM共用同一Q/indices/KV/cache/workspace/output，base/cached_ids/runtime_ids三份已加载二进制各5次完整冷启动，100GB0/20µs两链路，共30pass。Graph API实查3个不同函数、相同launch形状、每图128个attention节点；不是动态链接错用同一函数。每pass检查输出、映射及524288行实际cache载荷，全部通过。0µs下cached慢0.272%、runtime快0.044%；20µs两候选快约0.6%。`allocation_audit.json`保留全部逐步样本/miss和源/库SHA。此诊断不替代矩阵，未建立各波动的因果，也未解决32%–34%主差距。handle72917已exit0。

W早先GPU3小请求同分配诊断也补做Graph函数核对：两个独立函数，各1024个attention节点，`same_process_dispatch.json`/audit_controls已确认；handle91360 exit0。原独立进程3.879%负差和同分配0.01049%差异均保留。

最新只有两个CPU调度器等待GPU0/1外部PID806097（/mnt/data/mozi-elastic-20261022.0jc96E/env/bin/python），不终止它。先核实原handle/PID，不重复：

| handle | 父PID | 内容 | 状态 |
|---|---:|---|---|
| **82401** | **806233** | W/recheck_sxm_slow.py，R128/MTP8 g100lat0/2/5/10完整5pass前后夹测；再R40无LRU正常PCIe0.3夹测 | WAITING_GPU，尚未开始GPU工作 |
| **28183** | **814047** | W/small_floor_ref.py，MTP1/R1正常构建完整1024×3pass纯device参考 | WAITING_GPU，尚未开始GPU工作 |

全部896性能格已测量，但当前只有892格可带完整参考审计：旧原矩阵`floor_floor_mtp1_b1.json`未记录mla.cuh SHA（MTP8/R1记录有SHA）。完整审计handle98052因此exit1，失败日志`audit_missing_floor_math_sha_failure.log`保留。没有据git commit猜测dirty旧math来源；补测同硬件参考，不覆盖原floor或5304矩阵。新auditor保留4个MTP1/R1控制格的实际时间/配对/guard检查于pending_references、pending_reference_pairs和pending_reference_timings.csv，未伪造两参考。新ref完成后会自动优先用于这四格。`coverage.json`明确measured896完整、audited892未完整，cached_measured_complete=true而cached_planned_complete=false。

正常SXM矩阵新增最大负差为未改kernel的R40/MTP8 noLRU PCIe0.3慢2.849%；R128/MTP8 g100lat0/2/5/10慢1.1%–1.75%仍保留，尽管端点full5pass和同分配诊断未复现大幅负差。因此新夹测覆盖这些确实影响保留的疑点；不能仅用代码身份或同分配诊断放宽门槛。生产仍52919，整体goal active，32%–34%主差距未解决。

检查点提交`3d101924`（719个自有文件/414个stagedJSON）及`d8c7131e`（299个自有文件/151个stagedJSON）均通过diff check和JSON全解析。新慢格夹测/补参考脚本及审计修正另在本检查点，运行数据会继续增长。其后R完整原件、诊断与W审计更新已纳入本检查点；W正常矩阵仍在运行，后续raw/state继续增长。不要只checkout旧提交丢掉当前工作区。

### 2.7 已完成阶段诊断（不重复启动）

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

### 2.8 脚本注意事项

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

登录`ssh -n kaituo1000_login21`（luyouy@166.111.143.25），用`bash -lc`。原快照`temp/glm_trace/tune_general_20261031/h800_queue_20261031.txt`，最新再查结果为W目录`h800_queue_20261031_2313.txt`（实际23:13:30），先读实际结果；此前五任务仍Priority/Dependency。

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

后续提交`ce341a02`保留中batch派发，`43e2f835`/`b58193fb`保全拒绝候选，`448c4b3b`保存CTA预算和共享页号修正。W/H/QK/U已在3d101924保全，R和W后续原件保存在本检查点；当前仍可能有W运行数据继续增长。

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
