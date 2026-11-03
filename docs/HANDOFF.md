# 交接：远端 KV sparse MLA kernel 持续优化

更新于 **2026-11-03 17:51 SM80同warp流水线调研/CPU可行性完成；新探针GPU预检占用未运行，生产cab508dc/FAR8507eab0保持，Asia/Shanghai**。本文是当前入口；旧交接完整保存在 [历史记录](HANDOFF_history_before_20261031.md)。冲突时优先读当前用户要求、本文和实际进程/数据。

## 1. 任务目标与授权

**最新用户约束（2026-11-01 18:32）："每次不用验证这么多，确认机制有效、在不断优化就行"。** 后续迭代优先实现与少量有针对性的正确性、关键场景同参数对照；不再为每个候选重复105+stress全套、全部sanitizer、双顺序五pass和156格矩阵。已有验证可复用；发现具体疑点才补针对性检查。有效信号可先保留为当前优化，明确未覆盖范围，持续推进；不要把本文件旧全面门槛机械套成每轮必做或据此询问许可。单kernel、数学/LRU/载荷正确、真实计时和不改模拟器/参考仍保持。

持续优化 FlashInfer 的远端 KV sparse MLA decode **单 cooperative kernel**，融合跨query去重、跨步LRU、远端读取、attention及结果合并。在指定带宽/延迟下显著优于正确的先搬后算Memcpy，尽量接近逐步`max(纯device计算, 必需远端搬运)`的完美重叠参考；低远端流量时尽量接近纯device。

用户最新要求：**“还能继续优化吗？让我们的kernel在各种场景下都表现优秀。”** 已授权同范围的实现、实验、修复和回归，不需反复确认。此前完成交接后，用户已要求继续 active goal；本轮已经恢复独立候选实现与验证。

覆盖小/中/大batch、MTP1/8、有/无LRU、50/100 GB/s、附加延迟0/2/5/10/20 µs、物理PCIe/NVLink。允许按形状和路径选择不同调度，但必须区分代码、调参和路由收益，避免某格变快而其他场景回退。保持单kernel、数学计算和缓存载荷/状态正确。拆kernel或明显改变目标的架构决策需用户确认。

**不优化HiSparse**；如需对比，按原生部署接口运行，不替它接入多步去重。当前主任务是我们kernel对Memcpy和纯本地计算的比较，不模拟整个模型其他部分。

## 2. 当前运行状态与新一轮进度

**最新用户提出学习其他A100工程，让所有warp访存穿插计算；本轮完成官方源码调研和新结构可行性核算，未新增GPU结果/生产候选。** CUTLASS v3.9.2 `mma_multistage::mac_loop_iter`将future cp.async分散在当前warpMMA之间，stage边界才wait；FA2 v2.7.4 forward先V读再QK、nextK读再softmax/PV，utils预取下片段再MMA。它们同CTA拥有输出，不能照搬成逐引用far读取。相同base24 trace未缓存引用562.109MB/步而unique247.981MB，若取消launch去重会多2.267倍远端字节，不能沿用unique参考。

新依据：当前PVFeature PE已global→packedregs，数值region不读shared QPE，但load_q和storage仍保留8192B；仅该typedpath可考虑回收，canonical其他路径不能删。CPU实际sizeof已编译运行：原147968→compact139776，保留双32KVstage+2×10row独立farbuffer后162816dynamic，加假定static2304为165120≤166912/block。不是完整kernel资源/占用/正确性证明。108×20row×1152=2.488MB名义在途，大于100GB×20us=2MB，但CAS包频率、ready与复用、registerpressure仍可能使吞吐失败。新完整kernel假设保留8mathwarps/双stage，静态粗unique任务，cp.async向量分布到MMA之间，无QKscore跨SM交接；未实现，不原样重跑旧减stagecofetch/hybrid。

`temp/glm_trace/research_sm80_pipeline_20261103/`已有独立samewarp微探针（purefetch/bulkasyncthenQK/distributedcpasyncbetweenMMA，24/108CTA100GB0/20），compiledREG117/168/76、stack/local0，未查SASS真实交错。独立localQ/K、不含真实worker/LRU/O/softmax，不能替代完整attention。GPU3编译后外部VLLM约37GB占用，120秒稳定idle预检到期，CUDA子进程未启动、无result.json/GPU正确性性能，waiterterminal。下个硬件窗口若合理，仅跑该新目录run.py，不恢复旧completed筛选。sources.json是官方固定tag web读取的URL/函数，直接urllib下载SSL失败，无外部源码快照/伪造SHA。生产cab508dc/math34af/FAR8507、参考、矩阵均未改；GPU0/1/2/H800未使用。[研究报告](mla_host_sm80_pipeline_research_20261103.md)、feasibility.json/README为入口。

**17:29最新：用户已授权在维持带宽下利用fetch CTA冗余计算，支持局部warp分工及有依据地增加fetch SM；本轮两版完整实现及诊断已完成，无新生产加速。** 串行QK穿插探针补完：99.49/98.52→62.80/62.80GB/s，拒绝。新的fetch CTA内4loader+4QK、保留双64行＋12KiB feature工作区，24CTA微探针100GB0/20仍99.22/98.04GB/s并算完3361真实QK块；32CTA相近。payload/采样QK/arrival≥deadline检查通过，FAR header/reserve/tile不改，128loader wait同步适配；微探针不含真实worker消费，不能算端到端收益。

完整pull交接仅改fetch分支，保留原worker完整布局，gen-tagged所有权、每worker2score槽，已就绪KV，worker消费后跳过原QK，produced==consumed、零未释放lease。candidate9583397b用12KiB/两次32KV QK；fused最终SHA见manifest，16KiB共享Q的64KV pair＋预解析KV指针，dynamic164352/static2304/REG255/STACK0/LOCAL0，总166656≤block166912；实际SetAttribute/Occupancy替代不适用的+4096保守guard。普通worker64位状态/进度仍增加开销。GPU3/NUMA1 BF16MTP8R128K2048LRU4096 staged f24/off、256×3新同参base3307.840/3303.360µs；candidate3859.403/3875.499（慢16.67%/17.32%），fused3760.395/3782.347（慢13.68%/14.50%）。全部拒绝不安装。

fused消除约99%的原helper等待，真实分担2333/2440 pairs（7.12%/7.45%），但完整fetch只有95.96/92.64GB/s；32fetch＋20µs恢复97.30GB/s、分担3222（9.83%），总3824.597µs反慢于fused24，同参base32为3430.773。不要继续单靠加fetch数。新同binary100GB20/24CTA关闭控制（一冷32步pass）无交接3482.752、仅维护交接但helper不认领3724.352、完整辅助3774.080µs；交接维护增加241.600µs，辅助的额外读取/score传递与所省QK净效果再慢49.728µs。不是某条指令因果拆分。后续若继续此方向，需要新的粗粒度所有权/交接机制减少普通CTA热循环操作，而非原样重做pull或扫前瞻/warp/fetch数字。

新增长短/fused/f32共5项定向检查（output/LSE、QPE stride80、padding空/单元素/奇尾、3轮cache状态、Graph×2、全有效cache载荷/映射），长case要求非零消费。864次完整kernel计时（8×96＋3×32），另18串行/36空间微探针；原始失败和全部负差保持。无新整套pytest/sanitizers/full1024/FP16/物理链路/H800结论。首uint64ptr编译失败first_compile，空队列broadcast及fused预GPU修订有归档，勿运行这些archive。全部runner terminal，无等待器，GPU3末检0MiB0%；GPU0/1/2及H800不动。生产cab508dc/math34af/FAR8507及原矩阵保持，整体110%仍未完成。[结果报告](mla_host_fetch_qk_assist_results_20261103.md)，原件`temp/glm_trace/tune_fetch_qk_assist_20261103/{README.md,audit.json,manifest.json}`；两个probe目录均completed，勿重跑，构造make脚本非幂等resume。

以下16:31设计与16:04诊断是历史入口；原“probe未GPU运行/唯一待测”已由上文替代。

**当前用户希望优先利用fetch CTA冗余计算，进入新QK辅助设计讨论。** 新方向为保留84worker完整布局和24个64行双buffer，只让fetch算有界已就绪QK，原worker消费score并跳过相应QK，原softmax/PV保持。不同于旧全部CTA role/cofetch/两遍QK；尚未实现真实交接，不能称已有分担或收益。两个前瞻paired槽/worker约2.625MiB，必须有epoch/generation/claim及drain；CLAIMED后不能盲目回退复算或复用。全CTA等待内单纯穿插的显式等待机会折算约0.18–0.19ms，并不保证覆盖目标0.36–0.39ms；后续fetch分支内warp分工需要正确loader组/barrier协议，未实现，FAR模型不能改变。

已冻结`temp/glm_trace/probe_fetch_qk_assist_20261103/probe.cu`微探针：24CTA双64行、原v2 FAR，在每包等待前执行真正64head×32KV×576 QK并发布score；purefetch/assist实编REG30/54、STACK0/LOCAL0，CPU2048输出坐标/operand/PE→CKV576覆盖通过。只有独立local QK数据，无真实worker消费/LRU/84CTA并发；哪怕微探针通过也不是端到端收益。首漏cstdint编译失败存first_compile，已修。GPU3外部flashcheck连续占用，120秒稳定空闲预检到期，CUDA/probe未启动、无result.json/数值性能；原driver和preflight_failure保留，所有waiter terminal，不重启旧任务。下个空闲窗口只运行此新目录run.py做机制可行性，若吞吐受损先拒绝该实现，不扫参数。生产cab508dc/math34af/FAR8507及诊断原件不改；[设计与边界](mla_host_fetch_compute_design_20261103.md)。

**11/03当前用户要求先梳理差距，讨论前判断fetch CTA占SM是否主因；诊断已完成，没有实施新优化。** GPU3/NUMA1同卡farHBM/v2、BF16MTP8R128K2048LRU4096/f24/staged/splitoff、256位置。相同当前attention实现、同一步完整kernel刚完成的remap/cache载荷、同binary84/108计算CTA重放：108为2442.389µs，84为3047.989µs，慢605.600µs/24.80%；3冷pass，192次全输出逐元素相等、state不变，轮换顺序信号稳定，108个不同SM实测。重放从kv_indptr恢复退出已清零的无padding行长度，不执行fetch/LRU/prepare，不能当端到端优化或替换device参考。该资源代价与正式约0.63–0.66ms差距同量级，强力支持fetch占用是主因，但不能称96.7%已归因或收益可直接回收。

当前完整时间线0/20µs：前置189.088/187.296µs，fetch跨度2523.392/2552.672µs，fetch后attention尾部699.808/659.520µs；24fetch CTA后续仅算53.156/50.344行（总1024），最后ready至全部fetch返回约4µs。fetch SM服务预算/108约550.648/555.889µs，不是可直接扣除墙钟；尾部不能再与容量损失相加。暖步真正阻塞wait次数约26/31，计时含ready检查及插桩，不能直接扣成等待。100GB搬运预算~2.48ms与device~2.56ms接近，争用最容易暴露；低/高带宽趋势是机制解释，本轮未重扫v2带宽曲线，也不推广所有batch/物理NVLink。旧减fetch/退休/cofetch负结果继续有效，不原样重做。到110%仍需省0.36–0.39ms，前置全消除也不够。

独立trace/capacity实际REG252/254、STACK0/LOCAL0/static2304/dynamic147968；时间线相对正式均值扰动+2.81%/+1.41%，各一点一pass，只作诊断。64个trace launch+96个完整准备+192个重放，采样参考/全输出state/三份最终524288有效cache槽全载荷检查通过。无新大验证/全窗口/物理链路/H800结论。首driver完成0µs后第二次VMM分配OOM，原件first_driver保留；之后按每设置独立进程只补20µs/capacity。全部runner terminal，GPU3末检0MiB0%，无等待器；生产和模拟器未改。入口[诊断报告](mla_host_fetch_sm_diagnosis_20261103.md)、`temp/glm_trace/diagnose_fetch_sm_v2_20261103/{README.md,analysis.json,run_state.json}`。构造器make_diagnostics/fix_replay_length非幂等resume命令；不要重跑已接纳诊断。

**11/03最新v2性能重测已完成：** `temp/glm_trace/rebench_100gb_v2_20261103/`，GPU3/A100-PCIE/NUMA1同卡farHBM模拟，非NVLink；BF16/h64/K2048/MTP8R128/LRU4096/f24/item64/splitoff/staged，256位置×3冷pass。100GB0µs ours3327.371µs、Memcpy5813.493、同轮正常device2558.933、逐步max2701.022（高23.19%）；20µs ours3360.555、Memcpy5824.800、参考2701.192（高24.41%）。引用命中76.737/76.734%，unique247.915/247.957MB/步；不可套完整窗口~79%。110%目标2971.124/2971.311，还需降低10.71/11.58%。0→20均值只差33.184µs约1%，pass波动重叠，不当精确延迟因果量。

24步计算主导/8步传输主导；0µs两类平均高于参考611.285/671.539µs，20µs639.801/718.049µs；首冷步传输参考6210.835µs，实测6792.533/6816.085，仅高9.37/9.75%而绝对仍多约0.6ms。支持继续检查准备/SM争用/尾部，但未新做阶段插桩，不能直接归因。84CTA线性粗估3290.057µs只是待验证假设。当前EMU实际REG254/STACK0/static2304/LOCAL0。

Memcpy本地attention本轮明确绑定正常device库，远端仍v2 gather+必要cache scatter，同输入同容量离线LRU；元数据标memcpy_path_version3、far_emulator_version2，不与旧v1Memcpy作严格新旧对照。复用已验证v2二进制，实际加载路径/SHA均记录；device参考本轮相同32步新测，原mean(stepmax)公式保持。3格/480计时launch/30有效段，数值/guards/源与binary/miss审计通过，没有新kernel机制或全套/sanitizer/full1024/物理链路/H800。首子进程因外部flashcheck重启在CUDA前被预检挡住，原失败保留；稳定空闲10秒后成功，全接纳raw开始/结束无外部GPU3任务。所有runner均terminal，无等待器；生产、旧5304矩阵及旧数据保持。[报告与初步分析](mla_host_100gb_v2_retest_20261103.md)。后续再讨论优化方向，勿重复旧拒绝方案或把描述性分组当因果阶段拆分。

**11/03最新：用户明确要求“先修复吧，先保证我们的模拟准确然后我们讨论优化方向”，已授权本次模拟器修复，优先于旧“不改模拟器”限制。** FAR升级为8507eab01aa705c507d0ee389f1216dfc3c29d50844fbf509a0339fd6023d716；host仍cab508dc、math34af不改。CAS原子max-plus计费，发出时预约、stage记录各warp最晚deadline、等待前补齐CTA同步；纳秒预算向上取整<1ns/包，L不占传输时钟，移除1秒截短。Memcpy gather统一发出时预约，旧432x16正确CAS校准只约51GB/s而失败，原件保存在emu_fix/cas_contention；最终64页/48单bufferCTA=3072在途行，与f24双64页相当。公共emu.py新模块/build_v2，拒绝旧kernel头混用新gather。

GPU3/NUMA1：32组16000次并发预约、3组算术、2组late-warp stage检查全部正确；6真实payload搬运格全载荷一致。24CTA双64页fetch模拟100GB0/20实测99.53/98.78GB/s，50GB49.91/49.75；最终gather15组校准全过，100GB五延迟98.17–99.15，50GB49.69–49.88，20µs chase约20.48µs，原5%/10%+600ns判据、全部样本保持。normal/EMU构建、3项BF16staged/direct output/LSE/cachepayload/Graph集成、公共binding载荷/Graph及旧头拒绝通过。首CPU兼容错误和oracle错误归档first_attempt，非最终失败。软件开销已记录：302MB搬运微核nohooks486.4µs、disabled558.4、unlimited enabled695.0；不是完整MLA因果份额，不能直接扣除。它是校准的payload/到达模型，仍非物理NVLink等价，只验证单活动launch/module及50/100GB0–20µs。

**旧5304矩阵及此前所有性能都是v1，不改值、不重标v2；旧冻结runner不得混用新gather。** attention110%目标继续未完成，不能将模拟修复计作kernel加速。后续讨论差距或优化收益前，用同一v2、新JIT/新原始文件做少量关键新对照；device参考公式保持。没有本轮新MLA性能/完整trace/矩阵/sanitizer/FP16/SXM/NVLink/H800结论。所有本轮runner结束，GPU3资源重新查询，GPU0/1/2未使用。当前报告[模拟器修复与校准](mla_host_emulator_fix_20261103.md)，原始目录`temp/glm_trace/emu_fix_20261103/`及audit.py为入口。

以下14:24审计段落及更早记录是旧模型历史，FAR当前SHA以上述v2入口为准。

**11/03 14:24模拟器审计新发现（优先阅读）：用户追问100GB真实性，未改生产cab508dc或FAR5e409。** 原reserve的atomicAdd后conditional atomicMax不是原子max-plus，空闲恢复时并发预约可漏算字节预算。`temp/glm_trace/emu_semantics_20261103/`保留CPU合法交错反例和调用原header实际reserve的独立GPU3诊断：2/24CTA同issue空闲burst各2000次均预算不符，预忙对照各2000次全正确；24CTA service span737..8844ns，预期17688ns。这是构造场景并发正确性证据，无payload，不是fulltrace发生率或耗时影响。此前SXM432CTA/16row饱和校准99.72..99.96GB/s仍如实保留，但不证明24CTA/64row瞬态严格100GB或物理等价。现有AB是在同一未改模拟器内的结果，不能再承诺严格物理100GB或把~24%差距全归kernel。延迟加在虚拟传输完成deadline上，实际copy耗时与其重叠，并非真实copy后再睡满L。预约在consume时按recorded issue追记，排队按预约顺序。无模拟器修复/参考变更/旧数据覆盖，建议先隔离版本审计正确性与生产发生率，再新测；所有诊断任务已结束。[审计与原件](../temp/glm_trace/emu_semantics_20261103/README.md)。

**11/03接续最新：生产已保留PV feature调度，SHA cab508dcb6e83b7a953d94a8e54e30e0aaf60bb1d7ba98480fc66893edcf4384；不回退7547或更早版本。** `tune_pv_feature_20261103/`直接以7547为控制，32KV PV先载入两P片段，再逐输出特征依次执行KV half0/1 MMA；每元素累计次序、P/rowsum/联合MDO/barrier/QK/refill/fetch/精确LRU/FAR保持，不组合拒绝的rescale。2048地址核对、2项定向BF16检查及实际单108CTA/block32x4x2/dynamic147968/active1 Graph通过。normal/EMU新REG254/STACK0/static2304/LOCAL0。GPU3/NUMA1/MTP8R128/LRU4096/f24off/256×3，正向100GB0/20快1.029%/0.561%、50GB持平；反向100GB快1.257%/1.894%，正常同卡HBM快1.438%。按轻量偏好逐字节安装实测头，retention绑定原件，无生产root重编译/新全套/sanitizer/full1024/FP16/SXM/物理PCIe/NVLink/H800新结论。参考仍原同硬件逐步max，短窗口仍高约24%，110%未完成，不能套旧SXM绝对值或称低于计算参考。[本轮报告](mla_host_a100_compute_schedule_20261103.md)。

原唯一待续square_smem已完成：c69a2643 normal/EMU构建成功，REG255/STACK0，两项检查通过，100GB0/20慢3.668%/4.383%、50GB慢0.163%，拒绝。首编译失败件仍保存，最终build记录已正确绑定c69a，square家族不再重启。另新pv_rescale将原O缩放交错到首PV MMA，2项检查通过、REG254/STACK0，但100GB慢0.329%/0.141%、50GB持平，拒绝。此接续共6项成功定向检查、24格12对/4608计时launch/288段，另1实际Graph probe；所有raw/正负差/失败记录保持。全部build/run/confirm工具session均terminal，GPU3结束0MiB/0%；GPU0/1仍其他VLLM占用，GPU2不动。math34af/FAR5e409/原5304矩阵、H800队列源码均保持。后续从cab508dc继续真正新机制，不重跑本轮完成的筛选，也不将静态指令/栈减少当性能因果分解。

以下11/03上午迁移段落及更早条目是历史，生产和活动任务以上述新入口为准。

**11/03上午迁移历史：**用户希望新机制逼近100GB下逐步参考110%，也希望探索能否低于原“理论下限”；允许优化指定本地计算参考所对应的计算实现，不允许改模拟器/参考/实际远端载荷制造突破。GPU3空闲后已完成旧shared_claim有效补测，慢5.72%/6.65%/3.70%，拒绝，不再恢复旧SXM不完整runner。新anchor（8单位上界锚点）修正单元素LSE取整问题后3项通过，100GB0快0.29%/20慢1.67%，拒绝；原失败稿和容差保持。新square32×32warp内Q/K双复用，2项通过但100GB慢4.54%/5.08%；直接发布P的square_p也2项通过，慢6.94%/8.11%，均拒绝。normal/EMU square REG255/STACK16，square_p REG255/STACK88，皆static2304，不称零spill。今日24格12对/4608计时launch/288段，新增7项定向检查、复用旧2项，无新全套/sanitizer/原实例全等价/实际Graph节点probe/物理链路/FP16/SXM/H800性能结论。

**迁移时唯一待续为square_smem**，目录`temp/glm_trace/tune_qk_square_20261103/`：仅将square_p的32个长期QPE packed words换成读取已有shared QPE，针对实际STACK88，不能预断定收益。首30e7稿编译失败是新bool square_smem与旧控制constexpr square_smem重名，现已改成square_smem_enabled并同步header/include/generator。最终候选SHA **c69a2643f887b642471c621cddd318e715a92e292c3ed1b35341ecfc6a24aeb2**；**修正后未编译、无GPU验证/性能、未安装**。首失败件在failed_smem_dispatch/；build_square_smem.json仍记录旧30e7失败，不能误认为最终头编译过。用户要求迁移后无新构建/GPU任务。

重新预检GPU3后用指定Python先执行`build_smem.py`，成功再`run_smem.py`（两项定向检查＋smem256_三点新旧短对照），最后`audit_performance.py`；如考虑保留，再确认新分派/实际资源/单kernel，别重复旧大门槛。不要重跑make_smem_pe.py/fix_smem_dispatch.py等不可重复构造器，不恢复已完成旧runner。所有本轮handles已terminal，ps确认无活动runner或等待器；精确状态在`migration_state.json`。11:11GPU3 0MiB/0%，GPU0/1外部VLLM约38GB、GPU2外部服务，不动；NUMA1/共享锁继续。

GPU3当前前32步参考核对：24步计算主导、8步传输主导，逐步max2699.864µs，当前base3347.701µs，110%2969.850µs，需再降11.29%。hypothetical计算快10/20%对应参考2556.862/2497.469µs，仅代数分析，不是实测/替换参考；更快计算可能低于指定device实现参考，但不能突破真实必需字节/带宽约束。`bound_audit.json`绑定原件。SXM全窗口约3.29/3.31ms对2.691ms属于另一硬件/窗口，勿混拼。

H80011/03 11:12:58只读查：1004362/1004363 COMPLETED0，1040680仍FAILED1（CUDA静态库缺失，未重试）；1089379 PENDING/Priority、无依赖、估计11/05 00:49:44不保证，1089380 PENDING/afterok:1089379；watcher1174145活、4160646已结束。两排队cwd仍tuned_20261029，日志生产树temp/h800/logs，不含当前优化，未覆盖或重复提交。非login ssh无Slurm PATH，使用bash -lc或/rmprog/slurm/v24.05.1/bin。完整查询在`h800_login_query_20261103.txt`。

生产7547、canonical数学34af、FAR5e409及原5304矩阵保持；整体目标未完成，未声明新收益。最新报告[11/03机制与迁移](mla_host_a100_innovation_20261103.md)，上轮[110%报告](mla_host_a100_overlap110_20261102.md)为历史。本轮任务源码/数据已保留，其他用户dirty文件不动。

**20:00当前活动轮：用户要求更大优化，将100GB下staged/direct最优压到逐步参考110%以内。** 已完成两遍QK/PV与全局Q＋64KV双stage的GPU3/NUMA1定向筛选，均通过2项检查但100GB分别慢56%–57%和43%–44%，拒绝。新elastic_fetch经856CPU队列覆盖及normal/EMU构建；GPU3预检19:48遇外部525539，在CUDA前exit1；19:50发现GPU0/1释放，取得gpu01锁后转SXM/NUMA0完成2项定向检查和256×3三对，100GB0基本持平、20慢5.964%，也拒绝，不扫描退休边界。三版源码/原始结果及失败记录均在各`temp/glm_trace/tune_{two_pass,global_query,elastic_fetch}_20261102/`。6项定向检查/18格9对，不重跑整套测试/矩阵。生产754706f8保持，原方案不要重启。

**20:08最新：shared_claim构建和两项定向检查通过，候选性能仍待有效测量。** 同pool最多4行共享hash先去重，再全局CAS，owner临时page ID放在LRU前未用的entry_victim/victim_of，原dense rowprefix resolve再复制，原fetch/math/LRU/FAR保留。`tune_shared_claim_20261102/`，最终候选a47ca8b8；首CPU稿claim_entry会alias slot_pool，GPU运行前已纠正并完整存pre_gpu_alias/，不能运行首稿。handles45564/87149均编译terminal0，35058验证父在最后候选benchmark处exit1：新SXM三格base已完整，候选100GB0µs测完后资源检查发现外部581553/581554进入，raw complete=false，3440µs不得当有效回退数据，其余两候选格未运行。2项检查加入不等长pool/跨pool重复页/padding/全有效cache载荷和output/LSE/Graph，通过；source_review确认attention/fetch helper规范化命名后逐字同生产及scratch生命周期，未重复大验证。

所有本轮runner已结束，无后台等待器，不重启旧run_sxm.py或重跑基线。20:08 GPU0/1被VLLM585473/585474等占用各34495MiB，GPU3外部smoe546178仍占，GPU2不碰。后续重新预检，空闲后仅执行shared_claim的resume_candidate.py补候选三格，原不完整raw会自动归档，再audit_performance.py；同范围无需许可。当前有效21格/9对/4032计时launch/252段、共8项定向检查，另1不完整干扰格保留。canonical数学34af、FAR5e409、生产7547、原5304矩阵未改，无本轮正常链路/H800结果。110%尚未达到；已有SXM全窗口需由3289.824/3306.973降至约2960µs，仍需约10%/10.5%总时下降。[本轮报告](mla_host_a100_overlap110_20261102.md)和shared_claim/continuation_state.json为当前状态。

**19:02澄清：**用户追问batch重叠与LRU开关。当前无LRU10/30%初始远端不是LRU几乎全远端trace的开关消融，不再横向声称LRU导致差距扩大。相同R128trace实际unique672.920→259.582MB/步（引用命中78.94%、unique命中61.42%），100GB理想参考6.729→2.691ms，缓存实测3.290ms；参考变严不等于总时变差。100GB的R16→R128gap实际34.6/40.4%→22.2/22.9%，NVLink则10.9%→22.3%确实待优化。`lru_same_input_20261102/existing_accounting.json`是CPU核算，不是容量0timing。两点同输入capacity0补测runner66610在资源预检遇他人GPU0/1flashcheck，停止仅自有CPU waiter289478（无CUDA/无raw），父289459exit1；无活动runner、不抢占。准备脚本和状态保存，按具体疑点再续，不得把尚未运行补测当结果。生产754706f8保持。

**最新路由核对：用户要求两条路径按场景择优，并追问历史接近参考的记录。** `temp/glm_trace/route_select_20261102/`已完成当前生产754706f8、SXM0/1真实NV12/NUMA0的30格/15对。源码及既有正常/EMU BF16库SHA都保持，只有host_stage开关；MTP8/h64/K2048，LRU16/128请求完整1024位置×3冷pass，另无LRU16请求四原快照×0/10%/30%×3pass，正常NVLink及100GB0/20µs。16请求f16/split on，128请求f24/off，不做参数扫描。

择优结果：NVLink/R16/LRU4096 **direct557.725µs vs staged622.968，参考502.995，高10.88%**；NVLink/R128/LRU **staged3016.416 vs direct3335.189，参考2467.179，高22.26%**。当前大staged已有后续优化，不能再套旧“NVLink总选direct”。100GB0/20/R128/LRU均选staged3289.824/3306.973，距逐步参考22.24%/22.89%；R16/LRU仍差34.55%/40.37%，没有切路由就解决。R16无LRU远端10/30%，NVLink择优差1.87%/7.24%，100GB两延迟差13.38%–18.76%；0%负差控制也保留，不截断。原历史NVLink约7%是16请求、后半段steady与最快device样本参考，history_review.json绑定原件，不能拿它代表128请求LRU，也不把不同输入/配置/聚合口径直接当新旧回退。

`select_route.py`+`selected_routes.json`已提供启动/warmup/Graph前按实测键配置host_stage/fetch/split的选择器，仅适用测过的硬件/形状/LRU/链路/比例，未覆盖明确报错。不在计时内做选择，不在线逐launch切换已活跃cache/Graph，不擅改公共wrapper默认或全局NVLink路由。短K97/容量128两选中路径各3轮、每轮Graph两次replay，采样output/LSE和全部有效cachepayload/映射通过；性能回放实际LRU4096且原miss/命中诊断/输出检查保持。首验证pytest导入缺失发生在GPU工作前，原失败源和错误说明保留，改为抽取两个既有fixture辅助函数后只补该项。

9648计时launch/684段，全部逐步逐pass/慢路径/原始字节miss/源码SHA/两同硬件参考保留；15配置最小全样本均值是实测路由选择，不是新CUDA机制收益或普适最优承诺。原5304矩阵未改，未做整套pytest/sanitizer/大矩阵。原runner71476、首验证59366、修复56797均terminal（首验证1，其余0），无活动任务；GPU0/1此次空闲窗口已完成真实SXM验证，后续仍重新预检。生产仍754706f8，无CUDA改变。[当前择优报告](mla_host_route_selection_20261102.md)。整体大batch100GB约22%–23%当前SXM差距仍未完成，不用GPU3绝对时间混拼。

**最新生产为QK地址游标实测原件，SHA=`754706f8d8fba14a361ce6dcfc64a358446e5df36b225f392fa8a3e7756ded79`，已安装。** 从上轮SS48c2逐字节冻结基线继续，不回退到CPE/JQ。新目录`temp/glm_trace/tune_qk_cursor_20261102/`（QC）；paired CKV QK先计算Q/K起点，随后用已有xor/add推进swizzle地址，保持原unroll4/加载与MMA次序、QPE寄存器、joint MDO/PV、远端源hint、精确launch LRU、载荷/字节/FAR/ready/barrier和单kernel。16384项整数地址等价检查通过。

QC只做4项BF16定向检查（长queue/128池、容量512/4096、稀疏不等长池/空/奇尾、stride80、output/LSE/Graph/全部有效cache载荷），未机械重复旧全面门槛。EMU/normal各38个原BF16实例机器字/控制字/资源保持；目标REG255、STACK16、LOCAL0、static2304，EMU原STACK8/normal原0，不称无栈。实际Graph仍1kernel/108CTA/block32x4x2/dynamic147968/active1。短256×3 base→QC，100GB0/20快0.902%/0.611%、50GB20快0.041%；完整1024×3反向QC→base，100GB0 **3464.883→3413.173µs（1.492%）**、20 **3483.117→3438.675（1.276%）**、50GB **5502.875→5499.211（0.067%）**。正常256×3反向同卡HBM3157.451→3151.659（0.183%），实际HostPCIe9532.171→9531.669（0.005%，持平）。均GPU3/NUMA1/MTP8R128/LRU4096/f24off；同卡HBM不是NVLink。16格/8配对、6528计时launch/408段，两同硬件参考/所有逐步逐pass与正负数据保留。`retention.json`及snapshot绑定实测安装原件，没有FP16/SXM/NVLink/H800新验证、全矩阵、sanitizer重复或生产root再编译/测试声明。静态总指令/IMAD反而增加，不把源表达式减少当性能原因。整体100GB仍高于逐步参考约24%–25%，目标未完成。

同轮两个新机制已拒绝：`tune_cache_store_20261102/`只改LRU payload为st.global.cs，3项定向检查通过但100GB0/20慢0.017%/0.908%、50GB慢0.018%；`tune_query_copy_20261102/`只去掉item已不再读取的sharedQPE上传，4项定向检查通过但三点−0.316%/+0.135%/−0.073%，没有明确收益。两者都保留原件，不安装、不与QC组合、不扩大矩阵或扫数字。本轮共11项定向测试、28格/14对，失败/负差均保留。[机制及H800报告](mla_host_a100_followup_20261102.md)。

QC原handles4833/85090/31860/19193/57173、前两个候选的所有runner均已exit0，无本任务活动runner，不重启。GPU0/1仍他人flashcheck服务占用，GPU2不碰，GPU3最后空闲；后续GPU预检及共享锁保持。继续从754706f8实现新机制，用户轻量迭代偏好持续有效。

**H800新状态（11/02 15:14只读查询）：**1004362 COMPLETED0（02:33–02:41）、1004363 COMPLETED0（14:43–14:47）、1040680 FAILED1（02:49，nvcc缺libcudadevrt/libcudart_static，未进入GPU实验）；1089379已解除依赖但PENDING/Priority，估计11/05不保证；1089380 PENDING/afterok依赖。原watcher4160646自然结束，1174145仍活，不重启旧watcher。原host10份+peer4份JSON和环境/脚本均下载并绑定SHA在cache_store/h800_archive_manifest.json；实际旧header29667eaf、四步旧trace/steady末半段口径，不能当当前优化/正确Memcpy或assistant1024结果。旧host大batch输旧Memcpy，旧peer大batch有正收益，均保留，不能替代当前版本H800验证。未覆盖排队源码、未重复提交。bulk失败已定位库搜索路径，不改共享CUDA shim或原queued脚本；尚未重试。

以下是上轮及更早历史状态，生产/活跃任务以以上最新入口为准。

**上轮生产为SS实测原件，SHA=`48c2a724548c2ac554400b26592a1212e159cd279cd1a48da191b5139cd86f73`，已保留。** SS=`temp/glm_trace/tune_stream_source_20261101/`，直接基线为已保留CPEa436；原CPE冻结为SS/base.cuh，不要回退。唯一机制为dedicated fetch去重far-source cp.async的L2 evict_first固定fraction1.0提示，attention/staging/cachestore/元数据策略及数学/精确LRU/实际字节/FAR模型/ready/claim/barrier完全保持；九个attentionhelper逐字等价、fetch_policy.diff隔离两处hint设置/调用。实际三构建各36条带desc的LDGSTS，描述符高字0x12f00000/低0，非动态cache命中或因果份额。原38EMU/76normal实例全部指令/资源/机器字与控制字相同；新REG255/EMU STACK8/normalBF16FP16 STACK0/static2304/LOCAL0、实际单108CTA/256thread/dynamic147968/active1 Graph。115项数值/全载荷/LSE/Graph测试和首BF16/K97三sanitizer均通过。

SS关键对照GPU3/NUMA1/MTP8R128/LRU4096/f24off，短256×3：100GB0/20快0.416%/0.128%，50GB慢0.021%；完整1024×5反向100GB0/20 **3478.936→3455.752、3509.139→3481.059µs（+0.666%/+0.800%）**，正向 **3480.251→3463.062、3500.221→3486.115（+0.494%/+0.403%）**，50GB两序快0.015%/0.080%。正常HBM3247.563→3236.062（+0.354%）、真实PCIe9967.752→9951.293（+0.165%）。22格/11配对、两参考/全部正负差/实际miss/命中/逐步逐pass/guard/源码与binary SHA保存，[SS报告](mla_host_a100_stream_source_20261101.md)。GPU3同卡独立far HBM不是NVLink；没有本轮SS SXM/H800结论，不拼不同时轮CPE/JQ绝对时间。

**18:32用户明确要求减少逐轮验证后，停止SS尚在等待的CPU矩阵父58746/PID3545628，terminal143；当时无子进程、无矩阵/SXM启动，没有信号给GPU任务或他人。** scope_change.json及原state保存；已经进行的native对照自然完成。原验证55492和完整确认64042均terminal0，compile19576/test83290/stress45910亦terminal0。按新偏好，用已有115测试/38-76等价/实际路径/关键完整两序与native证据保留**逐字节实测48c2头**，focused_retention冻结原件并绑定安装SHA；没有重复新生产root编译/整套测试、没有156矩阵/SXM。这个局部改进可继续迭代，不能再机械套旧全面门槛。所有SS runner已结束，详continuation。canonical math34af、模拟器5e409、原5304矩阵e3fd保持；整体目标未完成。

**后续默认：每机制用针对性正确性和少数关键新旧对照，复用既有证据，发现疑点才增测；不再逐轮105+stress/三san/full两序五pass/156矩阵。** 本文下方及旧runner的全面验证要求是历史，不得据此增加新用户未要求的大验证流程或反复询问。当前未发现待处理GPU故障，无需重启任何LO/SS/SP旧runner。

**新LO=`temp/glm_trace/tune_local_order_20261101/`已实现并拒绝，生产CPEa436不变。** 头7f891959…0d6d1，只在原CPE大cached fully-staged且1<maxK≤2048派发新typed实例。whole-row item成功claim后、load_q前，用动态shared做resident/near尾部bitonic排序；miss前缀和prepare/resolve/LRU准入/淘汰/fetch/flags/payload/grid协议全部原样。query的KV多重集合保持，在线FP32/P量化分组随顺序改变，不声称逐位相同。原38EMU/76normal指令/资源/全部机器字及控制字相同；新REG255/STACK8/static2304/LOCAL0、dynamic147968/108CTA/256thread/active1。正常105+12=117项通过（新增K129逐行解码全部remap multiset/重复页/padding/尾部有序/命中槽保护/全部cache载荷/LSE/Graph），实际新单kernel和首BF16/K97三sanitizer均通过。

LO实际CPE正常构建观测连续16epoch、MTP8R128/LRU4096，采样epoch0/7/15×每8请求中1个，共48组remap。output及全部有效cache载荷检查通过。原8query/32页对齐窗口的总页集合366714→261335（少28.736%），references389557保持；只是CPU局部性代理，非实测L2事务/命中/stall/性能。实际固定GPU3/NUMA1/CPE→LO/256位置×3冷pass/f24off：100GB0 **3411.584→3883.040µs（−13.819%）**、20 **3419.467→3903.851（−14.165%）**、50GB20 **5269.845→5283.712（−0.263%）**。排序/同步/global回写的组合未取得收益，不能据代理改善声称加速或精确归因开销份额。6格/3对/1152计时launch/72segment、两参考和全部正负raw保存，[LO报告](mla_host_a100_local_order_20261101.md)。handles5272/81111/1240/85028/63486/59927均实际terminal0，确认rejected_short_screen，不扩full/native/SXM、不扫排序数字。

18:08 H800只读复查原五任务仍Priority/Dependency、两个原watcher活、AGENTS/MEMORY91b070/e3df236不变，无新提交/排队源写入。后继SS=`temp/glm_trace/tune_stream_source_20261101/`正在独立编译/验证，头48c2a724…86f73：仅给dedicated fetch的去重far-source cp.async加L2 evict_first hint，未改attention/staging/cache的策略或任何数学/LRU/FAR字节/同步/参数。原源页每launch去重后只读一次，attention读staging/cache，因此测试其cache污染假说；不预承诺收益。实际父/handle见该目录continuation和ps，不能从cachehint或静态代码推断改善。

**新SP=`temp/glm_trace/tune_sparse_prepare_20261101/`已完成拒绝，生产仍CPEa436。** 冻结头f998f831…37ec，直接以当前CPE生产为基线。临时batch pool有从pool_row_start×maxK起始的有界稀疏staging区间，warp一次分配pool内rank，先发布自己全部8候选中的owner，再等待其他owner的物理slot；prepare直接写负staging索引。后续pool计数扫描替代遍历全部远端reference的resolve，dense fetchjob二分映射物理槽；保留原LRU/fetch重叠、shared→staging→cache双写、16/64行选择，不增workspace/kernel/grid barrier。派发额外要求maxK正且整除256，保证prepare_parts1/segment=maxK/区间对齐；其他原路径literal不变。stamp精确LRU选择及current-hit保护保持，但pool内miss admission顺序从row-rank变为warp到达，不宣称各次cache状态相同。

SP正常105+10=**115项**通过，新增不等长pool/零miss前缀、K256/padding奇尾/淘汰/全部payload/LSE/Graph/QPE stride80；原CPE38EMU/76normal指令/资源/机器字与控制字全部相同。新EMU及正常BF16/FP16均REG255/**STACK8**/static2304/LOCAL0，不称零栈。实际新kernel单108CTA/block32x4x2/dynamic147968/active1 Graph和首匹配BF16/**K256**的三sanitizer零错；不是K97或全trace穷尽证明。CPU72case核对给定poolcounts下物理区间、job二分/16-64行flags和实际FAR行数；八个mathhelper规范化命名后与CPE逐字相同。首未编译eea2稿固定64job，编译前修为原16/64选择，旧header/generator/manifest保存在precompile_review/；两个CPU观察Path(.)路径错误记录，无GPU影响。

SP固定GPU3/NUMA1/CPE→SP、MTP8R128/LRU4096/f24off、256位置×3冷pass：100GB0 **3410.709→3410.336µs（+0.0109%）**，100GB20 **3431.851→3455.925（−0.7015%）**，50GB20 **5270.219→5106.923（+3.0985%）**。实际unique从约215243–215268/步降到207692–207714/步（约少3.5%），reference命中约76.73%→78.38%，50GB收益包含admission顺序变动的流量影响，不能归为纯准备优化。两参考仍按原公式和各自实际unique计算，全部正负时间/命中/逐步逐pass保留；不是修改模拟器或参考。

**SP原验证54327/PID3378954和确认10845/PID3393485均已实际terminal0**；确认state=rejected_short_screen，预设两100GB均需正收益门槛失败，未扩full/native/SXM/矩阵，不扫pool/chunk/fetch/spin参数。编译5533、正常测试21723、stress40797也均terminal0；GPU3曾等待外部3382102/3390931自然结束，未终止他人。6格/3配对、1152计时launch/72segment、两参考及合法原阈值reference诊断齐全；不同于此前destination坏诊断，SP负索引仍指staging区。报告[SP](mla_host_a100_sparse_prepare_20261101.md)、decision/checkpoint/continuation保存证据。所有本轮自有runner已结束，主目标仍active；不要重复本轮SP或destination/claim/HR/QP/QS等已拒绝机制。

**当前生产已升级为实测CPE原件，SHA=`a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15`。** 原JQ7193已冻结为生产前基线，不要回退。11:23:52保留父28985/PID2351788实际terminal0，补矩阵父36222/PID2351762亦terminal0：156矩阵完整、GPU3/SXM完整1024×5pass两顺序及正常链路确认通过；先冻结审计再逐字节安装a436，没有未测清理。生产root独立jit_production_cached_pe_normal正常BF16/FP16重建、**111项测试、76个全部原与新实例指令/资源/机器字及控制字、实际单108CTA/block32x4x2/dynamic147968/active1 Graph均通过**。[CPE最终保留审计](../temp/glm_trace/tune_cached_pe_20261101/final_retention_audit.json)、cached_pe_retention和completion_resume_state绑定原件SHA；17:11再次核对通过。canonical math34af/模拟器5e409/原5304矩阵e3fd保持，整体goal active。

CPE共**200格/97配对、407968计时launch/25552segment**，两参考、逐步逐pass及实际命中率齐全。60改变路径配对未加权降低均值0.454%、范围−0.225%至+1.619%，负差原样保留；原两个未改HBM>1%负差的完整夹测均未复现，原慢格不删。SXM完整100GB0/20反向较JQ快1.732%/1.869%，正向快0.965%/0.899%；正常实际NVLink3159.205→3079.042µs（快2.537%），PCIe9950.082→9942.459（快0.077%）。这些是同轮对JQ的收益，不能跨轮拼绝对时间当组合收益。SXM正向仍高于逐步max24.94%/25.67%，离目标仍远。

CPE原15217/PID2084930与第一保留47513/PID2331207均terminal1：最后native矩阵开始前，外部GPU1 PID2344687进入，在run_setting资源预检退出，未创建该次raw/计时，生产未动。原状态/全部失败log保留resource_preflight_failure/。确认终止后只用新36222补原计划缺失20格，日志/state追加，保留父28985以独立resume状态等待，外部任务自然结束后完成。**全部CPE runner已结束，不要重复启动15217/47513/36222/28985。** broader_validation_state和completion_state保留失败历史，当前完成状态看matrix_resume_state、completion_resume_state和final_retention_audit，不因旧failed误判当前生产。

**新预解析目标槽 `temp/glm_trace/tune_destination_20261101/` 已实现并拒绝。** 头3c744e…a7749，以CPEa436为直接基线；提前精确LRU、淘汰oldresident并将slot_page设pending，额外两grid barrier间把现有32bit miss remap改为最终cache槽或保留staging。原f24/64row双缓冲真实读后每页只写一个最终位置；全部256writer fence+CTAbar后release发布cache slot_page，tile wait acquire。原perKV地址解析器保持，无旧direct_cache逐KV查victim依赖；额外LRU/remap/同步开销明确，非单一省写因果证明。

该候选113正常/长cache-payload/LSE/Graph测试通过，原CPE38EMU/76normal指令/资源/机器字及控制字全同，新REG255/STACK0/static2048/dynamic147968，实际单108CTA/256线程/active1与首BF16/K97三sanitizer通过。GPU3/NUMA1/CPE→candidate、MTP8R128/LRU4096/f24off/256×3pass：100GB0/20慢3.773%/2.864%，50GB20慢1.916%，停止，不扩full/native/SXM或扫LRU/barrier参数。6格/3配对、1152launch/72segment，全部原件保留。

destination原父33355实际terminal1发生在**计时完成后的审计**：旧reference-hit诊断将remap低于cache_slots当命中，新候选的获准miss也落在cache槽，因此漏算，原报告reference命中率无效。审计已将该字段置null并保存原报告值/理由，实际unique counter、时间、两参考仍独立核验；不能用坏诊断宣称高命中。追加正常四步冷/热/淘汰/热的双版独立诊断证明冷miss41904引用/97unique、热0/0及全部载荷，旧阈值只在新候选漏计。首诊断脚本因cache backing尚未lazy初始化而在读views处退出，失败源/log保留，加入eager初始化并清空后两版通过；没有改kernel或回写原raw。其余原build/test/stress均terminal0；详细状态见continuation，勿重启。

17:14 H800只读复查仍原五任务Priority/Dependency，估计11/05；原两个watcher4160646/1174145活，AGENTS/MEMORY哈希91b070/e3df236不变，无新Slurm提交或排队源写入。下面所有“CPE尚未安装/正在等SXM”均为历史快照，以当前生产及最终保留原件为准。

**最新claim实验 `temp/glm_trace/tune_claim_probe_20261101/` 已完成，两版均拒绝，生产7193不变。** 直接基线是当前生产7193，不是尚未采用的CPE。claim_probe头95ff4054…f5c1c，仅prepare先GPU scope relaxed原子load，非零跳过CAS；claim_store头15aa954a…c81e保留全部原CAS，仅唯一owner发布编码row/rank时用relaxed GPU原子store替代无返回值atomicExch。0→1→编码及原grid阶段保持；规范化新kernel名/prepare调用后，整个cooperative生命周期与JQ逐字相同。数学、LRU/FAR/payload、256线程/108CTA、147968dynamic/2304static/32KV双stage/f24双fetch保持；每版REG251/STACK0/LOCAL0，原36EMU/72normal机器字/控制字及资源全同。

两版各105+8，共226项独立正常测试通过，新增两项覆盖跨CTA全共享97页/互斥2048页及跨launch清空；全部cache载荷/LSE/Graph/K2048及K65 stride80保持。每版实际单108CTA/block32x4x2/active1和首匹配BF16/K97三sanitizer均通过。初generator边界断言在header生成前失败；新增两stress漏Graph预warmup导致接口拒绝，修复后8项全过；validator抢先读取旧失败state的一次立即退出亦存档。原105+6的重复成功不重复计数；原失败source/log/state均保存test_fixture_failure/，包括nested失败log的gitignore已修正。

GPU3/NUMA1/MTP8R128/LRU4096/f24off短256×3：probe在100GB0/20慢2.091%/1.481%、50GB20慢0.961%，停止。store短快0.262%/0.434%，full反向1024×5快1.010%/0.704%，但full正向慢0.291%/0.058%；50GB两序快约0.1%，正常HBM慢0.123%、PCIe快0.055%。**原扩测父34409/PID2305614已实际terminal1，预设正向100GB收益gate拒绝，所有测量子步骤exit0；不重启、不放宽gate、不扩156矩阵或SXM。** 28格/14配对、22784计时launch/1424segment、两参考/逐步逐pass/命中率/全部正负差齐全；checkpoint_verified，完整矩阵明确未做。[claim报告](mla_host_a100_claim_atomics_20261101.md)、decisions及continuation保存证据。静态CAS/EXCH/load/store数只证明操作变化，非动态次数或因果分摊。

**CPE原父15217/PID2084930在外部2101252/2101253自然结束后，已做完SXM反向/正向/正常完整确认，现phase=sxm_matrix。** 100GB0/20反向快1.732%/1.869%，正向快0.965%/0.899%；50GB近持平、正常真实NVLink快2.537%、PCIe快0.077%。原子验证器2304430运行limited60格，前R64/R128已完成，MTP1R512进行中，然后自动native20格；原父会刷新审计/报告但不安装。先查15217和实际ps，**不要再等已结束外部PID、不要重复启动**。完整156矩阵/全部负差门槛和冻结保留后，还要新的生产root编译/测试/代码及实际Graph验证；不能仅凭这些正信号安装。09:06 H800只读查询仍原五任务Priority/Dependency、两个watcher活、AGENTS/MEMORY哈希不变，无新提交/排队源写入。

下面旧“CPE等待、无SXM数据”是当时时间快照，以上述最新进度及实际原件为准。主目标继续active，100GB大batch主差距仍未解决。

### 2.0 最新生产：JQ已保留；QPE全item片段复用验证中

**当前生产header已升级为JQ等价清理源，SHA=`7193255e2fdf83d3baa9347dea1a02b892c29ed746da719818e30fdf4066d469`。原b6ee…e72a是本轮基线，不要按下面旧快照回退。** 实测JQ源a2f37…c4b7c保持冻结；清理只改host局部别名，全部36EMU/72normal实例指令、资源、机器字/控制字全相同。canonical math34af、模拟器5e409及原5304矩阵e3fd均不变。

原JQ确认72075和自动保留1398均已实际terminal0，原PID1640200/1693759已结束。GPU0/1旧外部1780064/1780065自然结束的窗口完成最后JQ三格；156矩阵、full两顺序/正常NVLink-PCIe三方确认和全部门槛通过后，1398逐字节安装7193，再从生产root新建独立jit_production_joint_normal编译正常BF16/FP16，109项测试、72实例代码核验及实际单108CTA/block32x4x2/dynamic147968/active1 Graph全部通过。**安装和生产验证均完成，不要再等待或重启这两个handle。** [保留记录](../temp/glm_trace/tune_joint_qk_20261101/joint_qk_retention.json)、[最终当前状态审计](../temp/glm_trace/tune_joint_qk_20261101/final_retention_audit.json)及audit_final_retention.py绑定root SHA、测试/代码/Graph原件。completion_state.json保留循环最后一次missing/live旧字段，但phase/complete已完成；最终审计用实际raw/coverage/ps明确missing为空、原producer无存活，勿把旧字段当未完成。

JQ共210格/110个对原b6配对，156矩阵完整；60改变路径配对未加权降低均值1.602%，范围−0.188%至+5.253%，全部正负差保留。SXM完整1024×5pass：反向100GB0/20较b6快5.228%/4.886%，正向快3.839%/3.948%；正常NVLink较b6快3.683%，但较JM慢1.530%，没有隐藏；PCIe较b6快0.051%。[JQ报告](mla_host_a100_joint_qk_20261101.md)和主报告§7.9已改为已保留，原5304格/图1–6不改。当前100GB仍高于逐步max约25%–27%，整体goal active，不能标完成。历史performance_audit/源manifest里的candidate_adopted=false是测量时快照；当前采用状态以retention和final_retention_audit为准。

**后继CPE=`temp/glm_trace/tune_cached_pe_20261101/`正在验证，尚未安装。** 头SHA=`a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15`，基线是冻结JQ a2f37（与当前7193代码等价）。每item把完整64特征QPE按原MMA片段读入每lane16个uint32，成对和奇尾KV tile全程复用；保持PE→CKV MMA、jointMDO/PV/refill、256线程/64head/2stage/CTA32/f24及LRU/模拟器。原QPE共享分配/初始copy仍在，隔离复用机制；区别于N9每tile重读QPE和9warp服务角色。原JQ36/72实例机器字全同；新REG255，EMU STACK8、正常BF16/FP16 STACK0，static2304/dynamic147968/active1，实际单kernel和synccheck/racecheck通过。SASS LDSM198→190、LDG85→101、HMMA299/BAR58不变；R12–R27在成对循环前加载后不覆写，循环内无LDL/STL，整kernel的8B stack仍如实记录，不以静态数当性能证明。

CPE通过105+6=111正常测试，新增K2048/63tile奇尾、128池/512和4096容量/空单元素/padding/LSE/全部cache载荷/Graph，以及大形状K65/QPE head stride80。GPU3 MTP8R128/f24off/full1024×5：反向100GB0/20较JQ快1.471%/1.323%，正向快0.842%/0.598%；50GB快0.058%/0.138%；正常HBM快1.315%，物理PCIe快0.019%。GPU3两矩阵76格已完整；十个100GB矩阵配对快0.725%–1.363%，50GB接近持平，R64 PCIe慢0.225%也保留。两个原路径HBM负差MTP1/R1慢1.133%、R256慢1.078%，已追加完整5pass前/候选/后：R1为41.616/41.637/42.203µs（较两端均值快0.650%），R256为1061.371/1067.660/1063.863µs（慢0.475%）；均未复现>1%，原负差不删。当前104格/49配对、175520计时launch/11024segment、两参考完整。[CPE报告](mla_host_a100_cached_pe_20261101.md)、controls_audit与checkpoint记录完整边界。

**当前唯一CPE父runner为15217/PID2084930，validate_more.py；子screen PID2104175仍live，等待新的GPU0/1外部2101252/2101253（checkfreq_formal_runner.py）。** 旧JQ/外部进程都已结束，不要误等旧PID。15217已做完本地76格及刷新；会继续SXM反向/正向/正常NVLink-PCIe共16格full确认，全部过门槛后再做SXM80格矩阵，最后刷新审计/报告。它不会安装。先查该handle、broader_validation_state和实际ps，不要重启或重复启动。SXM没有CPE性能数据，不能拿GPU3独立HBM当NVLink或跳过MTP1R512。111测试/本地通过不能替代所有SXM门槛和新保留root验证。

CPE其他handles95227/36824/61566/49401/66271/66530/62457/40637均terminal0。最初generator签名空格断言在生成候选前失败；编译前又修正const数组引用与原MMA签名不兼容，未编译28d149源/manifest和说明都保留。只有a436被编译测量，没有GPU失败或重试。当前生产7193已独立验证，不要安装未完成的CPE。

CPU-only真实trace抽样另核验简单prepare交错：原row32重复率0，2row×16col约1.355%，8row×4col约1.734%，不是性能或实际atomic节省；不值得按这个简单布局直接实现/扫参数。followup_resource_review.json的五个16row attention半buffer现已具体实现并测量，结论为拒绝，详见下段HR；该旧分析中“未实现”的状态已过时。

H800本轮07:29只读复查原五任务仍Priority/Dependency，两个原watcher4160646/1174145 live；AGENTS/MEMORY哈希不变，CPE/h800_context_hashes.txt、h800_queue.txt及带最新哈希/时间的h800_queue_latest.txt保存原件，无新Slurm提交、排队源码或远端临时目录写入。

**本轮新HR=`temp/glm_trace/tune_half_ring_20261101/`已完成，五half attention环和恢复双fetch缓冲的copy控制均拒绝。** 直接基线为冻结CPEa436（仍未采用），不是生产7193。五个16row物理half组成两个32KV tile和一个提前half，hi/lo PV释放后旋转，提前加载仅near/resident/staged HBM，host chunk等待点、f24/64row双fetch缓冲、真实LRU/FAR post-read hooks及单kernel均保留。私有P half映射`uint4_offset=row*2+(column^((row/4)&1))`，保持量化P、rowsum、PE→CKV MMA/降序PV；半环dynamic158336/static2304、REG255/STACK32/LOCAL0，实际单108CTA/block32x4x2/active1。每版本原38EMU/76normal实例全部指令/资源/机器字/控制字相同。

HR首版no_copy_pe移除未用sharedQPE后，sizeof storage139904让通用fetch **安全地自动选择单缓冲**，不能隔离QPEcopy移除。最初以为越界的假设已明确撤回，原错误判断存档；memcheck首次25错中打印的16项均cuGetProcAddress_v2 API查询错误，其余9项未打印，exit86/log保留，关闭API错误报告但保留设备检查的复查零错。新冻结no_copy_pe_fixed **7ff39d…f6e53** 用union padding恢复原sizeof/dynamic147968和fetch双缓冲，不能只改host allocation。新控制REG255、EMU STACK8/normal STACK0/static2304。原控制a051与半环0c444保持冻结，首轮九格完整移入历史名invalid_budget/raw/，实际分类为意外单fetchbuffer策略，不宣称内存越界；archived_policy_audit使用归档后路径验证，原半环100GB约19%负差保留。

修正后固定base→fixedcontrol→half_ring、GPU3/NUMA1、MTP8R128/LRU4096/f24off/256位置×3冷pass：100GB0/20µs半环相对CPE慢**18.520%/18.334%**，50GB20慢1.018%；copy控制慢0.258%/1.044%，50GB快0.0067%，没有采用信号。修正后9格6配对/1728计时launch108segment，两参考完整；首轮另9格6配对/1728launch108segment，不能删负差或把直接CPE收益写成生产收益。三版本各105+6，共333独立数值测试；半环测试跨两轮复用只计一次。新控制和半环实际Graph/synccheck/racecheck/memcheck全部通过，sanitizer限定首匹配BF16/K97launch，不是全部trace证明。32B栈/packing/loader/地址控制是可能开销，未测得各自因果份额。

HR所有自有handles **37515/15219/66285/39156/28263/90521/96143/77771/81297/12596/80121/62278均已terminal0**；首memcheck23782为terminal86且保留。不要重启任何HR runner，不扩full/native/跨形状、不扫half数/unroll/fetch/cadence。checkpoint_audit通过，生产7193/math34af/模拟器5e409/原5304矩阵e3fd保持。[HR报告](mla_host_a100_half_ring_20261101.md)、decision、continuation和全部原始证据已保存。本轮为progress，主目标仍active。

HR/next_mechanism_review.json提出的四QK/八PV warp分工现已实现、完整短筛选并拒绝，详见下段QP。该CPU-only旧文件中的“尚未实现”已过时；不要重新实现相同独立分工或重复HR/旧WR。

08:11再次确认CPE父2084930/子2104175仍live，GPU0/1外部2101252/2101253继续占用，phase仍sxm_full_confirmation，无CPE SXM性能结果；不可终止他人或重复启动。HR/h800_queue.txt记录本轮07:54只读核验，原五Slurm作业Priority/Dependency、两watcher和AGENTS/MEMORY哈希不变，无远端提交/排队源码写入。

**本轮新QP=`temp/glm_trace/tune_qk_pv_roles_20261101/`已完成拒绝。** 冻结头 **1252cdee93be85dd74f9fdd9a922b28e5327c3929dfbece1811021bf015648f4**，直接基线CPEa436（未保留），生产7193不变。单cooperative/108CTA/block32x4x3：z0四warp做完整32KV列的QK/softmax，z1/2八warp做两组256feature PV；同Q片段用于两个16KV列块和两个stage，QK不持O、PV不持score/QPE。生产角色补下一hi与当前lo-PV重叠。完整PE→CKV MMA和量化P/native rowsum保持，jointMDO的max/denominator在四QKwarp完整列内归约并广播，FP32分组变化而方程相同。空item d初始2对应旧两组之和。384线程改变prepare/LRU/fetch工作分配，明确为组合变更，未单独归因Q复用。

QP计算字段140032B、union padding保留147968B dynamic/sizeof和原64row双fetchbuffer/f24/3072在途行。namedbarrier1/all384保护initial/Ppublish/hi-free/lo-free，barrier2/QK128保护acquire/cp.async和KPE→P生命周期；全部384线程参加原prepare/LRU/grid/lastCTA协议。编译前静查补齐paired/odd QK→P的128线程同步，未编译旧头及SHA保存在precompile_review/，无GPU失败；依赖复制初次Path(.)相对parent错误保存在setup_recovery.json，修正绝对cwd时原父78438仍等待已有build/test，只补依赖、不重复启动。

QP正常BF16/FP16及EMU均**REG168/STACK8/static2304/LOCAL0**，有LDL/STL，不称无栈；实际单108CTA/384thread/active1。原38EMU/76normal指令/资源/完整机器字与控制字均相同，105+6=111正常数值/cache-payload/LSE/Graph/长K2048及K65stride80等测试通过；实际Graph和首匹配BF16/K97launch的synccheck/racecheck/memcheck全通过，memcheck关闭API错误报告而保留设备检查，不冒称全部trace穷尽验证。新静态EMU7128条/HMMA230/LDSM124/BAR59/LDL7/STL3、normal6920/BAR58，其余计数相同；不是动态性能或因果分摊。

固定GPU3/NUMA1/CPE→QP、MTP8R128/LRU4096/f24off/256位置×3冷pass：100GB0/20µs **3390.325→3520.373、3438.176→3539.104µs（慢3.836%/2.936%）**，50GB20 **5271.488→5258.517（快0.246%）**。6格/3对CPE配对、1152计时launch/72segment、两参考与实际命中率/逐步逐pass完整。拒绝独立分工，不扩full/native/全形状、不扫warp/unroll/fetch；全部负差和源保留，不能声称生产收益。handles **15541/92506/38976/78438全部已terminal0**，不要重启。checkpoint_audit通过，[QP报告](mla_host_a100_qk_pv_roles_20261101.md)、decision和continuation保存最终状态。本轮为progress，主目标约25%–27%差距仍未解决。

QP/next_service_review.json为**当时未实现、现已实现并拒绝（见下段QS）**的旧设计快照：把有界far-service progress放进四QKwarp的PV空档和host等待中。每8row payload分到128producer lane最多5uint4/20reg，区别于旧N9单servicewarp的18uint4/72reg；当前QP已测168reg/8Bstack可作资源和性能控制，但不能推断加入service后仍无大spill。两块8row共享buffer加计算区为158464B，若static不变总160768B；第三块8row拟寄存器，全108CTA的24row在途2592，小于原f24×128=3072，是策略变化。必须将阻塞host wait改为全128producer一致参与的progress，否则所有QK等待flag会死锁；还需跨tile/item状态、64row ready/真实cachefill、bounded defer和退出drain。旧regpipe3三块64row/ldcg/f12/16/20已失败，不能只重扫。这个CPU-only文件不能作为当前实现状态；实际service候选、测量和拒绝结论见下段QS。

08:32原CPE15217再次实际poll为live，父2084930/子2104175仍live，外部GPU0/1的2101252/2101253继续占用；phase仍sxm_full_confirmation，没有CPE SXM新数据，不重复或终止他人。QP/h800_queue.txt记录08:17只读查询，原五Slurm作业仍Priority/Dependency、watcher4160646/1174145活、AGENTS/MEMORY哈希91b070/e3df236不变，无新远端提交或排队源改写。

**本轮新QS=`temp/glm_trace/tune_qk_service_20261101/`已完成，两种QK增量远端服务均拒绝。** 直接CPEa436基线和QP1252控制均冻结，生产7193不变。两版沿用单108CTA/block32x4x3：四QKwarp持服务上下文，在PV空档/hostflag等待中由全部128producer匹配progress推进，八PVwarp只做O；结束attention行后先drain服务再原退出协议。全部384线程先用原helper选完各poolLRU victim，再gridbarrier，然后立即cachefill无需defer；全108CTA静态strided attention行与远端chunk，替代原专用f24分区和claim/steal。输入f24仍在但不是原搬运分工，不能称策略/在途预算相同。

首版`qks`头 **cabe65e10626720653dc8596c3cf5347833b9ff902729ae5a40bc4bd4bc9f4b4**，两个8row shared stage加一个8row寄存器意图stage，最多5uint4/producer lane；每joint pair在PV-hi推进一subjob。dynamic158464/static1408、REG168/**STACK152**/LOCAL0，EMU和normal均有37LDL+37STL静态位置，不能称寄存器payload全留寄存器。名义108×24=2592在途行，小于原3072。首版CPE→QP→qks同参短筛选100GB0/20µs **3417.323→3756.373、3447.435→3914.688（慢9.922%/13.554%）**，50GB20慢10.756%。首轮9格/6配对原件及initial_performance_audit完整冻结，未被修正版覆盖。

修正版`qks_shared`头 **f3964bfb7dc6f0897e7e60ce6b492c11bf2116fd72589ea8edae602e4c989cba**，去掉persistent寄存器payload，改三块7row shared stage。driver只读查得166912B/block上限，140032计算字段+1408实际static下7是最大等大三stagepacking，无数字扫描；dynamic164224/static1408合计165632，host用实际cudaFuncAttributes.sharedSizeBytes/maxoptin守护派发，实际Graph确认新kernel108CTA/384线程/active1。REG168、**EMU STACK32、normal BF16/FP16 STACK72**，仍有LDL/STL，不称无栈。每实际PV tile推进一subjob（paired两次、odd一次），是另一个调度变化，不能单归因栈下降。64row chunk分九7row+一1row，所有子块完成才原64row readyrelease；上限2268在途行且1row尾块更低，非首版2592或原3072。真实读后FAR TILE、FAR WAIT后payload写staging/cache，模拟器/1152字节和参考均不变。

QS修正版独立四方CPE→QP→原qks→shared、GPU3/NUMA1/MTP8R128/LRU4096/f24off/256位置×3冷pass：100GB0/20µs **3405.088→4124.811、3416.256→4500.384（慢21.137%/31.734%）**，50GB20 **5269.536→6470.869（慢22.798%）**。两版都停止，不扩full/native/全形状，不再扫service行数/warp/unroll/fetch/cadence；更小栈没有收益，不把barrier/小块/FARwait/upfrontLRU/staticrowtail/spill的组合开销当已定位因果。两轮共21格/15个对CPE配对、4032计时launch/252segment、全部逐步逐pass/实际命中率/两参考/正负控制齐全。

两版各105+6=111，共**222项新正常数值/全cache-payload/LSE/Graph测试**通过；复用QP111不重复计数。每版QP原40EMU/80normal完整指令/资源/机器字与控制字相同，复用QP38/76证明链又绑定CPE实例身份。实际单新kernel与三项首匹配BF16/K97sanitizer均通过，memcheck关闭API报告而保留设备检查，非全部trace穷尽证明。八个attention数学helper规范化命名后与QP逐字相同，QP相对CPE的FP32分组变化仍保留。7row jobs独立slot集合/字节oracle45case通过，只作结构证明。首版编译前补ready共享标志迭代同步并用const引用传未用寄存器参数，旧未编译头/manifest在precompile_review/；最终CPU状态观察曾相对路径读错，修正cwd后重写观察，observer_path_correction记录，无GPU影响。

QS全部自有handles **54093/99106/9430/35507/39763/98380/78913/17744均已实际terminal0**，不要重启；checkpoint_audit通过，[QS报告](mla_host_a100_qk_service_20261101.md)、decision和continuation保存证据。生产7193/math34af/模拟器5e409/原5304矩阵e3fd保持。09:01–09:03实际poll/进程查询确认原CPE15217仍live，父2084930/子2104175与外部GPU0/1的2101252/2101253仍活，phase仍sxm_full_confirmation；无CPE SXM新数据，不重复启动或终止他人。QS/h800_queue.txt08:40只读查原五作业Priority/Dependency、两watcher活、AGENTS/MEMORY91b070/e3df236不变，无远端提交或排队源码写入。

本轮是progress，主目标约25%–27%差距仍未解决。后续勿重复HR/QP/QS/N9/cofetch/hybrid/WR/四缓冲等已拒绝路线，也不要仅凭更多在途或更少栈推断收益。可继续查prepare/resolve的索引与缓存依赖、行内/跨MTP局部性，但本轮没有实现或测量新prepare候选，也未证明剩余差距由prepare独自造成；已有阶段表只能给下一机制选题依据。先读当前源码/旧prepare失败证据及CPE实际进程，再独立实现真正的新机制。

下面WH/N9/LD/JP等记录保留当时的失败/拒绝证据，其中“生产b6”“72075/1398等待”等是旧时间快照；当前采用状态和活动进程以上述入口及实际文件为准。

**新WH=`temp/glm_trace/tune_warp_histogram_20261101/`已完成拒绝。** 只在原cached/staged ROW_MERGE（rows×headtiles≤32）、A100/2stage/CTA32派发新Params实例；保持所有原JQ36EMU/72normal实例指令/资源及机器字/控制字全相同。新helper保留warp聚合，用8×256-bin私有直方图避免warp间相同epoch计数争用，warp0归并回原lane负责的八个bin，再执行原radix rank/prefix和thread-major同龄淘汰。无新barrier，LRU/缓存载荷/attention数学/fetch/planner/模拟器不变；额外8192B static，新static10240+dynamic147968=158208B，实际108CTA/block32x4x2/active1。新REG246EMU/244normal、STACK0/LOCAL0。

WH通过正常105+4测试。新增4项BF16/FP16×capacity512/4096，注入同龄、单字节和四字节stamp差异，用CPU精确槽位oracle核对淘汰/保护hit/全部cache载荷，并测output/LSE、空/单元素/padding和零miss Graph重放。普通首个冷cache launch可能skip radix，因此另用4096空闲旧槽、776个miss和四字节差异的实际Graph launch做synccheck/racecheck（skip首个匹配warm launch，仅检查下一匹配launch），都零错误；原冷launch诊断也完整保留，不冒称穷尽验证。

WH固定GPU3/NUMA1、f16/split on、256位置×3冷pass、JQ→WH、MTP1/R1与MTP8/R1：100GB0µs分别慢1.047%/0.358%，100GB20µs慢0.027%/快0.037%，50GB20µs快0.056%/0.111%。没有稳定收益，拒绝，不扩full1024×5/native/跨形状或继续私有直方图行数/warp分组/归并布局扫描。12格/6个对JQ配对、10368计时launch/648segment、两参考/命中率/源码快照齐全，109测试及checkpoint通过；原handles16107/31851/84049/24248/69911全部terminal0，无失败/重试。基线是冻结JQ a2f37，不是root b6；[报告](mla_host_a100_warp_histogram_20261101.md)。本轮未改变生产。

06:32:41复查原JQ72075父PID1640200、1398父PID1693759仍live；completion_state仍仅缺最后JQ三格，GPU0/1外部1780064/1780065仍在，不动、不复制原runner。H800本轮只读队列仍原五任务Priority/Dependency、原watcher4160646/1174145 live；AGENTS/MEMORY哈希与已读快照相同，保存JQ/h800_context_hashes_0620.txt与h800_queue_0628.txt，无新远端提交/源码写入。整体goal active，约25%–29%差距未解决；本轮实现和拒绝验证属于实质进展，非blocked。

**新N9=`temp/glm_trace/tune_nine_warp_20261101/`及N9S=`temp/glm_trace/tune_nine_warp_static_20261101/`已完成，两版拒绝。** 单108CTA/block32x9x1，保留完整64head的8计算warp，另1服务warp；前后288线程准备/退出，计算组虚拟y/z和named barrier1/256线程。QPE直接按同一MMA片段从global读取，shared QPE8KB缩为128B对齐padding；计算shared139904B，两个8行cp.async辅助stage18432B，第三8行payload在源码声明为每线程18个uint4，希望进寄存器。dynamic158336/static1664=160000B，实查active1；**两版REG168/STACK248/LOCAL0，有LDL/STL，不是栈零或全寄存器缓冲。** 后一版只把stage和payload展开为固定表达式/18个标量，未消除栈。原JQ36EMU/72normal指令/资源及机器字/控制字全相同，两版共218测试（长LSE/空单行/padding/淘汰/Graph/全cache载荷），实际单kernel及各synccheck/racecheck均通过。

N9分工/在途预算明确改变：108服务warp×24行=2592行，旧JQ24fetchCTA×128=3072；输入API f24不表示仍用24专用CTA。QK unroll1为288线程寄存器预算实现，不是数值扫描。仍原FAR模型与post-read hook、64行ready flag、即时/有界deferred cachefill，不改模拟器/参考。基线base.cuh是冻结JQ a2f37，不是生产b6。GPU3/NUMA1/MTP8R128/LRU4096/f24off/256×3冷pass：N9在100GB0/20慢55.426%/56.289%、50GB慢9.247%；追加正常HBM/PCIe诊断也慢69.533%/18.722%，不能只归因模拟器。N9S固定stage版100GB慢54.168%/55.261%、50GB慢8.632%。两版全部负差，停止，不扩full/跨形状或继续warp/缓冲/unroll/stage布局扫描；栈是否为主因未证明。[合并报告](mla_host_a100_nine_warp_20261101.md)。

两目录合计16格/8配对、3072launch/192segment、两参考/逐步逐pass/源码快照齐全，checkpoint通过；所有N9 handles40640/65540/23187/14304/25015/80800、N9S88439/76252/10004/96169/73043均terminal0，无失败或重复启动。不要重启。生产保持b6，原JQ72075/1398继续独立等待；本轮未安装任何N9源码。

**新LD=`temp/glm_trace/tune_lane_denominator_20261101/`已完成拒绝。** 原生PV每tile用一次TensorCore rowsum复制分母至四个lane；新候选以相同量化BF16/FP16 P逐lane累加各自四列，仅在item末用xor2/xor1合并，再用原跨warpgroup normalize。joint MDO缩放因子quad相同，实数方程相同但FP32归约/缩放分组变化；d初始1/4保持空item d=1，奇尾同新状态。所有PV barrier、QK/MDO/PV输出次序、2stage/32KV/fetch64双buffer/LRU/模拟器保持，不是JP同步机制或unit-scale/anchor方案。

LD原34EMU/68normal指令/资源及机器字/控制字全相同，新BF16 REG252/FP16 REG253、STACK0/LOCAL0，实际正常单108CTA/block32x4x2/active1/dynamic147968/static2304。105+4测试全通过；长测试新增LSE，含K2048/128池/512及4096容量/空、单元素、padding/Graph多次重放/全cache载荷。固定JQ→LD→b6、GPU3/NUMA1/256×3pass/f24：100GB0/20较JQ慢0.275%/0.723%，50GB慢0.039%，拒绝，不扩full/native/跨形状或归约间隔扫描。9格/6个对b6配对/3三方点、1728launch/108segment，两参考完整；原99321/1359/92142/22672全部terminal0，无失败重试。checkpoint通过，[报告](mla_host_a100_lane_denominator_20261101.md)。

LD的前置SASS检查还证实：原JQ CKV循环已有K_A/Q/K_B在首MMA前加载，并预取下一feature，故没有重复实现简单grouped-K源码重排。qk_load_schedule_review.json+摘录冻结证据；LD新静态HMMA299→296、FADD4→28、SHFL53→57、总6952→6992，BAR58和LDSM198不变，仅作已测机制检查，不作动态性能或因果证明。生产仍b6；原72075/1398目前仍存活，等待同一GPU0/1外部1713079/1713080，不能重复启动或绕过门槛。

**新JP=`temp/glm_trace/tune_joint_pv_20261101/`已完成拒绝。** 在JQ现有paired QK+joint MDO后，一次量化/rowsum/发布两个P tile，4个原生PV barrier变1个，再保持hi-PV/refill/lo-PV/refill；原joint MDO barrier保护KPE→P覆写，P stage与m_wg分离，原补数barrier和奇尾原生PV保留。无tile/unroll/fetch/cadence扫描。原34/68指令资源及机器字/控制字全相同，新REG252/STACK0/LOCAL0，105+4正常长cache载荷测试及实际单108CTA/256线程Graph均通过。单独synccheck零错、racecheck首个匹配新kernel launch零hazard（BF16/432行/K97，不作穷尽或trace性能证明）。

JP固定JQ→新→b6、GPU3/NUMA1/256×3pass/f24，100GB0/20µs相对JQ慢0.182%/1.115%，50GB快0.048%，拒绝，不扩full/native/跨形状。9格/6个对b6配对、3三方点、1728launch/108segment、两参考齐全，相对b6正差也保留。全部50732/77642/96800/53029/60928/59627 terminal0；88832因短测漏拷plans.cpp预检exit1，冻结plan补齐后只恢复缺失短测，原failedstate/log保留；47636首次sanitizer等待测试触发attach timeout，核实无GPUcontext后仅终止自有collector/child exit143，失败log/stop JSON保留，第二次launch-timeout0成功。不要重启JP任何runner。生产未改，[报告](mla_host_a100_joint_pv_20261101.md)，checkpoint通过。

**新AM=`temp/glm_trace/tune_address_map_20261101/`已完成拒绝。** 复用whole-row闲置partial_o存uint64行地址，prepare/resolve连同remap写指针、padding compaction复制同样disjoint mover，沿用原grid同步，不增分配/barrier/kernel。派发明确uint64(rows)×max_kv_len×8≤partial_capacity×64×512×sizeof(O)，不足回原生产路径。canonical math/fetch/LRU/真实cache载荷不变。原34/68全等价，新REG252/STACK0/LOCAL0，实际单108CTA/block32x4x2通过。105+4长测试及1容量不足/全backing哨兵/Graph/padding/cache载荷测试，共110项通过。JQ控制复用原binary/gates不算新增测试。三方256×3pass：相对JQ的100GB0/20慢2.812%/3.159%，50GB慢0.493%；相对生产100GB20快0.742%也保留。9格/6对生产配对、3三方点、1728launch/108segment、两参考完整，全部10839/97365/62831/20361/3002/25750/2690terminal0。停止，不扫地址布局/容量参数。[报告](mla_host_a100_address_map_20261101.md)。

JQ本轮两项>1%GPU3未改路径负差已经64659完整5pass前/候选/后夹测（terminal0）：MTP1/R256 HBM原慢1.113%，夹测1053.639/1054.992/1054.030µs（约慢0.110%）；MTP8/R64无LRU零远端HBM原慢1.223%，夹测1205.146/1203.712/1203.814（约快0.064%）。原负差不删除，controls_audit保留每pass及两端，不仅凭代码身份放宽门槛。

硬件计数器诊断已单独完成（41849terminal0；56668首版因pytest导入缺失在GPU工作前exit1，失败源/log保留，改用make_wrapper后成功）。GPU3合成1024×2048/MTP8全共享/25%新页，application replay每次重建LRU/epoch，输出及327680有效cache槽全载荷通过；两版实际Tensor指令均71827456，short-scoreboard26.98→19.31%、long-scoreboard5.39→11.13%、barrier8.48→9.05%。默认Ncu cache控制、原CSV/source/限制见counters_audit.json；不计trace收益/参考，stall不能相加当时间分解。它提示shared读取/加载重叠取舍，不证明真实trace全部剩余差距因果。

JM的完整156矩阵已终，178总格/89配对、两参考完整，无>1%负差；30328/98811均terminal0，勿重启。JQ原66919 limited60格和18187 native20格也terminal0，全部156矩阵格通过审计；总207格/107个对b6配对，60个改变路径配对未加权降低均值1.602%、范围−0.188%至+5.253%，全部正负格保留。原两项>1%未改路径负差均有上述完整5pass夹测。

只剩JQ原72075完整5pass正向的JQ三格；原1713079/1713080自然结束后的窗口已完成JM三格，随后新外部进程进入，CPU screen正在等GPU0/1进程1780064/1780065（/mnt/raid/ny_work/envs/flashcheck/bin/python）。反向EMU3格、正常NVLink/PCIe三方和正向base/JM六格均完整。不得终止其他用户、重复启动72075或绕过1398门槛；没有当前自有GPU工作时也不说明72075死掉。GPU3本轮JP全部terminal，原JQ本机69652/63115/74895早已terminal0。

SXM完整反向三方首组已完成：JQ相对生产100GB0/20µs快5.228%/4.886%、50GB20µs快0.367%；相对JM的额外收益较GPU3小，分别约0.13%/0.24%，不能将全部生产收益归给Q复用。正常完整SXM：base/JM/JQ的NVLink3305.725/3136.016/3183.982µs，JQ较b6快3.683%，但较JM慢1.530%；PCIe9954.458/9951.677/9949.350µs，JQ较b6快0.051%。native_control_review.json保全这项组合负差，不声称Q复用均更好。正向完整JM/JQ仍待，不可跳过。

当前已测JQ full100GB距逐步max：GPU3约26.94%/27.89%，SXM约25.42%/26.53%，仍明显高于10%目标。50GB约6%差距，不能将优化3%–5%写成已解决主目标。H80005:42只读快照仍原五作业Priority/Dependency，预估11/05，两个watcher活、AGENTS/MEMORY哈希不变；没有修改/重新提交远端排队源。

**最新WR=`temp/glm_trace/tune_warp_roles_20261101/`空间warp分工两版已结束，均不采用。** 同256线程CTA，物理z0/1四warp取数，z2/3四warp计算映射到虚拟z0/1，named barrier2/1各128线程；全CTA做原prepare/LRU，grid同步后分角色，结束全CTA/grid同步后原lastCTA复位。原32head/sharded32KV/2stage计算shared110848B，加独立2×16row payload36864B共147712≤原dynamic147968B；每query两headitem直接输出。所有108CTA取数，实际在途3456row比原f24的3072多12.5%，明确是策略变化，未暗称预算相同。FAR waiter仍物理(0,0,0)，模拟器/字节不变。

首版warp_roles的105+4及实际单108CTA/block[32,2,4]/active1通过，但改为可折叠trait helper仍让22/34EMU、44/68normal原实例指令变化（puredevice精确相同），strict gate拒绝且无性能。warp_roles_isolated只复制专用math/host helpers、原定义字面不变，原34/68全部精确等价，新REG246/STACK0/LOCAL0/static1408B；105+4及同实际launch通过。两版共218项正常测试。隔离版f24同参三链路256×3pass：base3589.781→3715.349µs（100GB0慢3.498%）、3596.171→3780.747（100GB20慢5.133%）、5285.131→5290.805（50GB20慢0.107%）。6格/3配对、1152计时launch/72segment、两参考完整。停止此原型，不继续调warp/chunk/fetch/cadence。

WR首次生成在编译前静查发现插入索引错误，错误generator/header保留；首版编译等价失败及全部SASS/测试保留。首个性能runner23211缺plans.cpp在source preflight退出；11870成功完成base后，candidate资源preflight遇外部PID1500489进入退出；只补candidate的72315最终exit0。原日志追加、state和失败保留，无计时raw覆盖。WR所有handles已terminal；checkpoint_audit.py通过。[报告](mla_host_a100_warp_roles_20261101.md)。F/S/WR本轮合计763项测试、42格/24配对、8064计时launch/504segment，生产无变更。

**KV64=`temp/glm_trace/tune_kv64_20261101/`已完成拒绝。** 原64head/256thread使用native sharded64KV/1stage、shared147968B、原fetch64row双buffer/f24及精确LRU。原math canonical34af完全不改，FP32在线分组不同。原34EMU/68normal实例精确等价，109正常/长cache-payload测试通过；实际单108CTA/block32x4x2/active1、dynamic147968B/static2304B。新REG255/STACK8/LOCAL0，有STL/LDL，不得称stack0。三链路256×3pass、candidate→base：100GB0/20慢1.576%/1.795%，50GB快0.024%近乎持平；6格/3配对、1152计时launch/72segment、两参考完整。全部34389/54291/62484/40753/86019/21765/92528已terminal0；checkpoint通过。[报告](mla_host_a100_kv64_20261101.md)。不再扫KV/stage/fetch/cadence。本轮F/S/WR/KV64共872测试、48格/27配对、9216计时launch/576segment，无生产收益。

**本轮新JQ=`temp/glm_trace/tune_joint_qk_20261101/`已完成109项测试及完整反向三方确认，正常/跨形状验证中。** 以JM typed1190…aa59为算法控制，复用旧P冻结的paired-QK helper（原unroll4，不扫数字），两score tile共用Q fragment后沿用相同joint max/d/O和逐tile PV/refill。它要求两stage都ready，失去JM首QK与第二tile加载的重叠，可能抵消Q读取复用；方程及各score的PE→CKV MMA次序不改，仍单cooperative/原147968B/f24/精确LRU。原P配对QK单独失败不能外推这个组合收益；JM所有固定源/SXM队列继续原样验证。

JQ新REG251/STACK0/LOCAL0/static2304B，原34EMU/68normal实例精确相同，实际单108CTA/block32x4x2/dynamic147968/active1；正常105+4长cache载荷/Graph测试通过。新SASS6952条/HMMA299/LDSM198是unroll4静态数，不等同动态work/带宽/性能。冻结JM控制复用原binary及109测试证据，不算本轮新增测试；reused_joint_control.json列原件SHA，benchmark记录实际同字节host/math。

JQ短256×3pass、base→JM→JQ：100GB0µs生产3553.003、JM3464.789、JQ3433.067（JQ较生产快3.376%、较JM0.916%）；100GB20µs3554.848/3466.709/3452.907（2.868%/0.398%）；50GB20µs5285.024/5277.195/5273.344（0.221%/0.073%）。9格/6配对/1728launch108segment，两参考完整。短审计冻结short_performance_audit.json，不安装。full反向三方runner **75294**已terminal0：JQ→JM→base、1024位置×5pass、三设置，固定confirmation_plan.json。100GB0µs生产3620.643、JM3520.424、JQ3484.302，JQ较生产快3.766%、较JM1.026%；20µs3651.909/3540.707/3511.080，快3.856%/0.837%；50GB5521.166/5514.400/5506.459，快0.266%/0.144%。短+full18格/12对生产配对、13248launch/828segment，两参考完整；composition_comparisons.json另列6个三方点，不能混淆额外JM收益和对生产总收益。原88681/28946/71720/94692/81003/95130/52989全部terminal0，不poll/restart。当前JQ三个GPU3 runner：**69652**正常1024×5pass三方2链路（base→JM→JQ）；**63115**changedR64/R128×50/100×五延迟40格；**74895**正常改变形状/小中形状/无LRU控制36格。均在共享锁下串行，不重复启动。R64limited已配对10点，100GB快2.664%–3.335%，50GB0.211%–0.379%；nativeR64 HBM快2.923%、PCIe慢0.188%保留。两轮源和参数固定，无数字扫描。生产b6保持。后续候选选择须看全部结果及真实SXM，不因栈消失或短筛选采用。

**当前JM=`temp/glm_trace/tune_joint_mdo_20261101/`联合重缩放初测有收益，尚未采用。** 保持原32KV/2stage/64head/256thread/147968B、fetch64行双缓冲/f24/精确LRU。两个score tile共用一次max和d/O重缩放，原PV仍逐tile降序，第一个stage在第二PV时补future tile。区别于旧配对QK两次MDO及KV64单stage等待；方程相同、FP32在线归约分组变化。root/private math canonical34af完全不改。新host helper与单cooperative kernel保持原退出协议。

首版joint_mdo和joint_mdo_after定义顺序控制均105+4/实际单108CTA通过，但4/34EMU、8/68normal原实例指令文本改变，puredevice精确相同；strict gate拦截，二者无性能launch。after只移动相同新增helper/kernel不能消除差异，全部失败/source/SASS保留。第三版joint_mdo_typed用专属继承traits隔离helper模板实例，字段/布局/算法与after逐字重建断言相同，原34EMU/68normal全精确等价后才进入性能。三版共327正常/长cachepayload/Graph测试；实际单108CTA/block[32,4,2]/active1，dynamic147968/static2304B，最大绝对误差0.004128。新REG255/STACK16/LOCAL0，不能称stack0，未放宽等价或正确性门槛。

JM短256×3pass、base→typed，100GB0/20快0.653%/2.700%、50GB20快0.176%；已冻结short_performance_audit.json，6格3对/1152launch72segment。按预先confirmation_plan完整1024×5pass反向typed→base，100GB0/20快2.677%/2.902%，50GB快0.146%。正常1024×5pass的独立同卡HBM快3.002%，物理PCIe快0.0045%近乎持平。GPU3 HBM不是NVLink。纯device及逐步max均按原同硬件参考/公式，保留所有正负格与每步/每pass，尚无安装。

上述首三组已完整，12594/83243均terminal0。第二组完整正向base→typed的16920也已terminal0，100GB0/20快2.134%/1.881%、50GB快0.103%，保留独立cohort/forward_confirmation_plan.json。GPU3跨形状两矩阵现已全部完成，61915/12133均terminal0：61915 limited，MTP8/R64及R128×50/100GB×0/2/5/10/20µs×新旧，共40格；12133 native，R64/R128两链路改变路径、MTP1/R1/R256和MTP8/R28/R32原路径控制、有/无LRU及0/.1/.3远端，共36格。全LRU1024位置×3冷pass，无LRU四快照×3比例。固定源无fetch/cadence扫描，runner可跳完成格/只补缺项，state/log追加不覆盖。全部GPU3共享锁，等待日志的其他python也可能是本任务其他串行runner，以pid/command区分，不误称外部冲突。

JM/audit_performance.py对W的fresh小参考只填缺失键、不改变已有参考选择；原每步max公式与reference时间不改。当前report/checkpoint是进度快照，列活动runner及未完成raw，不代表candidate通过全部gate。GPU3合计98格/49配对、114080计时launch/7184segment、两参考无缺失、raw无未完成；coverage.json核验40/40limited与36/36native计划完整。改变的十个100GB配对快1.079%–3.633%，50GB快0.049%–0.174%；最大负差为未改无LRU HBM远端0.3慢0.646%，其余负差全部保留，无>1%负差。不要再poll/restart61915/12133。SXM已有两CPU验证器30328（limited60格）/98811（native20格）等待GPU0/1同时空闲、共用gpu01锁；源和计划固定，无新远端Slurm提交，不能重复启动。SXM改变形状MTP8/R64/R128及MTP1/R512、五档延迟，正常实际NVLink/PCIe另含小/中控制。GPU0/1仍其他用户占用，真实SXM/NVLink及全覆盖仍待，矩阵后还需实际SXM大形状完整5pass确认，**不得仅凭GPU3这几格安装**。

JM旧所有build/test/codegen/stress/probe/screen handles已terminal，详见continuation_state.json，不再poll。最新H80003:38:29五原作业仍Priority/Dependency，但调度估计已推迟11/05，两个watcher活；hash核对AGENTS/MEMORY与已读快照完全相同，无新提交/源码写入。当前[JM报告](mla_host_a100_joint_mdo_20261101.md)、continuation和实际进程为准。此前F/S/WR/KV64全部拒绝，已保全872测试/48格27对；JM327测试与正向信号单列，主目标仍active，大100GB约32%–34%历史差距未解决。

**本轮最新F=`temp/glm_trace/tune_feature_qk_20261101/`已完成，两版拒绝。** dim_split使用两个32-head item、128线程、2stage/16KV，两warp group分别计算PE和CKV各半、交换4096B score再相加；全576特征保留，FP32归约顺序变化。duplicate_control保留相同布局/padding但重复完整QK。新REG231/255、stack/local0，正常实际单kernel216CTA/128线程、dynamic78080B/static2176B、active2；EMU静态HMMA207/315。每版105+4正常/长序列全cache载荷测试，共218；原34EMU/68normal实例全等价，审计核验私有math e5968及68证明后才沿用canonical参考。f24的100GB0/20µs dim慢9.605%/27.923%。唯一追加的f48在途容量等价控制（48×32×2=24×64×2）仍比同参base慢2.408%/3.276%，约4ms不能替代原推荐f24约3.5ms；50GB正负差均保留，不再扫参数。18格/12配对、3456计时launch/216segment、两参考无缺失。[报告](mla_host_a100_feature_qk_20261101.md)。初次generator断言在生成候选前失败的原件保留，性能前已修复私有math provenance，未混入错误SHA格。

**后继S=`temp/glm_trace/tune_unit_scale_20261101/`三版也全部结束并拒绝。** 原布局64head/256thread、CTA32/2stage、108CTA/active1、dynamic147968B/static2304B、f24和LRU保持。unit_scale先算原scale，仅精确!=1才乘d/O；stable_scale保留原scale但检查max变化/非finite；early_stable在有限max不变时直接跳过rescale指数和d/O。后两版数学恒等式相同但FP32舍入改变，无逐位相同声明。三版各105+4通过，共327项；各原34EMU/68normal实例全等价，实际单108CTA新kernel，REG244/242/242、stack/local0。100GB0/20µs分别慢7.651%/6.646%、4.411%/5.069%、4.137%/4.161%；50GB全部略慢。三独立同轮cohort共18格/9配对、3456计时launch/216segment、两参考完整，不扩大full/native/跨形状，也不继续调整此条件。[报告](mla_host_a100_unit_scale_20261101.md)。primary SASS证实FMUL→FFMA→EX2→unit比较/真实BRA；独立相同max浮点表达式65536对中26440个scale非精确1、最大误差3.5763e-7。它不是实际attention分支频率或性能因果证据，独立源码/SASS/日志保留。

F/S全部自有进程已terminal；early105/stress原handles24991/28346已exit0，96149在最终查询时已不保留，完整compile log及后续代码生成/测试/benchmark均成功，不能重启。F/S各checkpoint_audit.py均通过，总545项测试/36格/21配对/6912计时launch/432segment；生产b6、root math34af、模拟器和原5304格SHA不变。GPU3现有外部PID1460325，不动，后续GPU工作继续共享锁/空闲等待。H80002:30:16只读复查仍五任务Priority/Dependency、原两个watcher活，AGENTS/MEMORY哈希与已读快照相同，详见S/h800_queue.txt；无新提交或源码写入。

**大batch100GB约32%–34%主差距仍未解决，goal active。** 空间warp角色隔离已按本节开头实现、通过正确性但无性能收益，不能重做；当前按JM的SXM跨形状及JQ的正常/跨形状验证继续。旧重复QK、feature分工、重缩放、时间交替cofetch/hybrid及空间warp分工均不重做；所有新候选仍须旧实例等价、实际路径/全cache载荷测试，再同参性能。

**最新C=`temp/glm_trace/tune_cofetch_20261101/`已完成并拒绝。** 原型af610…bf6f将attention减为1stage，腾出2×16行service buffer，所有CTA计算中/等待时推进取数，所有pool先做LRU再grid barrier；独立控制020b…0b41只减attention stage、用padding保留原64行fetch双缓冲。两版均105正常+2长队列/全cache载荷测试（共214项），34EMU/68normal原库内实例全等价，实际单kernel/108CTA/256线程/1CTA驻留已核实。cofetch静态38272B+动态111104B、REG244EMU/242normal；control静态2304+动态147968、REG240，均stack/local0。GPU3同参256×3pass的100GB0/20µs原型慢38.940%/53.030%，control慢11.794%/12.284%，50GB慢15.761%/0.470%。9格/6配对、1728计时launch/108segment、两参考完整。原control preflight外部counter失败及仅补缺失候选的重试保留；全部C runner已terminal。见[报告](mla_host_a100_cofetch_20261101.md)。

后继HC=`temp/glm_trace/tune_hybrid_cofetch_20261101/`两版也已完成并拒绝。hybrid 9168…8259保留原2stage attention和专用64行双缓冲主通道，worker用额外2×4行缓冲辅助；原gap flag只在prepare/compaction完成后复用为16行job cursor，last-CTA退出复位。16行ready须等worker四个subchunk全部落地，dedicated每次领四个job。新实例REG255、STACK440EMU/432normal，100GB慢107%–109%，50GB慢58%。独立hybrid_static 7cac…df25用编译期角色区分，把64行service移出attention accumulator活跃区间，保留同样的调度/cadence8/缓冲/job/数学/LRU/模拟器；新REG245、STACK0，但100GB仍慢13.881%/14.911%，50GB慢12.855%。两版均105+2正常测试（共214项），34EMU/68normal原实例全等价，实际单新kernel/108CTA/256线程、dynamic147968+static11264B、1residentCTA；12格/6配对、2304计时launch/144segment、两参考完整。[报告](mla_host_a100_hybrid_cofetch_20261101.md)。

**C/HC全部CPU/GPU runner已terminal，四版本合计428项正常测试、21格/12配对、4032计时launch/252segment。** 不扩展完整1024位置/正常物理链路/全场景矩阵。C已提交6a909b27；HC后续检查点只保全已拒绝实现与证据，生产仍b6。正常BF16/FP16、实际Graph、长队列/全payload、原实例等价及两参考均不是主目标完成证明；整体goal继续active，主差距约32%–34%仍未改善。本轮属于progress，不是blocked：新机制、寄存器活跃区间修正及控制证据已完成；下一步不要重新扫描这些候选。

生产提交为 **0f82e9de**。本轮P=`temp/glm_trace/tune_qk_pair_20261031/`实现不同KV tile共享Q fragment，确认历史regpipe3只改fetch buffer而未覆盖该机制；第一版75ea…d6f9f正常105测试通过、34EMU/68normal原实例相同、新REG255/STACK0，但GPU3同参256×3pass的100GB慢8.1%–8.5%。独立提前页号版adc6…f155fb同样105通过、原实例相同，新STACK8；三链路耗时降低−0.130%/+0.115%/+0.067%，没有稳定收益。两版均实际Graph确认单个新PairedQK kernel、108CTA/256线程、147968B dynamic shared、1CTA/SM。12格/6配对、2304计时launch、两参考完整，均拒绝。[报告](mla_host_a100_qk_pair_20261101.md)。首次base因外部counter_payload进入而preflight失败，原日志/state及仅重试base保留；SXM始终等待外部GPU0/1进程，取消自己的未启动runner，无SXM性能结果。

F=`temp/glm_trace/tune_fetch_warp_20261101/`的新warp_rows仅将大cached whole-row fetch读/staging写/cache写按每warp一行布局，复用72向量的地址，保留64行双缓冲/数学/LRU/模拟器。源码a156…404e；105正常测试、34EMU/68normal原实例等价、实际单新WarpRows kernel均通过；新REG244/STACK0，目标静态指令7888→7744。GPU3同参256×3pass，100GB0/20µs慢0.764%/0.030%，50GB20µs慢0.088%，无采用信号，6格/3配对与两参考完整。[报告](mla_host_a100_fetch_warp_20261101.md)。三候选均不扩展完整窗口/正常物理链路/全形状矩阵；原负差和全部源快照保留。

P/F所有CPU/GPU runner已terminal；P/continuation_state.json与F/continuation_state.json记录全部handle，不重启旧runner。新增restore_includes.py会核验既有树或从冻结host/math恢复缺失树，不覆盖既有include。此次不是blocked：已实现三个独立调度候选并完成315项正常测试、完整原实例等价和9个短窗口配对，但没有新增生产改进，历史100GB约32%–34%主差距仍未解决，goal继续active。

W补充矩阵896格/448配对的测量与两参考审计均完整，cached_planned_complete=true、pending_references=0。96个改变的LRU配对原矩阵耗时降低未加权均值2.028%、最大11.603%，最初四个大形状100GB负差仍保留；完整5pass前后夹测后最差−0.357%。未改路径R40PCIe0.3原−2.849%在夹测中+0.004%，其余原路径由代码身份和已有夹测/实际不同函数的同分配证据核对，原负差未删。保留标准明确不承诺每次独立内存分配都更快。

SXM正常构建、MTP8/R128、full1024×5pass、f24/off：真实NVLink3349.277→3229.662µs（3.571%），PCIe9937.678→9957.666µs（慢0.201%）。100GB完整确认仅约0.3%收益，不能声称解决历史32%–34%差距。整体goal仍active。

`retain_cached_ids.py --install`已实际通过门槛并逐字节安装已测b6原件；W/cached_ids_retention.json及retention_snapshot冻结完整审计/覆盖/编译/测试/慢格理由。生产root/include、独立jit_retained_normal、正常BF16/FP16 CPU编译及105项测试再次通过，记录tests_production_gpu3.json（13.16s）。测试source5845…保持不变。

上一轮本机GPU/CPU调度器均已terminal：W原矩阵92282、完整确认86595、慢格夹测82401、补小参考28183、生产编译37851和生产测试28695全部exit0；R的四确认和三版诊断也均exit0；H/Q/U拒绝候选无活动任务。**不要再poll或重复启动这些handle。** 最终audit83501和coverage/report已成功。41741初次将生产测试记录当候选记录而缺compile_exit_code，失败日志audit_production_record_schema_failure.log保留，已显式区分两种记录并复核通过。旧审计98052因单请求floor缺mathSHA而失败的日志保留；新同硬件normal小参考已补齐，没有覆盖原floor。

H800五个旧Slurm作业11/01 01:20:24仍Priority/Dependency，两个watcher活，保存HC/h800_queue.txt；没有覆盖排队源或重复提交。所有自有本机runner已结束，不必等待其他用户任务。下一步在b6上继续新机制推进大100GB目标；不要重复ready/prebind/scatter/四缓冲/tail/半head双驻留/重复QK/配对QK/warp按行取数/本轮单stage或hybrid辅助取数，或只扫cadence/unroll/fetch参数。已完成等待/CTA预算诊断支持查共享资源，但这些拒绝证据说明不能只凭更多CTA参与搬运推断收益。特征分工和单位重缩放已按本节开头实现并拒绝；后续方向以上述空间warp角色隔离的待验证设计为准。勿改模拟器/参考。

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

1. 复查源码SHA、dirty状态、GPU进程、Slurm；读已完成阶段诊断，不重复启动。所有§2.0旧runner已结束，按最新continuation_state核实，不因历史段落的WAITING字样重启。现有生产与已完成矩阵均已保全，不重做整轮实验。
2. 以新目录`base.cuh`和上轮正式确认作基线。重点定位100GB下fetch/attention争用和尾部；物理NVLink可比较原地读，小batch另查prepare/同步。
3. 独立include/JIT实现少量有依据候选，短窗口筛选后完整trace、同参数新旧/反向多pass确认；不要重跑原矩阵、重复已失败候选或重采trace。
4. 保留候选需覆盖大小batch、MTP1/8、50/100GB及高低延迟、有/无LRU、PCIe/NVLink；检查默认未走分支的代码生成/性能，防编译器引入回退。不要按测试结果暗中挑最优。
5. 正确性/缓存载荷/Graph回归通过后，保存每步/每pass、命中率、两参考、源码SHA、guard/失败，更新Markdown和审计，提交本任务自己的成果。新诊断和正式收益分开表述。
6. H800任务完成后先读硬件结果、修正比较口径，再隔离迁移当前优化；不覆盖原排队代码、不重复提交。A100模拟与H800实测分开。

目标仍是持续逼近下限；完成一轮参数扫描不等于全任务完成。已有30%差距不能写成“已接近极限”，也不要为了汇报隐藏慢格。回退需修复或拒绝候选并记录依据。

下一agent的goal文字见 [NEXT_AGENT_GOAL.md](NEXT_AGENT_GOAL.md)。
