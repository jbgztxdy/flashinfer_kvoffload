# fetch CTA参与必要计算：QK辅助的可检验设计

**接续实测：串行穿插带宽降到约63GB/s，4+4空间分工微探针能维持约98–99GB/s；两版真正交接及32fetch完整试验均慢，未保留。最新[结果报告](mla_host_fetch_qk_assist_results_20261103.md)优先；下文是设计时快照，原“无GPU结果”不再代表当前。**

用户希望优先利用fetch CTA中的冗余计算资源。当前建议是：**保留84个worker的完整attention布局和24个64行双缓冲fetch通道，仅让fetch CTA辅助计算有界、已就绪的QK块，再由原worker消费分数、执行原softmax/PV。** 这是新设计和独立微探针，不是已经有效的优化；生产host/math/FAR及正式参考保持cab508dc/34af/8507eab0。

## 为何不直接在fetch内调用整行attention

两个64行fetch缓冲占147456B，生产dynamic shared147968B；完整attention则另需共享Q、KV/P stage，跨tile的O还要每线程128个FP32累加寄存器。两套布局目前通过时间复用共用同一片shared，不能同时占用。这是数据存储及生命周期冲突，Tensor Core空闲本身不能消除它。

整行attention在84/108计算CTA下差约0.606ms，但这不是可全部利用的fetch空等。当前插桩0/20µs平均每fetch CTA有约781/822µs的FAR wait、约45µs物理cp.async wait；除以完整fetch时间约为三分之一。折算108SM的显式等待服务机会约183.5/192.6µs，低于达到110%尚需的356/389µs。计时有插桩、barrier和约1µs分辨率，**不是严格收益上限或因果分解**；它提示只替换显式等待可能不够，需要进一步利用复制/地址处理期间未用的Tensor Core管线。

## 新机制与旧失败的区别

旧cofetch让全部CTA推进小块fetch，缩小attention stage；旧WR/N9/QKS改变全部108CTA的计算角色、线程数、stage或寄存器预算；旧两遍QK/PV为整item保存所有分数并重读KV。本设计均不采用这些改变：

- 84个worker继续使用完整8warp、双32KVstage、pairedQK/jointMDO/PV；只有命中辅助结果的QK调用被替换。未命中时保留原本地计算路径。
- 24个fetch CTA保持原64行双缓冲、必要字节、LRU/cache写入与ready发布。先不减少fetch CTA、不缩chunk、不扫描退休/服务cadence。
- 辅助者只持小块FP32 score，不持整行O、m/d或P/PV状态；Q/K来自已经本地可用的真实载荷。不得在far deadline或staging ready之前读取远端数据计算。
- 原worker仍载入所需KV做PV，辅助不会少算PV或省略缓存写回。QK采用原生fragment映射、完整576特征及PE→CKV累计顺序；需要GPU数值验证，CPU坐标覆盖不能替代它。

## 有界分数交接

worker发布带epoch、row、原native hi/lo pair的前瞻任务。任务必须满足整对KV可用；fetch辅助者不能等待尚未ready的KV，避免“取数者等计算、计算者等取数”的环。

建议的第一版任务所有权是原子状态EMPTY→OFFERED→CLAIMED→READY→CONSUMED；worker到达时若任务仍是OFFERED，可CAS取回并自行计算；若已经CLAIMED，则辅助者必须完成，不能直接复算或复用该槽。辅助函数一旦认领就完成整个有界块并发布，不跨FAR wait留半个任务，因此没有依赖worker的进度环，但仍可能形成性能等待，必须测量。

每个worker仅提供两个在途paired块，存放于现有闲置partial_o backing。每对64head×64KV的FP32分数16KiB，84worker×2槽为2752512B（2.625MiB）；不是原两遍方案的512MiB全量分数。每个槽须有generation，旧helper释放前不得复用；结束时全部CLAIMED任务必须drain，再进入原退出/epoch重置。服务任务只有被原worker消费且跳过相应QK时才计入“分担”，不可把重复QK或忙碌Tensor Core当收益。

这能避免大O状态和大shared要求，但有真实代价：辅助者需要额外读取Q/K，并写回FP32分数，worker再读取；队列/原子/发布和可能等待都要计时。短前瞻可能提高同一KV在L2中的复用，但目前没有命中证据，不能据此把额外读取当免费。

## 两种调度层次

第一步是完整CTA的短QK量子：下一包远端已发出后，在原FAR wait之前执行一个独立、已就绪的QK块。原FAR header、reserve/deadline/wait函数、到达/发布语义均不改。这能先否定“一个小QK块都会把100GB通道拖慢”的实现，不能直接证明端到端有收益。

如果量子成本合适，再考虑**只在24个fetch CTA内部做warp分工**：部分warp保持复制/发布，其他warp计算辅助QK，保留84个原完整worker；辅助块仍覆盖完整64head的一部分KV，不把整个attention改为32head或添加重复QK。这与旧“全部108CTA分半”有结构差异，目的是让Tensor Core与复制指令真正并行推进，而非只填全CTA等待。

空间分工的难点是原fetch及FAR wait含全CTA barrier。不能直接把helper丢进某个warp分支后继续调用这些barrier；必须设计匹配的loader组同步、stage完成/复用协议。原v2每包reserve、最晚deadline、真实读取完成与release发布的约束都须保持，不能绕过模拟等待制造收益。当前微探针刻意不做这项改动；角色分工版也尚未编译或实现。

## 已做与明确未做

已实现独立探针`temp/glm_trace/probe_fetch_qk_assist_20261103/probe.cu`：真实64行双buffer cp.async搬运，24CTA，v2 FAR原hooks；在每包等待前穿插一个64head×32KV×576的真实BF16→FP32 QK，并发布完整FP32分数。Q/K是另外分配的本地数据，不提前使用far载荷。对照有纯fetch、fetch＋QK、独立QK；固定100GB0/20µs、交替运行顺序、三样本，最终检查完整远端payload与四个job的全部QK坐标。它没有接入真实trace/mailbox消费者、LRU或同时运行的84worker，**即使微探针通过，也不能声称端到端优化有效**。

CPU编译通过，实际purefetch/assist分别REG30/54、STACK0/LOCAL0，固定dynamic shared147968B；这是微探针的资源，不是未来整kernel资源。CPU枚举确认2048个输出坐标唯一且完整、每warp A/B各256元素覆盖、576特征PE→CKV次序完整。首编译漏cstdint导致错误，原源/日志存first_compile，GPU前已修复。

GPU3随后被其他用户flashcheck连续子进程占用，稳定空闲预检120秒到期，**没有启动探针GPU进程、没有result.json、没有GPU正确性或性能结论**。preflight_failure.json保留实际观察，CPU waiter已退出，无后台等待器。旧diagnose结果没有重跑，GPU0/1/2未使用。

下一硬件窗口先完成这个已冻结探针，不运行构造器、不重复旧矩阵。若显著拖慢fetch，应拒绝当前直接global-fragment实现，不把负差推为所有辅助计算都不可能；若能维持吞吐，再接入有界真实工作交接，核对被替代QK量、cache载荷/数学和两点端到端新旧结果。最终评价仍是总耗时和必要字节，不能用“Tensor Core动了”作为采用理由。
