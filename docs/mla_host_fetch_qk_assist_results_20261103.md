# fetch CTA内搬运与QK并行：机制可行，完整版本未获收益

用户明确要求在维持带宽时利用fetch SM冗余计算，并允许尝试fetch CTA内部warp分工、必要时增加fetch SM。本轮已完成串行穿插、4+4空间分工微探针，以及真正替代原QK的两版完整单cooperative kernel；另测32fetch CTA并做同一binary关闭辅助对照。**机制可执行，但两个完整版本均慢，未安装；生产cab508dc/math34af/FAR8507保持，110%目标未达到。**

全部使用独占GPU3/A100-PCIE/NUMA1，far是同卡独立HBM上的v2串行化/到达模拟，不是NVLink。正式参考仍为同硬件相同32步的指定正常device实测与本路径实际unique字节预算逐步取max再平均；没有改参考、模型或省略cache操作。

## 先验证带宽和真实QK能否重叠

原串行探针已在本轮空闲窗口补完：24CTA、双64行，等待前插入一个完整64head×32KV×576 QK。0/20µs纯fetch为99.49/98.52GB/s，穿插后仅62.80/62.80GB/s，拒绝该实现。真实payload及四个job的全部QK坐标参考通过，不把算术正确当性能可用。

随后只在fetch CTA内分组：四warp搬运，四warp执行QK；两个64行fetch缓冲保持147456B，另加12KiB feature工作区，总dynamic160256B。QK按64特征块coalesced cp.async载入shared、XOR swizzle后ldmatrix/MMA，减少直接global-fragment路径的操作数准备成本。普通attention CTA尚未加入此微探针。

原v2 FAR header/reserve/tile/wait_until完全保持；只将loader的前后可见性同步限定到128个实际loader，保留全部包预约、最晚stage deadline、真实读取完成及发布顺序，helper只读本地数据。微探针记录的所有arrival均不早于deadline，全部payload和采样QK通过。这不外推为任意stream/设备的完整模拟器验证。

| fetch CTA数 | 0µs纯fetch / 4+4辅助 GB/s | 20µs纯fetch / 4+4辅助 GB/s | 同时完成的QK块 |
|---|---:|---:|---:|
| 24 | 99.44 / **99.22** | 98.67 / **98.04** | 3361/3361 |
| 32 | 99.40 / **99.36** | 98.62 / **98.28** | 3361/3361 |

资源为purefetch/4warp-only/4+4共32/32/70寄存器，STACK0/LOCAL0。同样为单CTA内真实复制和MMA并行，不使用空MMA或预读取未到达的far数据；但分数没有被真实attention worker消费，因此这只是机制可行性，不是端到端收益或饱和QK容量。

[微探针审计](../temp/glm_trace/probe_fetch_warp_assist_20261103/analysis.json)、[原串行探针](../temp/glm_trace/probe_fetch_qk_assist_20261103/result.json)。

## 真正接入原attention的两个版本

完整版本保留256线程、单108CTA cooperative grid、原worker完整8warp及双32KVstage、joint softmax/PV、LRU和真实缓存写回。仅24个fetch CTA分为128loader/128QK helper。helper从已就绪的真实remap/cache/staging取数，输出native layout FP32 scores，worker消费后跳过对应pairedQK。

队列采用pull方式：worker发布当前行/进度，helper固定前瞻四对并认领未计算pair；每worker两个score槽，64位任务状态包含行代次和唯一所有权。未ready任务在MMA前取消，worker可收回；已经认领且开始计算的任务必须完成，worker不会复算。全部CLAIMED任务完成后再结束fetch helper，原last-CTA退出/epoch协议保留。普通worker用原pre-QK barrier广播认领结果、jointMDO原barrier保护分数消费后的槽复用。

active84worker分数空间2.625MiB，现有partial_o backing实际预留128CTA的4MiB score加144KiB control，无新增GPU allocation。控制结构1152B/CTA，两个slot及128个pair状态，原56MiB idle backing足够。私有派发限制SM≤128、行数≤65535、K≤8192；并非公共全形状部署。

- `candidate`：每对QK分别处理两个32KV块，12KiB工作区、dynamic160256B。实测REG255/STACK0/static2304/LOCAL0。
- `fused`：同一64KV pair共用Q载入，helper group barrier36→18；预先解析四个稳定KV行指针，避免每个feature块重复追索remap。16KiB工作区、dynamic164352B，实测REG255/STACK0/static2304/LOCAL0，总shared166656B，低于实际166912B/block上限。原保守`dynamic+4096`检查会错误排除它，私有派发改由实际cudaFuncSetAttribute/Occupancy核验；长案例真实consumed计数确认没有落到旧路径。附带少量wait/fetch计时，仅诊断用途。

首次uint64_t与unsigned-long-long指针类型编译失败保留first_compile。GPU前补齐空队列broadcast重用barrier，原源/CPU构建存pre_gpu_broadcast。fused的首CPU稿及后来页指针/launch guard修订保存在pre_gpu_fused_review；这些归档稿未跑GPU，不应当作实测最终源。

## 完整新旧短对照

BF16、h64、KV512+64、topk2048、MTP8/R128、LRU4096/请求、staged/split off、256位置/32步×三冷pass。CUDA Graph/event，只计单attention launch；CPU上传、诊断reduction、参考及载荷校验在事件外。每个设置独立子进程，避免VMM映射残留。以下同参数比较保留全部正负差：

| 实现 | fetch CTA | 链路 | 生产 µs | 新版 µs | 慢于同参生产 |
|---|---:|---|---:|---:|---:|
| candidate | 24 | 100GB0µs | 3307.840 | 3859.403 | **16.67%** |
| candidate | 24 | 100GB20µs | 3303.360 | 3875.499 | **17.32%** |
| fused | 24 | 100GB0µs | 3307.840 | 3760.395 | **13.68%** |
| fused | 24 | 100GB20µs | 3303.360 | 3782.347 | **14.50%** |
| fused | 32 | 100GB20µs | 3430.773 | 3824.597 | **11.48%** |

base24先/后顺序分别对应0/20点，fused复用同次新基线，不宣称精确的独立机制因果。当前波动不足以改变大幅负差结论。fused比candidate快也不能算生产加速。

| 完整版行为均值 | candidate24，0/20µs | fused24，0/20µs | fused32，20µs |
|---|---|---|---|
| 真正被消费并替代的QK pairs/32768 | 1380 /1395 | 2333 /2440 | 3222 |
| 分担QK比例 | 4.21% /4.26% | 7.12% /7.45% | 9.83% |
| 消费方曾等待的任务比例 | 99.28% /98.65% | 约0.009% /0.009% | 约0.008% |
| fetch有效payload带宽 | 未插桩 | **95.96 /92.64GB/s** | **97.30GB/s** |
| helper结束落后最后fetch | 未插桩 | 14.92 /13.50µs | 14.89µs |

fused当前计数说明辅助结果基本能提前到达，不能继续用“worker等helper”解释其主要负差。另一方面，真实worker并发、额外Q/K读取、score传递与控制操作使微探针约98–99GB/s不能完全保留；20µs下增至32确实恢复了部分带宽，也增大分担量，但普通worker从84减至76，总时反而比fused24再慢约42µs。这个原型不支持继续靠加fetch SM解决。

## 同binary控制指出交接热循环成本

针对fused仍慢的问题，另建同一诊断binary，用host_item_entries运行时选择三种模式；该scalar在整行路径不改变attention分块。所有模式保留全部attention数学、prepare/LRU/fetch/cache真实操作，未计算辅助的QK都由原worker计算。dynamic/static/寄存器、block/grid一致；100GB20µs、24fetch、同输入各一冷32步pass：

| 模式 | attention µs | 分担pairs | fetch GB/s |
|---|---:|---:|---:|
| 关闭交接、关闭辅助 | **3482.752** | 0 | 97.34 |
| 维护worker交接状态、helper不认领 | **3724.352** | 0 | 97.41 |
| 完整辅助 | **3774.080** | 2448 | 92.85 |

维护行/pair状态、进度发布、所有权原子等操作合计增加约**241.600µs**；helper辅助、额外读取/分数交换与所省QK计算的净效果又慢约**49.728µs**。这是同binary运行时控制差值，不能进一步精确拆成某条原子或HBM读指令时间；仅一pass，不作小百分比收益结论。关闭版仍有约0.18ms相对生产差距，包含改后的fetch组织、资源/代码生成及未完全移除的selector路径，未严格分离。

因此下一可检验方向应先降低普通CTA热循环的交接成本，例如一次认领一组pair/粗粒度任务归属，保持未辅助计算尽量走原直线路径；并减少辅助操作数准备/数据传递。当前细粒度pull实现不能仅通过调整前瞻距离或继续增加fetch数来保留，也未证明“所有fetch辅助计算都无效”。

## 验证和交付范围

新增5项定向检查：candidate两项、fused两项、fused32一项；K65/K2048、空/单有效/奇尾/padding、QPE stride80、三轮淘汰、output/LSE、Graph两次重放及所有有效cache载荷/映射。长案例必须有非零真实assistance，produced==consumed，结束无leased槽。性能每个pass抽查0/16/31步、每步三query输出，全部通过；每格最终524288有效槽载荷一致。控制三模式也执行完整数学参考和cache载荷检查。

共864次完整kernel计时launch（8正式比较格×96＋3诊断模式×32），另串行18、空间36次微探针计时launch；原始warmups和全部失败/负差保留，没有新整套pytest/三类sanitizer、完整1024窗口、正常物理链路、FP16或H800结论。最初串行探针的旧GPU占用超时仍是历史；本轮接纳任务的资源预检/末检均无外部GPU3进程。

入口：[完整审计](../temp/glm_trace/tune_fetch_qk_assist_20261103/audit.json)、[复现与源码](../temp/glm_trace/tune_fetch_qk_assist_20261103/README.md)。所有runner均terminal，无后台等待器；GPU0/1/2和H800队列均未触碰。两个完整候选均拒绝，冻结保留作诊断证据，不回退或安装生产源码。
