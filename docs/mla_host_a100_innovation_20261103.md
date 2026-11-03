# A100：新机制筛选与迁移入口（2026-11-03）

当前生产仍为754706f8，**本轮没有新生产收益，110%目标未完成**。用户11:10左右要求迁移至新agent；本机没有本任务活动runner，不再启动新的GPU或编译任务。首接续点为square_smem的已修源码重编译，见本文末尾及migration_state.json。

11:11实际查询GPU3空闲（A100-PCIE、0MiB、0%）；GPU0/1被其他VLLM任务占用约38GB，GPU2也有其他服务，不使用、不终止。继续使用GPU3共享锁和NUMA1，运行前必须重新预检。下列性能全部是GPU3同卡独立far-HBM模拟，不是NVLink，不与SXM绝对时间拼接。

## 已完成的新旧对照

BF16、64heads、KV512+64、K2048、MTP8/R128、LRU4096/请求、fetch24/split off、256位置×3冷pass，CUDA Graph/event，原模拟器、参考公式、计时和实际cache payload不变。每一候选各有同轮7547冻结基线；保留全部正负差、逐步miss、命中与两同硬件参考。

| 候选 | 100GB0µs基线→候选µs | 100GB20µs基线→候选µs | 50GB20µs基线→候选µs | 处理 |
|---|---:|---:|---:|---|
| shared_claim旧候选有效补测 | 3369.227→3561.835（慢5.72%） | 3386.037→3611.211（慢6.65%） | 5269.440→5464.373（慢3.70%） | 拒绝 |
| anchor缩放锚点 | 3359.765→3349.952（快0.29%） | 3387.264→3443.979（慢1.67%） | 5269.728→5269.899（持平） | 拒绝，无稳定收益 |
| square 32×32 QK | 3347.701→3499.573（慢4.54%） | 3390.677→3562.944（慢5.08%） | 5266.464→5277.984（慢0.22%） | 拒绝 |
| square_p直接发布P | 3377.131→3611.637（慢6.94%） | 3358.325→3630.741（慢8.11%） | 5267.765→5281.888（慢0.27%） | 拒绝 |

shared_claim复用昨日两项BF16定向正确性与同一a47ca8b8原件；今日只做GPU3三点新旧对照。昨日受外部任务干扰的不完整SXM格继续排除，不需要恢复其旧runner。decisions_gpu3_20261103.json和continuation_state已标记拒绝。

anchor把base2 exponent缩放锚点向上取整到8单位，延长稳定区间，warp一致跳过未改变锚点下的d/O重缩放。区别于此前仅在原最大值不变时跳过的unit_scale。实数softmax方程不变，BF16P量化和FP32缩放分组改变。首a0a07稿单元素行output正确但LSE偏约0.005，2项失败/1项通过；没有放宽容差，而在请求LSE时另外累加FP32 exp分母，输出仍用与quantized P一致的分母。最终ac18beac通过3项检查：长/奇尾/空/单元素、Graph/全有效cachepayload，另含大logits和后续最大值重缩放。没有稳定性能收益，拒绝。首失败件在failed_lse/，不可当最终实测源码。

square让原8warp中的4warp分别做32head×32KV块，一个warp内Q/K都复用两次，16KiB共享FP32分数交换恢复原8warp布局，后续原jointMDO/PV/refill保持。4096个分数坐标各有唯一写者和读者；2项定向output/LSE/Graph/全有效cachepayload通过。动态shared164352B，实际normal/EMU REG255/STACK16/static2304；不能用源级load减少当作已证实动态收益。

square_p在同一新QK布局中直接量化并发布P至原KPE/P缓冲，native量化P rowsum经共享sum交接给PV，省FP32分数往返及native P重发布，保留原PV MMA和hi/lo refill次序；FP32归约分组变化。2项定向检查通过，动态shared149248B；实际normal/EMU REG255/STACK88/static2304，不能声称零spill。首CPU稿重复dispatch变量名在GPU前修正，失败件存failed_direct_dispatch/。最终实测03925d9c仍慢，拒绝。

四组共24格/12配对、4608计时launch/288段；今日新增成功定向检查7项，复用shared_claim原2项。没有重复全套pytest/sanitizer/矩阵、原kernel逐实例机器码等价、单kernel实际Graph节点probe或正常物理链路/FP16/SXM/H800新测。不将任何未做项写成通过。已拒绝机制不再扫描数字或原样重做。

## 理论参考的准确解释

原参考是逐步`max(指定本地实现的GPU计算时间, 本路径实际unique miss载荷/带宽)`后平均，是特定计算实现与理想传输组合的乐观参考。计算实现更快时可以低于其中计算主导步骤的参考；实际必需字节/物理带宽约束仍在。不能改模拟器、改参考、用`max(mean)`、省略cache写入或改变attention数学制造“突破”。

本轮GPU3前32步，24步计算主导、8步传输主导：compute均值2556.960µs、unique传输均值2479.478µs、逐步参考2699.864µs，当前基线3347.701µs，110%为2969.850µs，仍需降11.29%。假设仅计算快10%/20%，同字节的逐步参考会成为2556.862/2497.469µs；这只是代数分析，不是候选实测，也没有替换正式参考。[绑定原件的分析](../temp/glm_trace/tune_qk_square_20261103/bound_audit.json)。SXM完整1024窗口原值约3.29/3.31ms、参考2.691ms，属于另一硬件/窗口。

## 当前唯一待续候选

square_smem只把square_p的32个跨item存活QPE packed words改为从已有shared QPE按原MMA片段读取，针对实测STACK88的寄存器压力；不改QK/P发布/归约/PV/barrier/fetch/FAR，不扫unroll或几何参数。移除长期QPE并不保证消除栈或取得收益，需实测。

首30e7ddab稿CPU编译normal/EMU均exit1：新`const bool square_smem`与保留的旧square控制分支`constexpr size_t square_smem`同scope重名。已将新布尔改为`square_smem_enabled`，同时更新生成器、冻结header和include原件，最终SHA为**c69a2643f887b642471c621cddd318e715a92e292c3ed1b35341ecfc6a24aeb2**。首失败头、编译日志、build记录在failed_smem_dispatch/。**修正后尚未编译，无GPU正确性/性能，不安装。**用户要求迁移后没有启动构建。

接手重新预检GPU3后，在仓库根使用指定Python执行`tune_qk_square_20261103/build_smem.py`；成功再`run_smem.py`，仅两项定向检查和三点同轮新旧短测。`run_smem.py`使用新smem256_ cohort、smempaths插件及smem_state，原square/square_p结果不覆盖。然后audit_performance.py，资源审计可按需要只扩review_codegen.py中的新类型；只有真实收益才保留并继续。不要重跑make_smem_pe.py/fix_smem_dispatch.py，它们是不可重复的构造/失败归档脚本。当前include/JIT目录仍在，且.gitignore排除；若缺失，用冻结base/candidate/math重建，严禁旧JIT冒充新SHA。

所有旧handles均terminal，详细见[migration_state.json](../temp/glm_trace/tune_qk_square_20261103/migration_state.json)。当前源码生产与数学/FAR保持7547/34af/5e409。

## H800只读状态

11/03 11:12:58 Asia/Shanghai经login shell查询：1004362 host COMPLETED0、1004363 peer COMPLETED0；1040680 bulk FAILED1，nvcc找不到libcudadevrt/libcudart_static，尚未重试/无bulk GPU结果。1089379 fi-tuned-assist-host PENDING/Priority、依赖已解除，估计11/05 00:49:44（不保证）；1089380 fi-tuned-assist-peer PENDING/afterok:1089379。watcher1174145仍活，4160646已结束，不重复启动。

两个queued任务cwd均/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer_tuned_20261029，日志在生产树temp/h800/logs，源码不含本机最新优化，不覆盖/不重复提交。原完成host/peer数据为旧29667eaf/四步trace/steady口径，不能当当前kernel或正确Memcpy新基线。首次非login shell无Slurm PATH，已用`bash -lc`成功复查；Slurm bin为/rmprog/slurm/v24.05.1/bin。[完整状态原件](../temp/glm_trace/tune_qk_square_20261103/h800_login_query_20261103.txt)。
