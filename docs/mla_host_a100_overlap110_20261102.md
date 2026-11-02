# A100：朝逐步重叠参考110%推进（2026-11-02）

本轮已实现四个新机制：两个结构性 attention 候选及fetch提前转计算均完成轻量筛选，拒绝安装；共享hash先去重候选通过定向正确性，但其他用户任务在性能测量期间进入，结束资源检查拒绝了该格，尚无有效候选性能数据。**尚未达到110%，没有新增生产加速。** 当前生产仍为754706f8，不能将失败或待测候选当成新生产版本。

当前SXM实测R128/MTP8/LRU4096，100GB/s的0/20µs选择staged，耗时3289.824/3306.973µs，逐步max参考2691.342/2691.017µs。110%对应2960.476/2960.119µs，仍需降低约10.01%/10.49%。这用的是同硬件逐步`max(device计算, 实测unique miss字节/带宽)`再平均；参考是指定本地实现和理想搬运的乐观参考，非严格数学下界。本轮GPU3绝对时间不与它拼接。[当前路由比较](mla_host_route_selection_20261102.md)继续有效。

前两个候选使用空闲GPU3/A100-PCIE、NUMA1，far为同卡独立HBM上的受限链路模拟；不是NVLink。后两个候选在GPU0/1释放后的空闲窗口、持gpu01锁/NUMA0使用真实SXM NV12源上的同一100/50GB模拟。BF16、64heads、KV512+64、K2048、MTP8/R128、LRU4096/请求、fetch24/split off、256位置×3冷pass，CUDA event/Graph，上传/CPU规划在事件外。每个已完成候选各有同轮基线、逐步miss/命中/实际字节/输出检查和两同硬件参考；原模拟器、计时、参考和5304格矩阵均未改，不混拼两硬件绝对时间。

| 硬件 | 新机制 | 设置 | 同轮生产基线µs | 候选µs | 耗时变化 |
|---|---|---|---:|---:|---:|
| GPU3 | 整item两遍QK→PV | 100GB/s，0µs | 3364.085 | 5265.792 | 慢56.53% |
| GPU3 | 整item两遍QK→PV | 100GB/s，20µs | 3373.845 | 5309.003 | 慢57.36% |
| GPU3 | 整item两遍QK→PV | 50GB/s，20µs | 5272.683 | 5893.579 | 慢11.78% |
| GPU3 | 全局Q片段＋64KV双stage | 100GB/s，0µs | 3354.005 | 4844.981 | 慢44.45% |
| GPU3 | 全局Q片段＋64KV双stage | 100GB/s，20µs | 3398.891 | 4863.840 | 慢43.10% |
| GPU3 | 全局Q片段＋64KV双stage | 50GB/s，20µs | 5267.659 | 5621.536 | 慢6.72% |
| SXM0/1 | fetch提前转计算 | 100GB/s，0µs | 3230.187 | 3233.963 | 慢0.12% |
| SXM0/1 | fetch提前转计算 | 100GB/s，20µs | 3246.357 | 3439.968 | 慢5.96% |
| SXM0/1 | fetch提前转计算 | 50GB/s，20µs | 5285.035 | 5278.005 | 快0.13% |

两遍候选先完成整个item的QK，将FP32分数写入放大的既有partial_o backing，求整行max后重读KV完成固定max的exp/PV/分母，尝试让O在QK时不活跃并消除逐tile输出重缩放。工作区在事件/Graph外同样扩大给新旧两版；R128/K2048需约512MiB的分数空间。实数attention方程保持，FP32归约和低精度P量化分组变化；代价为额外KV及分数读写。长度65/2048两项定向检查包含padding/空行/奇尾、QPE stride80、三轮淘汰、Graph两次重放、抽样output/LSE以及全部有效cache槽payload/映射，均通过。原LRU/真实远端载荷保持。但性能显著回退，不扩大验证，也不把回退精确归因到某项未测stall。

第二候选将Q共享矩阵腾给64行KV的第二stage，保持147968B dynamic shared，按MMA片段从全局缓存读取Q_NOPE，item内QPE仍复用；原生64KV QK/MDO/PV替代32KV配对流程。它区别于已拒绝的64KV单stage，尝试同时减少tile轮数并保留流水。两项相同范围定向检查通过，但100GB慢43%–44%，拒绝。首次CPU构建的片段坐标在GPU运行前经review修为原生连续32KV/warpgroup，旧头、日志和构建记录保存在pre_gpu_layout/，未产生旧稿GPU数据。

前两组共4项定向检查、12格/6配对、2304计时launch/144段。原始数据：[两遍候选](../temp/glm_trace/tune_two_pass_20261102/performance_audit.json)、[全局Q候选](../temp/glm_trace/tune_global_query_20261102/performance_audit.json)。

第三候选保留完整原attention、prepare/resolve、精确launch LRU、fetch64行双缓冲/source hint/cache写入/ready协议与FAR hooks，只改变fetch任务分配。先由全部fetch CTA覆盖约前三分之一chunk，再让前1/3 fetch CTA转入原attention反向工作队列，其余CTA覆盖搬运尾段。f24时从24个fetch CTA转为16个持续取数、8个提前计算；每个远端chunk仍恰有一个搬运者。它与worker辅助取数/额外warp/多buffer方案不同，检验提前释放SM的收益能否抵过较少fetch CTA的吞吐损失；固定一个策略，未扫描边界/比例。

856个CPU队列覆盖检查和正常/EMU BF16构建通过，候选SHA031277ef。19:48:34 GPU3被占，预检exit1，尚无CUDA子任务。19:50复查发现GPU0/1空闲，转SXM源完成两项定向检查及三个新旧短对照：100GB0µs基本持平、20µs慢5.96%，50GB快0.13%，拒绝，不扫描退休边界/比例。提前释放计算SM在该策略下没有端到端收益，但未测出严格因果时间拆分。[原始对照](../temp/glm_trace/tune_elastic_fetch_20261102/performance_audit.json)。

第四候选shared_claim用16384槽共享hash，在同pool最多4行内先去重miss，再向原全局claim表做CAS。全局CAS仍跨group/pool选owner；owner page ID先放entry_victim/victim_of，原rowprefix resolve再复制至dense unique_pages并发布slot_pool，原后resolve grid barrier之后才由LRU覆写victim buffer。没有inline等owner、稀疏fetch job或新增grid barrier，attention/fetch helper规范化名字后逐字相同。maxK≤2048、每组最多8192页的派发界限及scratch生命周期已检查；首CPU稿的claim_entry/slot_pool别名在GPU前修正，旧件存pre_gpu_alias/。候选a47ca8b8，未安装。[源码审查](../temp/glm_trace/tune_shared_claim_20261102/source_review.json)。

第四候选两项定向检查通过，新增不等长pool、跨pool重复页/padding，抽样output/LSE及全部有效cache载荷/映射通过。三格新基线完整，候选100GB0µs记录生成后，结束资源检查发现GPU0/1新进程581553/581554，runner退出1，raw的complete=false；该格保留且**不纳入有效性能结论**，其余两格尚未运行。20:04复查GPU0/1已有外部VLLM进程、GPU3也有外部任务，均不抢占。`resume_candidate.py`已备好，仅在新预检空闲后补候选缺失三格，旧raw会按原benchmark规则归档；无需重跑基线或定向检查。

合计8项定向检查，完整有效21格/9配对、4032计时launch/252段（包含第四候选3格完整基线）；另1格受资源干扰而不完整的候选raw保留。未重复整套pytest、sanitizer、完整窗口或全矩阵，没有本轮正常物理链路或H800性能结论。所有本轮runner已结束，没有等待器或后台任务；生产754706f8、canonical数学头34af、FAR模拟器5e409保持。110%目标仍未完成。
