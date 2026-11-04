# 原生FA2 direct + 外部LRU（LRU不计入attention计时）

GPU0计算、GPU1远端HBM/NV12，完整当前GLM5.3 assistant64k trace（1024位置、128步），R128/MTP8/BF16/64heads/K2048，容量4096页/请求。目标是让原生FlashInfer attention和我们的融合kernel面对同一逐步LRU后页面状态，且只把attention event计入时间。

实现方式：先用当前生产direct host-cache路径在事件外回放每一步，记录真实device LRU的remap、命中slot、miss页与payload；随后在新的GPU0 near+cache VMM中将相同miss页写入对应cache slots，向原生FA2传入等价重映射索引。每一步的LRU决定、索引上传、miss payload写回均在event外；原生kernel只直接访问GPU0 cache hit和GPU1 miss页。没有把miss页预先搬入本地再计时，也没有开启融合kernel/模拟器。原生FA2实际Graph逐步核对128个BatchMLAPagedAttentionKernel，未见DirectPV/CleanService。

结果：native原生FA2 + 外部精确LRU 平均 **2603.355µs/步**，三pass 2599.672/2601.080/2609.312µs。原生attention event包含本地命中/远端miss读取、QK/softmax/PV及输出写回；LRU规划、上传、miss填充和校验在event外。平均远端引用509.523MB/步、占全部2.097M引用的21.090%，unique miss约225524页/步。

同一进程配对的一次生产融合direct（其device LRU和写回仍在event内）为 **2907.176µs**；原native外部LRU约快10.46%。这是定向配对，不替代三pass native均值。完整三pass native的产物同时保留；两者事件边界不同，差值包括融合kernel的LRU/claim/writeback/staging控制成本。之前无LRU native direct为9459.469µs，说明主要差别来自约2.4GB→0.51GB的远端流量下降，而不是仅数学循环。

正确性：step0/16/127各3行输出对reference，atol/rtol=.01；全部最终cache载荷由生产donor核对；VMM属性明确near/cache为GPU0、far为GPU1。native库SHA20caf1e9dcebb439dd3b2f7985e00d921a8c35d4f5f7a39aaba889d251c08755；生产融合库同时在配对进程加载但原生计时只统计BatchMLAPagedAttentionKernel。GPU开始前稳定idle10秒、共享锁，结束无foreign。生产代码未改。

注意：为了精确复用device LRU，miss payload复制使用事件外临时GPU操作；这符合“LRU外置且不计入原生kernel”要求，但也意味着原生kernel测试不包含miss页的缓存写回/预取开销，只有它实际发出的远端引用计时。若要比较完整系统端到端时间，应把外部LRU/填充一起加回。

原始：`native_direct_lru_replay_20261104.json`（3pass）、`native_direct_lru_pair_20261104.json`（配对）、对应bench/run/state/log与`native_direct_verification.json`。实际执行源码SHA有exists保护；首版本修复后重跑，未覆盖失败raw。
