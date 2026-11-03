# 连续 unique 包所有权负结果

从 clean 冻结头隔离实现，将20页包的round-robin ownership改为按CTA均衡连续区间；计算行顺序、包大小、真实unique读、staging/cache写入、发布协议与数学不变。不是动态行队列或resident尾部排序。生产未改。

CPU44组ownership exact partition通过。正常构建GPU3/NUMA1、同卡HBM，两项K65/K2048定向数学/LSE/padding/Graph/cache载荷检查通过。BF16/MTP8R128/K2048/LRU4096/256位置32步×3冷pass，完整单kernel CUDA event计时：3661.151990、3666.751996、3698.751979µs，均值3675.551988µs。较历史clean3128.949µs明显更慢，但不是同轮严格单因素配对；无需追加100GB试验，拒绝安装。

索引helper新增运行期整数计算，连续供数与原轮转计算行也未建立局部关联；当前证据不能将退化精确归因于其中一项。不得声称连续所有权本身已被所有设计否定。后续若再做局部性，必须建立实际query/unique依赖对应，不能只换包分配顺序。

源码/实际加载库SHA及每步计时见contiguous_source.json、contiguous_native_unlimited_f24.json；失败/检查/runner状态见compile_contiguous.log、correctness_contiguous.log、contiguous_state.json。far_rows=0是继承的无诊断计数，不是无实际远端读取。新候选未另做Graph函数名probe，不将窄检查推广为全部场景。
