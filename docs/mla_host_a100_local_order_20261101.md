# A100：重排resident/near尾部以改善跨MTP局部性（2026-11-01）

候选拒绝，不扩大该机制。生产保持已保留验证的CPEa436。

原whole-row attention item只由一个成功claim的CTA执行。它在load_q前复用动态shared区域，按signed物理行索引bitonic排序已经在HBM/LRU的尾部，再用原attention计算。miss前缀、prepare/resolve、LRU admission/eviction、farfetch、真实payload/flags、claim及grid协议保持。同一query的KV多重集合保持，在线FP32缩放/量化P的分组随遍历次序改变，不能声称逐位相同。

在原CPE大cached fully-staged派发上加1<maxK≤2048条件；原定义及其派发保留。保持单108CTA/256线程/32KV双stage/dynamic147968B/f24/64行双fetch缓冲，无新增workspace或grid barrier。排序最多66轮CTA比较同步，另有读写/同步，且把尾部写回global；这些成本可能超过局部性收益。

冻结候选SHA：`7f891959e2da8d2307dd10eb3b36b96506fbf524509629a3c27557cf0d20d6d1`；直接基线CPEa436。当前验证phase：`short_screen_complete`。

真实trace观察使用CPE正常构建、GPU3同卡独立HBM、MTP8/R128/LRU4096，连续16个epoch，采样epoch0/7/15和每8个请求中的一个，共48组实际remap。output采样及所有有效cache载荷检查通过；原物理索引快照gzip及源码SHA保留。按8query相同进度的32页窗口合并，页集合由366714降为261335（28.74%），总reference 389557保持。这是CPU局部性代理，不能当作实际L2事务/命中率、stall、并发或kernel加速；观察本身未测性能。

| 验证 | 当前结果 |
|---|---|
| 正常105项 | 通过 |
| 长队列/重排等12项 | 通过 |
| 原38实例指令/资源/全部机器字和控制字 | 全部相同 |
| 原76实例指令/资源/全部机器字和控制字 | 全部相同 |

EMU及正常BF16/FP16新目标均REG255/STACK8/static2304/LOCAL0，8B栈如实记录；正常CPE控制目标原为STACK0。新增K129测试对全部行解码cache/staging索引，核对页多重集合和尾部有序，检查重复页、padding、Graph、LSE、命中槽保护和全部cache载荷。另保留长K2048、奇尾、128池/不等长池、跨CTA共享/互斥页、K65/QPE stride80等覆盖。

| cohort/链路 | CPE µs | 候选 µs | 耗时降低 | Memcpy µs | 纯device µs | 逐步max µs |
|---|---:|---:|---:|---:|---:|---:|
| localorder256_f24_/g100_l0 | 3411.584 | 3883.040 | -13.819% | 6058.027 | 2556.960 | 2699.672 |
| localorder256_f24_/g100_l20000 | 3419.467 | 3903.851 | -14.165% | 6072.480 | 2556.960 | 2699.933 |
| localorder256_f24_/g50_l20000 | 5269.845 | 5283.712 | -0.263% | 8485.301 | 2556.960 | 4959.876 |

当前6格/3配对、1152计时launch/72有效段，两参考、逐步逐pass/实际命中与unique、源码SHA及全部正负格保留。

实际Graph与首匹配BF16/K97 launch的三sanitizer结果以原件为准，memcheck关闭API报告、保留设备检查；不称全部trace穷尽证明。GPU3独立far HBM不是NVLink。完整1024位置多pass、正常物理链路、实际SXM/NVLink和全形状回归仍是任何保留的前提。数学头、模拟器、两参考公式及原5304格矩阵均保持，整体目标未完成。

[实现、原件与当前状态](../temp/glm_trace/tune_local_order_20261101/README.md)。
