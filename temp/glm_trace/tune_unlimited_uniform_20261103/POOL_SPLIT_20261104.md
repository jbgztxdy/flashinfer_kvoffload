# Pool内任务拆分对照

poolbounds固定真实pool包ownership，恢复原逐行轮转计算得到poolrows，避免不可拆分8行pool导致16行CTA。进一步poolstripe将每pool的原20页packet区间按该pool query行owner分散，保持每包唯一owner、原ready索引/数学/LRU/真实stage/cache写回、单kernel和8mathwarp。计算仍逐行轮转。与旧动态rowqueue不同，无计算任务原子；与全局round-robin不同，producer限定为该pool的query行owner。

|正常GPU3同卡HBM，MTP8R128K2048LRU4096，256位置32步×3coldpass|pass0µs|pass1µs|pass2µs|均值µs|
|---|---:|---:|---:|---:|
|poolrows|3929.791987|3933.824010|3986.144014|3949.920004|
|poolstripe|3460.800007|3461.439997|3511.904001|3478.048002|

相比poolbounds5427.893µs，恢复行均衡显著改善；再分散包改善11.95%。但仍慢于clean约3128及原生产staged约3030，均拒绝安装、不追加100GB模拟。这些是跨轮定向对照，无同分配夹测，不作精确单项因果分摊。不能称pool局部性成功或100GB目标完成。

两候选各2项K65/K2048数学/LSE/padding/QPEstride80/Graph/cache全有效载荷通过；性能采样及最终全cache载荷通过。poolstripe180组CPU随机包区间partition通过；真实最后一步max110packet/mean82.25，比整pool不均衡降低但仍逊于全局轮转。构造O(pools×rows_per_pool)且每tile读间接packet列表的成本均保留。两份raw header与实际加载库SHA经pool_split_audit.json重新核验。无新sanitizer/完整矩阵/物理/EMU结论，生产81093保持。

停止继续换pool映射数字；如继续需用等负载、等间接列表控制分离其成本，再决定是否重新设计共享机制。源码及runner均独立保留，不重跑构造器覆盖raw。
