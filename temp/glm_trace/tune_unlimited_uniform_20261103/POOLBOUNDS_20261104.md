# 真实pool边界直接构造

复用resolve持久化row_slot_base[pool_row_start]，pool的packet区间为[ceil(first/20),ceil(last/20))。每CTA leader O(num_pools)计数，线程并行写所拥有区间，去掉每CTA扫描全部包。保留pool%grid所有权与同pool计算行、原ready索引、真实unique载荷/LRU/数学。180组CPU随机含空pool exact packet partition通过；GPU两项K65/K2048数学/LSE/padding/QPEstride/Graph/cache载荷通过。性能与实际库SHA见poolbounds_native_unlimited_f24.json，源码SHA见poolbounds_source.json。生产不变，尚未解决静态pool负载，不外推100GB。

正常GPU3同卡HBM BF16MTP8R128K2048LRU4096、256位置32步×3冷pass5417.696044/5416.383982/5449.600011µs，均5427.893346µs；与poolparallel5457.664028接近，仍远慢于clean约3128。拒绝安装，不追加100GB。该控制表明全量扫描消除后剩余主差距仍在，不能再将退化主要归于扫描；负载、wait和代码生成仍未分清。
