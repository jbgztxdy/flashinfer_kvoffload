# 实际pool所有权首版负结果

区别于此前均分：使用actual slot_pool[chunk*20]%grid给每个原20页packet分配owner，同CTA计算row_pool%grid的行。每CTA等待时推进其所有pool的包；ready仍为原全局chunk索引，unique字节/真实cache与staging写入/数学/LRU/单kernel保持。包列表放whole-row未用partial backing，与原flags分离；producer/count复用原未用诊断字段，移除会覆盖它们的诊断输出。

每CTA leader两次串行扫描全局包元数据（一次计数前序owner/自身数量，一次写自身packed list），有显著构造成本风险。静态pool mod grid还可能不均衡。本版不是已优化低频调度。

正常GPU3同卡HBM BF16MTP8R128K2048LRU4096、256位置32步×3cold pass：8459.840089、8461.632043、8460.192114µs，均8460.554749。两项K65/K2048数学/LSE/padding/QPEstride/Graph/cache全有效载荷检查通过；完整性能采样与最终cache载荷通过。拒绝安装，不测100GB；不能由端到端结果精确分摊扫描、负载和等待，也不能否定pool局部性概念。raw/source/库SHA和失败/runner日志保留，生产81093保持。

若后续保留此思路，必须用prepare已有row-prefix/pool边界构造包列表，避免每CTA串行扫描全部包，并量化prepare与compute负载；不得原样重跑。
