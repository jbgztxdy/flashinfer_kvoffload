# 每CTA内部反序unique包：拒绝

依据实际依赖诊断与逆序KV消费，测试保留round-robin ownership、仅反转各CTA内部20页包顺序；不改行计算次序、数学、真实unique字节、cache/staging写入及ready协议。不同于连续分区、resident/near排序或fetch提前退休。索引helper带运行期计算，未声称等指令单因素控制。

CPU40组分区检查、GPU两项K65/K2048数学/LSE/padding/QPEstride/Graph/cache有效载荷通过。正常GPU3/A100-PCIE/NUMA1同卡HBM，BF16/MTP8R128K2048LRU4096、256位置32步×3冷pass：3426.144004、3445.919998、3481.759988µs，均值3451.274663µs。此前同源clean诊断单pass3113.055989µs，历史3pass约3129µs；跨轮非严格夹测，但明显无采用信号。拒绝安装，不追加100GB模拟，不能从负差精确分摊索引、等待和数据顺序成本。

全部逐步/库SHA见reverse_packets_native_unlimited_f24.json；源码SHA/分区见reverse_packets_source.json；编译/检查/runner原件独立保存。生产81093保持，无新物理链路、完整矩阵或sanitizer结论。far_rows0为无计数而非零载荷。该结果不支持仅反转生产顺序即可解决100GB差距。
