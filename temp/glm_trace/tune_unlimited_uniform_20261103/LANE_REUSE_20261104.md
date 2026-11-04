# 同线程远端buffer复用去CTA屏障：拒绝

从next_packet_qk_issuebar出发，只删除真实stage/cache stores之后、下一包issue之前的额外CTA barrier，保留编译器memory clobber、原MDO发布屏障及模拟器全CTA已发出后预约屏障。20页1440个uint4向量CPU检查严格一对一线程所有权，stores与next cp.async覆盖同线程地址。依据NVIDIA CUDA Programming Guide的generic-proxy async thread规则：先前同线程普通访问先于LDGSTS；不涉及TMA异步proxy。出处：https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html 。不是仅因测试通过而假定内存序正确。

GPU3/A100-PCIE、NUMA1、同卡独立HBM源；BF16、64heads、MTP8/R128/K2048/LRU4096、256位置32步、各1冷pass。正常及batch_call严格issuebar版各K65/K2048两项数学/LSE/padding/Graph/cache载荷通过；正式短筛全部最终524288有效cache槽载荷通过。正常candidate3075.872µs、随后原提前发包control3095.712µs，快0.641%。严格100GB20µs candidate4488.992、随后control4498.368，仅快0.208%；0µs candidate4303.136、随后control4315.104，仅快0.277%。反序20µs control4489.984、随后candidate4500.864，候选慢0.242%，方向翻转，拒绝不安装。

这说明本轮小差异不足以证明屏障可回收净收益；不否定上一轮提前发包相对迟发控制的6.36%信号，不把两个收益相加。源码next_packet_lane.cuh SHA256 f4bf9b2aa93ecc602c0481b7e87e1b725eec057ff3b1c48f6b45f33c7b3fbfc7；正常和EMU独立include/JIT、逐步raw、实际加载库SHA与硬件/NUMA绑定在next_packet_lane*、lane_control*、lane_issue_control*、lane_reverse* JSON。FAR仍复用batch_call 6ad89848，生产三header和默认staged路由未改。所有runner已结束；无物理链路/H800/full1024/sanitizer新结论，100GB目标仍未完成。
