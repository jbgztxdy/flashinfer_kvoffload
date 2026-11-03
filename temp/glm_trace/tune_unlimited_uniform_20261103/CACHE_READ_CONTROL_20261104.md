# 最终cache读取路径的双写控制

single_destination的提前victim解析、最终cache读地址、原global remap和packet-ready均不变，仅恢复clean原stage+cache真实双写。所有必要计算仍在计时内，诊断版本非待保留优化。

正常GPU3同卡HBM BF16MTP8R128K2048LRU4096/256位置32步×3coldpass3373.663999/3392.288007/3427.999996µs，均3397.984001。单目的地3320.319995约低77.664006µs（2.29%）；但二者都慢于clean约3128。结果支持省真实写入有收益信号，却不足抵消最终cache地址/读取布局综合成本；跨轮非同分配严格消融，未控SASS资源，不精确归因。当前cache直读形式停止，不继续小变体。

性能脚本数学采样和全部最终有效cache载荷通过，复用前驱两项定向检查，未重跑suite/sanitizer。source/实际加载库SHA重新核验见audit，全部raw保留。无新硬件字节或100GB成绩，生产81093保持。
