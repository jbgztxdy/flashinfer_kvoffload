# LRU select-all真实频率诊断

在clean实际run_host_batch_lru原选择前插桩，GPU3正常BF16MTP8R128K2048LRU4096/256位置32步1pass。pool miss4096次，select-all2446（59.7168%），但select-all且low!=high仅283（6.9092%），其余all-same stamp原radix已经跳过。victims总4619788。每步原计数lru_frequency_summary.json/raw保存。统计当前一次cache admission实例，不外推全部形状/trace。

由于精确保留victim顺序仍需额外count(high)扫描，机会仅6.9% pool项，暂不实现新CUDA选择分支。不是证明永远不值得优化，但不作为当前约22%端到端差距主要方向。插桩添加counter atomics和init grid barrier，3081.728µs只诊断，不能与其他轮次推断加速。性能脚本数学采样和全部最终cache载荷通过；未重复suite/sanitizer。source/实际加载库SHA由raw记录并再核验。生产81093不变，runner结束，继承far_rows打印字段此处是victim计数而非远端流量。
