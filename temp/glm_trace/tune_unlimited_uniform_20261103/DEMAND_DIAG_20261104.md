# 需求hint实际采用率

对demand_phase独立normal插桩，GPU3/NUMA1同卡HBM，BF16MTP8R128K2048LRU4096/256位置32步一pass。额外global计数开销保留，3175.744µs仅诊断不作优化成绩。五计数均在实际claim处；每步hint_success+fallback_success==ceil(unique/20)通过。

```json
{
  "hint_success": 29.03125,
  "hint_already_claimed": 39.125,
  "no_hint": 10803.03125,
  "fallback_success": 10734.15625,
  "fallback_skipped_claimed": 29.03125,
  "successful_hint_fraction": 0.002697272531952082,
  "per_step_claim_conservation": true
}
```

只0.2697%生产包按hint成功认领。大多数包在实际缺页前已fallback认领，当前机制主要是全局fallback队列加额外claim，不是有效需求调度。停止继续微调此版actual-wait hint；若重设计须提前识别待消费包并在本地计算窗口发起，保持unique读且不预订未来模拟预算。不能从normal采用率外推100GB，也不能说所有需求调度失败。性能脚本数学采样、全部最终cache载荷检查通过；未重复完整正确性suite/sanitizer，生产未改。raw/source/实际库SHA独立保留。
