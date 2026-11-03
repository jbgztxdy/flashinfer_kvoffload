# 实际依赖诊断

正常GPU3同卡HBM，clean冻结kernel，无计时kernel改动；各32步一pass，事件外从最后一步remap统计。首64个计算位置均本地（1024行），故起步远端等待假设被否定。进一步定位每行逆序计算中首次含far的64位置对，其依赖统计：

```json
{
  "rows": 1024,
  "unique": 177881,
  "mean_max_ticket": 41.595703125,
  "max_ticket": 82,
  "mean_producers": 13.5341796875,
  "own_fraction": 0.009618283784598685,
  "mean_first_far_pair_end": 374.75
}
```

仅最后一步，不是等待时间/完整trace因果证明；own_fraction按远端引用计。原始、实际源码/加载库SHA和全部时间在clean_dependency*_native_unlimited_f24.json。未改生产、参考或模拟器。下一设计应利用本地前缀计算窗口与真实首far依赖，不按均分近似声称局部性。
