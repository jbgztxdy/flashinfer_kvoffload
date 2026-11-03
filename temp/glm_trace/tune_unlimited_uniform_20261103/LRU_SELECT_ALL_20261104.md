# 精确LRU select-all特殊情况前置验证

当前run_host_batch_lru即使victims==evictable仍走radix-select。此时threshold必为highest evictable key，rank为key==high的数量。不能简单用threshold=UINT_MAX/rank0替换：虽然选相同集合，会改变当前thread分组的victim列举顺序和page-slot配对。新形式必须保留threshold=high和精确rank，才逐位保持原victim顺序。

model_lru_select_all.py随机460case覆盖capacity1/65/256/512/4096/8192、被当前launch保护的UINT_MAX、跨字节epoch、多tie。CPU镜像原thread-stride/packed scan列举，threshold/rank/victim_slots顺序完全相同。仅CPU等价，不是GPU速度或launch LRU完整证明。尚未实现CUDA候选。

值得注意：所有evictable stamp相同时原radix已跳过，冷空cache通常不获益；统计high数量还需额外shared扫描，可能抵消优化。下一步须先测真实select-all且low!=high的发生频率，不应直接构建性能候选或更改tie策略。生产未改。
