# 运行时stall补采

与basic相同GPU3正常clean目标、512query/K2048正确性用例首launch。SchedulerStats：80.42%周期无eligible warp，0.24eligible warp/scheduler，2active warp/scheduler。默认WarpStateStats缺pcsamp，无归因；随后明确采集受支持的ratio指标，结果：

```json
{
  "smsp__warp_issue_stalled_barrier_per_warp_active.pct": "17.857892",
  "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "12.619576",
  "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": "20.499511",
  "smsp__warp_issue_stalled_wait_per_warp_active.pct": "22.209754"
}
```

这些为warp-active比例，非端到端可回收耗时，也非正式R128热trace/100GB。barrier、long scoreboard、short scoreboard、fixed-latency wait同时存在，不支持将瓶颈全归搬运或全归barrier。报告/日志/csv独立保留，两次pytest均通过，生产未改。下一实现需针对具体依赖链而非仅减少静态代码数。
