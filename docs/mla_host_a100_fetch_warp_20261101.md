# A100：warp 按行取数筛选（2026-11-01）

候选不采用。三组短窗口对照均未变快，生产保持 `0f82e9de` / `b6eea095…e72a`，主目标仍 active。

原 fetch 按连续向量分配线程，在每次展开中解析页号、行内偏移和 cache victim。新候选让一个 warp 负责一行，一次解析该行地址后完成 72 个 16B 向量的远端读取、staging 写入和 cache 写入。64 行 chunk、双缓冲、就绪标志、fence、LRU 选择/绑定、deferred copy、attention 数学/循环和模拟器 hook 均保持原样。只对已有 cached whole-row、CTA_KV32、至少 4×SM 行的分支启用。

代价是每行要执行三轮 warp 指令，最后一轮只有 8/32 lane 有效。减少地址计算能否抵消这部分开销需要实测，不能凭指令数推断性能。新目标实例的静态指令从 7888 减至 7744；两版均 REG244、STACK0、SHARED2304、LOCAL0，并未改善驻留 CTA 数。

正常 BF16/FP16 构建及 105 项 GPU3 测试通过。原 34 个 EMU / 68 个正常库内 kernel 的指令和资源逐一相同。实际正常构建 Graph 只有一个 `WarpRowsHostBatchParams` kernel：108 CTA × 256 线程、147968B 动态 shared、1 resident CTA/SM。432 行、K97 的采样最大绝对误差 0.004128。

GPU3 为 A100-PCIE-40GB，模拟 far 源为同卡独立 HBM。MTP8 / 128 请求、LRU4096/请求、staged、fetch24、split off，256 个 trace 位置 × 3 次冷启动 pass，同参数 base 先、候选后，所有步及 pass 算术平均：

| 链路 | base µs | 候选 µs | 耗时降低 | 候选 Memcpy µs | 纯 device µs | 候选逐步 max µs |
|---|---:|---:|---:|---:|---:|---:|
| 100GB/s、0µs | 3516.245 | 3543.093 | -0.764% | 6056.523 | 2556.960 | 2699.713 |
| 100GB/s、20µs | 3556.683 | 3557.749 | -0.030% | 6081.824 | 2556.960 | 2699.844 |
| 50GB/s、20µs | 5280.053 | 5284.683 | -0.088% | 8490.443 | 2556.960 | 4959.719 |

两种 100GB/s 条件下仍高于逐步 max 约 31.2% / 31.8%。没有稳定采用信号，因此不扩大完整 1024 位置、5 pass 或跨场景矩阵。没有 SXM / 正常物理链路性能结论。

[性能审计](../temp/glm_trace/tune_fetch_warp_20261101/performance_audit.json)覆盖 6 格 / 3 配对、1152 次计时 launch、72 个有效 segment、两种参考、命中率、unique miss、guard、逐步/逐 pass 和源码 SHA；pending reference 为零。源码 SHA 为 `a1566bf9cc2c959aa61bd5c157ce33fc20d2fee23bfc568e560cc80cccae404e`。生产、模拟器、参考与原 5304 格矩阵均未改。

H800 于 2026-11-01 00:29:43 +08 查询：五个原 Slurm 作业仍为 Priority / Dependency，两个 watcher 存活，没有覆盖排队源码或重复提交。所有本候选 runner 已结束。
