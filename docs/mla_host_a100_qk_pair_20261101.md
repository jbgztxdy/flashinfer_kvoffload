# A100：不同 KV tile 的 QK 配对（2026-11-01）

两版均不采用，生产保持 `0f82e9de` / `b6eea095…e72a`。恢复提前页号读取后，短窗口结果在 ±0.14% 内，没有稳定改善主目标。

`qk_pair` 让两个不同 KV tile 共用一次 Q fragment 的 ldmatrix，分别维护 score。每个 score 仍按 PE、CKV 的顺序执行 MMA，softmax/PV 仍按原 tile 降序运行。只影响既有 cached whole-row、CTA_KV32、至少 4×SM 行的路径；单 cooperative kernel、fetch、LRU、数学头及模拟器不变。历史 `regpipe3` 增加的是 fetch 寄存器缓冲，未实现这个 attention 机制。

第一版在 PV 后读取下一对页号。独立第二版 `qk_pair_prefetch` 将这两次读取提前到 QK/PV 前，恢复原 loop 的页号延迟隐藏；配对 helper、展开 4、其他调度不变。两版均在 pair 入口等待两个 KV stage，不能将它们与原逐 tile 流水的差异仅归因于 Q fragment 复用。

每版均通过正常 BF16/FP16 构建和 105 项 GPU3 测试。反汇编证明全部 34 个 EMU、68 个正常构建原库内 kernel 的指令及资源逐一相同。新 kernel 都使用 REG255、SHARED2304、LOCAL0；第一版 STACK0，提前页号版 STACK8，未将后者写成零 spill。两版实际 Graph 派发均为一个新 `PairedQKHostBatchParams` kernel：108 CTA、每 CTA 256 线程、147968B 动态 shared、每 SM 驻留 1 CTA。432 行、K97 的采样最大绝对误差均为 0.004128。

GPU3 为 A100-PCIE-40GB，受限链路的 far 源是同卡独立 HBM，不能称作 NVLink。以下均为 MTP8 / 128 请求、LRU4096/请求、staged、fetch24、split off，256 个 trace 位置 × 3 次冷启动 pass；保留所有步和所有 pass 的算术均值。每版使用其独立 cohort 的同参数 base。

| 候选 | 链路 | base µs | 候选 µs | 耗时降低 | 候选 Memcpy µs | 纯 device µs | 候选逐步 max µs |
|---|---|---:|---:|---:|---:|---:|---:|
| qk_pair | 100GB/s、0µs | 3544.085 | 3831.371 | -8.106% | 6028.597 | 2556.960 | 2699.640 |
| qk_pair | 100GB/s、20µs | 3550.539 | 3851.669 | -8.481% | 6061.003 | 2556.960 | 2699.946 |
| qk_pair | 50GB/s、20µs | 5283.595 | 5304.576 | -0.397% | 8481.259 | 2556.960 | 4960.085 |
| qk_pair_prefetch | 100GB/s、0µs | 3544.864 | 3549.483 | -0.130% | 6044.811 | 2556.960 | 2699.793 |
| qk_pair_prefetch | 100GB/s、20µs | 3573.824 | 3569.728 | +0.115% | 6059.712 | 2556.960 | 2699.813 |
| qk_pair_prefetch | 50GB/s、20µs | 5284.107 | 5280.576 | +0.067% | 8478.496 | 2556.960 | 4959.704 |

第一版 base 的首次 preflight 遇到其他用户 `counter_payload`，失败日志和 state 保留；仅重试 base 成功，未删除有效候选样本。SXM runner 一直等待其他用户释放 GPU0/1；在 GPU3 已无采用信号后取消了自己的等待进程，没有发出 SXM launch，也没有 SXM 性能结论。未扩大到完整 1024 位置、多形状或正常物理链路确认。

[性能审计](../temp/glm_trace/tune_qk_pair_20261031/performance_audit.json)完整覆盖 12 格 / 6 配对、2304 次计时 launch、144 个有效 segment、两种参考、命中率、unique miss、guard、每步/每 pass 和源码 SHA，pending reference 为零。两版源码 SHA 分别为 `75ea7044f3ba677b1dc67b828651c8cb013dfd6bb88a7d2ff43a3f81922d6f9f`、`adc6aef1541bc46699771e823e8da75d9dbcd00861435cede95534a7d7f155fb`。本轮未改生产、参考或原 5304 格矩阵。

提前页号版在两个 100GB/s 点仍高于逐步 max 约 31.5% / 32.2%，没有解决历史约 32%–34% 的主差距。此证据支持停止当前配对调度；不再仅扫描展开数字或凭正确性通过扩大矩阵。
