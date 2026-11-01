# A100 attention 半缓冲环与未用 QPE copy 控制

更新于 2026-11-01 08:14 CST。五 half attention 环和恢复双 fetch 缓冲的 copy 控制均不保留。

**当前生产仍为已验证 JQ，SHA `7193255e…d469`；整体优化目标继续 active，大 batch 100 GB/s 距逐步 max 参考约 25%–27%。** 本实验的直接基线是尚未保留的 CPE `a43617…2b15`，不能把这里的差值写成生产收益。CPE 原 SXM 验证器保持独立运行。

## 实现与约束

原 attention 的两个 32 KV stage 改为五个 16 行物理 half：四个组成当前 hi/lo tile，第五个在 paired QK 前预取下一 tile 的前半；hi-PV 释放两 half 后填下一 hi 的后半及下一 lo 的前半，lo-PV 释放后填最后半。每对结束环起点倒退一格。仅对 near/resident/staged HBM 提前预取，远端 chunk 的 acquire 等待仍在原释放点。原 f24、64 行 fetch 双缓冲、FAR post-read hooks、真实 LRU 载荷及退出协议保持。

P half 使用 64 Q×16 KV 的 2 KB packing：`uint4_offset=row*2+(column^((row/4)&1))`，保留原 FP16/BF16 P 量化、rowsum、PE→CKV MMA 和降序 PV。256 线程/64 head，dynamic 158,336 B 加 static 2,304 B；正常及 EMU 新实例 REG255/STACK32/LOCAL0。不能把 LOCAL0 写成没有栈。奇尾采用同一 helper 的 unroll4；这是静态代码组织差异，未扫描 unroll/tile/fetch 参数。

布局枚举只证明写入无洞/无重叠、边界和逻辑 bank 映射，不能替代实际 CUDA 或 bank transaction 证据。111 项正常数值/cache/Graph 测试、实际单 kernel Graph、synccheck、racecheck 及新增 memcheck 另行核验。

恢复后的两 stage 控制 `no_copy_pe_fixed` 只去掉 CPE 已不读取的 shared QPE 分配和初始 copy，另用 union padding 保留原 147,968 B 存储/launch 预算及 fetch 双缓冲。其计算布局 139,904 B，完整存储 147,968 B。它用于检查半环之外的 copy/layout 成本，不把半环总差归因于单独预取。

## 先前控制的误判与完整归档

首版 `no_copy_pe` 缩小 `sizeof(SharedStorage)` 后，通用 fetch helper 会自动将 BUFFERS 从 2 选为 1。最初认为它会越界的判断已撤回：一个 73,728 B buffer 位于 139,904 B 内。只增加 host launch allocation 也不会恢复双缓冲，因此本次使用新的冻结 union-padding 控制，并重新编译/测试。

首轮九格保存在 `invalid_budget/raw/`；这个目录名反映初始假设，实际分类为意外单 fetch 缓冲控制。原 hash、正负差、日志和当时审计均保留。首轮半环相对 CPE 的 100 GB/s 慢约 19% 同样保留，未以归档为由抹去。`archived_policy_audit.json` 使用归档后的真实路径核验。

memcheck 首次报告 25 项错误并退出 86，打印的前 16 项均为 cuGetProcAddress_v2 invalid argument，另外 9 项未打印；原日志完整保存。复查使用 `--report-api-errors no` 保留设备内存检查；旧单缓冲控制、恢复双缓冲控制及半环首个匹配 kernel launch 均零设备错误。此范围不等于所有 trace/形状已做 sanitizer。

## 修正控制后的固定对照

GPU3 为 A100 PCIe、NUMA1，EMU 的远端 KV 在同卡独立 HBM；这不是物理 NVLink。共享锁及运行前后资源检查通过。MTP8/R128、LRU4096、f24、host split off，固定 base→control→half_ring 顺序，256 位置×3 个冷启动 pass；三设置均未改。短筛选不能替代完整 1024×5 pass 和跨场景门槛。

| 设置 | 候选 | CPE µs | 候选 µs | 降低 | Memcpy µs | device µs | 逐步 max µs | 高于 max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| g100_l0 | half_ring | 3385.707 | 4012.725 | -18.520% | 6037.749 | 2556.960 | 2699.747 | 48.63% |
| g100_l20000 | half_ring | 3399.371 | 4022.624 | -18.334% | 6047.339 | 2556.960 | 2699.955 | 48.99% |
| g50_l20000 | half_ring | 5268.181 | 5321.835 | -1.018% | 8483.520 | 2556.960 | 4959.789 | 7.30% |
| g100_l0 | no_copy_pe_fixed | 3385.707 | 3394.443 | -0.258% | 6053.835 | 2556.960 | 2699.614 | 25.74% |
| g100_l20000 | no_copy_pe_fixed | 3399.371 | 3434.859 | -1.044% | 6066.571 | 2556.960 | 2699.979 | 27.22% |
| g50_l20000 | no_copy_pe_fixed | 5268.181 | 5267.829 | +0.007% | 8471.776 | 2556.960 | 4959.573 | 6.22% |

所有逐步、逐 pass 时间和命中率保存在 raw；两参考沿用原同硬件数据与逐步 `max(C_t, unique_miss_t×1152/BW)` 公式，不调整模拟器、参考选择或原 5304 格。

修正后九格/六个对 CPE 配对、1,728 次计时 launch、108 个有效 segment；首轮另九格/六配对、1,728 launch/108 segment。三版本共 **333 项独立数值测试**（各 105+6），半环测试跨两轮复用，只计一次。修正后对照的两版本共 222 项。每版原 38 EMU/76 正常 BF16/FP16 kernel 的完整指令、资源、机器字和控制字保持相同。实际新 kernel 都是单 108 CTA、block[32,4,2]、每 SM active1；控制 dynamic147,968 B，半环158,336 B。

## 结论与复现

半环在修正后的同轮对照中，100 GB/s 的 0/20 µs 分别慢18.520%/18.334%，50 GB/s＋20 µs 慢1.018%；这与首轮约19%的100 GB/s负差信号一致。恢复双缓冲的 copy 控制分别慢0.258%/1.044%、50 GB/s快0.0067%，没有采用信号。停止这些原型，不继续扫描 half 数、unroll、fetch 或 cadence，也不扩展完整 trace/native/全形状回归。额外32B stack、packing/loader与地址控制开销是可能原因，现有静态证据不能给出各自的性能因果份额。生产7193保持；父CPE的完整SXM验证仍独立等待。

原 CPE 的 SXM 完整确认和 80 格矩阵仍由原父 runner15217/PID2084930推进；本 checkpoint 观察 phase 为 `sxm_full_confirmation`。此状态应结合最新实际进程及 broader_validation_state 读取，不要重复启动。

冻结源、`make_candidate.py`、`make_fixed_control.py`、`restore_includes.py`、独立 JIT 的代码生成摘要/SASS、测试、Graph 和 sanitizer 原件均在 [实验目录](../temp/glm_trace/tune_half_ring_20261101/)。重建缺失 include 用 restore_includes；CPU build 按 compile.py 的 TUNE_VARIANT 与独立 workspace。原生产、math、模拟器和矩阵 SHA 由 checkpoint_audit.py 核验；自动验证/benchmark 已结束时不要重启原 runner。

[修正后审计](../temp/glm_trace/tune_half_ring_20261101/performance_audit.json)、[首轮审计](../temp/glm_trace/tune_half_ring_20261101/archived_policy_audit.json)、[checkpoint](../temp/glm_trace/tune_half_ring_20261101/checkpoint_audit.json)、[纠正记录](../temp/glm_trace/tune_half_ring_20261101/invalid_budget_review.json)。
