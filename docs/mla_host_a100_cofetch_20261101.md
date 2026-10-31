# A100：CTA 同时推进 attention 与取数（2026-11-01）

当前原型和单 stage 控制版均拒绝，生产保持 `0f82e9de` / `b6eea095…e72a`。100GB/s 主差距仍未解决。

原型 `cofetch` 将 attention 的两个 KV stage 减为一个，将空出的 36864B 用于两个 16 行远端取数缓冲。所有 CTA 各自负责循环分配的 disjoint chunk，每八个 KV tile 推进一次取数；遇到未就绪 tile 时先完成自己的取数，退出前排空剩余队列。所有缓存池先完成原 LRU 选择，再经一个全 grid barrier 开始搬运。attention 的 MMA、softmax/PV 顺序、缓存容量与载荷、单 cooperative kernel 均保持；原模拟器的 issue/wait hook 与字节计算直接复用，没有修改模拟器或参考。

独立 `stage1_control` 也减少一个 attention stage，但保留原专用 fetch CTA 调度及 64 行双缓冲。padding 将其动态 shared 保持为原 147968B，避免把 fetch 缓冲数量同时改变。控制版用来观察减少 attention stage 的成本，不能据它精确相减分摊原型的各项开销。

每版通过 105 项原正常测试和 2 项新增长队列测试，共 214 项。新增测试使用 128 个池（超过 108 SM）、MTP8/K2048、容量 512/4096、多轮淘汰、零 miss、padding、Graph 捕获/重放，检查采样数学结果以及所有有效缓存行的映射和逐字节载荷。它覆盖旧 K97 whole-row 测试未充分触及的队列多轮复用。两版全部 34 个 EMU / 68 个正常原库内 kernel 的指令及资源逐一相同。

| 新实例 | EMU REG | 正常 REG | STACK / LOCAL | static shared B | dynamic shared B | 实际驻留 CTA/SM |
|---|---:|---:|---|---:|---:|---:|
| cofetch | 244 | 242 | 0 / 0 | 38272 | 111104 | 1 |
| stage1_control | 240 | 240 | 0 / 0 | 2304 | 147968 | 1 |

实际正常构建 Graph 均只有一个新 kernel，108 CTA × 256 线程，采样最大绝对误差 0.004128。没有把 CPU 编译成功当作新路径被执行的证据。

GPU3 A100-PCIE，同卡独立 HBM 模拟 far，MTP8/R128、LRU4096/请求、staged、fetch24/off、256 位置 × 3 次冷启动 pass，同参数 base：

| 候选 | 链路 | base µs | 候选 µs | 耗时降低 | 候选 Memcpy µs | 纯 device µs | 候选逐步 max µs |
|---|---|---:|---:|---:|---:|---:|---:|
| cofetch | 100GB/s、0µs | 3556.096 | 4940.832 | -38.940% | 6028.661 | 2556.960 | 2699.747 |
| cofetch | 100GB/s、20µs | 3543.595 | 5422.752 | -53.030% | 6035.509 | 2556.960 | 2700.025 |
| cofetch | 50GB/s、20µs | 5284.096 | 6116.896 | -15.761% | 8484.032 | 2556.960 | 4960.203 |
| stage1_control | 100GB/s、0µs | 3556.096 | 3975.509 | -11.794% | 6039.509 | 2556.960 | 2699.810 |
| stage1_control | 100GB/s、20µs | 3543.595 | 3978.880 | -12.284% | 6060.117 | 2556.960 | 2699.782 |
| stage1_control | 50GB/s、20µs | 5284.096 | 5308.917 | -0.470% | 8486.667 | 2556.960 | 4959.583 |

减少 attention stage 已有明显成本；当前所有 CTA 的 16 行取数队列也没有带来足够收益。因此不扩大完整 1024 位置、正常物理链路或跨场景矩阵。此结果不等于否定所有 CTA 辅助取数设计，也没有证明全部退化来自某一个等待或指令。

[审计](../temp/glm_trace/tune_cofetch_20261101/performance_audit.json)保存 9 格 / 6 配对、1728 次计时 launch、108 个 segment、两参考、每步/每 pass、命中率、unique miss、guard、全部慢格及 source SHA。首次 control 的全局 preflight 因其他用户 counter 进入退出，原失败日志/state 保留；确认原 runner terminal1 后，仅补缺失的两候选，均完成。补测后段的独立监视器有 89 个一秒采样、零采样冲突；它不覆盖早先样本，也不能证明亚秒级独占。所有正式格仍有原资源预检/末检。

原型 SHA：`af61098d178108843357bee2218d6f6719a2f3355591a5efe6f77dc6eebbbf6f`；控制 SHA：`020b565681fa948aef764d8d4faa795bd91867f0b6e9a3134c38016aa82b0b41`。H800 00:45:46 +08 五个原作业仍 Priority/Dependency，两个 watcher 存活；无本轮 H800 结论。

下一独立实验保留原两 stage attention 和专用 64 行取数主通道，用剩余 shared 给 worker 配两个 4 行辅助缓冲。该候选仍需完整编译、载荷/进度和同参数性能门槛，不能由本原型的正确性外推。
