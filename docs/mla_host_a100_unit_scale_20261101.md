# A100：在线 softmax 单位重缩放分支（2026-11-01）

三版均拒绝，生产保持 0f82e9de / b6eea095…e72a。跳过数学上单位重缩放的分支没有改善大 batch 100GB/s 的耗时。

只对已有 cached whole-row、至少 4×SM 行、完整 staging 的 SM80 路径增加独立 traits/params；保持 64 heads、256 线程、32 行 KV/2 stage、108 CTA、单 CTA 驻留、原 fetch24、QK/P/PV、tile 顺序和精确 LRU/真实缓存载荷。全部都是单 cooperative kernel，模拟器和原 5304 格矩阵未改。每版 105 项原正常测试和 4 项 BF16/FP16 长序列测试通过，共 327 项；长测试含 K2048、128 个池、容量512/4096、淘汰、零 miss、padding、Graph 重放及所有有效缓存载荷。

- unit_scale 保留原表达式计算 o_scale，仅当其精确等于1时跳过 d/O 乘法；score exponentials 不变。新 REG244。
- stable_scale 保留原 scale 计算，在最大值改变或 scale 非有限时才重缩放，跳过相同有限最大值下由 FMA 舍入产生的微小非单位 scale。新 REG242。
- early_stable 将检查移到 rescale 指数之前，在最大值不变且有限时直接跳过 rescale 指数和 d/O 乘法，非有限最大值仍计算原式。新 REG242。

后两版利用有限状态下 exp2(0)=1 的数学恒等式，改变 FP32 舍入，不能称为逐位相同，也不承诺维持有限数乘巨大 scale 溢出后的旧 NaN 伪影。没有改变 attention 方程或截断特征。原 traits 的检测恒为 false；每版全部原34个EMU/68个正常BF16/FP16库内实例（含纯device）指令和资源逐一相同。审计要求匹配每版私有 math/host SHA 和正常68实例等价证明才复用 canonical math 的同硬件参考，未更改参考数据或公式。

实际正常 BF16 Graph 每版只有一个对应新 traits kernel：grid[1,108,1]、block[32,4,2]、dynamic147968B/static2304B、active1。所有新实例 STACK0/LOCAL0，实际 probe 最大绝对误差0.004128。资源减少不能推断性能收益。

GPU3为A100-PCIE/NUMA1，far源是同卡独立HBM模拟，不是真实NVLink。MTP8/R128、LRU4096/请求、256位置×3次冷启动pass、f24/split off；所有步/pass算术均值，保留每步/每pass、命中率、unique miss及两种参考。纯device参考2556.960µs。三组各自独立同轮base，顺序分别候选→base、base→候选、候选→base；不能将跨组绝对时间拼作收益。

| 版本 | 链路 | 同轮 base µs | 候选 µs | 耗时降低 | 候选 Memcpy µs | 候选逐步 max µs |
|---|---|---:|---:|---:|---:|---:|
| unit_scale | g100_l0 | 3541.728 | 3812.715 | -7.651% | 6040.640 | 2699.736 |
| unit_scale | g100_l20000 | 3575.093 | 3812.683 | -6.646% | 6059.403 | 2699.972 |
| unit_scale | g50_l20000 | 5282.357 | 5304.299 | -0.415% | 8480.053 | 4959.942 |
| stable_scale | g100_l0 | 3542.091 | 3698.325 | -4.411% | 6052.587 | 2699.868 |
| stable_scale | g100_l20000 | 3547.712 | 3727.541 | -5.069% | 6057.867 | 2699.984 |
| stable_scale | g50_l20000 | 5281.365 | 5294.731 | -0.253% | 8488.629 | 4960.224 |
| early_stable | g100_l0 | 3558.667 | 3705.888 | -4.137% | 6051.552 | 2699.721 |
| early_stable | g100_l20000 | 3563.051 | 3711.307 | -4.161% | 6059.360 | 2699.806 |
| early_stable | g50_l20000 | 5285.195 | 5297.803 | -0.239% | 8491.616 | 4960.282 |

第一版100GB慢6.646%–7.651%，stable慢4.411%–5.069%，early仍慢4.137%–4.161%；50GB三版均略慢。因此停止本系列，不继续调整条件、cadence、unroll或几何，不扩大完整1024位置/5pass/物理链路确认，不采用。

为检查精确等于1的条件，保留了两项独立诊断。primary kernel的静态SASS依赖链为FMUL→FFMA→EX2→unit比较，并有真实BRA跳过约64条FMUL，不只是谓词乘法。独立CUDA表达式probe从两个数组加载65536对相同最大值，26440个算得scale不精确为1，最大误差3.57627868652e−7。它证明该浮点反例存在，不能说明真实attention的最大值稳定比例、实际分支频率，或将全部性能负差归因于舍入；诊断不计作性能或参考。源、编译日志、SASS和结果均保留。

[完整审计](../temp/glm_trace/tune_unit_scale_20261101/performance_audit.json)含18格/9配对、3456次计时launch/216个有效segment，参考无缺失、raw无未完成项。三个host/math SHA以及最终拒绝理由见decisions.json；所有源码快照可校验。与前一feature-QK实验合计545项测试、36格/21配对、6912次计时launch/432segment；两组均没有生产收益。100GB大batch约32%–34%主差距仍在，goal active。
