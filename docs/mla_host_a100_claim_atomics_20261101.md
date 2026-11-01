# A100：prepare claim 原子访问实验（2026-11-01）

两个候选均不采用。claim probe 短筛选回退；claim store 短测和完整反向有正信号，但完整正向未确认100GB收益，扩展门槛失败。原失败状态和全部正负样本保留，未启动跨形状或SXM扩测。

直接基线是当前生产JQ `7193255e…d469`。两版只复制并修改prepare helper：probe先用GPU scope relaxed原子load查看claim，非零时跳过CAS；store保留全部原claim CAS，仅唯一owner发布编码row/rank时，把不使用返回值的atomicExch改为GPU scope relaxed原子store。claim的0→1→编码状态及prepare/resolve/clear阶段边界保持。规范化专用kernel名与prepare调用后，整个cooperative生命周期与生产JQ逐字相同。

原256线程、108CTA、32KV/2stage、dynamic147968B/static2304B、fetch24及64行双搬运缓冲保持；数学、精确launch级LRU、真实cache载荷、模拟器与参考均未修改。每版REG251/STACK0/LOCAL0；36个EMU及72个正常BF16/FP16原实例的指令、资源、机器字与控制字全部相同。

两版各105项正常测试加8项stress，共226项独立测试通过。stress覆盖长K2048、奇尾、128池、容量512/4096、空/单元素/padding、LSE/全部cache载荷、Graph多次重放、K65/QPE stride80，以及跨CTA共享97页/互斥2048页和多launch claim清空。原六项stress因fixture修复重复执行，独立测试数没有重复计入。实际Graph为单108CTA/block[32,4,2]/active1；每版首个匹配BF16/K97 launch的synccheck、racecheck、memcheck均零错。memcheck关闭API错误报告而保留设备检查；这些检查不代表全部trace的穷尽证明。

静态目标代码中每个原实例CAS8/EXCH9/strong GPU load14/store3；probe为8/9/22/3，store为8/1/14/11。它们确认实际生成的操作变化，不是运行期原子次数、stall份额或性能因果证明。内存协议推理与NVIDIA官方规范链接见memory_model_review.md。

| 候选/cohort | 位置/pass | 链路 | JQ µs | 候选 µs | 耗时降低 | Memcpy µs | 纯device µs | 逐步max µs |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| claim_probe/claim256_f24_ | 256/3 | g100_l0 | 3408.544 | 3479.808 | -2.091% | 6055.509 | 2556.960 | 2700.544 |
| claim_probe/claim256_f24_ | 256/3 | g100_l20000 | 3443.413 | 3494.400 | -1.481% | 6068.160 | 2556.960 | 2700.504 |
| claim_probe/claim256_f24_ | 256/3 | g50_l20000 | 5270.720 | 5321.376 | -0.961% | 8482.421 | 2556.960 | 4968.261 |
| claim_store/claim_store256_f24_ | 256/3 | g100_l0 | 3451.104 | 3442.048 | +0.262% | 6068.181 | 2556.960 | 2699.707 |
| claim_store/claim_store256_f24_ | 256/3 | g100_l20000 | 3471.819 | 3456.757 | +0.434% | 6074.869 | 2556.960 | 2699.675 |
| claim_store/claim_store256_f24_ | 256/3 | g50_l20000 | 5272.352 | 5267.029 | +0.101% | 8479.701 | 2556.960 | 4958.031 |
| claim_store/claim_store_forward1024_f24_ | 1024/5 | g100_l0 | 3491.835 | 3501.998 | -0.291% | 6284.045 | 2561.848 | 2744.362 |
| claim_store/claim_store_forward1024_f24_ | 1024/5 | g100_l20000 | 3523.350 | 3525.389 | -0.058% | 6313.542 | 2561.848 | 2744.832 |
| claim_store/claim_store_forward1024_f24_ | 1024/5 | g50_l20000 | 5507.402 | 5502.256 | +0.093% | 8888.314 | 2561.848 | 5187.603 |
| claim_store/claim_store_native1024_f24_ | 1024/5 | native_hbm | 3289.982 | 3294.027 | -0.123% | 4083.781 | 2561.848 | 2561.848 |
| claim_store/claim_store_native1024_f24_ | 1024/5 | pcie | 9967.173 | 9961.702 | +0.055% | 13327.390 | 2561.848 | 9592.671 |
| claim_store/claim_store_reverse1024_f24_ | 1024/5 | g100_l0 | 3520.917 | 3485.370 | +1.010% | 6276.672 | 2561.848 | 2744.511 |
| claim_store/claim_store_reverse1024_f24_ | 1024/5 | g100_l20000 | 3541.848 | 3516.906 | +0.704% | 6305.821 | 2561.848 | 2744.783 |
| claim_store/claim_store_reverse1024_f24_ | 1024/5 | g50_l20000 | 5507.731 | 5501.730 | +0.109% | 8888.581 | 2561.848 | 5187.644 |

共28格/14配对、22784计时launch/1424有效段，待参考0项、未完成raw0项。完整确认使用1024位置×5个冷启动pass，短测256位置×3pass，固定MTP8/R128/LRU4096/f24/split off。逐步/逐pass样本、实际命中率、unique miss、数学校验、guard、二进制及源码SHA均保留。两参考仍按同硬件原纯device及逐步max公式计算，未改时间或挑最好pass。

所有性能数据来自GPU3 A100-PCIE/NUMA1：受限far为同卡独立HBM，native_hbm也不是NVLink；pcie为实际Host pinned搬运。没有本实验的SXM/NVLink或H800性能结论。

首次generator边界断言发生在header生成前；新加两项stress最初漏掉Graph capture前warmup，由接口在GPU执行前拒绝，105+6项已通过。修正fixture后8项全部通过。修复validator曾读取尚未被新stress覆盖的旧失败状态，立即退出；等stress完成后原样重跑通过。原失败源码、日志、状态及修复记录均在test_fixture_failure/，没有删去失败或计入有效性能。扩测父进程34409实际terminal1是预设正向收益门槛拒绝，非GPU错误；不重启或放宽门槛。

生产仍为JQ7193，canonical math34af、模拟器5e409和原5304矩阵e3fd保持。claim probe SHA：`95ff4054a653135668daa8dc41b6fe6abbcb285b82da1254eb66c795be4f5c1c`；claim store SHA：`15aa954ac63ce52024439446ca9dde583875eaf5afa24fd8df343075ecd9c81e`。大batch100GB相对重叠参考的主差距仍未解决，整体goal active。

[原始记录与复现入口](../temp/glm_trace/tune_claim_probe_20261101/README.md)。
