# A100：去重远端源读取使用L2 evict-first提示（2026-11-01）

已按用户要求的关键场景验证保留实测源码，当前生产为48c2a724…86f73。

仅专用fetch中的去重far-source cp.async增加L2 evict_first策略，固定fraction1.0。每个源page在本launch去重后只读取一次，attention读取staging/cache，源数据与计算数据的复用方式不同。测试其硬件cache竞争假说；它只是性能提示，不能推断缓存命中或收益。attention、staging/cache写入和元数据读取策略、数学计算、精确LRU、ready/claim协议及FAR模型均保持。

新typed派发条件与CPE相同：单108CTA/256线程、32KV双stage、dynamic147968/static2304B，原64行双fetch缓冲和f24、1152字节/unique页、原commit/wait/barrier/FAR post-read hook全部保持。原38EMU/76normal实例指令、资源、机器字和控制字精确相同；九个attention helper规范化命名后逐字相同。新REG255，EMU STACK8、正常BF16/FP16 STACK0、LOCAL0，与CPE资源级别相同。

105+10=115项正常数值/长队列/LSE/全cache-payload/Graph测试通过；实际Graph仍单108CTA/block32x4x2/dynamic147968/active1。首匹配BF16/K97 launch的synccheck/racecheck/memcheck全零错误；memcheck关闭API报告、设备检查保留，不作穷尽证明。三种构建静态目标均有36条带desc的LDGSTS，策略描述符高字为0x12f00000、低字为0；静态指令确认提示进入二进制，不是动态访问次数、命中率或stall因果份额。

| 硬件/cohort | 位置/pass | 链路 | CPE µs | 候选 µs | 耗时降低 | Memcpy µs | 纯device µs | 逐步max µs |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| GPU3/streamsource256_f24_ | 256/3 | g100_l0 | 3419.307 | 3405.099 | +0.416% | 6075.680 | 2556.960 | 2699.721 |
| GPU3/streamsource256_f24_ | 256/3 | g100_l20000 | 3434.603 | 3430.219 | +0.128% | 6098.571 | 2556.960 | 2699.869 |
| GPU3/streamsource256_f24_ | 256/3 | g50_l20000 | 5268.939 | 5270.048 | -0.021% | 8484.512 | 2556.960 | 4959.885 |
| GPU3/streamsource_forward1024_f24_ | 1024/5 | g100_l0 | 3480.251 | 3463.062 | +0.494% | 6289.736 | 2561.848 | 2744.745 |
| GPU3/streamsource_forward1024_f24_ | 1024/5 | g100_l20000 | 3500.221 | 3486.115 | +0.403% | 6311.555 | 2561.848 | 2745.485 |
| GPU3/streamsource_forward1024_f24_ | 1024/5 | g50_l20000 | 5505.429 | 5501.014 | +0.080% | 8888.016 | 2561.848 | 5189.447 |
| GPU3/streamsource_native1024_f24_ | 1024/5 | native_hbm | 3247.563 | 3236.062 | +0.354% | 4060.872 | 2561.848 | 2561.848 |
| GPU3/streamsource_native1024_f24_ | 1024/5 | pcie | 9967.752 | 9951.293 | +0.165% | 13323.957 | 2561.848 | 9594.519 |
| GPU3/streamsource_reverse1024_f24_ | 1024/5 | g100_l0 | 3478.936 | 3455.752 | +0.666% | 6281.941 | 2561.848 | 2744.705 |
| GPU3/streamsource_reverse1024_f24_ | 1024/5 | g100_l20000 | 3509.139 | 3481.059 | +0.800% | 6312.762 | 2561.848 | 2745.560 |
| GPU3/streamsource_reverse1024_f24_ | 1024/5 | g50_l20000 | 5505.550 | 5504.747 | +0.015% | 8887.570 | 2561.848 | 5189.544 |

当前审计22格/11配对、21632计时launch/1352有效段，待参考0项、未完成raw0项。全部正负格、逐步逐pass、实际miss/命中率、guard、源码和binary SHA保持；使用全样本算术均值，每pass冷启动，不取最好样本。两参考仍为原同硬件纯device及逐步max公式。

完整确认状态：`local_confirmation_complete`；扩展状态：`stopped_before_matrix_per_user_scope`。

GPU3为A100-PCIE/NUMA1，受限far与native_hbm均为同卡独立HBM，不能称NVLink。SXM0/1若有结果来自真实NV12/NUMA0，绝对时间不混拼。完整156矩阵、SXM完整两顺序确认、正常物理链路及大范围回归尚未执行。用户18:32明确要求减少逐轮验证，已停止尚未启动任何矩阵的CPU扩展父；已有数值/代码/实际路径及完整本地关键对照用于本轮机制判断。实测原件逐字节安装，未重复生产root新编译/整套测试，后续按具体疑点补测，不声称所有形状/SXM均已验证。

当前实测源码SHA：`48c2a724548c2ac554400b26592a1212e159cd279cd1a48da191b5139cd86f73`。canonical数学34af、FAR5e409及原5304矩阵e3fd保持。整体目标仍未完成。

[原始记录与复现入口](../temp/glm_trace/tune_stream_source_20261101/README.md)。
