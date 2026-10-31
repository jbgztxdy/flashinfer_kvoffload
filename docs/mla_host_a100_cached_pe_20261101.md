# A100：在整个 attention item 内复用 QPE MMA 片段（2026-11-01）

QPE片段复用候选在GPU3完整两顺序100GB对照均有正信号，76格本地矩阵及两项负差夹测已完成；SXM确认/80格矩阵仍待，尚未安装。

对照是冻结 JQ `a2f37f…`，不是原生产 b6。候选从global按原ldmatrix.x4的MMA布局读取64个QPE特征，每lane保存16个uint32 packed word，在该item所有成对及奇尾KV tile中复用；PE→CKV MMA累加次序、联合max/d/O缩放、量化P/PV/refill和全部576特征保持。原QPE共享分配和初始copy也保留，以隔离复用机制。

保持原256线程、64head、32KV/2stage、108CTA、fetch24专用CTA及精确launch级LRU/真实cache载荷。只在原大batch cached fully-staged JQ条件下派发新traits/params/kernel；没有tile/unroll/fetch参数扫描。这不同于N9中每tile重读global QPE的288线程服务warp方案。canonical math、模拟器、参考和原5304格矩阵不变。

| 构建 | 新资源 | 原JQ实例代码证明 |
|---|---|---|
| EMU BF16 | REG:255 STACK:8 SHARED:2304 LOCAL:0 CONSTANT[0]:672 TEXTURE:0 SURFACE:0 SAMPLER:0 | 36实例指令/资源精确相同 |
| 正常BF16/FP16 | REG:255 STACK:0 SHARED:2304 LOCAL:0 CONSTANT[0]:672 TEXTURE:0 SURFACE:0 SAMPLER:0 | 72实例指令/资源精确相同 |

另逐项验证36EMU/72normal实例的Ampere机器字和控制字。正常105项测试和新增6项均通过，共111项。新增覆盖BF16/FP16、K2048/63tile奇尾、128池、容量512/4096、空/单元素/padding、output/LSE、淘汰及全部cache载荷、Graph多次重放，以及大形状K65/QPE head stride80。实际Graph仍为单108CTA/block[32,4,2]/dynamic147968B/static2304B/active1；首个匹配launch的synccheck和racecheck均零错误。

静态EMU反汇编为LDSM198→190、LDG85→101、HMMA299不变、BAR58不变；总6952→6968条。固定CUDA13.1/sm80原件中，16个QPE word在成对tile循环前读入R12–R27，循环内不覆写它们，也无LDL/STL。8B stack对应的LDL/STL位于该循环范围外，因此整kernel仍有stack使用。此指令/活跃区间证明不等于动态work、stall分摊或性能收益，详见static_instruction_review.json及完整SASS。

| 硬件/cohort | 位置/pass | 链路 | JQ µs | 候选 µs | 耗时降低 | Memcpy µs | 纯device µs | 逐步max µs |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| GPU3/cached256_f24_ | 256/3 | g100_l0 | 3429.867 | 3421.216 | +0.252% | 6060.043 | 2556.960 | 2699.815 |
| GPU3/cached256_f24_ | 256/3 | g100_l20000 | 3470.827 | 3450.539 | +0.585% | 6077.728 | 2556.960 | 2699.867 |
| GPU3/cached256_f24_ | 256/3 | g50_l20000 | 5270.613 | 5270.752 | -0.003% | 8485.728 | 2556.960 | 4959.953 |
| GPU3/cachedforward1024_f24_ | 1024/5 | g100_l0 | 3508.710 | 3479.150 | +0.842% | 6295.194 | 2561.848 | 2744.781 |
| GPU3/cachedforward1024_f24_ | 1024/5 | g100_l20000 | 3528.082 | 3506.997 | +0.598% | 6317.298 | 2561.848 | 2745.595 |
| GPU3/cachedforward1024_f24_ | 1024/5 | g50_l20000 | 5508.011 | 5500.387 | +0.138% | 8888.483 | 2561.848 | 5189.467 |
| GPU3/cachednative1024_f24_ | 1024/5 | native_hbm | 3296.746 | 3253.398 | +1.315% | 4085.365 | 2561.848 | 2561.848 |
| GPU3/cachednative1024_f24_ | 1024/5 | pcie | 9963.818 | 9961.877 | +0.019% | 13295.499 | 2561.848 | 9594.053 |
| GPU3/cachedreverse1024_f24_ | 1024/5 | g100_l0 | 3517.976 | 3466.235 | +1.471% | 6274.264 | 2561.848 | 2744.870 |
| GPU3/cachedreverse1024_f24_ | 1024/5 | g100_l20000 | 3534.850 | 3488.069 | +1.323% | 6307.341 | 2561.848 | 2745.535 |
| GPU3/cachedreverse1024_f24_ | 1024/5 | g50_l20000 | 5508.152 | 5504.982 | +0.058% | 8886.904 | 2561.848 | 5189.439 |

当前审计104格/49配对，175520计时launch/11024有效段；待参考0项、未完成raw0项。每步/每pass、全部正负差、命中率、unique miss、队列guard、源码SHA与原件均保留。使用所有有效样本的算术均值，每个pass冷启动。

| 回归计划 | 已审计 / 计划格数 | 状态 |
|---|---:|---|
| gpu3/limited | 40 / 40 | 完整 |
| gpu3/native | 36 / 36 | 完整 |
| sxm/limited | 0 / 60 | 待完成 |
| sxm/native | 0 / 20 | 待完成 |

两项未改路径原负差（MTP1/R1 HBM慢1.133%、R256 HBM慢1.078%）均保留；追加同参数完整1024位置×5冷pass的前/候选/后夹测如下。代码等价和未复现不证明唯一因果，也不承诺每次独立分配都更快。

| MTP/请求 | 前基线 µs | 候选 µs | 后基线 µs | 较两端均值降低 |
|---|---:|---:|---:|---:|
| 1/1 | 41.616 | 41.637 | 42.203 | +0.650% |
| 1/256 | 1061.371 | 1067.660 | 1063.863 | -0.475% |

GPU3为A100-PCIE/NUMA1，受限far是同卡独立HBM，不能称为NVLink；SXM0/1数据来自实际NV12/NUMA0。MTP1/R512受限链路必须在SXM完成，不能拿GPU3代替。原JQ保留及生产root109项验证已完成，当前生产为7193255e…d469；本候选继续使用与该清理源全部36/72实例机器字等价的冻结a2f37f控制。完整回归、正常物理链路、SXM完整确认和所有负差门槛均通过后，才可考虑冻结保留和root新编译/测试。

初始generator因签名空格差异触发断言，在创建候选前失败；随后编译前检查发现新helper的const数组引用与原MMA签名不兼容，改为仅供MMA读取的非const引用。失败源/说明及未编译28d149版本均保留，未测量或混入其时间。

当前已测候选SHA256：`a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15`。整体目标继续active，100GB大batch剩余差距未解决。

[原始记录、状态与复现脚本](../temp/glm_trace/tune_cached_pe_20261101/README.md)。
