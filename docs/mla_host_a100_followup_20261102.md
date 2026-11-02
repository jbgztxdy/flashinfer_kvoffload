# A100 后续机制实验（2026-11-02）

从已保留生产 SS `48c2a724…86f73` 开始，继续独立 include/JIT 与冻结基线。本轮按用户“每次不用验证这么多，确认机制有效、在不断优化就行”执行定向检查和少量新旧对照，未重复全套 pytest、sanitizer 或矩阵。总体优化目标仍未完成。

前两个候选均未安装。LRU cache streaming store 的三个性能点没有收益；去除未使用的 shared QPE copy 也没有明确收益，不因源码工作量减少就保留。

| 机制 | 链路 | 原生产 µs | 候选 µs | 耗时降低 | Memcpy µs | 纯 device µs | 逐步 max µs |
|---|---|---:|---:|---:|---:|---:|---:|
| LRU cache streaming store | g100_l0 | 3382.848 | 3383.424 | -0.017% | 6070.347 | 2556.960 | 2699.797 |
| LRU cache streaming store | g100_l20000 | 3388.320 | 3419.072 | -0.908% | 6086.517 | 2556.960 | 2699.872 |
| LRU cache streaming store | g50_l20000 | 5270.549 | 5271.499 | -0.018% | 8478.912 | 2556.960 | 4960.140 |
| 去除未使用的 shared QPE copy | g100_l0 | 3366.304 | 3376.939 | -0.316% | 6046.059 | 2556.960 | 2699.742 |
| 去除未使用的 shared QPE copy | g100_l20000 | 3393.323 | 3388.725 | +0.135% | 6067.755 | 2556.960 | 2699.963 |
| 去除未使用的 shared QPE copy | g50_l20000 | 5267.189 | 5271.019 | -0.073% | 8484.523 | 2556.960 | 4960.471 |

全部为 GPU3/A100-PCIE/NUMA1、BF16/64 heads、KV512+64、topk2048、MTP8/128请求、LRU4096/请求、f24/split off、256位置×3冷 pass。受限 far 来自同卡独立 HBM，不是 NVLink。每个候选为同轮同参数 base→candidate；不同轮绝对时间不混拼。全部逐步、逐 pass、真实 miss/命中率、正负差、源码和二进制 SHA、正确 Memcpy 及同硬件纯 device/逐步 max 参考保留。逐步 max 先对每步计算 max，再求均值。

cache store 候选 `0805d7c0…acbba` 只把专用 fetch 的 LRU payload 写入改为 `st.global.cs.v4.u32`，当前 launch 使用的 staging 写入、源 hint、ready/fence/commit/wait、精确 LRU、真实字节与数学保持。其依据是 staging 更快被消费、cache 供后续 launch 使用的复用距离不同。[NVIDIA PTX cache operators](https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators) 将 st.cs 定义为带 evict-first 的性能提示；它不改变内存一致性语义，也不证明实际命中率或收益。三个 BF16 定向测试覆盖稀疏/不等长池、128请求长队列、容量512/4096、output/LSE、Graph多次重放与每个有效cache槽逐字节载荷。正常及 EMU 各38原 BF16实例机器字、控制字和资源相同；目标REG255/static2304/LOCAL0，EMU STACK8、normal STACK0，25条静态 STG.EF。此处不是动态store数或命中率证据。

query copy 候选 `9faf0219…9f5df` 保留 item 级 QPE 寄存器读取和全部 MMA 次序，仅用专用 query loader 跳过不再读取的 shared QPE 上传。保留原 shared 分配/布局和 CKV query copy，不声称容量或occupancy收益。四项 BF16定向检查在上述三项之外覆盖QPE head stride80/K65。正常及EMU各38个原 BF16实例机器字/控制字/资源保持；目标资源级别同生产。三个性能点为混合微小正负差，停止扩展。

两个候选各6格/3对、1152计时launch/72有效段，两参考均无缺失。没有本轮FP16/SXM/NVLink/H800或完整1024位置/全形状性能声明。没有重复sanitizer；数学、LRU与协议未改变，已有广泛证据继续复用。

原始入口：[cache store](../temp/glm_trace/tune_cache_store_20261102/README.md)、[query copy](../temp/glm_trace/tune_query_copy_20261102/README.md)。

H800 原队列已有真实结果：1004362于11月2日02:33–02:41成功；1040680于02:49因nvcc链接缺少libcudadevrt/libcudart_static而失败，没有bulk GPU数据。1004363也已于14:43–14:47成功；15:14查询时1089379已解除依赖等待Priority，1089380仍等afterok依赖。原watcher4160646自然结束，1174145仍活；详见本轮h800_queue_1514.txt。未覆盖排队源码、未重复提交作业。

已下载1004362的10份及1004363的4份原始JSON、环境与原脚本快照，h800_archive_manifest.json绑定SHA。实际旧header为 `29667eaf…93b36`，benchmark为4步旧trace、末半段steady统计，并非当前assistant1024、正确Memcpy完整口径或当前优化实现。示例旧MTP8/128请求/f32为本文10492.6µs、旧Memcpy8431.6µs、floor1209.9µs；旧结果存在负差，但不能用于当前生产优劣或当前大100GB差距的推断。原NVLink旧MTP8/128请求/f64为本文2425.8µs、旧Memcpy3459.3µs；随机gather167.33GB/s、DMA177.56GB/s。它同样是旧四步口径。原始JSON和日志保存于cache-store实验目录的h800_archive及h800_completed_logs.txt。后续H800新机制须隔离迁移与构建验证，不能用A100结果代替。

## QK 地址游标已保留

生产更新为实测原件 `754706f8d8fba14a361ce6dcfc64a358446e5df36b225f392fa8a3e7756ded79`。paired CKV QK原循环每个dim重算row/column/swizzle偏移；候选只初始化Q/K游标一次，随后用已有advance_offset_by_column的xor/add推进。16,384项整数地址检查覆盖32lane×2group×4warp×Q/K×32dim。保持原unroll4、Q/K fragment加载和每score的MMA次序、联合MDO/PV、QPE片段、远端源hint、精确LRU、实际KV字节和FAR模型。没有合并前两个拒绝候选。

| cohort | 链路 | 原生产 µs | 候选 µs | 耗时降低 | Memcpy µs | 纯device µs | 逐步max µs |
|---|---|---:|---:|---:|---:|---:|---:|
| focused256_ | g100_l0 | 3395.680 | 3365.056 | +0.902% | 6095.573 | 2556.960 | 2699.861 |
| focused256_ | g100_l20000 | 3419.488 | 3398.581 | +0.611% | 6105.205 | 2556.960 | 2699.991 |
| focused256_ | g50_l20000 | 5270.741 | 5268.597 | +0.041% | 8482.240 | 2556.960 | 4959.960 |
| native256_ | native_hbm | 3157.451 | 3151.659 | +0.183% | 3948.565 | 2556.960 | 2556.960 |
| native256_ | pcie | 9532.171 | 9531.669 | +0.005% | 12564.405 | 2556.960 | 9170.949 |
| reverse1024_ | g100_l0 | 3464.883 | 3413.173 | +1.492% | 6285.821 | 2561.848 | 2744.828 |
| reverse1024_ | g100_l20000 | 3483.117 | 3438.675 | +1.276% | 6303.667 | 2561.848 | 2745.362 |
| reverse1024_ | g50_l20000 | 5502.875 | 5499.211 | +0.067% | 8882.805 | 2561.848 | 5189.460 |

初筛256位置×3冷pass为base→candidate；reverse1024为完整1024位置×3冷pass、candidate→base；native256为正常构建256位置×3冷pass、candidate→base。全部沿用上述BF16/MTP8R128/LRU4096/f24off/GPU3/NUMA1参数。不是SXM或物理NVLink。没有重复五pass正反向或156格矩阵。

4项定向BF16检查通过；38EMU/38normal原BF16实例机器字、控制字和资源逐一相同，未新验证FP16实例。新目标REG255/STACK16/LOCAL0/static2304B，栈由EMU8B/normal0B增至16B，不称零spill。实际Graph仍单108CTA/block32x4x2/dynamic147968B/active1，K97采样max abs0.004128。未重复sanitizer或生产root新编译；安装文件与独立正常/EMU构建、实测源码逐字节一致。

源码表达式减少不代表静态指令减少：EMU总6968→7008、IMAD1344→1423、LOP3355→363，HMMA299/LDSM190保持。原循环及完整SASS、二进制SHA和address_codegen_review.json保留；没有把指令数或猜测的stall份额当性能原因。收益仅由同参数实测支持。

合计16格/8对，6528计时launch/408有效段，无未完成raw/缺失参考。全部实测配对正差，物理PCIe近乎持平。轻量证据足够保留本轮机制；仅此卡/此形状得到性能确认，FP16、MTP1、其他形状、SXM/NVLink/H800未新覆盖。整体100GB仍高于逐步参考约24%–25%，总体目标未完成。

[本轮QK游标源码及运行入口](../temp/glm_trace/tune_qk_cursor_20261102/README.md)，retention.json和retention_snapshot冻结了逐字节安装证据。所有本轮runner已结束，不重启。
