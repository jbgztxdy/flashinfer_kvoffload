# 全 CTA unique 搬运与 QK 并行：三版短筛

2026-11-04，GPU3/A100-PCIE、NUMA1、同卡独立 HBM 远端源；BF16、64 heads、MTP8/R128/K2048、LRU4096/请求、256 位置/32 步。均为正常构建、单 cooperative kernel、原 unique 页搬运、真实 staging/cache 双写；不是物理 NVLink 或 100GB/s 仿真。生产三个 header 未改。与旧“QK 后写回”不同，本次尝试在 QK 执行时让下一包远端 cp.async 仍在途，而旧版在 QK 前已 wait_group<0>。

SM80 的 cp.async.commit_group/wait_group 按每线程的提交顺序等待；wait_group<1> 只容许最新一组未完成，不能跳过同一线程较早的远端组而只等待较新的本地组。见 NVIDIA PTX ISA 的 cp.async.wait_group 定义。跨 CTA 唯一页发布仍等真实远端 payload 与 cache/staging 写回。

| 版本 | 结果 |
| --- | --- |
| far_qk_overlap | 在本地两个异步 stage 之后才发远端包，QK 前 wait_group<1>、QK 后 wait_group<0> 并写回。K65/K2048 两项定向检查通过，但完整 32 步 Graph 首次 replay 报 CUDA illegal memory access；**失败，不计性能**。wait_tile 可在延期 issue 之前执行 service_step，旧 ticket 的远端 buffer 与下一包元数据错配。失败日志和未完成 JSON 原样保留。 |
| sync_local_far_qk | 保留原 refill/ticket 时序，全部计算 warp 用同步 128b 本地 KV 载入隔离远端 cp.async。两项定向检查、32 步×3 冷 pass 的采样数学与最终所有有效 cache payload 通过。三 pass 3657.760 / 3665.120 / 3715.200µs，均 **3679.360µs**。 |
| warp_isolated_far | 原时序、四个既有计算 warp 负责远端 cp.async 与同步本地 KV，另四个继续异步本地 KV；八 warp 均执行 QK/PV。两项定向检查、32 步×3 pass 采样数学与最终所有有效 cache payload 通过。三 pass 6056.896 / 6055.424 / 6056.992µs，均 **6056.437µs**。 |

同轮 fresh clean 控制，1 冷 pass **3096.832µs**；与 sync/warp 第一 pass 相比，候选分别慢 **18.11%/95.58%**。sync/warp 三 pass 均值与 1 pass 控制的差值不是严格相同 pass 数配对；但损失幅度足以拒绝。三者的 unique miss 约 215k 页/步、约 248MB/步，未用 direct 的约 563MB/步作 unique 分母。

两版成功完成完整 trace 的候选又各做一次 K97 单 launch Graph 节点核对：实际都是单个 BatchMLACleanServiceKernel、grid [1,108,1]、block [32,4,2]、dynamic shared 163072B、active 1 block/SM，并核对两行数学输出。原始节点和源码 SHA 见 dispatch_probe_sync_local_far_qk.json、dispatch_probe_warp_isolated_far.json；这避免把回落其他 kernel 的时间误认成候选收益或退化。

原始数据：clean 控制 sync_local_control_clean_native_unlimited_f24.json，sync 为 sync_local_far_qk_native_unlimited_f24.json，warp 为 warp_isolated_far_native_unlimited_f24.json；全部记录每步时间、header SHA、实际加载 .so 路径和 SHA、硬件/NUMA/配置。失败的 far_qk_overlap_native_unlimited_f24.json 不完整，不能用作性能值；对应 far_qk_overlap_native.log 保留非法访存栈。三版的 test、build 和 runner state/log 均在本目录。

源码 SHA256：clean 29835918684dd24c76e15e27366e7e330268c3dd66c939eaffb1462b54ab2742；far_qk_overlap cfc8dcb5fdea840a00b8ba97b6a6295d74ddeb4271b46a2b55c708d45038f2e6；sync_local_far_qk c4322115e6383e5179592c0749d18716d8e1a2419dc7f5e4796b4a8e5909107d；warp_isolated_far cabf9e0db8c3d5b8afd89fdd210a3a66593c0f25cc7e5cb17e0648f3cfb20e07。

实际加载 host_overlap_v2 库 SHA256：clean ccc28bc51265e1233b82ec2c3ce06ebdf718140016370e86a3cc09fe86d8dbc9；sync 88f64a2101ba32440c04466490000a10ed1d66c410efc0187aae8fe0f0b123eb；warp c9d190b45b0dd8d7401ca5fc2e5a51779f0b0dc597686b0273ef650e0cc38bb1。失败版编译所得库 SHA256 0be420dad86970c4d97c54a3de3296407d3305a765121a25304bc030f668bd07；报错前未捕获 /proc/maps，故不称已记录实际加载库。

生产 header SHA256 81093ba5ab33cc5d84b2df04c702975697387b4054ebe8f1991f558f1bd6cdd2，canonical mla.cuh 34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3，FAR 8507eab01aa705c507d0ee389f1216dfc3c29d50844fbf509a0339fd6023d716。均未安装；无新 100GB/s、物理 PCIe/NVLink、H800、完整 1024 位置或 sanitizer 结论。后续不能原样重试本次迟发包、整块同步载入或四 warp 扁平 copy；下一候选必须同时保留高并行 unique 搬运和异步本地载入，并先证明跨 CTA ready/进度协议安全。
