# 远端链路模拟器 v2：并发计费修复与校准

用户明确要求先修好模拟器，再讨论kernel优化。本次已将生产模拟器升级为 **8507eab01aa705c507d0ee389f1216dfc3c29d50844fbf509a0339fd6023d716**。attention host头cab508dc、canonical数学头34af保持；参考公式及全部旧性能原件未改。**本次没有新的attention加速或110%达标结论。**

## 修复内容

旧版用atomicAdd后再conditional atomicMax恢复空闲时钟，并发请求可能丢失部分预算。新版用CAS将`max(clock, issue_time) + service_time`作为一次原子更新。每包传输预算为`ceil(bytes × ps_per_byte / 1000)`纳秒，向上取整误差小于1ns；50/100GB/s分别精确表示为20/10ps每字节。64页包在100GB/s下取整多计约0.098%，不会少算。

原先消费stage时才追记预算，新版在warp发出数据包时预约，并记录该CTA/stage所有warp的最晚deadline。等待前增加CTA同步，保证其他loader warp已发布deadline；调用方原有等待后同步继续保护消费及stage复用。旧State聚合布局保留以兼容手工初始化，prefetch无需再保存延迟计费状态。删除了将超过1秒的合法等待强行截短的逻辑。

模型现在明确为：`传输完成 = max(链路时钟, issue时间) + 传输预算`；`deadline = 传输完成 + L`。L不进入链路时钟，允许多个包在途。数据还必须完成真实读取及staging写入，才会发布ready。L=0不代表真实访问和软件操作耗时为零。

Memcpy gather也改为发出时预约。首次保留432CTA×16页布局时，正确CAS竞争使100GB档只能达到约51GB/s，**该次校准判为失败**，原源码及结果保存在`cas_contention/`。最终改为64页包、48个单缓冲CTA，3072行在途容量与24个双缓冲fetch CTA一致；不改变字节、限速方程或延迟。

公共Python绑定使用独立的`far_emu_gather_v2`模块和`build_v2`目录，并在JIT/GPU工作前核对kernel与gather的模拟器头是否一致，避免旧冻结kernel混用新gather。旧build及旧结果保留。

## 校准及定向验证

全部使用独占GPU3/A100-PCIE、NUMA1、同卡独立HBM和共享锁；不是NVLink实测。

| 测试路径 | 设置 | 实测有效载荷带宽 |
|---|---|---:|
| 24CTA、双64页buffer、真实cp.async搬运 | 100GB/s，0µs | 99.53GB/s |
| 同上 | 100GB/s，20µs | 98.78GB/s |
| 同上 | 50GB/s，0/20µs | 49.91 / 49.75GB/s |
| 最终Memcpy gather，五档延迟 | 100GB/s，0/2/5/10/20µs | 98.17–99.15GB/s |
| 同上 | 50GB/s，0/2/5/10/20µs | 49.69–49.88GB/s |

15组独立gather/依赖读校准（不限速/50/100GB/s×0/2/5/10/20µs）全部通过原5%带宽、10%延迟＋600ns判据，所有样本均检查；有限带宽各格均值的最大偏差1.833%。20µs依赖读约20.48µs，全部输出载荷一致。

32组并发预约、16000次构造试验全部正确：覆盖50/100GB/s、0/20µs、2/24CTA、空闲/预忙、等长/混合包。另3组算术检查覆盖空闲间隔、非单调issue时间、零字节及不限速；2组故意延迟warp的stage检查通过。6个真实载荷搬运格包含13页尾块，完整载荷一致。

normal/EMU BF16构建均通过。3项attention定向集成检查通过：staged K65/K2048及direct K65、output/LSE、三轮cache状态、所有有效cache槽payload/映射、Graph两次重放。公共emu绑定另通过实际载荷及Graph两次重放；旧模拟器头混用被拒绝。首构建的State兼容错误及诊断取整oracle错误原件保存在`first_attempt/`，不算通过项。

证据：[总审计](../temp/glm_trace/emu_fix_20261103/audit.json)、[并发与真实搬运](../temp/glm_trace/emu_fix_20261103/verification.json)、[15组校准](../temp/glm_trace/emu_fix_20261103/calibration.json)、[attention集成](../temp/glm_trace/emu_fix_20261103/integration.json)、[复现说明](../temp/glm_trace/emu_fix_20261103/README.md)。

## 解释范围与后续口径

这是经过校准的载荷串行化/数据到达模型，仍有软件开销。约302MB的搬运微核中，无hook约486.4µs、hook关闭约558.4µs、hook开启但不限速约695.0µs；这些是独立微核诊断，不能将差值直接从attention时间中扣除或当其严格因果拆分。50/100GB/s校准表证明目标带宽能在相应搬运路径上接近达到，不等于物理NVLink全资源等价。

同卡far读取仍消耗本地HBM/L2，CAS/同步/轮询也消耗资源。每模块的CTA stage槽要求同一时刻只有一个相关launch；不同模块的链路时钟独立。未验证并发多stream、任意带宽、物理NVLink或H800，不能外推。

所有原5304格及历次kernel筛选仍属于旧模型，保留原值及源码SHA；不得改标为v2。讨论新的100GB差距或kernel新旧收益之前，需要用一致v2源码、新JIT及新原始文件采集少量关键对照。原device参考公式不变；不能把模拟器修复或gather布局变化算作attention机制加速。本轮没有重跑完整矩阵或完整trace。
