# A100：联合MDO与Q fragment复用的组合（2026-11-01）

候选尚未采用，生产保持0f82e9de / b6eea095…e72a。短窗口组合版较生产快2.868%–3.376%，较同轮联合MDO控制再快0.398%–0.916%；需要完整窗口反向确认，不能把较生产的全部收益归于Q复用。

以已独立通过本机98格/49对的joint_mdo_typed（1190…aa59）为算法控制。两个32KV score tile已经同时保留在寄存器中，组合版复用旧P冻结的paired-QK helper，让每个Q fragment加载一次供两个tile使用，随后执行相同joint max/d/O重缩放和逐tile native PV/refill。每score的PE→CKV MMA累加次序保持，与JM相同的在线归约分组；canonical math头34af、原fetch64行双buffer/f24、64head/256thread/32KV/2stage/shared147968B、精确跨步LRU/实际cache载荷、单cooperative kernel均保持。它要求两个stage先ready，失去JM第一QK与第二tile加载的重叠，收益必须实测。复用旧helper的unroll4，不扫unroll/配对长度/cadence/fetch数值。

旧paired-QK-only使用两次独立MDO、无稳定收益；本组合只测试在已经受益的jointMDO中追加Q复用，未修改正在排队的JM源码/计划，不覆盖旧失败证据。派发仍仅SM80原cached whole-row、至少4×SM行、完整staging，其他kernel原34EMU/68normal实例逐一指令/资源相同。

本版新增105项正常BF16/FP16测试和4项长cache-payload/Graph测试通过，共109项。JM控制复用原独立binary及原109测试证据，reused_joint_control.json列精确原件SHA与同字节host/math，不冒充本轮新增测试。实际正常BF16 Graph只有一个新BatchMLAJointQKKernel，108CTA、block[32,4,2]、dynamic147968B/static2304B、active1，probe最大绝对误差0.004128。新EMU/normal均REG251/STACK0/LOCAL0，JM控制REG255/STACK16。新SASS6952条/HMMA299/LDSM198是unroll4静态计数，不能当作动态work、带宽或性能提升的证明。

GPU3为A100-PCIE/NUMA1，far模拟源是同卡独立HBM，不是真实NVLink。MTP8/R128、LRU4096/请求、f24/off。短256位置×3冷pass顺序base→JM→JQ；confirmation_plan.json规定完整1024位置×5pass反向JQ→JM→base。所有步/pass算术均值，冷启动不剔除；纯device同硬件参考、逐步max和完整Memcpy均按原口径，两参考与所有正负差保留。

| cohort | 位置/pass | 链路 | 生产µs | 联合MDOµs | 组合QKµs | 较生产降低 | 较联合MDO降低 | 候选Memcpyµs | 候选逐步maxµs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| composition256_f24_ | 256/3 | g100_l0 | 3553.003 | 3464.789 | 3433.067 | +3.376% | +0.916% | 6051.029 | 2699.652 |
| composition256_f24_ | 256/3 | g100_l20000 | 3554.848 | 3466.709 | 3452.907 | +2.868% | +0.398% | 6064.416 | 2699.896 |
| composition256_f24_ | 256/3 | g50_l20000 | 5285.024 | 5277.195 | 5273.344 | +0.221% | +0.073% | 8487.616 | 4960.298 |
| compositionfull1024_f24_ | 1024/5 | g100_l0 | 3620.643 | 3520.424 | 3484.302 | +3.766% | +1.026% | 6267.045 | 2744.776 |
| compositionfull1024_f24_ | 1024/5 | g100_l20000 | 3651.909 | 3540.707 | 3511.080 | +3.856% | +0.837% | 6298.947 | 2745.475 |
| compositionfull1024_f24_ | 1024/5 | g50_l20000 | 5521.166 | 5514.400 | 5506.459 | +0.266% | +0.144% | 8887.653 | 5189.514 |

完整审计当前46格/24个对生产配对、35776计时launch/2236segment，0份未完成raw；表只列已完成三方配对。三方结果另见composition_comparisons.json，不能把其中的对JM比较混入对生产配对数量。short_performance_audit.json冻结首次9格/6对、1728launch/108segment；后续追加不覆盖原件。当前活动runner以continuation_state.json和实际handle为准。完整/正常链路/跨形状/真实SXM门槛未完成前不安装，也不声称主目标完成。

JM仍有独立SXM验证器等待GPU0/1空闲；本实验没有远端Slurm提交、没有覆盖排队源或终止其他用户进程。原5304矩阵、生产头、root math和模拟器均未改。
