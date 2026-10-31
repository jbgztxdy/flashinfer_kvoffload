# A100：联合MDO与Q fragment复用的组合（2026-11-01）

JQ已通过全部门槛并保留，等价清理源SHA为7193255e…d469。生产root独立新JIT的正常BF16/FP16编译、109项测试、72实例指令/资源和实际单kernel Graph验证均已完成；原72075/1398已terminal0。GPU3完整1024×5pass反向三方确认，组合版100GB0/20µs较原b6快3.766%/3.856%，较同轮联合MDO控制再快1.026%/0.837%。正常独立HBM较原b6快3.311%、较JM快0.142%；物理PCIe较原b6慢0.031%、较JM慢0.043%，均保留。GPU3/SXM共156矩阵格已完整，60个改变路径配对降低范围−0.188%至+5.253%，均值1.602%，所有负差保留。SXM完整反向100GB0/20µs较原b6快5.228%/4.886%；正常NVLink base/JM/JQ为3305.725/3136.016/3183.982µs，JQ较原b6快3.683%但较JM慢1.530%，PCIe较原b6快0.051%。不能把较原b6的全部收益归于Q复用，也不能声称组合总比JM快。完整SXM正向确认已齐，后续保留与生产root验证以实际completion_state.json为准。

以已独立通过GPU3/SXM共178格/89对的joint_mdo_typed（1190…aa59）为算法控制。两个32KV score tile已经同时保留在寄存器中，组合版复用旧P冻结的paired-QK helper，让每个Q fragment加载一次供两个tile使用，随后执行相同joint max/d/O重缩放和逐tile native PV/refill。每score的PE→CKV MMA累加次序保持，与JM相同的在线归约分组；canonical math头34af、原fetch64行双buffer/f24、64head/256thread/32KV/2stage/shared147968B、精确跨步LRU/实际cache载荷、单cooperative kernel均保持。它要求两个stage先ready，失去JM第一QK与第二tile加载的重叠，收益必须实测。复用旧helper的unroll4，不扫unroll/配对长度/cadence/fetch数值。

旧paired-QK-only使用两次独立MDO、无稳定收益；本组合只测试在已经受益的jointMDO中追加Q复用，未修改正在排队的JM源码/计划，不覆盖旧失败证据。派发仍仅SM80原cached whole-row、至少4×SM行、完整staging，其他kernel原34EMU/68normal实例逐一指令/资源相同。

本版新增105项正常BF16/FP16测试和4项长cache-payload/Graph测试通过，共109项。JM控制复用原独立binary及原109测试证据，reused_joint_control.json列精确原件SHA与同字节host/math，不冒充本轮新增测试。实际正常BF16 Graph只有一个新BatchMLAJointQKKernel，108CTA、block[32,4,2]、dynamic147968B/static2304B、active1，probe最大绝对误差0.004128。新EMU/normal均REG251/STACK0/LOCAL0，JM控制REG255/STACK16。新SASS6952条/HMMA299/LDSM198是unroll4静态计数，不能当作动态work、带宽或性能提升的证明。

GPU3为A100-PCIE/NUMA1，far模拟源是同卡独立HBM，不是真实NVLink。MTP8/R128、LRU4096/请求、f24/off。短256位置×3冷pass顺序base→JM→JQ；confirmation_plan.json规定完整1024位置×5pass反向JQ→JM→base。所有步/pass算术均值，冷启动不剔除；纯device同硬件参考、逐步max和完整Memcpy均按原口径，两参考与所有正负差保留。

| 硬件/cohort | 位置/pass | 链路 | 原b6µs | 联合MDOµs | 组合QKµs | 较原b6降低 | 较联合MDO降低 | Memcpyµs | 纯deviceµs | 逐步maxµs | 高于max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPU3/composition256_f24_ | 256/3 | g100_l0 | 3553.003 | 3464.789 | 3433.067 | +3.376% | +0.916% | 6051.029 | 2556.960 | 2699.652 | 27.17% |
| GPU3/composition256_f24_ | 256/3 | g100_l20000 | 3554.848 | 3466.709 | 3452.907 | +2.868% | +0.398% | 6064.416 | 2556.960 | 2699.896 | 27.89% |
| GPU3/composition256_f24_ | 256/3 | g50_l20000 | 5285.024 | 5277.195 | 5273.344 | +0.221% | +0.073% | 8487.616 | 2556.960 | 4960.298 | 6.31% |
| GPU3/compositionfull1024_f24_ | 1024/5 | g100_l0 | 3620.643 | 3520.424 | 3484.302 | +3.766% | +1.026% | 6267.045 | 2561.848 | 2744.776 | 26.94% |
| GPU3/compositionfull1024_f24_ | 1024/5 | g100_l20000 | 3651.909 | 3540.707 | 3511.080 | +3.856% | +0.837% | 6298.947 | 2561.848 | 2745.475 | 27.89% |
| GPU3/compositionfull1024_f24_ | 1024/5 | g50_l20000 | 5521.166 | 5514.400 | 5506.459 | +0.266% | +0.144% | 8887.653 | 2561.848 | 5189.514 | 6.11% |
| GPU3/compositionnative1024_f24_ | 1024/5 | native_hbm | 3395.507 | 3287.741 | 3283.066 | +3.311% | +0.142% | 4062.962 | 2561.848 | 2561.848 | 28.15% |
| GPU3/compositionnative1024_f24_ | 1024/5 | pcie | 9959.808 | 9958.626 | 9962.880 | -0.031% | -0.043% | 13291.848 | 2561.848 | 9594.868 | 3.84% |
| SXM0/1/compositionsxmforward1024_f24_ | 1024/5 | g100_l0 | 3485.006 | 3368.302 | 3351.232 | +3.839% | +0.507% | 6239.928 | 2467.179 | 2691.358 | 24.52% |
| SXM0/1/compositionsxmforward1024_f24_ | 1024/5 | g100_l20000 | 3513.358 | 3384.733 | 3374.661 | +3.948% | +0.298% | 6262.078 | 2467.179 | 2691.080 | 25.40% |
| SXM0/1/compositionsxmforward1024_f24_ | 1024/5 | g50_l20000 | 5527.763 | 5522.358 | 5517.030 | +0.194% | +0.096% | 8970.160 | 2467.179 | 5191.881 | 6.26% |
| SXM0/1/compositionsxmnative1024_f24_ | 1024/5 | nvlink | 3305.725 | 3136.016 | 3183.982 | +3.683% | -1.530% | 4502.181 | 2467.179 | 2467.179 | 29.05% |
| SXM0/1/compositionsxmnative1024_f24_ | 1024/5 | pcie | 9954.458 | 9951.677 | 9949.350 | +0.051% | +0.023% | 13285.968 | 2467.179 | 9598.859 | 3.65% |
| SXM0/1/compositionsxmreverse1024_f24_ | 1024/5 | g100_l0 | 3561.822 | 3379.960 | 3375.597 | +5.228% | +0.129% | 6250.685 | 2467.179 | 2691.335 | 25.42% |
| SXM0/1/compositionsxmreverse1024_f24_ | 1024/5 | g100_l20000 | 3579.909 | 3413.069 | 3404.986 | +4.886% | +0.237% | 6281.184 | 2467.179 | 2691.146 | 26.53% |
| SXM0/1/compositionsxmreverse1024_f24_ | 1024/5 | g50_l20000 | 5536.534 | 5524.480 | 5516.242 | +0.367% | +0.149% | 8970.432 | 2467.179 | 5190.612 | 6.27% |

完整审计当前210格/110个对原b6配对、386904计时launch/24258segment，0份未完成raw；表只列已完成三方配对。三方结果另见composition_comparisons.json，不能把其中的对JM比较混入对原b6配对数量。short_performance_audit.json冻结首次9格/6对、1728launch/108segment；后续追加不覆盖原件。当前活动runner以continuation_state.json和实际handle为准。GPU3两矩阵76格及SXM两矩阵80格均完成，原66919/18187已terminal0；JM原30328/98811也terminal0。两个未改路径原负差1.113%/1.223%在5pass前后夹测中未复现：MTP1/R256 HBM约慢0.110%，无LRU零远端HBM约快0.064%，controls_audit.json保留全部前/候选/后样本与原负差。原34/68实例机器字含控制字也完全相同，不以此掩盖原独立分配波动。保留源码只改host局部变量名称，36/72实例机器字全相同，并额外通过105+4测试；retention脚本已实际安装等价清理源，并完成root独立新JIT的109项测试、72实例代码核验与实际单kernel Graph。全部156矩阵及两顺序完整确认均齐，保留记录joint_qk_retention.json绑定生产验证SHA；这不等于主目标完成。当前已测SXM100GB仍高于逐步max约25%–27%，GPU3约27%–28%，主目标未解决。

独立硬件计数器诊断使用GPU3固定1024×2048、MTP全集共享、25%新远端行的合成输入，不是GLM trace性能点。Nsight Compute采用application replay，每次重新构建LRU/epoch；两版输出和全部327680有效cache载荷通过。实际Tensor指令数均71827456；short-scoreboard等待26.98%→19.31%，long-scoreboard5.39%→11.13%，barrier8.48%→9.05%。这些warp stall比例不能相加当时间分解；它与shared读取减少/等待两个stage的取舍一致，不能单因解释trace差距。诊断时间、默认cache控制、原始CSV/日志/配置/源码SHA和限制见counters_audit.json，不纳入性能审计或参考。初次诊断导入pytest缺失，在GPU工作前退出；原失败源/日志保存，改用原make_wrapper后成功，没有安装依赖或改测试环境。

JM和JQ的独立SXM矩阵已在GPU0/1自然空闲后完成；本实验没有远端Slurm提交、没有覆盖排队源或终止其他用户进程。原5304矩阵、root math和模拟器均未改；生产头的安装与验证以保留记录为准。
