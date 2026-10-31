# A100：双32KV tile共用在线softmax重缩放（2026-11-01）

候选尚未采用，生产仍为0f82e9de / b6eea095…e72a。短窗口100GB0/20µs初测快0.653%/2.700%；第一组完整1024×5pass反向快2.677%/2.902%，正常独立HBM快3.002%、物理PCIe基本持平。第二组完整正向也已完成（快2.134%/1.881%），GPU3跨形状两矩阵全部完成；真实SXM/NVLink确认仍待。100GB大batch历史约32%–34%主差距尚未解决，goal active。

保留64head/256线程、32KV/2stage、dynamic147968B、原fetch64行双buffer/f24、精确跨步LRU和真实缓存载荷。两个降序tile先分别计算原native QK，再对两个score fragment共同求max，只计算一次d/O重缩放；各自的P指数和native PV仍按原tile降序执行。第一个stage在第一个PV结束后补下一tile，让其加载与第二个PV重叠，随后补第二个stage。原PE→CKV dot次序不改，完整576特征保留，attention方程相同但FP32在线归约分组不同，不承诺逐位相同。canonical数学头34af、模拟器及原5304矩阵均未改；新联合状态更新仅在私有host头内。派发仅原cached whole-row、至少4×SM行、完整staging的SM80路径，仍单cooperative kernel。

该机制不同于早期配对QK的两次独立MDO，也不同于KV64/单stage的双缓冲损失；不涉及fetch/cadence/unroll扫描。

joint_mdo首版和joint_mdo_after定义顺序控制均通过105+4正常/长cache载荷/Graph测试及实际单kernel实查，但各有4/34EMU、8/68normal原实例指令文本改变（puredevice精确相同），strict gate拦截，二者均无性能launch。只移动新增helper/kernel到原kernel后不能消除变化。joint_mdo_typed用专属继承traits隔离模板helper实例，字段/布局/算法与after版相同，原34EMU/68normal实例全部指令/资源精确相同后才允许性能。没有放宽或归一化反汇编检查，三个冻结源码/JIT及失败日志全部保留。

typed版同样通过105+4，共三版327项测试。长测试覆盖BF16/FP16、K2048、128个池、容量512/4096、淘汰、零miss、padding、Graph复用及所有有效cache payload。实际正常BF16 Graph均只有一个对应新kernel，108CTA、block[32,4,2]、dynamic147968B/static2304B、active1，probe最大绝对误差0.004128。typed新REG255/STACK16/LOCAL0，不能称stack0或将栈代价当作已证明的唯一性能原因。

GPU3为A100-PCIE/NUMA1，far模拟源是同卡独立HBM，不是真实NVLink。正常native_hbm也是独立同卡HBM，pcie才是真实物理Host链路，绝对时间不能拼接为NVLink收益。MTP8/R128、LRU4096/请求、f24/split off；所有步/pass算术均值、包含冷启动。短256位置×3pass为base→typed；预先保存confirmation_plan.json规定full1024位置×5pass反向typed→base，正常两链路base→typed。纯device同硬件参考短窗口2556.960µs，full使用对应全步参考；逐步max按每步compute及unique miss原式计算。

| cohort | 位置/pass | 链路 | 同轮base µs | 联合MDO µs | 耗时降低 | 候选Memcpy µs | 候选逐步max µs |
|---|---|---|---:|---:|---:|---:|---:|
| joint_shapes_gpu3_limited_ | 1024/3 | g100_l0 | 3632.371 | 3500.413 | +3.633% | 6270.659 | 2744.839 |
| joint_shapes_gpu3_limited_ | 1024/3 | g100_l0 | 1830.488 | 1807.597 | +1.251% | 3180.339 | 1373.042 |
| joint_shapes_gpu3_limited_ | 1024/3 | g100_l10000 | 3630.352 | 3544.944 | +2.353% | 6294.091 | 2744.988 |
| joint_shapes_gpu3_limited_ | 1024/3 | g100_l10000 | 1842.480 | 1805.816 | +1.990% | 3198.645 | 1372.449 |
| joint_shapes_gpu3_limited_ | 1024/3 | g100_l20000 | 3629.784 | 3550.603 | +2.181% | 6313.973 | 2745.084 |
| joint_shapes_gpu3_limited_ | 1024/3 | g100_l20000 | 1861.275 | 1823.819 | +2.012% | 3206.635 | 1372.558 |
| joint_shapes_gpu3_limited_ | 1024/3 | g100_l2000 | 3619.477 | 3529.309 | +2.491% | 6274.728 | 2745.511 |
| joint_shapes_gpu3_limited_ | 1024/3 | g100_l2000 | 1828.115 | 1808.384 | +1.079% | 3189.365 | 1373.027 |
| joint_shapes_gpu3_limited_ | 1024/3 | g100_l5000 | 3630.229 | 3530.597 | +2.745% | 6282.779 | 2745.240 |
| joint_shapes_gpu3_limited_ | 1024/3 | g100_l5000 | 1853.317 | 1815.075 | +2.063% | 3195.824 | 1372.471 |
| joint_shapes_gpu3_limited_ | 1024/3 | g50_l0 | 5497.451 | 5489.221 | +0.150% | 8880.432 | 5187.551 |
| joint_shapes_gpu3_limited_ | 1024/3 | g50_l0 | 2879.221 | 2876.093 | +0.109% | 4490.165 | 2648.242 |
| joint_shapes_gpu3_limited_ | 1024/3 | g50_l10000 | 5507.904 | 5502.752 | +0.094% | 8876.651 | 5189.636 |
| joint_shapes_gpu3_limited_ | 1024/3 | g50_l10000 | 2885.416 | 2884.011 | +0.049% | 4509.565 | 2648.438 |
| joint_shapes_gpu3_limited_ | 1024/3 | g50_l20000 | 5517.592 | 5509.160 | +0.153% | 8899.715 | 5187.320 |
| joint_shapes_gpu3_limited_ | 1024/3 | g50_l20000 | 2896.653 | 2894.080 | +0.089% | 4512.507 | 2648.593 |
| joint_shapes_gpu3_limited_ | 1024/3 | g50_l2000 | 5499.805 | 5493.811 | +0.109% | 8870.515 | 5189.811 |
| joint_shapes_gpu3_limited_ | 1024/3 | g50_l2000 | 2878.312 | 2876.016 | +0.080% | 4506.312 | 2648.459 |
| joint_shapes_gpu3_limited_ | 1024/3 | g50_l5000 | 5504.368 | 5494.811 | +0.174% | 8885.725 | 5187.628 |
| joint_shapes_gpu3_limited_ | 1024/3 | g50_l5000 | 2881.117 | 2877.872 | +0.113% | 4502.235 | 2648.636 |
| joint_shapes_gpu3_native_ | 1024/3 | native_hbm | 42.668 | 42.176 | +1.155% | 39.074 | 32.295 |
| joint_shapes_gpu3_native_ | 1024/3 | native_hbm | 1049.374 | 1054.674 | -0.505% | 1227.563 | 770.722 |
| joint_shapes_gpu3_native_ | 1024/3 | native_hbm | 920.053 | 914.939 | +0.556% | 1051.288 | 750.813 |
| joint_shapes_gpu3_native_ | 1024/3 | native_hbm | 885.843 | 884.725 | +0.126% | 1129.477 | 751.733 |
| joint_shapes_gpu3_native_ | 1024/3 | pcie | 53.934 | 53.722 | +0.394% | 61.899 | 32.453 |
| joint_shapes_gpu3_native_ | 1024/3 | pcie | 3201.845 | 3201.849 | -0.000% | 4031.264 | 3026.086 |
| joint_shapes_gpu3_native_ | 1024/3 | pcie | 2401.384 | 2402.181 | -0.033% | 3206.661 | 2170.405 |
| joint_shapes_gpu3_native_ | 1024/3 | pcie | 2792.765 | 2786.339 | +0.230% | 3695.747 | 2596.837 |
| joint_shapes_gpu3_native_ | 1024/3 | native_hbm | 3378.088 | 3273.859 | +3.085% | 4056.536 | 2561.848 |
| joint_shapes_gpu3_native_ | 1024/3 | native_hbm | 1770.957 | 1719.307 | +2.917% | 2033.621 | 1248.016 |
| joint_shapes_gpu3_native_ | 1024/3 | pcie | 9961.509 | 9958.528 | +0.030% | 13305.459 | 9594.769 |
| joint_shapes_gpu3_native_ | 1024/3 | pcie | 5163.272 | 5172.909 | -0.187% | 6779.731 | 4896.678 |
| joint_shapes_gpu3_native_ | 1024/3 | native_hbm | 1201.749 | 1202.347 | -0.050% | 1239.637 | 1239.040 |
| joint_shapes_gpu3_native_ | 1024/3 | native_hbm | 1240.320 | 1227.947 | +0.998% | 1299.115 | 1241.856 |
| joint_shapes_gpu3_native_ | 1024/3 | native_hbm | 1452.971 | 1462.357 | -0.646% | 1408.683 | 1241.515 |
| joint_shapes_gpu3_native_ | 1024/3 | pcie | 1202.005 | 1202.176 | -0.014% | 1242.795 | 1239.040 |
| joint_shapes_gpu3_native_ | 1024/3 | pcie | 1584.469 | 1591.808 | -0.463% | 2538.155 | 1321.049 |
| joint_shapes_gpu3_native_ | 1024/3 | pcie | 3971.584 | 3971.669 | -0.002% | 5074.603 | 3815.651 |
| typed256_f24_ | 256/3 | g100_l0 | 3502.496 | 3479.637 | +0.653% | 6059.989 | 2699.865 |
| typed256_f24_ | 256/3 | g100_l20000 | 3570.688 | 3474.293 | +2.700% | 6085.931 | 2700.010 |
| typed256_f24_ | 256/3 | g50_l20000 | 5287.552 | 5278.251 | +0.176% | 8482.475 | 4959.682 |
| typedforward1024_f24_ | 1024/5 | g100_l0 | 3623.770 | 3546.453 | +2.134% | 6303.896 | 2744.917 |
| typedforward1024_f24_ | 1024/5 | g100_l20000 | 3643.818 | 3575.277 | +1.881% | 6320.954 | 2745.550 |
| typedforward1024_f24_ | 1024/5 | g50_l20000 | 5521.758 | 5516.075 | +0.103% | 8886.258 | 5189.564 |
| typedfull1024_f24_ | 1024/5 | g100_l0 | 3612.686 | 3515.981 | +2.677% | 6272.926 | 2744.775 |
| typedfull1024_f24_ | 1024/5 | g100_l20000 | 3650.123 | 3544.189 | +2.902% | 6297.626 | 2745.509 |
| typedfull1024_f24_ | 1024/5 | g50_l20000 | 5521.693 | 5513.618 | +0.146% | 8888.888 | 5189.531 |
| typednative1024_f24_ | 1024/5 | native_hbm | 3399.480 | 3297.413 | +3.002% | 4065.243 | 2561.848 |
| typednative1024_f24_ | 1024/5 | pcie | 9961.226 | 9960.776 | +0.005% | 13300.317 | 9594.757 |

[完整审计](../temp/glm_trace/tune_joint_mdo_20261101/performance_audit.json)当前98格/49配对、114080计时launch/7184segment，保存每步/每pass/命中率/unique miss/guard/两参考/源码快照；所有正负差保留。原short_performance_audit.json冻结首次6格3对，不被full追加覆盖。GPU3两矩阵按冻结计划完成76格，合计当前98格/49对、114080次计时launch/7184segment。100GB十个改变路径配对快1.079%–3.633%；所有负差保留，最大为未改无LRU HBM远端0.3慢0.646%。coverage.json明确GPU3完整、SXM80格仍待。活动runner可能仍有未完成raw（当前0份），未纳入表格与收益声明。327测试的通过不等于跨形状/完整链路性能gate已完成；只有全门槛完成才可能安装。

H80003:38:29只读复查五个原任务仍Priority/Dependency，调度估计推迟到11/05；两个原watcher仍活，AGENTS/MEMORY哈希与已读快照相同，无源码写入或重复提交。当前runner和接续任务以continuation_state.json及实际进程为准，不因本报告启动重复任务。此前F/S/WR/KV64已拒绝872项测试、48格/27配对；本候选的正向信号必须单独核实，不覆盖负差。
