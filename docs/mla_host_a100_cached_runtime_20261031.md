# A100：缓存页号流水的运行期判断与同分配诊断（2026-10-31）

候选尚未采用，生产保持ce341a02/52919…ff2c5。本轮没有解决历史大batch 100GB/s约32%–34%的差距。

W/cached_ids把有LRU的whole CTA32页号读入shared，同时把host_batch_caching设为编译期true。当前runtime_ids只将这个判断恢复为原pool_capacity非零检查。缓存限定派发保留，无LRU与原generic/whole路径继续用原kernel；没有改数学、LRU策略、实际载荷、搬运字节、模拟器或参考。

新host头SHA=`85ef148314998a7973c615e1f3a3853ea2be027ce17bdc2ccfbc162495948a93`，math头不变。105正常测试通过；所有原32个EMU及64个正常库内kernel指令/资源一致，新增目标REG244、STACK0、LOCAL0。新缓存目标与最初fetch_ids的缓存目标在EMU及正常BF16/FP16中的逐条指令和资源完全一致，无LRU回退路径没有引回。此为代码等价证明，不能替代新性能确认或证明前版负差的因果。

## 独立进程新旧对照

均为MTP8/R128、LRU4096、staged、fetch24/off。short为256位置×3次冷启动，full为1024位置×5次；同参数、相同trace绑定、CUDA Graph/event、前后资源检查和实际cache载荷检查。GPU3模拟far是同卡独立HBM，不是NVLink。正常PCIe按NUMA1/0分别绑定，不拼两种硬件的绝对时间。

| 硬件 | 组/位置/pass | 链路 | base µs | 候选 µs | 耗时降低 | Memcpy µs | 纯device µs | 逐步max µs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| GPU3 | runtime256_f24_/256/3 | g100_l0 | 3553.824 | 3545.568 | 0.232% | 6050.347 | 2556.960 | 2699.780 |
| GPU3 | runtime256_f24_/256/3 | g100_l20000 | 3595.104 | 3557.963 | 1.033% | 6077.760 | 2556.960 | 2699.887 |
| GPU3 | runtime256_f24_/256/3 | g50_l20000 | 5303.157 | 5283.659 | 0.368% | 8486.805 | 2556.960 | 4959.850 |
| GPU3 | runtime_full_gpu3_limited_/1024/5 | g100_l0 | 3594.384 | 3610.840 | -0.458% | 6276.912 | 2561.848 | 2744.843 |
| GPU3 | runtime_full_gpu3_limited_/1024/5 | g100_l20000 | 3646.453 | 3632.654 | 0.378% | 6301.250 | 2561.848 | 2745.576 |
| GPU3 | runtime_full_gpu3_limited_/1024/5 | g50_l20000 | 5538.021 | 5520.938 | 0.308% | 8887.699 | 2561.848 | 5189.609 |
| GPU3 | runtime_full_gpu3_native_/1024/5 | native_hbm | 3428.794 | 3380.966 | 1.395% | 4051.211 | 2561.848 | 2561.848 |
| GPU3 | runtime_full_gpu3_native_/1024/5 | pcie | 9938.162 | 9967.008 | -0.290% | 13352.786 | 2561.848 | 9594.396 |
| SXM0/1 | runtime_full_sxm_limited_/1024/5 | g100_l0 | 3458.869 | 3481.446 | -0.653% | 6247.298 | 2467.179 | 2691.449 |
| SXM0/1 | runtime_full_sxm_limited_/1024/5 | g100_l20000 | 3505.330 | 3509.606 | -0.122% | 6275.539 | 2467.179 | 2691.093 |
| SXM0/1 | runtime_full_sxm_limited_/1024/5 | g50_l20000 | 5533.976 | 5528.355 | 0.102% | 8975.494 | 2467.179 | 5191.518 |
| SXM0/1 | runtime_full_sxm_native_/1024/5 | nvlink | 3309.995 | 3249.069 | 1.841% | 4444.296 | 2467.179 | 2467.179 |
| SXM0/1 | runtime_full_sxm_native_/1024/5 | pcie | 9941.090 | 9949.539 | -0.085% | 13323.669 | 2467.179 | 9597.975 |

完整确认中，SXM100GB0/20µs为慢0.653%/0.122%，正常NVLink快1.841%，PCIe慢0.085%；GPU3的100GB0µs慢0.458%、20µs快0.378%、50GB快0.308%，正常独立HBM快1.395%、PCIe慢0.290%。这些新原件不支持强的100GB改进，也不证明恢复运行期判断稳定优于cached_ids。仍需完成跨形状门槛，不安装生产。

## 同进程、同内存的三版诊断

SXM0/1、完整1024位置、MTP8/R128、LRU4096、f24/off，在同一套Q/indices、KV、workspace、cache和输出上，对base/cached_ids/runtime_ids交替各5次冷启动，分别测100GB0/20µs。CUDA Graph API确认每图128个attention节点，三版函数指针不同、grid/block/shared一致。不是错误地让两个模块都调用同一个函数。

| 链路 | base µs | cached_ids µs | runtime_ids µs | cached_ids降低 | runtime_ids降低 |
|---|---:|---:|---:|---:|---:|
| g100_l0 | 3469.472 | 3478.920 | 3467.944 | -0.272% | 0.044% |
| g100_l20000 | 3528.437 | 3507.466 | 3506.693 | 0.594% | 0.616% |

30次均验证采样输出、resident/slot双向映射和最终524288行真实缓存KV载荷。不同pass的同stamp并行入槽次序使miss略有变化，未声称逐步命中相等。结果显示该大形状100GB收益较小；它支持继续检查分配/运行环境影响，但没有给各波动建立净因果归因。原独立进程−1%～−2.2%等全部慢格保留；该诊断不替代完整矩阵，不放宽门槛。

W早先单请求GPU3 PCIe同分配诊断也补做了dispatch核对：两个图各1024个attention节点，实际函数不同。原约3.9%独立进程负差和同分配约0.01%差异均保留。

本轮独立性能26格/13配对，其中20格为四组full确认，另6格short。完整原件：[性能审计](../temp/glm_trace/tune_cached_runtime_20261031/performance_audit.json)、[同分配审计](../temp/glm_trace/tune_cached_runtime_20261031/allocation_audit.json)、[前版代码比较](../temp/glm_trace/tune_cached_runtime_20261031/predecessor_comparison.json)。全部source、两参考、逐步/逐pass、guard、失败与负差均保留。
