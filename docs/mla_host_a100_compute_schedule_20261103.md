# A100：保留 PV 按输出特征调度（2026-11-03）

生产已由754706f8升级为实测原件 **cab508dcb6e83b7a953d94a8e54e30e0aaf60bb1d7ba98480fc66893edcf4384**。新机制在GPU3短窗口的100GB/s两种运行顺序中均有收益，按用户轻量迭代偏好保留；**110%目标未完成，没有低于原计算参考的突破**。本轮还完成square_smem，并拒绝另一个O缩放调度候选。

全部性能使用GPU3/A100-PCIE、NUMA1，同卡独立far-HBM；不是NVLink。BF16/h64/KV512+64/K2048/MTP8/R128、LRU4096/请求、f24/split off、256位置×3冷pass，CUDA Graph/event，CPU准备在事件外。每个比较都有同轮冻结7547基线，逐步实际miss/命中、原始样本、两同硬件参考和资源检查保持。未改模拟器、参考、实际载荷或必要计算。

## 保留的机制与结果

`pv_feature`直接从7547出发，原生32KV PV先把两个P片段载入寄存器，再按输出特征依次执行KV half0、half1的MMA。每个输出元素的MMA累计次序保持，P量化/rowsum/发布、联合softmax/O/d缩放、barrier、QK、hi/lo refill、fetch、LRU和FAR保持。它没有组合下文被拒绝的O缩放调度，也没有扫描unroll/stage/fetch参数。

| 顺序／链路 | 基线 µs | 新版 µs | 耗时降低 | 新版逐步参考 µs |
|---|---:|---:|---:|---:|
| base→候选，100GB0µs | 3368.565 | 3333.909 | 1.029% | 2699.752 |
| base→候选，100GB20µs | 3400.981 | 3381.888 | 0.561% | 2700.075 |
| base→候选，50GB20µs | 5266.528 | 5266.528 | 持平 | 4959.413 |
| 候选→base，100GB0µs | 3387.819 | 3345.227 | 1.257% | 2699.645 |
| 候选→base，100GB20µs | 3407.733 | 3343.179 | 1.894% | 2699.924 |
| 候选→base，正常同卡HBM | 3149.259 | 3103.957 | 1.438% | 2556.960 |

当前短窗口100GB高于逐步参考23.49%–25.25%；反向两点约23.9%/23.8%。不能把约1%的局部收益当作110%目标完成，也不能套到旧SXM完整1024窗口绝对时间。正常同卡HBM仍包含融合metadata/LRU/fetch，不是新的纯device计算参考。

2048项整数地址核对通过；9个周边helper规范化类型/命名后逐字相同，旧host头前缀与canonical数学头保持。normal/EMU实际新实例 **REG254、STACK0、static shared2304B、LOCAL0**。静态HMMA299、FMUL397保持，EMU总指令7008→6920；这些不是动态操作数或严格因果时间分解。目标SASS、资源和库SHA已保存。

两项BF16定向检查通过：K65/K2048、512行/64池/容量512、padding/空/奇尾、QPE stride80、3轮cache状态、output/LSE、Graph两次重放以及全部有效cache槽payload/映射。实际正常BF16 Graph另确认一个`BatchMLAPVFeatureKernel`、108 CTA、block[32,4,2]、dynamic shared147968B、每SM active1。性能回放实际容量4096，不能将合成检查的全槽验证说成完整trace每步全槽验证。

只在既有large cached whole-row SM80条件下新增typed dispatch；旧路径保留。安装的是逐字节实测header，没有安装后清理或重写。未重复生产root编译、整套测试、sanitizer、全部原实例机器码等价、完整1024窗口或矩阵；没有本轮FP16、SXM、物理PCIe/NVLink或H800结果。既有7547路由记录仍是旧源码实测，本轮没有更新其数据或宣称新源码重新择优。

证据：[性能与原始文件审计](../temp/glm_trace/tune_pv_feature_20261103/performance_audit.json)、[实际分派](../temp/glm_trace/tune_pv_feature_20261103/dispatch_probe_pv_feature.json)、[保留记录及冻结证据](../temp/glm_trace/tune_pv_feature_20261103/retention.json)。

## 两个拒绝的候选

| 候选 | 100GB0µs基线→候选 µs | 100GB20µs基线→候选 µs | 50GB20µs基线→候选 µs | 决定 |
|---|---:|---:|---:|---|
| square_smem c69a2643 | 3357.067→3480.192（慢3.668%） | 3355.883→3502.965（慢4.383%） | 5266.283→5274.880（慢0.163%） | 拒绝 |
| pv_rescale ab8cf8d5 | 3360.341→3371.403（慢0.329%） | 3396.437→3401.216（慢0.141%） | 5267.584→5266.869（快0.014%，持平） | 拒绝 |

square_smem修正版normal/EMU构建通过，实际REG255/STACK0/static2304/LOCAL0，square_p的STACK88降到0仍没有端到端收益。两项定向检查通过；原首稿变量重名编译失败件继续保留。至此square家族全部完成，不重启或扫数字。[完成状态](../temp/glm_trace/tune_qk_square_20261103/decisions.json)。

pv_rescale将联合MDO对每个O元素的既有乘法移至其首个PV MMA前，尝试交错标量乘法与tensor core。每元素算术、P和barrier保持；normal/EMU REG254/STACK0，两项检查通过，但100GB无收益，拒绝。源审查和静态SASS减少不能代替性能证据。[决定和三点原始对照](../temp/glm_trace/tune_pv_rescale_20261103/decision.json)。

本次接续共新增6项成功定向检查、24格/12配对、4608计时launch/288有效段，以及1项实际单kernel Graph probe。全部runner已结束，GPU3结束时0MiB/0%；GPU0/1仍外部VLLM占用，GPU2未使用。数学34af、FAR5e409、原5304矩阵保持，H800排队源码及任务未触碰。整体优化继续，已拒绝机制不再原样重做。
