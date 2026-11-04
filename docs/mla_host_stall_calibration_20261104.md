# 固定场景stall校准：探针约100GB，完整trace过度节流

用户授权固定场景经验节流。GPU3 A100-PCIE/NUMA1，同卡HBM，无物理NVLink结论。独立header/include/JIT，无全局CAS/锁/broker；每CTA每stage记录当前真实已发远端字节和issue，消费前等last_issue+ceil(bytes*固定系数)+latency。globaltimer重查+nanosleep64退避；不是固定sleep的精确时间假设。

先独立校准、后冻结：108CTA/8warp/双32rowstage/dynamic147968，真实cp.async32行、完整写回、约254.8MB载荷逐字节验证。每stage收费far8/32行；系数扫描0/200/400/600/800/1000/1400/2000ps/B。冻结2000ps/B，独立探针far8约97.66GB/s、far32约105.24GB/s。不是直接按100GB物理ps/B；额外系数补偿固定并发和双stage重叠，仅经验校准。参数冻结记录绑定probe raw SHA，未根据attention时间再调。

完整原生FA2+外部精确LRU重放，与前次native_lru100完全同mapping SHA4f31c2fd8ee21ae4a016c5c09815141e96ca7c7e7b245de805bee4e1dec60ae1，R128/MTP8/C4096/K2048/BF16/完整128步，1pass，cache-hit重建与索引传入在event外，远端miss仍真实读取。结果：

| 条件 | 平均µs |
|---|---:|
| 前轮同GPU正常构建纯device参考 | 2771.912 |
| 前轮同GPU正常构建native+LRU回放 | 2773.792 |
| 本轮stall编译binary disabled | 2954.160 |
| 本轮enabled unlimited0 | 3498.360 |
| 冻结stall，名义100GB，0µs | 10091.752 |
| 冻结stall，名义100GB，20µs | 15948.192 |

同逐步far平均509.256MB/步，自身逐步max(device,bytes/100GB)参考5092.560µs不改。原CAS0/20约88926/89710µs，仅作为模型软件成本的不同实现对照，不当作物理kernel加速成绩，不扣controls修正成绩。

冷热诊断：冷步0µs24965.120µs，按真实引用2414.774MB/该步时间约96.726GB/s；热127步平均9974.639µs，全部128步没有bytes/time>100GB。20µs冷步31342.592µs/77.044GB/s，热127步15826.976µs。没有超过总kernel字节预算只是必要条件，**不是模型准确性证明**；完整trace热步出现明显过度节流。局部等待不能将闲置CTA/warp的额度共享、计算节奏和far分布不同于拷贝探针，给定系数的吞吐不再等于100GB。没有用完整算子重调系数“凑100”，无采样hardwarecounter/逐stage因果分解。

结论：不使用原全局高频CAS后不再约89ms，但**当前10.092/15.948ms仍只是经验stall模型输出，不能当作可信物理100GB原生direct成绩**。固定探针校准成功不等于完整attention模型验证成功。保留本负差与全部raw，不安装，不扩矩阵。

正确性/实际执行：copy探针全部payload通过；完整每条件step0/16/127各3行输出reference atol/rtol0.01通过；每条件Graph128个原生BatchMLAPagedAttentionKernel，actualEMU.so SHA d2322115bc80cded190f57367e2c80157e55488d666896f476ed6ec07eb1564b。实际source/数学/FAR SHA、mapping、资源开始结束（无foreign）、逐步时间俱全。GPU3稳定idle10秒，共享锁；不占GPU0/1/2。生产81093/34af/8507不变。1pass定向、无suite/sanitizer/物理/H800新结论。

文件：temp/glm_trace/stall_calibration_20261104/README.md、probe.cu、probe_raw.json、probe_state.json、frozen_parameters.json、stall.cuh、probe_frozen_header.cuh（probe编译原头）、full_summary.json、summarize.py。整kernel：temp/glm_trace/tune_unlimited_uniform_20261103/bench_stall100.py、run_stall100.py、stall100_state.json、stall100_stall.log、stall100_full_trace_gpu3.json。prepare.py非幂等，完成raw有exists保护；独立include/JIT被ignore。无pendingrunner。
