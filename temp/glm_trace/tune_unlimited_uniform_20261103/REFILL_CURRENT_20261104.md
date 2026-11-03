# 全CTA服务：前置current发布、删等待step首屏障

从clean全CTA unique staged冻结头出发，单cooperative kernel、108个8-warp计算CTA、原20页全局轮转包、双32KV计算stage、真实每unique远端页读及stage/cache必要写回、ready release/acquire、精确launch LRU全部保留。不改attention数学、包所有权/次序/大小、模拟器或参考。

原clean_service_step由leader先写shared current=done，再cp.async.wait_group0、CTA barrier、stores。新版本在已有refill末尾由leader同时更新done和current，借原PV第二barrier或step末尾barrier发布；等待step只做各线程wait_group0、随即由相同线程读取其自己搬入shared remote的16B向量并写stage/cache。stores后的CTA barrier及ready发布仍保留。主计算循环的KV stage barrier未删，只去掉冗余current写入。与旧ticketstep（保留首屏障、取消等待step末屏障、显式传ticket）不同。跨线程页号/victim由上一处已有CTA barrier发布；cp.async.wait_group本身只保证执行线程提交的异步拷贝完成，不替代跨线程同步。此语义参见[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)；GPU定向测试不能证明所有未来形状无竞态。

GPU3/A100-PCIE、NUMA1、同卡独立HBM远端源，BF16/64heads/MTP8/R128/K2048/LRU4096，256位置32步×3coldpass，单位µs：

| 路径 | pass0 | pass1 | pass2 | 平均 |
|---|---:|---:|---:|---:|
| refill_current | 3094.944 | 3107.584 | 3150.720 | 3117.749 |
| 随后新clean | 3096.640 | 3120.224 | 3162.688 | 3126.517 |

同轮小幅约0.280%优势，pass0仅1.696µs；反序补做的一次refill_current为3102.816µs，比新clean首pass3096.640慢6.176µs。两者非同一冷启动顺序，不能从反序一pass单独推断退化；综合没有稳定净收益，拒绝安装、不延伸100GB。近期当前生产staged3039.147µs（另一轮）明显更快，也不拿它作严格同轮差分。

两项K65/K2048定向数学/LSE/padding/QPEstride/Graph/cache映射和载荷检查通过；性能采样与最终全部有效cache载荷通过，原容差未放宽。针对首屏障去除，K65目标BatchMLACleanServiceKernel一次launch racecheck为0 hazards（0 errors/0 warnings）；不称完整K2048或跨launch sanitizer证明。Graph实际单目标kernel、108CTA、block[32,4,2]、dynamic163072B、active1，非旧路径回退。静态SASS目标函数BAR.SYNC由91降至86，REG255/STACK16/static1152维持；静态屏障次数不是动态执行次数或可回收耗时的测量。原始每步时间、实际header/加载库SHA与所有载荷、probe记录在raw/audit中。正常raw的far_rows=0是关闭的诊断字段，并非零远端载荷。

本轮未测新可信100GB、50GB、物理PCIe/NVLink、H800或全矩阵；生产header维持81093ba5。后续应针对等待/计算重叠的实际因果证据继续设计，不重复删同一道屏障或只凭SASS指令减少判断收益。
