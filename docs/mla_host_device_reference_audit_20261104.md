# 原生 device 参考 2.454ms 与旧图 2.755ms 对齐审计

结论：当前 assistant64k 完整1024位置/128请求/MTP8的原生 FA2 MLA 计算参考约 **2.46ms**。新增整段Graph连续128步、3pass，平均2459.195µs，分pass2451.528/2457.352/2468.704；旧逐步同步2454.085µs相差0.208%。当前direct2915.435µs相对新的连续参考高18.55%，而非18.80%。参考仍不含数据准备、LRU或远端搬运，不是路径专属传输/计算联合下限。

旧报告2.755ms也可复现，但不属于当前输入/测试条件：用原bench_sweep、旧overlap captures、原fraction0–30%、8rounds、attn-only重跑，128请求各格2756.5–2757.9µs。第二次0/30%小probe得到2692.9/2756.6µs。未安装旧头或回退源码。

## 实际二进制排除法

新参考、同进程新旧输入控制、旧脚本probe都实际加载同一个原生.so，SHA **20caf1e9dcebb439dd3b2f7985e00d921a8c35d4f5f7a39aaba889d251c08755**。实际symbol均BatchMLAPagedAttentionKernel，traits相同、108CTA、block32/4/2、dynamic147968、active1。旧probe Graph含多次该kernel，符合原每graph约8ms重复执行。不是新的DirectPV/CleanService或host融合kernel，也没有LRU。

历史commit dbbc02c6与当前canonical mla.cuh差异仅FAR宏接入；正常编译宏展开为空。helper make_wrapper文件SHA历史/当前相同，原生planner相关检查无改动；_core变化只在host buffer。历史raw没有保存原生.so SHA，因此无法声称历史机器码逐字相同；本次复现已经用同一实际库排除了“当前换了计算kernel才产生差距”。

## 定向实验

固定当前原生.so、GPU0/1、NUMA0、BF16/64heads/K2048/R128/MTP8，reference-only不启动多个GPU runner。主audit稳定idle10秒、共享GPU01锁，记录开始/结束无foreign。数据prep在事件外，索引按每步unique远端页compact到GPU0。

| 同进程控制 | 搬入后直接算 µs | 先写128MiB驱逐buffer再算 µs |
|---|---:|---:|
| 新assistant完整128步，每step单launch，3samples | 2452.995 | 2453.029 |
| 旧overlap fraction0，两index sets，每graph3launch | 2581.220 | 2607.332 |
| 旧overlap fraction30%，同上 | 2505.728 | 2537.870 |

驱逐操作不代表profiler证明每条L2都冷，但沿用旧脚本相同操作。**新trace下差0.035µs，否定了“新参考只是提前搬运留下热L2而快0.3ms”的此前猜测。** 上表旧trace与新trace仍有重复次数不同，不能精确作单因素分解。

后续统一每graph3launch、先replay一次再计下一次、8samples：新trace每16步取1步（8步样本）prepared2633.723/evicted2652.219；旧fraction0为2668.736/2769.472；旧30%为2626.603/2731.115µs。说明改变重复/执行方式会改变数字；这是新的小样本反复执行负载，不能替代完整128步参考，也不能把两表差值全归时钟或单一机制。

因此最后补**与融合kernel同样整段Graph连续128步**：每step prep（未计时）→原生attention（单独event），没有每步CPU同步；3pass2459.195µs，确认当前完整输入确实约2.46ms。prep耗时在event外但仍可能影响系统资源状态，此为指定数据准备方式的参考。连续运行采样SM1395/1410MHz、mem1215MHz，最高power245.77W；不是逐kernel周期/因果分析。

## 已知输入差别与解释边界

旧trace为overlap_ctx65536代理capture，新trace为GLM5.3真实assistant indexer capture。按(request,token)身份统计每步unique页：新完整128步平均584259（范围517397–698121），旧两sets为660892/649664；每步总引用均2097152。新输入平均独立页更少、复用更多。物理compact布局还随fraction改变：旧原图相同batch的全本地attn本身随fraction由2902.912到2755.413µs，说明“都在HBM”并不意味着相同访存负载。

原生attention矩阵计算形状相同，但读取地址/页重复关系不同；新旧不等时不意味着计算被删掉、原生算法改变或LRU进入参考。页计数不是硬件HBM字节或缓存miss计数，**不能据此把所有约0.3ms差距精确归因于页复用**。精确分解旧2.755与新2.459的每一微秒尚无counter/单因素证据，不作此声称；对当前参考取值的疑问已经通过同库、完整连续重测解决。

另：旧attn路径脚本注明staging仅供timing、未校验数据内容；本轮原参考会真实填入所需KV，并对采样输出验证。旧attn-only复现沿用其timing-only行为，不当作新增数学正确性验证。

## 原始证据

目录temp/glm_trace/tune_unlimited_uniform_20261103：

- floor_audit_20261104.json：完整新trace/两种准备条件/旧输入控制，264记录。
- floor_audit_sustained_20261104.json：小样本统一重复控制，非完整trace参考。
- floor_continuous_20261104.json、floor_continuous_clocks.csv：完整连续参考及实际库/Graph/时间。
- floor_old_exact_20261104.json、floor_old_probe_20261104.json、old_exact_dispatch.json：旧脚本重现与同库证明。
- trace_locality_audit.json及.py：纯CPU唯一页统计。
- make_floor_audit.py、bench_nvlink_floor_{audit,sustained,continuous}.py及run_nvlink_*对应脚本/log/state。

首old_probe未正确把keep_graph传到__init__，raw graph访问失败，首log保留；修复后retry2成功。前述所有计时均CUDAevent原始实测，不扣除/修正时间、不放宽正确性容差，生产未改，无新融合kernel收益。没有全面suite/sanitizer/100GB/物理PCIe/H800结论。
