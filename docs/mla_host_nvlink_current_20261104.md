# 11/04 当前方法在真实 NVLink 上复测

GPU0 计算、GPU1 远端 HBM，A100-SXM4-40GB，NV12，NUMA0。不开 FAR 模拟器，无人为带宽或延迟限制；本轮未重新校准链路 GB/s，不把 kernel 字节/耗时称为链路带宽。

BF16、64 heads、KV512+64、topk2048、128 请求、MTP8、每请求 LRU4096；assistant64k 原 trace/seed/bindings，3 cold passes，全部步骤算术平均。短筛256位置32步；正式完整1024位置128步。索引上传、诊断、校验在事件外，attention 和真实必要缓存/中转写入在事件内。进程开始前稳定 idle 10 秒、共享 GPU01 锁，开始/结束没有外部 GPU0/1 进程。

| 方法 | 256位置 µs | 完整1024位置 µs | 完整比纯device |
|---|---:|---:|---:|
| 指定纯 device 计算参考 | 2453.557 | 2454.085 | — |
| 当前优化 direct | 3013.675 | 2915.435 | +18.80% |
| 当前生产 staged | 2882.517 | 2967.971 | +20.94% |
| 全CTA/八warp unique staged，clean基础版 | 2968.896 | 3041.592 | +23.94% |
| 全warp下一包提前发，next_packet_qk | 2976.832 | 未扩测 | — |
| 全warp异步分组，deferred_packet_normal | 3006.603 | 未扩测 | — |

完整 direct 比当前 staged 快1.77%，全warp clean比 staged慢2.48%。短筛排序不同；不得仅凭短前缀下结论。完整三pass均值：direct2917.576/2903.496/2925.232；staged2946.208/2969.728/2987.976；clean3037.216/3030.264/3057.296 µs。没有反序重复，较小方法间差距仍有顺序/时钟漂移限制。两种近期变体短筛均未胜clean，不扩大矩阵。

完整实际源码模型远端字节：direct平均509.809 MB/步，staged259.601 MB/步，clean259.589 MB/步。direct按远端引用，staged/clean按unique miss；不共用传输分母。这些为remap/状态逐步核算，不是硬件计数器流量。

**2.454ms是指定device实现的计算参考，不是已经加入路径专属NVLink字节预算的逐步理想重叠参考。** 本轮没有新测链路带宽，不把上表差距称为严格传输/计算联合下限差距，不沿用100GB模拟参考。纯device将每步unique远端数据预先装入本地compact tail，搬运在事件外，再计原device attention；这和历史指定参考的做法一致。它不包含融合prepare/LRU/实际必要cache维护。

## 与历史比较

11/02相同GPU0/1、完整1024位置、128请求/MTP8/LRU4096的旧direct3335.189、staged3016.416、device2467.179 µs。当前direct2915.435较旧direct快12.585%，比device的差距由35.18%降为18.80%；这是跨日期实测对照，非同轮旧binary消融。旧16请求约531–558µs、高device约7–11%的格子是另一batch，不可与128请求混比。本轮没有重测16请求。

## 正确性与实际派发

所有五种远端路径的短筛，在每pass的step0/16/31上对三行完整输出与reference比较，atol/rtol均0.01，不放宽；结束核对全部524288有效cache槽真实载荷。完整三路径复用这些检查，在完整128步后再次核对全部cache；未新增全面pytest/sanitizer。

完整Graph实际检查：direct为单BatchMLADirectPVKernel，staged单BatchMLAPVFeatureKernel，clean单BatchMLACleanServiceKernel；均108CTA、block[32,4,2]、active1。纯device实际Graph及库SHA也保存。VMM实际allocation properties明确far位于GPU1，near/staging位于GPU0。

生产direct/staged header均81093ba5ab33cc5d84b2df04c702975697387b4054ebe8f1991f558f1bd6cdd2；clean29835918684dd24c76e15e27366e7e330268c3dd66c939eaffb1462b54ab2742。继续使用独立冻结include/JIT；生产代码/默认路由未更改。全warp未安装。

## 原始证据与失败保留

目录 `temp/glm_trace/tune_unlimited_uniform_20261103/`：

- `nvlink_full_20261104_{retained,clean}_nvlink_native_f24.json`、`nvlink_full_20261104_retained_staged_nvlink_native_f24.json`、`nvlink_full_20261104_floor.json`：完整逐步时间、实际派发、载荷、library SHA。
- `nvlink_20261104_*.json`：短筛全部正负结果；`nvlink_summary_20261104.json`绑定raw SHA与均值。
- `bench_nvlink*.py`、`run_nvlink*.py`、`resources_nvlink.py`、`nvlink_graph.py`及对应logs/state：复现及资源记录。完成的runner不要覆盖运行，raw有exists保护。
- 首轮额外旧`base`控制加载的是cab508dc旧direct（**不是当前staged**），3370.699µs；raw名字继承旧harness，为`production_service22_control_nvlink_native_f24.json`。其config明确staged=false；只保留为过时控制，不混入当前生产表。本轮随后用retained准确源码补当前staged。
- 首floor启动因runner错误指向不存在的include_floor而在CUDA前失败；`nvlink_floor.log`与初state保留，修正为retained include/JIT后`nvlink_floor_retry.log`成功。
- 汇总脚本初次把自己的list格式summary当raw读取而报错，已增加类型过滤；无GPU动作/计时影响。

本轮没有可信100GB/新延迟/物理PCIe或H800结论。
