# A100：100 GB/s 高延迟与小 batch 调优（2026-10-28）

用户明确了两个独立方向：100 GB/s 加较高访问延迟，以及小 batch。MTP8 的 128 请求对应 **1024 行，是大 batch**；小 batch 在本文单列，覆盖 MTP1 的 1/4/16 请求和 MTP8 的 1/2/4 请求。

## 最终改动与结果

保留的 kernel 改动只调整 `host_batch_fetch_rows`：fetch CTA 多于 8 个时，未命中页足够每个 fetch CTA 分到一块 64 行后，就使用 64 行搬运块；此前要达到三块才切换。这样在较多 fetch CTA 下提高在途字节量，并减少 16 行小块造成的串行等待。8 个及以下 fetch CTA 的选择规则保持。仍为单个 cooperative kernel，融合 prepare、去重、搬运、LRU 和 attention；模拟器未改。

配合 fetch CTA 和远端 attention item 大小调参，100 GB/s、附加 20 µs 的最终对照为：

| MTP | 请求 | query 行 | 旧版 f16/h64 | 新版推荐配置 | 新版耗时 | 耗时下降 | 下限 a | 下限 b | 新版÷a | 新版÷b |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 8 | 125.95 µs | f16/h64 | 101.89 µs | **19.1%** | 49.15 µs | 49.15 µs | 2.07 | 2.07 |
| 8 | 2 | 16 | 152.06 µs | f24/h64 | 151.55 µs | 0.3% | 76.80 µs | 76.80 µs | 1.97 | 1.97 |
| 8 | 4 | 32 | 272.90 µs | f16/h256 | 231.94 µs | **15.0%** | 110.59 µs | 110.59 µs | 2.10 | 2.10 |
| 1 | 4 | 4 | 101.18 µs | f32/h64 | 79.36 µs | **21.6%** | 46.08 µs | 46.08 µs | 1.72 | 1.72 |
| 8 | 16 | 128 | 852.48 µs | f24/h64 | 715.78 µs | **16.0%** | 497.66 µs | 497.66 µs | 1.44 | 1.44 |
| 8 | 64 | 512 | 2911.74 µs | f24/h64 | 2405.38 µs | **17.4%** | 1746.71 µs | 1232.90 µs | 1.38 | 1.95 |
| 8 | 128 | 1024 | 5609.98 µs | f24/h64 | 4444.16 µs | **20.8%** | 3582.52 µs | 2458.62 µs | 1.24 | 1.81 |

单位和配置：f 为 `host_fetch_ctas`，h 为 `host_item_entries`（远端 attention 任务的目标 KV 条目数）。同参数比较与调参收益不能混在一起：MTP8 单请求的 19.1% 来自搬运块选择；128 请求的主要收益来自 f16 改为 f24；MTP8 四请求的主要收益来自 h64 改为 h256。**旧版比较基线固定为 f16/h64，不代表旧版全部参数的最优值**，例如无附加延迟时四请求的旧版 f8 已接近本次 f16/h256。

大 batch 高延迟优化后，128 请求为 4.44 ms，与新版无附加延迟 f16/h64 的 4.44 ms 接近；仍为纯 device 计算下限的 1.81 倍，不能宣称达到纯 device 性能。小 batch 也没有普遍胜过 Memcpy+LRU：MTP8 单请求本文 101.89 µs、Memcpy 95.23 µs；四请求本文 231.94 µs、Memcpy 225.28 µs；两请求的前后差异在计时波动内。MTP1 单请求仅约 2.3% 改善，固定管理开销仍需进一步处理。

全部 0/20 µs 的同参数对照、较快和较慢格子、朴素方法及两个下限在 [`results.md`](../temp/sim_link/tune_100/results.md)。其中 0 µs、64 请求同参数耗时约增加 0.2%，没有删去。

## 测试设置与证据

- 本机 GPU0/1：A100-SXM4-40GB，NV12，NUMA0；BF16、64 头、576 维 KV、top-k 2048、64K 上下文、每请求 LRU 4096 槽。
- 当前使用此前 DeepSeek-V2-Lite 代理 top-k：MTP8 每次 pass 为 4 步，MTP1 为 32 步。每个最终配置 **7 次冷启动**；逐步中位数再对后一半步平均。正式 GLM 64K/1024 trace 尚在排队，不能把本轮结果称为正式 GLM 长步回放。
- 旧版 header 由 `git show c7921958:include/flashinfer/attention/mla_host_batch.cuh` 读取；按旧版/新版顺序串行跑，所有退出路径恢复新版，运行清单和源码哈希保存在 [`verification_runs.json`](../temp/sim_link/tune_100/verification_runs.json)。
- 带宽和延迟模拟器保持原版 SHA-256 `5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095`。100 GB/s、0/10/20 µs 的独立校准通过；20 µs 档实测 99.80 GB/s、20.55 µs。
- **75 项完整单元测试通过**，包括新增 16/24 fetch CTA，以及不足 2048 个独立远端页、末块不完整的 MTP 用例；日志 [`pytest.log`](../temp/sim_link/tune_100/pytest.log)。最终 13 份计时 JSON、63 个批次配置的本文每步采样行校验通过，最大绝对误差 **0.006007**，BF16 容差 1e-2。
- 下限 b 取正常构建的全 HBM FA2 attention 最好实测样本；下限 a 为逐步 max(b, 去重后未命中字节/模拟带宽) 的稳态平均，未计依赖延迟，是乐观下限。比值没有截断。
- 朴素方法保留原口径：CPU LRU 不计时；本次远端是 peer GPU，其 gather 计时。HiSparse 需 H800，本轮不测。
- 另保存新版 50 GB/s＋20 µs 的逐步正确性和性能检查，没有据此声称所有 50 GB/s 场景已优化完成。

## 未采用的候选与复现

四级紧凑 fetch 流水在部分小 batch 有收益，但固定 f16 时出现大 batch 回退；单 warp 前缀扫描收益不足；将 cooperative grid 缩到 32/64 个 CTA 会损失 attention 并行度，普遍更慢。这些改动已撤回，补丁及原始 JSON 留在 `temp/sim_link/tune_100/`；已验证补丁能从 c7921958 还原相应计时文件记录的 header 哈希。

初始 fetch 扫描：`bash temp/sim_link/tune_100.sh`；planner 扫描：`bash temp/sim_link/tune_planner.sh`。重跑最终旧版/新版对照：`python3 temp/sim_link/verify_100.py --output-dir temp/sim_link/tune_100_repeat`，它从 git 读取并校验旧版 header，要求指定目录没有同名输出；应在 GPU0/1 空闲时串行运行。生成已保存数据的报告：`python3 temp/sim_link/report_tune_100.py`。最终代码没有实验 grid 控制；复现已撤回的 grid 试验需要在独立基线 checkout 中应用所存补丁，不能直接运行 `tune_grid.sh` 并把无效覆盖当成实测。

H800 专用 TMA/attention 尚未在本轮验证，集群排队的原始调优作业保持其原源码。接下来应将本轮参数选择方法用于真实 GLM 长步 trace，再评估小 batch 的 prepare/LRU 固定开销与跨层复用。
