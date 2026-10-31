# 小 batch 每 warp LRU 直方图实验（2026-11-01）

每 warp 独立 LRU 直方图候选拒绝。六个同参数短窗口配对的收益不稳定，最佳仅0.111%，最差慢1.047%；不保留到生产。

对照为冻结 JQ `a2f37f…`，不是仍在等待原保留门槛的生产 b6。只改变原 cached/staged、rows×headtiles≤32 的 ROW_MERGE 路径；A100 上保持 108 CTA、256 线程、2 stage、32 KV、fetch16 和原小 batch planner。

原 helper 的八个 warp 向同一 256-bin 直方图写入已经 warp 聚合的整数计数。候选为每个 warp 分配独立的 256-bin 行，warp0 将八行求和写回各 lane 原来负责的八个 bin，再执行相同 radix rank/prefix scan。阈值、thread-major 同龄淘汰顺序、stamp、payload 和 attention 数学均保持；不增加 CTA/grid barrier。额外静态共享内存 8192 B，归并增加共享读取。

| 构建/实例 | 检查结果 |
|---|---|
| 受限链路 BF16 | 原 JQ 36 实例指令和资源相同；新增实例 REG:246 STACK:0 SHARED:10240 LOCAL:0 CONSTANT[0]:672 TEXTURE:0 SURFACE:0 SAMPLER:0 |
| 正常 BF16/FP16 | 原 JQ 72 实例指令和资源相同；新增实例 REG:244 STACK:0 SHARED:10240 LOCAL:0 CONSTANT[0]:672 TEXTURE:0 SURFACE:0 SAMPLER:0 |

机器字审计另外逐项比较原 36/72 实例的指令和控制字。正常构建原 105 测试及新增 4 测试通过；新增测试覆盖 BF16/FP16、capacity512/4096，相同 stamp、单字节/四字节 radix、精确淘汰槽位、受保护命中、全部缓存载荷、output/LSE、empty/single/padding 和零 miss Graph 重放。实际 Graph 和 synccheck/racecheck 证据单列保存。

| cohort | MTP / 请求数 | 链路 | JQ µs | 候选 µs | 耗时降低 |
|---|---:|---|---:|---:|---:|
| small256_m1_ | 1 / 1 | g100_l0 | 44.060 | 44.521 | -1.047% |
| small256_m1_ | 1 / 1 | g100_l20000 | 63.072 | 63.089 | -0.027% |
| small256_m1_ | 1 / 1 | g50_l20000 | 66.107 | 66.069 | +0.056% |
| small256_m8_ | 8 / 1 | g100_l0 | 92.448 | 92.779 | -0.358% |
| small256_m8_ | 8 / 1 | g100_l20000 | 116.331 | 116.288 | +0.037% |
| small256_m8_ | 8 / 1 | g50_l20000 | 134.016 | 133.867 | +0.111% |

共 12 格 / 6 配对，10368 计时 launch / 648 段。使用各步所有有效 pass 的算术均值；每 pass 冷启动，warm pass 排除，保留正确 Memcpy、同硬件正常纯 device 与逐步 max(计算,unique_miss字节/带宽) 两参考、命中率及源码 SHA。

不扩展完整1024位置×5pass、物理PCIe/NVLink或跨形状矩阵；也不继续扫描私有直方图行数、warp分组或归并布局。短测不构成跨场景收益证据。原JQ72075/1398继续等待最后三格确认，生产保留决策由其原门槛执行。

候选 SHA256：`093c4daef94fed17e6c9ee3d19472f4c66b5778ba2732c888a63fb59227c73ee`。canonical math、模拟器和原5304格矩阵均不变。整体目标继续 active；本实验不能证明大 batch 100GB 与重叠参考的剩余差距已解决。

[原始记录与复现脚本](../temp/glm_trace/tune_warp_histogram_20261101/README.md)。
