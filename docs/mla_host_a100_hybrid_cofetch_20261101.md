# A100：双 stage attention 与辅助取数（2026-11-01）

两版均拒绝，生产保持 `0f82e9de` / `b6eea095…e72a`。编译期区分搬运角色消除了大量栈使用，但仍没有优于现有调度。

此前单 stage 控制版在 100GB/s 下慢约 12%，因此本轮保留原两个 attention stage。专用 CTA 在动态 shared 中使用两个 64 行搬运缓冲，完成后加入 attention；worker 另有两个 4 行 shared 缓冲，每八个 KV tile 或等待未就绪数据时辅助搬运。一个原子 job cursor 分配不重叠的 16 行工作：worker 领一个 job，专用 CTA 每次领四个。worker 四个 subchunk 全部落地才发布 16 行 ready flag。原 gap flag 仅在 prepare/compaction 完成后复用为 cursor，退出时沿用 last-CTA 复位。所有 LRU 选择先完成，缓存数据和绑定各只有一个写者。仍为单 cooperative kernel，数学、容量、模拟器与参考不变。

第一版 `hybrid` 在 attention 中保留运行期 dedicated/worker 分支，导致新实例使用 REG255、EMU STACK440B / 正常 STACK432B。独立 `hybrid_static` 将两种 service 编译期区分，专用 64 行搬运在 attention 前完成；attention 的活跃 accumulator 区间只保留 worker 4 行代码。调度、job、缓冲、cadence8、数学、LRU、模拟器完全相同，没有扫描展开或 cadence。新实例变为 REG245、STACK0。EMU 静态指令 18712→11016，但不能将指令数下降等同于采用依据。

每版均通过正常 BF16/FP16 构建、105 项原测试和 2 项长队列/Graph/淘汰/全缓存载荷测试，共 214 项。每版全部 34 个 EMU / 68 个正常原库内 kernel 的指令和资源逐一相同。实际 Graph 均只有一个对应新 kernel，108 CTA × 256 线程、147968B dynamic shared、11264B static shared、1 resident CTA/SM，采样最大绝对误差 0.004128。

GPU3 A100-PCIE，far 为同卡独立 HBM 模拟，MTP8/R128、LRU4096/请求、staged、fetch24/off、256 位置 × 3 次冷启动 pass。两版分属独立 cohort，各有同参数 base；原版顺序 base→候选，修正版候选→base，全部步和 pass 算术平均：

| 候选 | 链路 | base µs | 候选 µs | 耗时降低 | 候选 Memcpy µs | 纯 device µs | 候选逐步 max µs |
|---|---|---:|---:|---:|---:|---:|---:|
| hybrid | 100GB/s、0µs | 3523.552 | 7347.307 | -108.520% | 6031.488 | 2556.960 | 2699.837 |
| hybrid | 100GB/s、20µs | 3554.549 | 7369.888 | -107.337% | 6064.768 | 2556.960 | 2700.013 |
| hybrid | 50GB/s、20µs | 5282.699 | 8368.747 | -58.418% | 8513.696 | 2556.960 | 4959.479 |
| hybrid_static | 100GB/s、0µs | 3559.456 | 4053.557 | -13.881% | 6032.576 | 2556.960 | 2699.648 |
| hybrid_static | 100GB/s、20µs | 3544.843 | 4073.397 | -14.911% | 6047.083 | 2556.960 | 2699.918 |
| hybrid_static | 50GB/s、20µs | 5284.704 | 5964.064 | -12.855% | 8480.736 | 2556.960 | 4959.410 |

修正版的两个 100GB/s 点仍高于逐步 max 约 50.2% / 50.9%，没有改善生产约 32%–34% 的主差距。因此不扩大完整窗口、正常物理链路或全场景矩阵；不再无新机制重做辅助取数或只扫描 cadence/fetch 数字。控制结果支持编译期角色拆分解决了该版的寄存器压力，但不提供所有性能退化的精确因果分摊。

[审计](../temp/glm_trace/tune_hybrid_cofetch_20261101/performance_audit.json)保留 12 格 / 6 配对、2304 次计时 launch、144 个有效 segment、两参考、命中率、unique miss、guard、每步/每 pass、全部慢格、源快照及编译证据。两版源码 SHA：`9168c83fdb5ea1e5f0281aa42f820d433135269922225c133d824f21a3588259`、`7cac4924a0871435b5edb915cdfd28de562c7951418fd520ea8321f98fb0df25`。原大栈版本完整保留，没有将其结果归给修正版。

H800 01:20:24 +08 查询仍为五个原作业 Priority/Dependency、两个 watcher 存活，未改排队源或重复提交。本轮没有 H800 / SXM 性能结论；所有本机 runner 已结束，主目标继续 active。
