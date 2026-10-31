接手 `/mnt/data/rtx/projects/flashinfer` 的远端KV sparse MLA单kernel持续优化。先完整阅读 `docs/HANDOFF.md`，按引用检查未提交源码、实验数据、已完成阶段诊断和H800队列，再自主推进，不要只给计划或重复已完成实验。

目标：保持单cooperative kernel、数学计算与LRU语义正确，使kernel在小/中/大batch、MTP1/8、有/无LRU、50/100 GB/s及0–20µs延迟、物理PCIe/NVLink下都表现更好，显著优于正确Memcpy基线，并持续缩小到纯device及逐步max(计算,unique搬运)重叠参考的差距。当前100GB大batch仍高参考约32%–34%，目标尚未达成。

继续当前dirty工作区，不要只checkout HEAD：上一轮整行优化、103项测试和报告尚未提交。`temp/glm_trace/tune_general_20261031/`的三组阶段诊断已完成；本机没有本项目GPU任务留跑，状态仍需复查。基于诊断实现候选，用独立include/JIT隔离，在同参数新旧对照、完整trace冷启动多pass、正确CUDA计时及跨场景回归后保留有效改进。记录缓存命中率、每步/每pass、两参考、源码SHA、失败和更慢格子，更新自包含Markdown及可复现审计，只提交本任务文件。

不优化HiSparse，不改计时/模拟/参考制造收益，不覆盖原5304格矩阵。不抢占或终止其他用户GPU进程；自有实验串行并查独占。集群GPU只经Slurm，查询已有五个H800任务，不重复提交或覆盖排队源码；有硬件结果后再隔离验证迁移。保持单kernel，拆kernel等目标/架构改变再询问用户。以具体实现和有证据的改进持续推进，诚实报告剩余差距与回退，不将局部收益推广为所有场景已逼近极限。
