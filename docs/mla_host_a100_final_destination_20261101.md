# A100：预先解析 miss 最终目标槽（2026-11-01）

候选拒绝，生产保持已完成保留验证的CPEa436。以冻结CPE为直接基线，GPU3三种设置均回退，不扩展完整trace、正常物理链路或SXM矩阵，也不扫描LRU/barrier/fetch参数。

候选先用原精确LRU选择victim，清除旧resident并把目标slot_page标为pending；两次新增grid barrier之间，把现有32bit miss remap改为最终cache槽或保留staging行。原f24专用fetch CTA使用64行双缓冲，真实far读取后每页只写一个最终位置；全部256个writer执行fence并参加CTA barrier后，逐页release发布slot_page。attention tile waiter acquire该标志或原staging chunk flag。per-KV地址解析器保持，不再像旧direct_cache候选那样逐KV依赖victim查询。

提前LRU、remap遍历、两个grid同步及逐页就绪轮询都是新增成本，省一次HBM写并未取得整体收益。保持原attention数学helper、完整576特征、32KV双stage、108CTA/256线程、dynamic147968B、实际远端字节和FAR hook、精确launch级LRU；模拟器及两参考完全保持。正常/EMU原CPE38/76实例的指令、资源、机器字和控制字精确相同。新REG255/STACK0/LOCAL0/static2048B；实际单108CTA/block32x4x2/active1 Graph通过。

105项正常加8项stress共113项独立测试通过，含长K2048/奇尾、128池、淘汰/全部cache载荷、LSE、Graph重放、K65/QPE stride80、跨CTA共享97页/互斥2048页及跨launch清空。首个匹配BF16/K97 launch的synccheck/racecheck/memcheck全零错误；memcheck关闭API报告而保留设备检查，不是全部trace穷尽证明。CPU结构oracle144case只检查给定victim映射的目标/载荷/保护hit，不替代GPU同步、LRU选择或性能验证。

| 链路 | CPE µs | 候选 µs | 耗时降低 | Memcpy µs | 纯device µs | 逐步max µs |
|---|---:|---:|---:|---:|---:|---:|
| g100_l0 | 3416.437 | 3545.333 | -3.773% | 6099.808 | 2556.960 | 2699.765 |
| g100_l20000 | 3464.501 | 3563.733 | -2.864% | 6076.448 | 2556.960 | 2699.860 |
| g50_l20000 | 5268.821 | 5369.760 | -1.916% | 8481.365 | 2556.960 | 4960.201 |

共6格/3配对、1152计时launch/72有效段。固定MTP8/R128、LRU4096、f24/split off、256位置×3冷pass，CPE→candidate。GPU3 A100-PCIE/NUMA1的far是同卡独立HBM，不是NVLink；本实验没有正常链路/SXM/H800结果。逐步逐pass、全部原始时间/guard、实际unique counter、源码/库SHA及两参考保持。

**原始候选reference命中率诊断无效，审计结果置null。** 旧脚本把remap指向cache区视为命中，本候选的获准miss同样指向cache区，因此只计未获准miss。原raw数值保留于原件，审计另列原报告值和失效原因，未用它宣称命中率提高；unique counter、时间和逐步max按独立路径核验。审计最初明确拒绝，原父33355 terminal1发生在所有计时完成之后，失败log/state/source保存diagnostic_failure/。

追加双版本、各四步正常构建诊断：运行前从resident统计cold/eviction的41904个reference miss和97个unique，warm为0/0；kernel unique counter及全部有效cache载荷/output均相符。原CPE post-remap阈值cold计41904，新候选计0，证明诊断与新编码不兼容。它不能恢复旧trace每步的reference命中率。该诊断首版在首次eager run前读取lazy cache views失败，两个exit1日志及源码保留；先初始化再清空cache后两版各四步通过，未修改kernel或原计时raw。

候选SHA：`3c744e78251fdc6787c6a68c0d248773857c859e1c09dba687c58ef1dfaa7749`。CPE生产SHA为a4361729…2b15；数学34af、模拟器5e409及原5304矩阵e3fd不变。主目标继续active，现有100GB差距仍未解决。

[实现、原始记录与状态](../temp/glm_trace/tune_destination_20261101/README.md)。
