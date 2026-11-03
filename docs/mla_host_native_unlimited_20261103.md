# 不限速：全 warp 迭代与 direct 计算专化（2026-11-03）

已保留新的正常 BF16 大 batch direct 计算分支。GPU3 同卡 HBM、完全未编译模拟器，最终2.862ms，对照原生产 staged3.029ms，耗时降低5.53%。相同 unique 字节的全 CTA staged 原型仍慢3.2%，未安装；因此不能把本轮收益表述为全 CTA unique 搬运已优于生产，更不能称100GB/s目标完成。

用户本轮要求先让不限速新 kernel 优于原生产，再讨论模拟。所有新性能是 GPU3/A100-PCIE/NUMA1 的正常构建，`FLASHINFER_EXTRA_CUDAFLAGS=''`，不是 enabled-unlimited 模拟器，也不是 NVLink。BF16、64heads、MTP8/R128、K2048、LRU4096/请求，真实 assistant trace256位置/32步×3冷pass，Graph/events覆盖完整单 kernel；规划/上传/校验在事件外。每步 actual miss、所有时间/负差、源码与实际加载库 SHA 均保存。

| 实现/对照 | 时间 µs | 对照结论 |
|---|---:|---|
| 初始生产 staged正常构建 | 3036.544 | cab508dc |
| 全 CTA：service 复用数学屏障 | 3404.149 | 慢12.11% |
| 合并为20页包 | 3262.763 | 慢7.45% |
| 完成自身搬运后切纯计算 | 3196.117 | 慢5.26% |
| 页内地址复用＋简化纯计算等待＋去诊断计数 | 3126.699 | 仍慢2.97% |
| 上述 staged反序确认 | 3128.949 vs3031.499 | 慢3.21%，拒绝 |
| 新 direct 计算首测 | 2856.469 | 对初始 staged快5.93% |
| 新 direct反序确认 | 2861.301 vs3029.035 | 快5.54% |
| 最终受限 BF16正常分支 | 2861.536 | 对最近 staged快5.53%，安装 |
| 非阻塞同 launch cache复用 | 3700.885 | 减少真实源读取仍慢，拒绝 |

表中跨行差值只表示迭代结果，不是严格单项因果分解。完整其他负结果在[audit.json](../temp/glm_trace/tune_unlimited_uniform_20261103/audit.json)，没有删慢格或混用计时分母。相同 stamp 下 admission tie可变化，不声称不同launch的 cache逐位完全相同；每格都检查最终全部有效载荷。

全 CTA 路线保留108CTA、每CTA完整8个计算warp、双32KV计算stage，回收目标路径未用的shared QPE后使用23KiB独立farbuffer；静态unique ownership保留去重与真实staging/cache写入。逐步减少service屏障、发布fence、地址重复计算及完成搬运后的检查，确实从旧正常诊断3.935ms降到约3.129ms，但仍未赢。PV之间分散cp.async、双操作数预加载、全完成计数、死farshared存QPE、warp相反服务次序、动态行队列、首读与LRU重叠等均没有进一步采用信号。寄存器直写版REG252/STACK0却3.387ms，不能从资源下降推断加速。该路线后续须针对新证据设计，不恢复原样筛选或扫cadence/fetch数。

新 direct 将已有 paired QK、item内 QPE复用、joint softmax rescale、PV feature调度接到原 direct inplace prepare/LRU/owner cache写回中。每CTA完整8warp计算、双stage，单108CTA cooperative kernel，REG254/STACK0/static1408/dynamic147968，实际Graph仅一个新`BatchMLADirectPVKernel`节点、active1。原始attention累加次序、quantized P分母、精确launch LRU和真实owner cache载荷写回保持，没有额外kernel或CPU规划结果代替设备工作。

**字节约束必须区分路径。** 新 direct约563.13MB实际远端引用读取/步，原 staged约248MB unique miss/步；它在HBM足够快时省去专用搬运/中转并使用较快数学，能够取胜，但不是unique-only远端实现。100GB/s下前者仅字节/B就约5.63ms，不能拿后者约2.48ms的传输参考宣称逼近下限。非阻塞cache复用在实际读取来源处按8lane整页统一选择/计数，实际约504.68MB，虽少10.36%却慢到3.701ms，未保留。

生产正常构建的轻量时间线补充：一pass32步event均3016.768µs，kernel内部span3007.808µs，prepare188.608µs，fetch阶段1366.016µs、相对起点最晚1554.208µs退出搬运，24个fetch CTA随后总计完成134.25/1024行计算。无miss对照生产2507.296µs，全CTA phase2572.192µs，release2667.520µs。时间线有少量插桩、无miss不是相同输入LRU消融，均只作诊断；这些数据支持“不限速时fetch占用代价较小且已会退役计算，新统一循环开销也重要”，不是可相加的严格因果拆分。[原始时间线](../temp/glm_trace/tune_unlimited_uniform_20261103/native_timeline_analysis.json)。

最终生产header为`81093ba5ab33cc5d84b2df04c702975697387b4054ebe8f1991f558f1bd6cdd2`。新增定义与dispatch均在`#ifndef FLASHINFER_FAR_EMU`下，另限制BF16、cached direct、至少4×SM行、单64head tile、原A100 shared guard；默认staged路由不改。剔除新增guard区域后，原source tokenstream与cab508dc一致（忽略空白）；这是源码范围证明，不是原全部机器码等价。canonical math34afad69/FAR8507eab0/逐步参考/原5304矩阵不变。[保留记录](../temp/glm_trace/tune_unlimited_uniform_20261103/retention.json)。

本轮18个候选各2项定向检查，共36项成功：K65/K2048、padding空/单有效/奇尾、QPEstride80、数学/LSE、3轮cache、Graph×2、全部有效cache映射/载荷；每性能格另采样0/16/31步数学、最终524288有效cache槽全载荷。发布协议独立检查38880000个uint4，errors0。有效正常计时2528launch；另有gate错误回落的192launch归档，不能算新候选成绩。新direct首gate误留staging容量条件导致回落，已承认、归档、修复并对direct/trycache/retained均确认实际Graph；queue/trycache首编译失败和trycache GPU前来源选择修订也保存。无新整套测试/sanitizer/full1024/FP16/SXM/物理PCIe或NVLink/H800结果，未重测模拟。

生产文件逐字安装最终已测冻结头，root JIT未额外重编；独立include/JIT被ignore，重建说明、raw、失败件和SHA在[实验入口](../temp/glm_trace/tune_unlimited_uniform_20261103/README.md)。其他用户文件、GPU0/1/2及H800队列均未操作。下一步讨论应分别围绕同unique staged的约3.2%正常差距和direct的带宽敏感性，不能假定消除专用fetch就一定获益。
