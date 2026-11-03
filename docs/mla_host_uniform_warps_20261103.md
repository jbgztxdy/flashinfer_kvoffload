# 全部CTA、全部warp交错访存与计算：首版实现及诊断

用户已授权实现同warp流水线方向。本轮已实现完整单cooperative kernel：**全部108CTA均以原256线程/8warp计算attention，同时承担唯一miss搬运；保留双32KV计算stage，无跨SM的QK分数交接。** 数学、载荷和搬运唯一性检查通过，但首版没有性能收益，未安装。生产cab508dc/math34af/FAR8507及正式参考保持，110%目标未完成。

当前结果还揭示一项模型适用范围：原v2的100GB校准不能直接外推到新108CTA×双10行小包通道；后者即使不计算attention也只有31–34GB/s。必须将这项软件限速开销与完整kernel自身开销分开，不能把9.5ms全部说成物理100GB下的架构失败。

## 已先完成同warp微探针

前轮已冻结的`research_sm80_pipeline_20261103/probe.cu`本轮稳定空闲后实际运行。比较同一组8warp纯fetch、整批发async后做真实QK、将18轮future-copy分散在36个MMA特征步之间；payload与四job的完整2048分数参考通过，原FAR字节/到达模型保持。独立local Q/K，没有真实attention工作替换。

| CTA数 | 模式 | 100GB0µs实际payload GB/s | 100GB20µs实际payload GB/s |
|---|---|---:|---:|
| 24 | 纯fetch | 99.43 | 98.56 |
| 24 | 整批async＋QK | 61.88 | 61.78 |
| 24 | 分散copy＋QK | 62.87 | 62.85 |
| 108 | 纯fetch | 99.19 | 98.40 |
| 108 | 整批async＋QK | 98.40 | 97.96 |
| 108 | 分散copy＋QK | 98.21 | 98.12 |

108CTA足以同时完成探针工作并接近目标带宽，但本轮没有证据证明“分散copy”比bulk更快。它只支持全部CTA参与具有可行性，不能据其没有O/softmax/LRU的微核推断完整attention达标。微核REG117/168/76、STACK0；不同codegen不能当作等指令或相同资源控制。

前轮占用失败state已保存在first_preflight，当前成功raw独立保存。没有重跑已完成cofetch、QKassist等旧筛选。

## 完整候选实现

冻结候选及SHA见[manifest](../temp/glm_trace/tune_uniform_warps_20261103/manifest.json)。仅私有typed路径，输入限制SM≤128、rows≤4096、K≤8192、cached staged whole-row；旧公共实现/默认派发不改。

- 回收目标PVFeature数值路径不再读取的8KiB共享QPE；query PE继续按原packed片段从global载入寄存器。Q-nope shared地址/向量布局保持；canonical其他kernel所需shared QPE没有删除。
- 保留两个32KV计算stage、pairedQK、joint max/O/d rescale及PV每元素累计次序；全部108CTA按row=CTA+k×grid静态计算自己的完整输出。
- 每CTA另有双10行far缓冲、shared service上下文；unique chunk=CTA+k×grid静态且恰分配一次。三轮16B复制在原pairedQK的特征步0/12/24之间发出，FAR TILE按有效页数×1152计费，不提前预约未发出的未来数据。
- 所有pool先执行原精确LRU选择，再经grid barrier开始service，避免等待victim时阻塞进度。这增加了串行准备工作，是候选的实际成本，不称免费。
- 使用原cp.async和v2 FAR wait/预约语义；仅在真实读完成且stage deadline到达后复制staging/cache并release ready。cache绑定、dedup槽清理和真实payload写回保留。flags在whole-row闲置partial_o backing中按10页分配，未扩大wrapper/GPU工作区。
- 每个CTA等待需要的KV时继续推进自己拥有的唯一miss，退出前drain自己的所有远端任务。末CTA沿用原游标和epoch重置。没有跨CTA score队列，也没有固定的fetch SM或loader warp角色。

声明dynamic163072B（包含176B service context及128B对齐）、实测static1152B、REG255、**STACK48B/LOCAL0**，SASS有LDL/STL，不能称零spill。共享内存及寄存器均限定每SM一个CTA。原f24输入scalar仍保留以满足私有派发，但这里不代表24个专用fetch CTA；实际全部108CTA服务。

52组CPU ownership覆盖0/1/9/10/11/尾包/大count和1/4/24/108grid，全部unique页及72个16B向量恰一次，FAR字节等于unique×1152。Q-nope4096个向量坐标逐项一致；PV、query PE、PE MMA及jointMDO四helper主体保持原数学。CPU证明不是GPU同步或性能证明。

首编译因两处外层模板缺Params失败，原源/日志/manifest在first_compile；GPU前已修正，并按实际sizeof固定163072。后续CPU/EMU BF16 entry构建通过，实际新typed实例资源和loaded library SHA均绑定。没有正常非EMU构建或所有原实例机器码等价验证。

## 完整trace短窗口结果

独占GPU3/A100-PCIE/NUMA1，同卡独立farHBM模拟，非NVLink；BF16/h64/K2048/MTP8R128/LRU4096、staged/split off，256位置=32步、三冷pass。Graph/event单kernel计时，输入上传、计数/数值/载荷校验在事件外；0µs先base后candidate，20µs反向。每路径按自己的实际unique miss构造原逐步max参考，未将两路径小幅miss差异强制抹平。

| 链路 | 同轮生产 µs | 全warp新版 µs | 相对生产 |
|---|---:|---:|---:|
| 100GB0µs | 3295.179 | **9687.957** | 慢194.00% |
| 100GB20µs | 3307.616 | **9449.024** | 慢185.67% |

生产对应逐步参考2701.156/2701.157µs；候选实际unique参考2701.240/2701.165µs。每步约248MB必需远端payload保持。候选发布约21524–21528个chunk，约12.3万/12.7万次等待进度循环；每个pass每步发布页数==actual unique、issue==publish==ceil(unique/10)，没有多搬或遗漏字节。

20µs比0µs总时小不是可采用的延迟收益，不能据此推断延迟有利；它们是不同运行及不同竞争/调度实例，当前负差远大于波动。

## 新小包通道的100GB适用性检查

由于候选大幅变慢，新做纯transfer对照，完全不执行attention/metadata/LRU，仅真实copy和原v2 hooks，固定约247.741MB载荷、13页尾部、每点三样本，全payload一致：

| hook设置 | 24CTA×双64行，0/20µs GB/s | 108CTA×双10行，0/20µs GB/s |
|---|---:|---:|
| hook关闭 | 522.91 /522.91 | **652.70 /646.31** |
| hook启用、不限带宽 | 423.95 /147.67 | **582.51 /118.11** |
| hook启用、100GB/s | **99.44 /98.56** | **34.26 /31.41** |

独立首个transfer对照也得到10行34.36/31.24GB/s，原件保留，未用后续数据覆盖。小包实际读写和20µs名义在途本身能支持超过100GB/s；强退化在启用带宽预约后出现，指向v2软件限速/预约竞争与高包频率的组合。不能仅凭此表精确分摊CAS碰撞、时间粒度、barrier或其他指令成本，也不能把header数学计费正确当成所有packetization的有效带宽均已校准。

这些结果不否定原24CTA×64行校准和生产3.3ms的口径。它们明确否定“原校准也保证新108CTA×10行能实现100GB/s”。候选9.5ms是当前模拟实现下的实际端到端结果，但不能称为等价物理100GB链路下的性能预言。本轮没有修改模拟器、跳过预约或改参考来挽救候选。

## 模型之外仍有实质开销

再做完整kernel不限速、0µs诊断，两版各同输入一冷32步pass，保持所有必要attention/LRU/cache操作：

| 诊断 | 生产 µs | 新版 µs |
|---|---:|---:|
| v2 hook启用、不限带宽0µs | **3004.960** | **3935.328** |

候选仍慢30.96%，并非全部问题都在限速预约。等待进度循环从约12万降至约1096，但细小service退休/发布、频繁同步、准备LRU顺序、静态row收尾及48B栈等仍可能共同影响；未完成严格因果分摊，不能直接把0.930ms全称为某一个barrier的成本。不限速只是诊断，不替换100GB参考，不称NVLink/物理链路实测。

## 决定和下一机制约束

**拒绝当前双10行统一服务实现，生产保持。** 本轮没有将“全部warp都计算”写成目标达成，也没有回退到旧源码。成熟dense软件流水线不自动解决远端稀疏去重、cache和小包限速开销。

后续需要新的包粒度或shared生命周期方案，同时降低普通attention热循环内的service同步；不能仅扫round/cadence/row数，或者继续沿用“2.49MB≥BDP因此带宽必满”的推断。若继续评估更细的取数组织，应先验证其独立有效带宽，再运行完整对照；模拟器适用范围需要单独记录或审计，不能在kernel候选内悄悄更改。

## 验证与归档

2项BF16定向检查通过，含K65/K2048、padding/空/单有效/奇尾、QPE stride80、三轮cache状态、output/LSE、Graph两次重放、全部有效cache映射/载荷；长case有实际非零miss和发布计数。每个性能pass抽查0/16/31步的三行数学，最终每格524288有效cache槽全载荷一致。无新整套pytest/sanitizer/full1024/FP16/正常物理链路/H800验证，也没有新Graph节点probe。

共448次完整kernel计时launch（4×96＋2诊断×32）、36次samewarp微探针、48次纯transfer诊断，全部实际源/库/输入绑定/正负差/计数保存。[审计](../temp/glm_trace/tune_uniform_warps_20261103/audit.json)、[复现入口](../temp/glm_trace/tune_uniform_warps_20261103/README.md)。所有runner已结束，无后台等待器；GPU3末检0MiB/0%，GPU0/1/2及H800排队任务未触碰。
