# 认领行包号查询去重复控制

从claimrow_quota冻结头出发，保留并行planner、精确q/q+1配额、包序和完整unique载荷/真实stage-cache双写/原数学/LRU。将每CTA limit/base只在初始化算一次，包号仅在ids中由warp0查询一次，leader写shared packet供issue/stores读取；phase/drain只查done<limit。不是改变配额或包大小，也不是旧ticketstep仅传不可变step ticket。新增三uint32字段占原结构尾部padding，dynamic shared仍163072。

共享复用协议：stores/issue的packet读取位于原CTA barrier之间；MDO barrier已完成全部stores后，publish只有leader读旧packet并st.release，然后由同一leader在ids写下一packet。其他warp在publish不读packet，因此不存在新旧值覆盖间的跨warp读写。下一次issue在原PV第一barrier之后读新packet，done在原第二barrier发布；最后一包按旧ticket<limit推进done到limit，不能按下一包sentinel跳过done推进。保留全部原有同步，未引入额外grid或CTA barrier。

两项K65/K2048数学/LSE/Graph/cache载荷检查通过。针对新增shared元数据复用，K65过滤BatchMLACleanServiceKernel一次launch的racecheck0errors/0warnings；不是完整K2048或全局release/acquire协议的sanitizer证明。性能事件外核对实际包表8877项与本次rowbase CPU映射逐项一致，ready epoch匹配，严格82/83包，保留7631/溢出1246；349959far引用中147872自用（42.2541%），原轮转同remap3423。完整最终cache有效载荷通过，容差未变。

正常GPU3/A100-PCIE/NUMA1同卡独立farHBM，BF16/h64/MTP8/R128/K2048/LRU4096，256位置32步×3coldpass。候选[3195.391982793808, 3187.8719851374626, 3256.4480155706406]µs，均3213.237328；此前quota3200.330672，跨轮未见改善。随后新clean控制[3108.415998518467, 3084.92798358202, 3179.2320013046265]µs，均3124.191994，候选慢2.8502%，三个pass均慢。真实miss计数逐步保存，未要求相同LRU等时戳次序/相同字节，不使用staged参考修饰结果。单向顺序控制非严格因果成本分解。

主函数同统计器10638→9942指令，REG255/STACK16/static1280保持。减少静态代码不能证明动态访存减少多少或单独归因时间；尚未采动态query/等待/规划时序，不能据此认定全CTA机制无潜力。停止当前包号查询局部变体，不安装、不测100GB，不声称达到目标。实际source/加载库SHA与全部raw、定向racecheck及控制保留。生产三header不变，无新全suite/完整矩阵/物理链路/H800结论。
