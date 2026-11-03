# 认领行局部性与并行前缀配额

先用原clean kernel事件外导出实际row_slot_base/remap（32步一pass、正常GPU3），无生产修改。实际末步unique177486、far引用349968；将每个20页包交包首slot的claimrow%108可把自用引用3159→171534（约0.90%→49.01%），首次far pair涉及producer平均13.366211→5.113281。但每CTA负载30–160包、均82.176，约1.95倍尾部，未直接构建这个无配额形式。这是末步算术分析，不是时间/带宽预测。

新完整kernel从clean冻结头出发，在resolve已产生本CTA shared row prefix之后、LRU覆盖shared之前，线程并行计算各owner已有区间包数，两个CTA scan得到excess/deficit prefix，再由线程并行逆查源owner与其至多约10个行区间，写本CTA只读包表。每CTA严格q/q+1包，先取自己的claimrow包，超额按prefix填缺口。无额外grid barrier、无动态逐包CAS/全局服务队列、无串行greedy规划。原unique页、真实stage/cache双写、包级release/acquire、全8warp数学与精确LRU保持。区别于旧poolquota：不是按pool粗划，也不由每CTA leader反复重建localfirst+spill greedy；是复用现有shared row prefix的并行计数/scan/逆映射。每包仍有只读列表查询成本，规划及全部global写入均在kernel事件内。

180CPU随机含unique0/不同CTA数/不整包证明exact partition及q/q+1；真实先导末步可保留7557/8875包在claimrow CTA，1318包溢出。GPU原型unique0直接返回planner，不读resolve未发布的旧global rowbase，避免重犯旧poolquota零miss问题。新增partial backing长度和planner shared边界检查。

实际GPU最终包表8857项，逐项等于用本次actual rowbase重新构造的CPU映射，完整0..8856恰一次，ready flags epoch匹配；108CTA配额82/83，保留7621包，溢出1236包。该步far引用348379、自用147704（42.3975%），同remap原轮转为3158（0.9065%）。这些是完整kernel实际planner/载荷协议的验证，不是硬件远端计数器证明。两项K65/K2048数学/LSE/Graph/cache载荷定向检查通过；性能数学采样和全部最终有效cache载荷通过，容差未变。

正常GPU3/A100-PCIE/NUMA1同卡独立HBM源，BF16/h64/MTP8/R128/K2048/LRU4096，256位置32步×3coldpass [3215.519994497299, 3212.6720026135445, 3172.8000193834305]µs，均3200.330672。未优近期clean3099.381或生产staged约3029；不安装，不扩展100GB模拟，不宣称全CTA unique方案已胜出。相对旧串行poolquota约3540更快仅为跨轮不同方案描述，不能当单项因果收益。主函数REG255/STACK16/static1280（原1152），同统计器10638指令；未分离planner、包序、列表查询或服务时序的动态成本，不能凭更高自用率推断更快。

raw、实际header/加载库SHA、CPU模型、首导出及全部计时保留。无新全矩阵/sanitizer/FP16/物理PCIe/NVLink/H800结论；正常far_rows0仅禁用计数，不是零远端载荷。生产三个header未改。
