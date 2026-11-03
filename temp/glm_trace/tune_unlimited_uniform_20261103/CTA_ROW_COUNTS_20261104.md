# 单CTA行计数前缀实验

在clean prepare_parts==1且segment<=2048时，每行只有一个CTA，8warp的fresh/host/local三计数由共享前缀及直接store替代原每warp3atomicAdd（每行最多24次）。页级host_slot CAS/EXCH及去重owner编码、resolve和LRU完整保留；非旧claim probe/store，也不是shared claim hash。多分片/更长行回退原prepare。利用prepare阶段空闲动态shared，无额外global workspace；三次CTA同步包含末尾scratch复用保护，未省略必要同步。

384CPU随机含padding/local/far重复/fresh情况证明，在相同CAS胜者下rank/计数与原warp0..7合法原子串行次序完全一致，非声称原运行次序逐位一致。两项K65/K2048 GPU数学/LSE/Graph/cache载荷通过，性能数学采样/全有效cache载荷通过。

GPU3/A100-PCIE/NUMA1同卡独立farHBM，BF16/h64/MTP8/R128/K2048/LRU4096，256位置32步×3coldpass [3068.7040090560913, 3076.2880071997643, 3141.247995197773]µs，均3095.413337，近期clean3099.381330约0.13%差异，无采用信号。unique均215559.541667 vs旧215250.020833，实际字节略有变化，不能按相同bytes推断收益。主函数同统计器10169指令/REG255/STACK16，不能单独从codegen归因。raw/header/实际库SHA已核验，未新增sanitizer/完整矩阵/100GB/物理链路结论，生产不变。
