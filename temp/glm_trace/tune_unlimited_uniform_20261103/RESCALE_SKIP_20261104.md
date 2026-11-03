# 恒等softmax缩放跳过实验

从clean全CTA unique staged冻结头出发，仅joint MDO在原exp2算出的factor精确等于1时跳过O/d乘1。分数exp、QK/PV、归一化、搬运/发布、精确LRU及真实stage/cache双写均保持。不同于已有包调度/同步/最终目的地实验。没有改变容差或远端字节。

正常GPU3/A100-PCIE、NUMA1、同卡独立HBM源、BF16/h64/MTP8/R128/K2048/LRU4096，256位置32步×3coldpass。候选逐pass [3140.6080052256584, 3132.224015891552, 3170.7199960947037]µs，均3147.850672；随后同参数未修改clean [3084.1279849410057, 3072.3840072751045, 3141.6319981217384]，均3099.381330。候选慢1.564%，每pass均慢，无采用信号；不安装、不扩展100GB模拟。顺序单向，不当严格因果性能分解。

两项K65/K2048定向数学/LSE/全mask/padding/QPE stride/Graph/cache载荷检查通过，性能数学采样与所有最终有效cache载荷通过。沿用旧容差。source和完整raw实际加载库SHA已核验；机器码有额外4个BRA/BSSY/BSYNC，REG255/STACK16与clean相同。这证明分支已生成，不能从静态代码单独归因退化。未统计动态因子分布/分支效率，未新增sanitizer/全矩阵/物理链路结论。

初构造器未限定重复数学函数导致assert，随后构建缺include失败；日志保留。新clean控制首次未放行禁用的service诊断白名单，计时后assert，incomplete原件保留；仅修runner白名单并使用新输出文件重试。未覆盖失败raw，未使用失败计时作成绩。
