# Pool列表并行构造负结果

在poolowner上仅将每CTA leader全量串行扫描改为256线程连续分段扫描；计数/前缀使用query加载前闲置shared，leader扫描256计数，再每线程写自身稳定列表。owner/包序/行分配保持，shared容量不变，真实字节和必要操作保持。

正常GPU3同卡HBM BF16MTP8R128K2048LRU4096、256位置32步×3cold pass5450.912043、5442.368001、5479.712039µs，均5457.664028µs。比串行poolowner8460.554749µs低35.49%，但远慢于clean约3128µs，拒绝安装不测100GB。跨轮非严格因果分解；结果说明串行构造确有重大可消除成本，不证明剩余退化全部来自负载或等待。

2项K65/K2048数学/LSE/padding/QPEstride80/Graph/cache全载荷通过，性能采样和最终全部cache载荷通过。原始/source/实际库SHA在poolparallel_*json，编译/runner日志保留。未做全suite/sanitizer/物理链路，生产81093不变。下一机制应避免每CTA遍历全部包并平衡pool实际工作，不能把本版称已优化局部性调度。
