# 固定20页包的全线程向量映射

从clean冻结头改producer cp.async及真实stage/cache stores的线程布局，20页所有权/服务时点/ready/原LRU和attention不变。原8lane/page共5warp每9向量轮；新1440uint4由256线程展平，5完整轮及160线程尾轮。不是换包大小、包序、配额、PV中分散copy或取消必要写回。0–20页共21CPU exact partition检查，每有效16B恰一次源读及原目标写；非硬件字节计数/内存序证明。

GPU3/A100-PCIE/NUMA1、同卡farHBM、BF16/h64/MTP8/R128/K2048/LRU4096，256位置32步3coldpass。flat_payload逐pass3183.455996/3193.440020/3241.247989µs，均3206.048002。主函数同统计器9308→11093指令，STACK16→32，REG255。由此补唯一loop控制：两payload循环保持rolled，向量分配/字节及协议完全相同，指令9341/STACK16，3124.384008/3163.744003/3193.408005，均3160.512005µs。两者都未优近期clean3099.381（跨轮非夹测），各2定向检查通过，性能采样及全最终cache载荷通过。原始/实际加载库SHA已核验。未把codegen变化单独认定退化因果。

停止此映射形式，未安装、不测100GB，无物理链路/全矩阵/新sanitizer结论；正常far_rows0为禁用的诊断字段，非零载荷声明。
