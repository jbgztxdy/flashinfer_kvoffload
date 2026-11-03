# 生产staged完整前缀tile专用地址计算

从当前生产81093ba5 header的正常BF16大batch staged PVFeature路径出发，仅在tile+1 < host_tiles时，给该完整32KV tile使用已知为负的staging行地址计算。host_tiles仍由该行实际Host前缀数决定；最后一个Host tile及其余混合/本地tile继续原resolve_host_batch_row。PVFeature的完整QK/softmax/PV、原取数CTA、跨query unique miss去重、远端源字节、staging/cache真实载荷与精确launch LRU不变。该代码改动发生在共用loader，但只有私有PVFeature调用显式STAGED_ONLY=true，其他调用仍走原分支；生产header未安装。

约82.9万个随机长度/起点/Host前缀tile的CPU整数边界检查中，约34.3万个被选中；每个被选tile严格位于Host前缀且完整，不是GPU数学或内存序证明。GPU K65/K2048两项定向输出/LSE/padding/Graph/cache载荷通过；256位置32步三冷pass性能采样与全部最终有效cache载荷通过，容差未变。实际Graph单BatchMLAPVFeatureKernel、108CTA、block[32,4,2]、dynamic147968B、active1，确认未回退原kernel。候选实际库和header SHA绑定原始计时。

GPU3/A100-PCIE、NUMA1、同卡独立HBM源，BF16/64heads/MTP8/R128/K2048/LRU4096、256位置32步×3coldpass：候选3003.936/3032.992/3077.376µs，均3038.101；随后同参数当前生产staged3025.856/3026.848/3059.232µs，均3037.312。候选慢0.789µs，约0.026%；pass方向混合，无收益信号。旧一轮生产staged3039.147µs只作上下文，不与本轮拼接归因。机器码同一统计器目标static指令6537→6661，REG254/STACK0/static2304保持，不能仅凭指令数说明动态损失。未装、不测100GB/50GB/物理PCIe/NVLink/H800或全矩阵；生产staged路由仍旧。

构造脚本首版字符串转义失败，随后构建缺include，原日志保留；修正后才有实际候选头及GPU成绩。第一次机器码分析脚本误用CleanService符号，第二次仍用了错误mangled长度，两个分析失败日志保留；第三次锁定实际Graph对应PVFeature符号，绑定实际加载库SHA。未把失败当通过。
