# 行内任务耗尽后服务关停

从clean冻结头增加每paired tile一致的service_live判断，任务耗尽后不再进入stores/publish_prepare/refill helper；原ACTIVE/pure行模板仍保留。与旧phase仅行边界切换不同，允许行内停止服务；与unifiedbody不同，未合并数学主体。ready消费检查、数学CTA barrier、真实unique搬运/必要stage/cache写回全部保持。

正常GPU3同卡HBM BF16MTP8R128K2048LRU4096，256位置32步×3coldpass候选3081.951991/3091.839992/3154.943995，均3109.578659µs；随后clean3098.207988/3111.520015/3143.872000，均3117.866668µs，仅快0.266%。前两pass快、第三pass慢，不能认定稳定净收益，更未胜原生产staged约3030；不装不测100GB。两项K65/K2048数学/LSE/padding/QPEstride80/Graph/cache有效载荷通过，性能采样/全部最终cache载荷通过。两raw实际加载库SHA重核见midrow_release_audit.json。无新全suite/sanitizer/物理结论，生产保持81093。
