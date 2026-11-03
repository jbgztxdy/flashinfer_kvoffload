# 真实载荷helper非内联：无采用信号

clean冻结头仅将clean_service_issue和clean_service_stores由forceinline改为noinline。与此前模拟broker noinline不同，本次正常无EMU，仅隔离实际cp.async源地址/指令和staging/cache载荷写入临时变量；所有权、顺序、数学和发布协议保持。未移除必要操作。

GPU3/A100-PCIE/NUMA1同卡HBM，正常BF16MTP8R128K2048LRU4096，256位置32步×3冷pass完整单kernel：3133.055992、3146.879993、3219.135970µs，均值3166.357319µs。历史clean3128.949µs、近期单pass3113.056µs；不是同轮夹测，无采用信号，拒绝安装、不追加100GB模拟。不声称差异已精确归因函数调用或寄存器；未额外提取资源统计。

两项K65/K2048数学/LSE/padding/QPEstride/Graph/cache载荷检查通过；性能脚本数学采样、全部最终有效cache载荷通过。raw每步时间及实际加载库SHA在service_calls_native_unlimited_f24.json，源码SHA在service_calls_source.json。far_rows0只是无计数。无新全suite/sanitizer/物理链路结论。生产保持81093。
