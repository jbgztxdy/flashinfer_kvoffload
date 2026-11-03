# 本地计算窗口提前需求hint

基于demand_phase，每paired tile的preQK前warp0读取实际future=min(hi-2,host_tiles-1)的remap，ballot选尚未claimed完整包；现有preQK CTA barrier发布hint，当前包发布后优先claim它。与旧实际缺页才hint不同，允许计算local尾段时提前选首far包。没有提前读payload/预约模拟字节，没有新增CTA barrier；额外warp0 metadata/atomic状态查询全部计时。独立normal构建，原数学/LRU/真实unique/cache/staging写入保持。

正常GPU3同卡HBM BF16MTP8R128K2048LRU4096，256位置32步×3coldpass3218.303993/3221.215986/3266.784005µs，均3235.434661，比demand_phase3187.637327更慢，未胜clean约3128或生产staged约3030。拒绝安装，不扫提前距离、不追加100GB模拟或采用率插桩。不能由总时间声称hint采用率未提高，未实测该项。

2项K65/K2048数学/LSE/padding/QPEstride80/Graph/cache全有效载荷通过，性能采样及最终cache全载荷通过。raw/header/实际加载库SHA重核见demand_lookahead_audit.json，全部pass保留。无新sanitizer/实际Graph函数名probe/硬件字节/EMU结论，生产81093不变，runner结束。
