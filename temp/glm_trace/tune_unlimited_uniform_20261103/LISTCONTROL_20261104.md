# 等负载间接包表控制

以poolrows为基础恢复clean的packet=CTA+ticket*grid顺序，保留间接列表和produced/issues控制字段，计算仍逐行轮转。列表在设备初始化构造，完整计时包含构造，没有CPU替代必要工作。unique读取、cache/staging写回、数学/LRU、单kernel保持。

正常GPU3同卡HBM BF16MTP8R128K2048LRU4096、256位置32步×3coldpass：3119.359992、3152.608015、3205.471985µs，均3159.146664µs。接近近期clean3127.680µs，明显快于poolstripe3478.048µs；跨轮非同分配夹测，不能精确相减分摊，也不把约1%差视稳定收益。结果削弱间接包表是pool版本主要退化来源的解释，优先处理pool分配带来的负载与供需顺序。未安装，不测100GB。

两项K65/K2048数学/LSE/padding/QPEstride80/Graph/cache有效载荷通过，性能采样与全部最终cache载荷通过。raw实际库/header SHA重新核验见listcontrol_audit.json，原始每步时间保留。无新sanitizer/完整矩阵/物理/EMU结论，生产81093保持。
