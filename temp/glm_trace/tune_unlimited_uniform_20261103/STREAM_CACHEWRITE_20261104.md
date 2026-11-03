# Miss-cache载荷streaming store提示

从clean仅将实际miss-cache的uint4写入改为st.global.cs.v4.u32，stage写入、读源提示、写回时点与线程、数学/LRU、CTA发布协议不变。与旧cache写回迁移不同，不改变调度或省写；目的检验缓存写回对当前stage/命中页的缓存争用，不预设有效。

正常GPU3同卡HBM BF16MTP8R128K2048LRU4096、256位置32步×3coldpass3100.159973/3100.607991/3148.800015µs，均3116.522660。与近期clean3117.866668接近，跨轮非夹测，无净收益证据，拒绝不测100GB。2项K65/K2048数学/LSE/padding/QPEstride80/Graph/cache全有效载荷通过；性能采样和最终全cache载荷通过。source/实际加载库SHA核验见audit，raw全部保留。未另做SASS目标store核查、L2因果counter/全suite/sanitizer/EMU或物理结论，不能从耗时持平推断L2不存在争用。生产81093不变，runner结束。

后续实际加载库机器码补证：clean STG.E.128静态140，候选普通86+STG.E.EF.128共54；store提示真实生效。REG同255/static1152，STACK16→8，静态指令9336→9431。代码生成不等价，所以时间持平不能单独归因L2提示。库SHA核对与资源/指令统计cachewrite_codegen.json。原“未另做SASS”只描述首测时状态，本段补上该项；仍无L2counter因果/新GPU时间。
