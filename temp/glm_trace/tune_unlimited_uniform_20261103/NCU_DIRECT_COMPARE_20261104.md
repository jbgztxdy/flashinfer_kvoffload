# Direct与全CTA stall对照

同GPU3正常构建、同K2048/512query/64request正确性生成器、首次目标launch，Nsight Compute采集；分别确认BatchMLADirectPVKernel与BatchMLACleanServiceKernel。direct使用512cache槽/request、原direct准备和逐引用字节；clean使用staged准备与unique字节，不能视为纯同步消融。

| warp-active比例 |clean %|direct %|
|---|---:|---:|
|barrier|17.857892|7.271044|
|long scoreboard|12.619576|13.031340|
|short scoreboard|20.499511|23.341265|
|fixed wait|22.209754|24.826677|

Direct eligible warp/scheduler0.274513，clean此前0.24。比例分母为各自active周期，不能将差10.59pp当端到端可回收时间；不同准备、载荷和代码生成均可能影响分布。该证据支持优先检查service相关同步而非只换包序或泛化LD优化，不证明删barrier安全。

两次目标用例均通过；direct raw ncu-rep/csv/log及summary独立保存，未改生产或参考，未做100GB外推。
