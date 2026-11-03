# 整pool绑定的负载尾部

从已保存真实最后一步pool counts做CPU算术审计，不重跑GPU。128pool/108CTA，pool%grid绑定：最多16计算行/CTA，均值9.48148，max/mean1.6875；最多5331unique页/CTA，均1644.84259，max/mean3.24104。原row轮转最多10行，max/mean1.05469；原20页包轮转约82.25包平均、最多83包。

按combined normalized work贪心分配完整pool可把最大unique降到3343，但128个不可拆分8行pool分到108CTA仍必有16行CTA。因此停止整pool不可拆分路线；不再只换映射数字或优化列表构造。未来需拆分pool内计算任务并允许空闲CTA接计算，同时unique包保持一次搬运；不能取消去重或偷换参考。

这不是严格端到端下界：不同CTA计算速率、缓存、服务重叠与等待尚在，不能将1.6875直接乘clean时间证明5.43ms因果。它是明确的资源负载风险和后续设计约束。raw依据clean_pooldependency_native_unlimited_f24.json，数值pool_load_tail_audit.json。生产不变。
