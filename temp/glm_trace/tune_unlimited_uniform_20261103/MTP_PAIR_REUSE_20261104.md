# 跨query计算复用前置分析

全assistant capture/layer/window前256位置，非重叠相邻MTP8 query对10752组。平均common71.9159%，原位相同0.3488%，同32tile共同5.7038%，union2623.161页。输入SHA、逐对统计mtp_pair_reuse.json，脚本analyze_mtp_pair_reuse.py；原selectedKV、LRU/remap前，不代表actual far变化。

候选与旧半head双驻留不同：一CTA两query×32head共享KV，仍一CTA/SM。但两query各64heads需要两CTA，union加载平均2623.161×2=5246.323页，对比原两64head单query CTA4096页增加28.08%；若全union masked计算也增加相同比例。不能用query交集直接宣称省计算/加载。避免无效union计算需要新的条件计算/score映射，不是免费优化。两query×64head则仅Q-nope128KiB+双CKV64KiB已192KiB，未计PE/P/farbuffer即超A100每block166912B，不能直接保留现布局。

因此不实现这个直接形式，不否定其他新布局。staged launch本已去重，跨query共同页不等于新增远端字节收益。无新GPU计时/生产修改，目标仍未完成。
