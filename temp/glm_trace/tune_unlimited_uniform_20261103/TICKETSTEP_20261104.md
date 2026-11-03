# 不可变ticket等待step

依据step_barrier_audit反例，等待step将done保存到线程局部ticket，经首wait_group/CTA barrier后，stores/publish_prepare/refill显式传ticket，避免shared current跨调用覆写。仅等待step取消末尾barrier，由下一wait_tile投票barrier发布done；drain显式保留barrier保证一致退出。计算融合服务路径保留原default current与屏障。真实数学、unique读、staging/cache写回、release/acquire不变。

两项K65/K2048定向GPU检查通过。正常GPU3同卡HBM BF16MTP8R128K2048LRU4096，256位置32步×3冷pass候选3092.991993/3115.327999/3168.192014µs，均3125.504002；随后同参数clean3098.688006/3113.184005/3171.168029，均3127.680014。仅快0.0696%，无明确收益，不安装、不追加100GB模拟。两者第三pass共同变慢，全部保留；不是同分配夹测或统计显著性结论。

raw/source/实际库SHA与两runner状态独立保存。无新全suite/sanitizer/100GB/物理结论，生产保持81093。
