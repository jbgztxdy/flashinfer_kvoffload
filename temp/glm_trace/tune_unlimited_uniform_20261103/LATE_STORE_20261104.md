# QK后真实写回候选

clean ACTIVE paired循环中，真实stage/cache stores从QK前移到QK后、softmax前。仍在jointMDO原barrier之后release ready，refill不提前；remote和计算stage互不覆盖。数学顺序、字节、LRU与必要写回不变。区别于PV分散copy与warp反相，本次移动真实global写回。

正常GPU3同卡HBM BF16MTP8R128K2048LRU4096，256位置32步×3冷pass：3111.071996、3130.655982、3188.032001µs，均3143.253326µs。近期clean同参数3127.680014µs；非同轮夹测，无采用信号，不装、不追加100GB模拟。两项定向K65/K2048数学/LSE/padding/QPEstride80/Graph/cache全有效载荷通过，性能步骤采样和全部最终cache载荷通过。全部raw/source/实际加载库SHA见late_store_*json。未做全suite/sanitizer/物理链路，生产81093保持。
