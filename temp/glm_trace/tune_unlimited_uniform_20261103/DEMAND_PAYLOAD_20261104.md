# 动态完整包认领GPU载荷协议探针

独立CUDA原型，不是完整attention kernel。2/24/108CTA×3冷epoch，每CTA256线程、20页shared buffer，需求候选CAS claim+fallback atomic cursor，结束后所有CTA继续drain。每个包真实源读一次、完整stage/cache各写一次，所有writer threadfence→CTA barrier→leader release ready，消费者acquire后逐uint4比较两份全部载荷。退出决定由leader广播，避免warp分歧退出屏障。

每组215053页（尾包13页）、10753包、真实源247741056B。9launch共96777包，每包copy计数精确1，payload errors0，done精确等包数。GPU3/A100-PCIE/NUMA1同卡HBM，稳定idle预检/共享锁，source/binary SHA及前后资源见demand_payload_run.json；原始run/build日志和源码独立保存。REG20/STACK0/spill0/shared23044。未计时，不将微核吞吐/正确性当完整kernel收益。

本探针使用普通global→shared加载，未集成cp.async/MMA交错、attention、精确LRU或launch连续cache状态；验证的是动态唯一认领、drain及载荷release/acquire协议。消费者在drain阶段验证已发布包，尚未覆盖attention中局部缺页等待时重入service。下一完整集成需将这些进度与现有attention屏障生命周期结合；不能仅凭微核通过称100GB目标完成。生产未改。

**字节表述更正：** 每包producer copy计数恰1，但探针验证会重读src；247741056B为producer字节而非所有源读取。没有计时/带宽结论。交错局部消费补证见DEMAND_CONSUME_20261104.md。
