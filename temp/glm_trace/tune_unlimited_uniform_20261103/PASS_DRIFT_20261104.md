# Pass漂移专项

只重解既有raw：7/7最近3pass运行末pass最慢，首末差中位1.6909%，完整文件SHA/pass均值见pass_drift_audit.json。不能据此直接修正时间或删末pass。

一次clean新诊断，事件外每pass前后nvidia-smi时钟/功耗/温度：3107.743986、3099.904001、3091.232017µs，末pass不再最慢。前快照SM1410/1410/1365MHz，后1230/1245/1230MHz；温55–59°C，mem1215MHz。快照并非事件内连续采样，查询会改变pass间空闲间隔，不能证明温度/时钟是此前漂移唯一原因。禁止将1230MHz后快照当整个kernel实际频率或做频率归一化成绩。

本轮只提示小于约1%的跨轮结果不够稳，不否定已明显退化候选；ticketstep0.07%无采用信号的判断保持。不改锁频/功耗/外部任务，不改生产。raw及全部源码/库SHA在clean_telemetry_native_unlimited_f24.json，检查原样包括数学采样和全有效cache载荷，runner结束。
