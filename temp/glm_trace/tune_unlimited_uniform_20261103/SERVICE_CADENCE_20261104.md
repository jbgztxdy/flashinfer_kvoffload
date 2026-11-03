# 服务补发频率的计算预算

复用clean_midrowcontrol完整3pass32步unique原件，不新增GPU跑分。每pair64KV对应20页producer packet补发，K2048每行32pair；108CTA/1024行。近似用每步mean_unique求包数，冷步最多250包/CTA、最后包issue需约249pair=7.78125行；热末步83包/CTA、82pair=2.5625行；32步平均3.0986行，平均占总计算行等价量32.68%。各步细值service_cadence_budget.json。

这是纯算术预算，不是时序或100GB预测：真实等待会推进额外packet，register/shared/发布/缓存载荷、不同CTA计算与bandwidth deadline均不在模型，按mean miss向上取整也只是近似，不能称实际issue时间。已有normal wait诊断每步约91次blocked step与约10765发布包，但受冷热混合和插桩影响，不能推出100GB也不缺包。

本模型不支持继续无依据扫描cadence：热步能在计算较早阶段发完自己包，冷步接近计算尾部。需要先直接观察冷/热实际在途、最晚发布与consumer stall，才能针对不同阶段改低频调度。保持用户先normal胜生产门槛，未额外开展不可靠模型新成绩。生产未改，目标未完成。
