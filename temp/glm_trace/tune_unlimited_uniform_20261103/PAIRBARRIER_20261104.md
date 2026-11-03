# Pure计算head-pair同步原型

clean pure行体jointMDO最大值交换仅同warp_idx、z0/1两warp读取，替为bar.sync id=threadIdx.y+1,count64。pure PV的P发布后读者也为同16-head组两warp，第二PV barrier改同64thread named barrier；第一PV barrier保护跨所有head的KPE读→P覆写，保持全CTA。ACTIVE服务body、全部KV装载/缓冲复用/真实写回发布仍CTA同步。不拆kernel/增加warp/删必要写回；区别于旧waitvote或删service_step barrier。

正常GPU3同卡HBM BF16MTP8R128K2048LRU4096、256位置32步×3coldpass3082.720011/3080.031998/3153.824002µs，均3105.525337。近期clean3117.866668约快0.40%，跨轮非夹测，第三pass波动仍在，无稳定收益或优于生产staged证据，未安装不测100GB。2项K65/K2048数学/LSE/padding/QPEstride80/Graph/cache全有效载荷通过；性能采样/最终cache载荷通过。新同步范围专项K65 synccheck1case errors0，非racecheck或完整同步证明。实际header/library SHA核对见pairbarrier_audit.json。

尚未提取SASS确认named barrier编译资源/动态次数，未测原R128实际Graph symbol或100GB。生产81093保持。

后续补证：实际库cuobjdump确认3处BAR.SYNC.DEFER_BLOCKING register-id,0x40，四组id由threadIdx.y+1动态选择；REG255/STACK16/static1152与clean一致，见pairbarrier_codegen.json。K65目标kernel一次racecheck0errors/0warnings，log为pairbarrier_racecheck_target.log。首未过滤racecheck范围过大，约数分钟后只终止本任务pytest3781887，退出15，原log保留，不算通过；收窄后目标1launch检查完成。不是完整K2048/R128 racecheck或全局memory race证明，无新性能成绩。
