# 20页 unique 包在完整 QK 前发下一包

2026-11-04。目标是把 staged 的跨 query unique 字节与 direct 已验证的 paired QK/joint softmax/PV 数学调度结合，并让全部108个CTA/八warp在计算时交错推进 unique 搬运；仍为单 cooperative kernel、真实 staging/cache 双写、精确 launch LRU。实验 GPU3/A100-PCIE、NUMA1、同卡独立 HBM 源，BF16/64 heads、MTP8/R128/K2048、LRU4096/请求，256位置32步。**这不是物理 NVLink，也不是已达到100GB理想参考。**

机制：从冻结 clean 全CTA头出发，保留一份20页远端 shared payload buffer和两个32KV计算 stage，只将每包 pages/victims 元数据改为两个160B轮换槽。预载当前与下一包；当前包完成真实远端copy、stage/cache写回后，新增一道CTA屏障，立即从另一元数据槽 issue 下一包的实际 cp.async，再执行当前 paired QK。原MDO屏障后仍按release发布当前ready，并为下下包预载元数据。wait_tile/service_step/drain沿同一ticket递推；没有提前发布尚未写完的载荷，也没有增加实际远端读取。

初版正常构建两项K65/K2048定向输出/LSE/padding/Graph/cache载荷通过，完整32步×3冷pass 3082.528/3089.440/3153.280µs，均3108.416µs；最近clean单pass3096.832µs不是本轮严格三pass配对，不能称不限速净胜。正常Graph实际一个BatchMLACleanServiceKernel、108CTA、block[32,4,2]、dynamic163200B、active1。生产staged约3.03ms仍更快，因此**不安装**。

模拟器审计发现初次batch_call集成可在其他warp发出该包全部cp.async前按完整20页预约；初次0/20µs 4303.744/4497.536µs及对照4332.672/4789.888µs只作为探索信号，**不能按严格已发请求口径当采用证据**。无hook误构建均未上GPU；正确宏启用且noinline broker的私有JIT另行保存，未覆盖任何旧raw。

正式隔离模型在两版clean/候选的每个service issue处先commit本线程copy，再CTA barrier，之后才让broker预约实际已发出的整包；同一batch_call、同一真实payload和max-plus计费。版本clean_next100_issuebar / next_packet_qk_next100_issuebar，FAR header SHA256 6ad898483f510695357d83282c416a4fc97c4d14772147f63a26fa09f453f4df。batch_call的broker仅函数noinline，语义不变；完整kernel软件hook成本仍在事件内，不扣除。先前同模型dense20包100GB校准约99.22/98.18GB/s（0/20µs），**未**据此宣称完整稀疏kernel或物理链路达到该带宽。

| 同模型1冷pass | clean原布局 | 双元数据槽、原PV时机控制 | 下一包QK前issue | 相比同布局迟发控制 |
| --- | ---: | ---: | ---: | ---: |
| 100GB/s，0µs | 4338.752 | 4337.888 | **4305.984** | 快31.904µs / 0.735% |
| 100GB/s，20µs | 4800.288 | 4789.408 | **4484.960** | 快304.448µs / 6.357% |

20µs候选相对clean快315.328µs/6.569%，32/32步逐步时间更短；相对同布局迟发控制也是32/32步更短。0µs相对clean快0.755%，28/32步更短。各版本数学采样和最终524288个有效cache槽真实KV载荷通过；候选与迟发同布局控制各2项带模型定向检查通过。控制只把refill从QK前恢复到原PV时机，保留两个元数据槽、相同20页包、同一issue后屏障与模型。双方warm trace unique miss约215k页/步，逐步细小差异如raw保留；不假定字节完全相同，也不借direct约563MB/步作staged分母。

候选broker追踪两次独立诊断launch（冷tick0与直接跳tick31，**不是**热32步均值）：26957请求恰收费539135页/621083520B；16505请求恰收费330091页/380264832B。两次均与kernel实际unique计数相等、包数恰为ceil(unique/20)、每请求end=max(previous_end,recorded_issue)+cost递推正确；收费ns含逐包向上取整，不篡改参考。追踪/符号拷贝都在正式CUDA event计时外。严格模型候选实际加载库 SHA256 e64658797ae94911f964bc6697d0289089863775347aa2c0cb3e340a933bd267；全部四格raw另记录实际库路径、SHA、每步时间/unique和硬件/NUMA。EMU单launch节点核对也是单目标kernel、108CTA、active1、163200B；该节点probe的运行时hook关闭，只证明编译版派发身份。

源码 SHA256：clean_issuebar 4a1fede269090ce24c8fe11caf8402895f01af495356563a8ca7ff35d9d5e24e；next_packet_qk_issuebar 1f820daa6f3635ee50b2bfdc3779571fd1482daf6b2ec04dfeec1d6140000096；next_packet_late_issuebar 2f99e020407bc5e123bf50a70a20b0d61397af1377957261ee47a9cbe34eb8ad。正常候选 next_packet_qk 232d410a53cf93adefd3ca83086c415a2d3387e23a7402390bd490e02e7b7d4b。CPU审计 audit_qk_next100.py 逐项重核完整raw、header及实际库SHA、全部32步/载荷/字节计数，汇总 qk_next100_audit.json。实际加载库、构建、失败、逐步raw、runner状态及节点probe保存在本目录；初次probe硬编码旧163072B而实际163200B导致断言失败，原日志留存，修正后通过。

额外尝试把当前ready在新增写回后屏障立即release：正常1pass3079.968µs；同batch_call 100GB20µs4501.856，对新近原发布时机对照4478.784慢0.515%，0µs4308.320也无采用信号。该早ready版不组合、不安装。

边界：严格模型端到端4.485ms仍远高于此前指定device/unique字节逐步参考约2.70ms，也不能直接与另一模拟实现下生产staged约3.3ms做相同软件成本比较。没有物理PCIe/NVLink、H800、FP16、完整1024位置或sanitizer新结论。生产81093ba5、canonical34afad69、FAR8507eab0及默认staged路由均未改。**保留下一包提前issue为实验候选，不宣称目标完成。** 下一阶段应在保持20页unique包/全warp计算与正确issue字节的前提下，降低服务软件开销并用物理链路/更可信的限速实现验证，不能以调参考或漏写cache制造110%。
