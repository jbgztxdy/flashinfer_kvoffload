# 带宽模拟：已发请求批量预约＋本地到达等待

本轮实现并测量了独立原型，证明批量预约能消除高并发小包的CAS竞争，但**尚不能作为所有新kernel的物理100GB/s替代模型**。生产模拟器8507eab0、正常direct81093ba5、数学头34af、正式参考和原矩阵保持。用户明确授权修复/换模拟思路；本轮改动全部在隔离include/JIT中，无生产安装。

模型仍按真实字节计费，不插入固定时长来冒充共享带宽。各加载warp在实际读发出后发布请求；一个请求warp临时获得broker锁，合并当前已发布请求，用32lane max-plus prefix scan计算每个包的`end=max(previous_end,issue)+ceil(bytes/B)`，再发布`end+latency`。消费者本地等待。没有专用CTA/SM、静态均分带宽或预订未发出的未来数据。锁转交有GPU acquire/release fence。每个CTA/warp只有一个尚未得到预约结果的邮箱；清除旧request bit早于发布应答，避免重用时清掉新请求。

先前v2预约原子正确性保持，但108CTA/10页小包平均~53次CAS重试，单请求最大>9000，24CTA/64页只有~0.5。插桩本身可能扰动时间，然而带宽34GB/s复现原证据。指数退避后19/14GB/s更慢，拒绝。

| 独立真实搬运形状 | 旧v2 100GB0/20µs | 批量原型 100GB0/20µs |
|---|---:|---:|
| 108CTA，10页包 | 约34/31GB/s | 约98/92GB/s |
| 108CTA，20页包 | — | 约99.3/98.4GB/s |
| 108CTA，32页包，逐warp4页预约 | — | 约72.8/72.7GB/s |
| 同32页包，当前stage全CTA合并预约 | — | **99.4/98.6GB/s** |
| 24CTA，64页包 | 约99.4/98.6GB/s | 约98.8/96.4GB/s |

不同表格行来自各自冻结校准，不当作严格同时同序因果拆分。50GB各dense形状约49.2–49.9GB/s。第一次未加broker acquire fence的batch_first约98.3/94.8，保留原结果，不归到最终batch。逐包向上取整的10页100GB成本116ns，相对理想115.2ns有约0.7%保守偏差，未偷偷合并舍入来改善成绩。

逐请求验证覆盖24组合、154368次预约：2/24/108CTA、1/8活动warp、50/100GB、0/20µs、混合1B/1152B/10页/64页、晚到warp、主动100µs空闲。每记录符合max-plus、总收费正确、消费不早于deadline；bad/early均0。另真实payload10/64页共约5万条记录逐请求核对且整载荷相同。不是物理链路公平性或所有调用方式的穷尽证明。

全CTA stage合并只适用于全部线程参与的loader：等所有当前stage真实copy已发出，再汇总实际far行数预约一次。不包含未来stage，所有计算仍等待两stage就绪。记录issue较最早warp保守，且新增CTAbarrier有开销，不能隐去。原native direct在新pairedQK循环里还必须补上真实FAR wait；仅开放normal-only dispatch而不补wait会绕过限速，本轮没有这样做。首direct编译State作用域错误在GPU前修复，失败件保留。

完整kernel试验均GPU3/A100-PCIE/NUMA1、BF16/MTP8R128K2048/LRU4096、256位置32步；staged/uniform各3冷pass，direct/call/scalar为1pass诊断。真实载荷、sampled数学、每个最终有效cache槽均检查。旧staged模型和新staged模型对照：

| 设置 | 旧v2 µs | 新批量模型 µs | 对原逐步参考 |
|---|---:|---:|---:|
| 100GB0µs | 3288.395 | 3307.787 | 旧高21.80%，新高22.52% |
| 100GB20µs | 3293.429 | 3320.501 | 旧高21.98%，新高22.98% |

参考复用原GPU3指定device实现前32步，按各自实际unique字节逐步max后平均约2700µs，没有新测/替换device参考。旧staged差距在本轮新模型下仍存在，不能全归因原CAS竞争；也不宣称22%已严格归因fetch占SM。

**完整新kernel仍有模型接入局限。** Uniform批量内联REG128/STACK944，100GB约11ms；broker noinline后REG255/STACK32，0µs约5.172ms；同binary hooksdisabled3.786ms、enabledunlimited0为3.920ms。资源/栈改变和运行成本都在结果内，不能把独立copy校准当成完整kernel模拟开销可忽略。

Direct批量CTA聚合版REG255/STACK8（static1536）100GB0/20约8.762/12.073ms；noinline STACK112约9.327/12.426ms，均不是采用结果。原CAS加CTA聚合虽dense32校准97.7/91.0GB/s，完整direct约19.69/21.51ms，更差。不要根据这些prototype时间排名物理100GB路径。Actual direct约563MB/步，按其自己实际引用字节算传输参考约5.63ms，不用staged的248MB分母。

同binary direct hooksdisabled2.890ms、enabledunlimited0为3.262ms、enabledunlimited20为8.762ms；说明泛化hook成本及双stage延迟敏感性都需要考虑，不能简单扣掉372µs或把所有额外耗时算到带宽模型。诊断两次实际launch（冷step0和直接跳到step31，不是热trace）读取全部broker记录，收费行数与remap真实远端引用精确相等，分别2.414GB/1.292GB，逐请求max-plus也正确，排除了这两次的重复计费。访问CUDA symbol和记录拷贝全部在事件外；timing时trace关闭。

更接近warm-cache请求的校准：108CTA、每32行仅8行算far、全部32行真实copy，实际far61.941MB。批量模型100GB0/20只有91.61/70.01GB/s，50GB为49.24/47.88。20µs下名义2stage在途约1.99MB，接近100GB×20µs=2MB且没有余量；加上控制/等待成本会低于目标。当前没有把在途不足与模型成本严格分解，**稀疏请求+高延迟属于尚未解决的适用边界**。不提前预约未来字节来伪造吞吐。

本轮有效完整kernel计时960launch，6个新集成版本×2定向检查=12成功，另payload校准/预约专项。没有新大suite/sanitizer/物理PCIe或NVLink/SXM/H800/完整1024矩阵或Graph节点probe。一个自有copy runner与staged20争用锁导致后者在CUDA前失败，原件保留并仅补缺失两格；另control catalog/assert、accounting Python局部变量错误在真实诊断前修复保留，没有丢失败/负差。所有源/库SHA、原始每步时间和控制记录在[实验入口](../temp/glm_trace/emu_packet_accuracy_20261103/README.md)与[audit](../temp/glm_trace/emu_packet_accuracy_20261103/audit.json)。

后续应先选稀疏few-far-rows/20µs这一具体疑点，量化同调用频率的到达和控制开销，再决定能否进一步降低预约频率；不能以dense校准通过就宣布模拟准确，也不能按结果反调限速器或改参考以实现110%。生产保持原版本，本轮保留的是可审计的实验原型和适用范围。
