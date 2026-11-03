# A100同一warp交错访存与计算：源码调研与新可行性条件

用户提出参考其他A100工程，让全部warp在访存和计算之间交错，而非固定部分SM搬运。**这个方向符合SM80成熟流水线的写法，有理由继续；“所有warp都参与”本身不能保证逼近逐步理想参考。** 关键是异步读取与当前MMA的依赖距离、在途量、资源预算，以及保留唯一miss搬运而不增加远端字节。本轮完成官方源码调研、结构核算和独立探针编译；没有新GPU性能或生产加速。

## 官方实现具体如何做

1. [CUTLASS v3.9.2 `mma_multistage.h`](https://github.com/NVIDIA/cutlass/blob/v3.9.2/include/cutlass/gemm/threadblock/mma_multistage.h)：`mac_loop_iter`在各次warp MMA之间调用`copy_tiles_and_advance`，将下一个global→shared stage的复制指令分组发出；stage边界才commit/wait，片段另用双缓冲。不是发完复制立刻等待，也不是一次算完大块后才继续搬运。相关函数约L269、449、458–555。
2. [FlashAttention-2 v2.7.4.post1 forward](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/src/flash_fwd_kernel.h)：约L337–380先发V读取再做QK，之后发下一K读取再做softmax/PV；同一个CTA保有完整输出状态。其[`utils.h`的gemm/gemm_rs](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/src/utils.h)约L123–167提前装载下一片段再执行当前MMA。
3. [NVIDIA Ampere调优指南](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)明确SM80的global→shared异步copy可以与计算重叠；同时寄存器、shared容量与驻留仍受约束。它不是Hopper的TMA/WGMMA机制，不能直接套Hopper专用持久化生产者/消费者写法。

这些工程的共同点是：读取未来块、计算已经就绪的当前块，尽量只在即将消费时等待；减少跨warp/CTA交接。不是要求每条load与MMA在同一时钟发射，也不保证两种资源同时达到各自独占峰值。[FlashAttention-2论文](https://arxiv.org/abs/2307.08691)报告A100特定稠密形状约50%–73%峰值FLOPs/s，不能当作我们的远端KV重叠比例或性能承诺。

本仓库当前attention侧其实已经有同warp的cp.async/MMA/PV流水；区别是远端unique miss仍由专用fetch CTA送到staging。要解决现有SM资源问题，新的改动必须覆盖这一层，而非只重复attention本地load排序。

## 稀疏远端场景还多了什么约束

dense GEMM/attention的future tile通常就是本CTA后续要用的连续操作数。当前kernel则先将跨MTP/query重复miss合并，搬运一次后让多个CTA读本地副本，并维护精确LRU/cache载荷。简单改成“各CTA直接把自己需要的far数据读回来”会破坏字节预算。

同一GPU3/BF16MTP8R128/LRU4096前32步新基线核算：unique miss平均215261页，247.981MB；未缓存引用平均487942次，562.109MB。如果按每次未缓存引用各自读远端，估计字节变成**2.267倍**。这只是同输入字节核算，不是已测的另一direct路径；足以说明不能省掉launch级去重并继续沿用unique参考。

因此可尝试的结构是：所有CTA兼做唯一miss任务的搬运与自己的attention计算，搬运任务有唯一所有者，仍向必要消费者发布KV ready；保留全载荷cache写入。**不再跨SM传递QK分数**，使上一版每pair所有权/进度维护约0.24ms的成本不必继续存在。KV的必要ready依赖仍在，绝非零同步。

## 与已经拒绝的cofetch有什么新差别

旧cofetch减掉一个attention stage以容纳2×16行servicebuffer，单stage控制本身约慢12%；并在每八个KV tile或等待时调用完整service流程。旧hybrid保留双stage，但只是专用fetch之外的两个4行辅助buffer，完整版本仍慢。旧QK分数辅助则额外读取Q/K、传递分数并付出细粒度所有权成本。它们都保留为负证据，不能仅改一个cadence数字重做。

当前有一个与旧实现时间点不同的新条件：生产PVFeature的QPE已经从global直接装入packed寄存器，paired/odd数值路径不再读取shared `q_smem_pe`；原`load_q`仍复制QPE、原SharedStorage仍留8192B。**仅在这一typed路径**可考虑回收这块冗余空间；canonical/native其他路径仍需它，不能全局删除。

C++静态sizeof已实际编译并在CPU运行确认：

| 存储 | 字节 |
|---|---:|
| 生产原attention dynamic | 147968 |
| 去掉目标路径冗余shared QPE，保留双32KVstage及原O布局 | 139776 |
| 再加两个10行远端buffer | **162816** |
| 加沿用的2304B static估计 | **165120** |
| 设备已知每block上限 | 166912 |

这不是最终kernel的实际资源/占用证明，static可能变化，寄存器也可能溢出。它说明不必像旧cofetch那样先牺牲一个attention stage，就能在源码存储预算中放入独立取数空间。10行来自这次shared/在途预算推导，不是已经扫得的性能参数。

若108CTA各有两块10行，名义在途2160行、2488320B；100GB/s×20µs为2000000B，名义余量约24.4%。旧2×4行若单独承担全部链路，则仅约0.995MB；这也不能简单移植成统一fetch主通道。BDP只给必要库存尺度，不保证实际带宽：issue/ready间隔、buffer复用、cache写入和虚拟link CAS包频率都会降低可用在途量。

建议的新假设为：保留全部8个计算warp和双stage，外层静态分配较粗的unique miss组，内层按CUTLASS方式将复制向量分散到已有QK/PV的MMA片段之间；仅在数据即将消费或buffer复用时等待。任务分配尽量离开热循环，避免跨SM score队列。等待别的CTA的KV时，必须继续推进自己拥有的miss任务以避免循环等待。cp.async group与stage生命周期需统一设计；不能只把现有阻塞service函数插进MMA循环。尚未实现或测量该完整kernel，不承诺110%。

## 本轮新探针与限制

独立`research_sm80_pipeline_20261103/probe.cu`已编译：相同八warp全部做QK和访存，双64行远端buffer＋12KiB独立local QK工作区；对照purefetch、先整批发出async后算QK、把18轮远端copy分散到36个MMA特征步之间。24/108CTA，100GB0/20µs。它使用原v2 FAR header和完整真实远端payload，不提前使用未到达的far数据；本地QK来自独立数据，不是完整trace工作替换。源码意图是指令交错，未查最终SASS的实际调度，不能据此声称真实硬件重叠已发生。

三个微核编译REG117/168/76，stack/local均0；工作量相同不代表代码生成同，不能把这些寄存器差异当收益。这个探针没有O/softmax/LRU/真实交接，因此也不能验证完整attention的资源与效果。

GPU3在编译后被其他用户VLLM占用；持共享lock的稳定空闲预检120秒到期，**未启动CUDA子进程、无result.json、无GPU数值/载荷/带宽结果**。waiter已退出，不再后台等待或重启旧实验。GPU0/1/2未使用。

正式host/math/FAR仍cab508dc/34af/8507，逐步max参考和旧原始数据不变。源码调研来自web工具对官方固定版本的读取；直接urllib下载遇SSL错误，目录没有外部源码快照，不伪造其哈希。来源与结构核算见[sources.json](../temp/glm_trace/research_sm80_pipeline_20261103/sources.json)、[可行性审计](../temp/glm_trace/research_sm80_pipeline_20261103/feasibility.json)。没有本轮新生产候选、GPU检查、性能结论或H800任务。
