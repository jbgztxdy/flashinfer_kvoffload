# A100：CTA 内固定取数/计算 warp 分工（2026-11-01）

空间 warp 分工未优于保留版本，拒绝。生产仍为 0f82e9de / b6eea095…e72a，100GB/s 大 batch 约32%–34%的主差距未解决。

此前cofetch/hybrid让整个CTA在取数与计算间交替。本候选在同一256线程CTA内固定划分四个取数warp和四个计算warp：block[32,2,4]，物理z0/1取数，z2/3计算时映射为虚拟z0/1，分别使用named barrier2/1、各128线程。取数的第一个线程仍为物理(0,0,0)，保持模拟器的issue-time/waiter、真实字节数和等待行为，没有改模拟器。取数的FAR等待只阻塞所属warp，计算按原ready flag acquire等待所需的staged tile。

计算使用原sharded QK、32head/32KV/2stage，每query两个独立head item直接写输出，无head合并，PE→CKV和在线softmax/P/PV均为原方程/顺序。只复用已验证32head Q-load索引修正。计算shared110848B，独立2×16行取数payload36864B紧接其后，总147712B放在原dynamic147968B中，互不覆盖。仍是单cooperative kernel。全256线程执行原prepare/resolve/compaction及每pool精确LRU，grid barrier后才分角色，结束后全CTA和grid同步、沿用原last-CTA复位。metadata和真实cache payload均保持。

所有108个CTA的取数子组静态轮转16行chunk，两缓冲总在途108×16×2=3456行，比原24×64×2=3072行多12.5%。API fetch_ctas同为24，但实际固定空间调度是算法变化，不能称为相同搬运预算，也不是只扫fetch参数。派发限原大cached whole-row、至少4×SM行、完整staging的SM80路径，其他分支继续原kernel。

首版warp_roles通过105+4正常/长序列cache-payload/Graph测试及单kernel实查，但改写原helper为默认可折叠的trait检测仍让22/34EMU和44/68normal旧实例的指令文本变化，包含交换LOP3操作数与reuse hint等；puredevice的BatchMLAPagedAttentionKernel仍逐项相同。不能据此推断等价性能，严格gate拒绝，未运行首版性能。独立修正版warp_roles_isolated复制专用helper、保持所有原helper字面不变，全部34EMU/68normal原实例指令/资源逐一相同才进入性能；未放宽审计。两版源码/include/JIT各自冻结，合计218项测试通过。

隔离版实际正常BF16 Graph只有一个新BatchMLAWarpRoleKernel，108CTA×256线程、block[32,2,4]、dynamic147968B/static1408B、active1；REG246、STACK0/LOCAL0。实际probe最大绝对误差0.004128。SASS可见两种128线程named barrier；这些是实现证据，不是实际并发度或收益证明。

GPU3为A100-PCIE/NUMA1，far是同卡独立HBM模拟，不能称为NVLink。MTP8/R128、LRU4096/请求、256位置×3冷启动pass，base→isolated，同轮全步全pass算术均值。纯device同硬件参考2556.960µs。逐步max按每步compute及unique miss计算，没有修改参考数据或公式。

| 链路 | 同轮 base µs | warp 分工 µs | 耗时降低 | 候选 Memcpy µs | 候选逐步 max µs |
|---|---:|---:|---:|---:|---:|
| g100_l0 | 3589.781 | 3715.349 | -3.498% | 6088.096 | 2699.859 |
| g100_l20000 | 3596.171 | 3780.747 | -5.133% | 6096.107 | 2699.833 |
| g50_l20000 | 5285.131 | 5290.805 | -0.107% | 8492.651 | 4959.840 |

三个点全部变慢，不采用，也不扩大full1024×5pass/正常物理链路/跨形状验证。停止此原型，不继续只调warp数量、chunk/fetch/cadence/unroll来追单格。四个计算warp的资源占用与额外head item等代价可能抵消分工收益，但没有运行时因果测量，不能把负差单因归于其中一项。

[性能审计](../temp/glm_trace/tune_warp_roles_20261101/performance_audit.json)保留6格/3配对、1152次计时launch/72有效segment、每步/每pass/命中率/unique miss/两参考与完整源码快照。初次生成在任何编译前被静态检查发现插入索引错误，原生成器/错误header保留；首版等价失败及全部编译/SASS/测试也保留。性能首次启动因缺少冻结plans.cpp在源码快照preflight退出，没有计时launch；补齐相同文件后只重试缺失格，追加日志/state。其他资源预检/末检失败见原state/raw/failures，未作为有效性能。没有终止其他用户进程；一次准备取消首版CPU等待器的安全断言发现测试已启动，未发信号，全部测试自然结束。

所有runner结束，生产/root math/模拟器/原5304格未变，goal active。继续新机制前先读本轮和F/S拒绝证据，不能把正确性/stack0/空间分工本身当作收益。
