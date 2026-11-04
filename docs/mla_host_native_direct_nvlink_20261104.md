# 当前真实trace，原生FA2直接读取NVLink远端KV

用户要求不预先搬运KV，直接用未改计算循环的FlashInfer原生kernel读取GPU1。GPU0/1稳定idle10秒、共享GPU01锁、NUMA0，完整assistant64k1024位置128步，R128/MTP8/BF16/64heads/K2048，3passes，整段Graph连续执行。没有LRU、缓存admission/写回或staging copy，没有模拟限速。Q/KV为随机BF16载荷、访问索引为真实trace，绑定和seed与当前device参考相同。

原生direct平均 **9459.469µs**，3pass9460.624/9458.480/9459.304µs。event只包原生attention，包含其真实远端读取、数学计算和输出写入；初始化、规划、索引上传与采样验证在event外。每步99.9525%选中引用为远端，平均2096156.164远端引用、按1152B/引用核算2414.772MB/步。按引用字节/整个kernel时间约255.276GB/s，仅模型有效载荷吞吐，不是硬件NVLink counter带宽。

Graph实查128个BatchMLAPagedAttentionKernel，各108CTA、block32/4/2、dynamic147968、active1；其余Graph节点为索引更新、检查输出采样，不在attention计时内。加载原生.so SHA20caf1e9dcebb439dd3b2f7985e00d921a8c35d4f5f7a39aaba889d251c08755，与当前全本地device参考完全一致。源KV VMM allocation properties为GPU1 HBM，near为GPU0；未使用我们的DirectPV或CleanService。

每个pass step0/16/127各3行完整输出reference检查通过，atol/rtol0.01。开始和结束无foreign进程；结束利用率采样不等于空闲，进程退出后显存已释放。不新增suite/sanitizer，生产未改。

比较边界：全本地连续参考2459.195µs（event外预先搬运），本次无LRU原生direct9459.469µs（真实远端引用约2415MB/步），我们的LRU4096融合direct2915.435µs（约510MB/步）。三者工作范围不同，不能把原生direct与融合direct差值全称作计算优化收益；LRU节省流量是关键差别。本次没有对原生kernel外接CPU/offline LRU。

原始目录temp/glm_trace/tune_unlimited_uniform_20261103：native_direct_continuous_20261104.json（逐步时间、引用数、库SHA、Graph、资源、bindings）、native_direct_verification.json、native_direct_retry_clocks.csv、bench_nvlink_native_direct.py、run_nvlink_native_direct*.py、对应state/log。实际执行harness已从SHA冻结副本恢复，当前源码SHA精确等于raw记录；运行时追加Graph断言未进入该进程，因此恢复实际执行版本并在native_direct_verification.json单独完成相同后验检查，不冒称执行了未加载的源码。

首runner在用户释放GPU后自动开跑，但校验输出使用Python list高级索引触发Graph捕获期间CPU→GPU临时索引错误；没有完整时间成绩。保留nvlink_native_direct_retry.log、native_direct_state.json和首时钟记录；改为提前创建GPU checkids/index_select后retry2成功，不修改kernel或容差。
