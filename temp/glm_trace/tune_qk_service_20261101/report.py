"""Report both service implementations, retaining all controls and failures."""
import json,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text())
a=read(p/'performance_audit.json');initial=read(p/'initial_performance_audit.json');m=read(p/'manifest.json');sm=read(p/'shared_manifest.json');c=read(p/'checkpoint_audit.json');decision=read(p/'decision.json')
idx={(r['cohort'],r['setting'],r['variant']):r for r in a['rows']}
lines=['# A100：QK 角色参与远端搬运','',
 '更新于 '+time.strftime('%Y-%m-%d %H:%M %Z')+'。**'+decision['summary']+'**','',
 '当前生产仍为 JQ `7193255e…d469`，主目标继续 active，100 GB/s 大 batch 距逐步 max 参考约25%–27%。直接性能基线是尚未保留的 CPEa436；QP1252为同参数调度控制，不能把本实验差值标作生产收益。','',
 '## 两个实现','',
 '沿用QP的单108CTA/block[32,4,3]：四QK/softmax warp和八PV warp。QK角色持有跨tile/item的服务上下文，在PV空档和Host flag等待中推进搬运；等待循环的所有128个producer lane执行匹配的progress/barrier步骤。消费者只等P/scale发布，不参与service。每个CTA结束自己的attention行后，producer必须drain剩余远端任务，再进入原退出协议。','',
 'LRU先由全部384线程按原helper选好所有pool victim，再用一次grid barrier，之后立即写cache，无待定victim需要defer。所有108CTA静态按row=blockIdx.y+k×gridDim.y分配attention；输入f24仍保持，但专用fetchCTA分区和claim/steal已被替代。它们都是本实验的策略变化，不把全部差值归因于QK空档或寄存器。缓存容量、launch epoch的LRU保护/淘汰规则和真实cache载荷保持。','',
 '首版qks `'+m['candidate_sha256']+'` 使用两个8行shared stage和一个8行寄存器意图stage，每producer lane最多5个uint4；计算字段140032 B加shared18432 B，dynamic158464/static1408 B。每个joint QK/PV对在PV-hi时推进一个8行subjob；Host等待和结束drain按需继续。实测REG168/STACK152、LOCAL0，有大量LDL/STL，不能称寄存器载荷全部留在寄存器。','',
 '修正版qks_shared `'+sm['candidate_sha256']+'` 将服务载荷全部放入三个7行shared stage，dynamic164224/static1408 B；实测REG168，EMU STACK32、正常BF16/FP16 STACK72、LOCAL0，仍有栈。7是按实际driver的166912 B/block上限、已测1408 B static及140032 B计算区推导出的最大等大三stage行数；未扫描行数/unroll/fetch。每个实际PV tile推进一个subjob（paired两次、odd一次），也是与首版不同的调度变化。不能将差值单独归给stack下降。','',
 'driver只读查询确认A100共享内存167936 B/SM、166912 B/block；修正版动态与静态合计165632 B。host派发使用cudaFuncGetAttributes的实际static和设备max-optin核验，再设置dynamic，避免旧“dynamic+4096”保守检查退回旧路径。实际Graph另证明新路径确实运行。','',
 '## 字节、数学与生命周期','',
 '首版每64行chunk分八个8行subjob；修正版分九个7行和一个1行，最后subjob严格截到chunk/count边界，全部子块写入后才发布原64行ready flag。两版均在真实读之后调用未修改的FAR TILE，FAR WAIT之后才将载荷写staging/cache；没有提前落HBM来隐藏模拟等待。新增static/shared布局不改模拟器或两个参考。','',
 '首版名义在途108×24=2592行，修正版上限108×21=2268行、遇1行尾块更低；原f24专用CTA为24×128=3072行，不声称预算相同。7行packing以独立slot集合oracle枚举45个grid/count组合，所有slot恰覆盖一次、发布chunk齐全、累计FAR字节等于unique×1152。这是结构证明，不替代实际CUDA或FAR运行时证明。','',
 '八个attention数值helper在规范化命名后与QP控制逐字相同，包含QK、softmax/P、PV、normalize输出相关路径；服务调用和状态生命周期另变。继承QP的完整576特征、PE→CKV MMA、量化P/native rowsum、joint max/O/d缩放。QP相对CPE已有FP32归约分组差异，不能从helper身份外推与生产逐位相同。','',
 '## 正确性与代码身份','',
 '两版各105+6正常BF16/FP16测试通过，共222项独立测试，涵盖K2048/63tile奇尾、128pool、512/4096容量、空/单元素/padding、LSE、全cache载荷/淘汰/Graph和K65/QPE head stride80。实际均为单108CTA/384线程新kernel、active1；每版Graph/synccheck/racecheck/memcheck均通过。sanitizer只检查首匹配BF16/K97新launch，memcheck关闭API错误报告而保留设备检查，不声称全部trace穷尽验证。','',
 '每版QP控制原40EMU/80normal BF16/FP16实例的指令、资源、完整机器字与控制字全同。复用的QP1252原件又证明CPE38/76全同；复用的111测试不计入本轮222。库与源码SHA、SASS/resource、实际dispatch及原证明链接都由checkpoint核验。','',
 '## 首版固定三方筛选','',
 'GPU3为A100-PCIe/NUMA1，EMU far在同卡独立HBM，不能称物理NVLink。两轮都使用共享锁和前后资源检查，MTP8/R128/LRU4096/f24/host split off，256位置×3冷pass，各设置算术均值；首轮CPE→QP→qks，完整原件不覆盖。','',
 '| 设置 | CPE µs | QP µs | qks µs | qks降低 | Memcpy µs | device µs | 逐步max µs | 高于max |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for setting in ['g100_l0','g100_l20000','g50_l20000']:
 rows={r['variant']:r for r in initial['rows'] if r['setting']==setting};b,q,n=[rows[v] for v in ['base','control_qp','qks']]
 lines.append(f"| {setting} | {b['ours_us']:.3f} | {q['ours_us']:.3f} | {n['ours_us']:.3f} | {1-n['ours_us']/b['ours_us']:+.3%} | {n['memcpy_us']:.3f} | {n['local_us']:.3f} | {n['reference_us']:.3f} | {n['over_reference']:.2%} |")
lines+=['','## 修正版固定四方对照','',
 '新cohort按CPE→QP→原qks→qks_shared完整重做同参对照，保留原服务版作为控制；未用旧cohort绝对时间拼接。','',
 '| 设置 | CPE µs | QP µs | 原qks µs | shared µs | shared较CPE降低 | shared较qks降低 |',
 '|---|---:|---:|---:|---:|---:|---:|']
newrows=[]
for setting in ['g100_l0','g100_l20000','g50_l20000']:
 rows={r['variant']:r for r in a['rows'] if r['setting']==setting and r['cohort']=='qks_shared256_f24_'}
 b,q,o,n=[rows[v] for v in ['base','control_qp','qks','qks_shared']];newrows.append(n)
 lines.append(f"| {setting} | {b['ours_us']:.3f} | {q['ours_us']:.3f} | {o['ours_us']:.3f} | {n['ours_us']:.3f} | {1-n['ours_us']/b['ours_us']:+.3%} | {1-n['ours_us']/o['ours_us']:+.3%} |")
lines+=['','| 设置 | shared µs | Memcpy µs | device µs | 逐步max µs | 高于max |','|---|---:|---:|---:|---:|---:|']
for n in newrows:lines.append(f"| {n['setting']} | {n['ours_us']:.3f} | {n['memcpy_us']:.3f} | {n['local_us']:.3f} | {n['reference_us']:.3f} | {n['over_reference']:.2%} |")
lines+=['','两轮共21格/15个对CPE配对、4032计时launch/252有效segment。逐步/逐pass时间、实际命中率、失败与源快照完整保留；两个参考仍为原同硬件纯device及逐步`mean(max(C_t,unique_miss_t×1152/BW))`，不是max(均值)。','',
 '## 结论与复现','',decision['detail'],'',
 '首版编译前复查修正ready共享标志的下一轮覆写竞争，并把未用寄存器参数按const引用传递，避免读取未初始化值。未编译旧源/manifest保存在precompile_review/，修正后cabe版本才编译测量。修正版独立f396源和JIT不覆盖首版。所有运行状态/handle与原始日志见continuation_state.json，结束的producer不重启。','',
 'CPE原父15217/PID2084930与其SXM验证保持独立；本checkpoint观察phase为`'+c['parent_cached_pe_phase']+'`，应结合最新实际进程读取。H80008:40只读查询仍原五作业Priority/Dependency、原两个watcher活、AGENTS/MEMORY哈希不变，无新提交或排队源变更。生产7193/math34af/模拟器5e409/原5304矩阵e3fd保持。','',
 '冻结源、make/restore脚本、独立JIT代码证明、全部raw和日志在[实验目录](../temp/glm_trace/tune_qk_service_20261101/)。[总审计](../temp/glm_trace/tune_qk_service_20261101/performance_audit.json)、[首轮快照](../temp/glm_trace/tune_qk_service_20261101/initial_performance_audit.json)、[checkpoint](../temp/glm_trace/tune_qk_service_20261101/checkpoint_audit.json)、[helper身份](../temp/glm_trace/tune_qk_service_20261101/attention_identity_review.json)、[7行字节/packing核验](../temp/glm_trace/tune_qk_service_20261101/shared_job_review.json)。','']
(root/'docs/mla_host_a100_qk_service_20261101.md').write_text('\n'.join(lines));print('REPORT_WRITTEN')
