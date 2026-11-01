"""Render the completed QK/PV role experiment, keeping historical matrices untouched."""
import json,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text())
a=read(p/'performance_audit.json');m=read(p/'manifest.json');c=read(p/'checkpoint_audit.json')
lines=['# A100 QK/softmax 与 PV warp 分工实验','',
    '更新于 '+time.strftime('%Y-%m-%d %H:%M %Z')+'。**候选拒绝，生产 JQ 保持 `7193255e…d469`，整体目标仍 active。** 100 GB/s 大 batch 仍距逐步 max 参考约 25%–27%。','',
    '本实验直接基线为冻结 CPE `a43617…2b15`，该父候选尚未保留。这里的差值不能标成相对生产的收益。新头 SHA `'+m['candidate_sha256']+'`，canonical math34af、模拟器5e409、原5304格e3fd均未改。','',
    '## 实现','',
    '单 cooperative CTA 改为 block[32,4,3]，共384线程。z0的四个warp计算64Q head的完整32KV列，并把同一Q片段用于两个16KV列块及两个stage；z1/2的八个warp仍各持128个FP32输出寄存器，计算原生PV的两组256特征。QK/softmax不持O，PV不持score/QPE。区别于旧WR的取数/计算分工，也区别于N9的独立服务warp。','',
    '完整PE→CKV的MMA顺序、576特征、量化FP16/BF16 P和原生TensorCore rowsum保留。两个tile共用joint max和O/d缩放；完整32KV列的max/denominator由四个QKwarp直接归约并广播，改变了FP32加法/归约分组，实数attention方程相同，不声称逐位等价。空item分母初始2对应旧两个warpgroup初始值之和，最终仍按原规则输出零与LSE。','',
    'P继续复用两个KPE stage。named barrier1的384参与者分别保护初始状态、P/scale发布、hi-PV释放、lo-PV释放；barrier2的128参与者同步QK角色的host acquire和cp.async可见性。生产角色在hi-PV释放后补下一hi，与PV角色计算当前lo重叠；在lo释放后补下一lo。原prepare/LRU/grid/last-CTA协议仍由全部384线程参加。','',
    '计算字段140,032 B，union padding保持sizeof/dynamic147,968 B，通用fetch按sizeof仍选原两块64行buffer；f24和实际在途3072行保持。每个fetch/prepare/LRU实例的线程分配由256变384，是本候选的一部分，不把全部差值归因于单独Q读取。LRU容量、epoch保护/淘汰规则、真实cache载荷和FAR post-read模型保持。','',
    '## 资源与验证','',
    '正常BF16/FP16和EMU均REG168、STACK8、static2304、dynamic147968，实际每SM active1；SASS有LDL/STL，不能称无spill或无栈。EMU目标静态7128条、HMMA230/LDSM124/BAR59/LDL7/STL3；正常目标6920条、HMMA230/LDSM124/BAR58/LDL7/STL3。静态数受展开及分工影响，不能当作动态工作量或性能原因。','',
    '105项原正常测试和6项额外测试通过，共111项。额外覆盖BF16/FP16、K2048/63tile奇尾、128池、512/4096容量、空/单元素/padding、output/LSE、淘汰及全部cache载荷、Graph重放，以及K65/QPE head stride80。Graph实际为单个108CTA/384线程新kernel。三项sanitizer的首个匹配BF16/K97新kernel launch均通过；memcheck仅关闭CUDA API错误报告，保留设备检查，不外推为所有形状或完整trace的sanitizer证明。','',
    '每版库内原38个EMU及76个正常BF16/FP16实例的指令、资源、完整机器字和控制字均相同，含纯device参考实现。新数值、缓存、Graph及代码证据与性能审计分别保存。','',
    '## 固定同参短对照','',
    'GPU3为A100-PCIe/NUMA1；EMU far payload是同卡独立HBM，不是物理NVLink。共享GPU锁和前后资源检查通过。MTP8/R128、LRU4096、f24、host split off，base→qkpv_roles，固定256位置×3个冷启动pass，三链路未扫参数。','',
    '| 设置 | CPE µs | 新候选 µs | 降低 | Memcpy µs | device µs | 逐步 max µs | 高于 max |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in a['pairs']:
    lines.append(f"| {r['setting']} | {r['old_us']:.3f} | {r['new_us']:.3f} | {r['reduction']:+.3%} | {r['memcpy_us']:.3f} | {r['local_us']:.3f} | {r['reference_us']:.3f} | {r['over_reference']:.2%} |")
lines+=['','6格/3配对、1,152次计时launch/72个有效segment，全部逐步/逐pass时间和实际命中率保留。两个参考沿用同硬件正常构建的纯device，以及逐步`mean(max(C_t, unique_miss_t×1152/BW))`；未修改公式、模拟器、原矩阵或基线时间。','',
    '100GB0/20µs慢3.836%/2.936%，50GB20µs快0.246%，未改善主要瓶颈。停止独立QK/PV分工原型，不扩完整1024×5pass/native/全形状矩阵，也不扫描warp数、unroll或fetch。寄存器预算、更多同步、加载分工、公共384线程分配及8B栈可能共同影响，当前证据不能确定各自因果份额。','',
    '## 失败与复现记录','',
    '编译前静查发现：一个QKwarp发布P可能覆盖其他warp尚在读取的KPE。paired和odd路径均补128线程producer barrier后才编译；初始源/manifest/辅助文件保存在precompile_review/，SHA和说明在precompile_review.json，没有GPU失败或性能数据。','',
    '验证脚本的依赖复制首次误用Path(.)的相对parent，找不到兄弟目录compiler.py；后续shell仍启动了原验证父进程78438。修正为绝对cwd并补齐依赖时，该父进程仍在等待原build/test gate；复用同一进程继续，未重复启动、未覆盖候选或重跑GPU测试。setup_recovery.json保存错误和修复边界。','',
    '原handles15541（normal105）、92506（EMU）、38976（extra6）、78438（codegen/Graph/sanitizers/screen/audit）均terminal0。continuation_state保存状态；不要重启这些producer。CPE原父15217/PID2084930和子2104175仍独立推进SXM验证；本审计观察phase为`'+c['parent_cached_pe_phase']+'`，读取最新实际进程后再判断，不重复启动。H800本轮08:17只读复查原五作业仍Priority/Dependency、原两个watcher活，AGENTS/MEMORY哈希不变，没有新提交或排队源码变更。','',
    '冻结源、独立JIT、完整SASS/resource、日志、源码快照及raw均在[实验目录](../temp/glm_trace/tune_qk_pv_roles_20261101/)。restore_includes.py恢复缺失include并拒绝不同内容；compile.py使用各variant独立workspace。checkpoint_audit.py绑定源/库SHA、全部原代码字、实际派发及审计；报告不需GPU即可重生成。','',
    '[性能审计](../temp/glm_trace/tune_qk_pv_roles_20261101/performance_audit.json)、[checkpoint](../temp/glm_trace/tune_qk_pv_roles_20261101/checkpoint_audit.json)、[决定](../temp/glm_trace/tune_qk_pv_roles_20261101/decision.json)、[静态代码核验](../temp/glm_trace/tune_qk_pv_roles_20261101/new_kernel_codegen_review.json)。','']
(root/'docs/mla_host_a100_qk_pv_roles_20261101.md').write_text('\n'.join(lines));print('REPORT_WRITTEN')
