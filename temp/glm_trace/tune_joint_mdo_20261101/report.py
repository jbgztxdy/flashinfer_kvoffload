"""Render audited joint-MDO evidence; positive screens remain unadopted until all gates."""
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
a=read(p/'performance_audit.json')
assert not a['pending_references']
table=['| cohort | 位置/pass | 链路 | 同轮base µs | 联合MDO µs | 耗时降低 | 候选Memcpy µs | 候选逐步max µs |','|---|---|---|---:|---:|---:|---:|---:|']
for r in a['pairs']:
 table.append(f"| {r['cohort']} | {r['positions']}/{r['passes']} | {r['setting']} | {r['old_us']:.3f} | {r['new_us']:.3f} | {100*r['reduction']:+.3f}% | {r['memcpy_us']:.3f} | {r['reference_us']:.3f} |")
text='''# A100：双32KV tile共用在线softmax重缩放（2026-11-01）

候选尚未采用，生产仍为0f82e9de / b6eea095…e72a。短窗口100GB0/20µs初测快0.653%/2.700%；第一组完整1024×5pass反向快2.677%/2.902%，正常独立HBM快3.002%、物理PCIe基本持平。第二组完整正向也已完成（快2.134%/1.881%），GPU3跨形状两矩阵全部完成；真实SXM/NVLink确认仍待。100GB大batch历史约32%–34%主差距尚未解决，goal active。

保留64head/256线程、32KV/2stage、dynamic147968B、原fetch64行双buffer/f24、精确跨步LRU和真实缓存载荷。两个降序tile先分别计算原native QK，再对两个score fragment共同求max，只计算一次d/O重缩放；各自的P指数和native PV仍按原tile降序执行。第一个stage在第一个PV结束后补下一tile，让其加载与第二个PV重叠，随后补第二个stage。原PE→CKV dot次序不改，完整576特征保留，attention方程相同但FP32在线归约分组不同，不承诺逐位相同。canonical数学头34af、模拟器及原5304矩阵均未改；新联合状态更新仅在私有host头内。派发仅原cached whole-row、至少4×SM行、完整staging的SM80路径，仍单cooperative kernel。

该机制不同于早期配对QK的两次独立MDO，也不同于KV64/单stage的双缓冲损失；不涉及fetch/cadence/unroll扫描。

joint_mdo首版和joint_mdo_after定义顺序控制均通过105+4正常/长cache载荷/Graph测试及实际单kernel实查，但各有4/34EMU、8/68normal原实例指令文本改变（puredevice精确相同），strict gate拦截，二者均无性能launch。只移动新增helper/kernel到原kernel后不能消除变化。joint_mdo_typed用专属继承traits隔离模板helper实例，字段/布局/算法与after版相同，原34EMU/68normal实例全部指令/资源精确相同后才允许性能。没有放宽或归一化反汇编检查，三个冻结源码/JIT及失败日志全部保留。

typed版同样通过105+4，共三版327项测试。长测试覆盖BF16/FP16、K2048、128个池、容量512/4096、淘汰、零miss、padding、Graph复用及所有有效cache payload。实际正常BF16 Graph均只有一个对应新kernel，108CTA、block[32,4,2]、dynamic147968B/static2304B、active1，probe最大绝对误差0.004128。typed新REG255/STACK16/LOCAL0，不能称stack0或将栈代价当作已证明的唯一性能原因。

GPU3为A100-PCIE/NUMA1，far模拟源是同卡独立HBM，不是真实NVLink。正常native_hbm也是独立同卡HBM，pcie才是真实物理Host链路，绝对时间不能拼接为NVLink收益。MTP8/R128、LRU4096/请求、f24/split off；所有步/pass算术均值、包含冷启动。短256位置×3pass为base→typed；预先保存confirmation_plan.json规定full1024位置×5pass反向typed→base，正常两链路base→typed。纯device同硬件参考短窗口2556.960µs，full使用对应全步参考；逐步max按每步compute及unique miss原式计算。

{table}

[完整审计](../temp/glm_trace/tune_joint_mdo_20261101/performance_audit.json)当前{rows}格/{pairs}配对、{launches}计时launch/{segments}segment，保存每步/每pass/命中率/unique miss/guard/两参考/源码快照；所有正负差保留。原short_performance_audit.json冻结首次6格3对，不被full追加覆盖。GPU3两矩阵按冻结计划完成76格，合计当前98格/49对、114080次计时launch/7184segment。100GB十个改变路径配对快1.079%–3.633%；所有负差保留，最大为未改无LRU HBM远端0.3慢0.646%。coverage.json明确GPU3完整、SXM80格仍待。活动runner可能仍有未完成raw（当前{incomplete}份），未纳入表格与收益声明。327测试的通过不等于跨形状/完整链路性能gate已完成；只有全门槛完成才可能安装。

H80003:38:29只读复查五个原任务仍Priority/Dependency，调度估计推迟到11/05；两个原watcher仍活，AGENTS/MEMORY哈希与已读快照相同，无源码写入或重复提交。当前runner和接续任务以continuation_state.json及实际进程为准，不因本报告启动重复任务。此前F/S/WR/KV64已拒绝872项测试、48格/27配对；本候选的正向信号必须单独核实，不覆盖负差。
'''.replace('{table}','\n'.join(table))
for key,value in [('rows',len(a['rows'])),('pairs',len(a['pairs'])),('launches',a['timed_launches']),('segments',a['accepted_segments']),('incomplete',len(a['incomplete']))]:text=text.replace('{'+key+'}',str(value))
(root/'docs/mla_host_a100_joint_mdo_20261101.md').write_text(text)
(p/'README.md').write_text('''# Joint online MDO over two32KV score tiles, 2026-11-01

Not adopted; productionb6eea095…e72a/0f82e9de, goal active. Short100GB0/20us positive0.653/2.700%; full reverse/native/cross-shape gates required. See [report](../../../docs/mla_host_a100_joint_mdo_20261101.md), confirmation_plan.json and performance_audit.json.

Original64head/256thread32KV/2stage/shared147968B/fetch/LRU remain. Two native QK score tiles share one max/d/O rescale; original PV runs per tile descending, refill first stage during secondPV. Same equation, different FP32 online grouping; canonical math header untouched. Distinct from paired-QK-only and64KV/one-stage, no parameter sweep.

First joint_mdo and definition-order control joint_mdo_after each pass109tests/actual108CTA but fail strict old-codegen gate(4/34EMU,8/68normal changed,puredevice exact); zero performance launches. joint_mdo_typed only uses a dedicated inherited traits type to isolate template helper usage: same fields/layout/algorithm, old34/68 exact proof then109tests/actualGraph. Three variants327tests total, frozen independently. TypedREG255/STACK16/LOCAL0,static2304B,one108CTA/block32x4x2kernel,active1. No stack-zero claim.

short_performance_audit.json freezes first6cells/3pairs,1152timedlaunches/72segments with both references. confirmation_plan.json predefines reverse full1024x5 and normal physicalPCIe/independentHBM1024x5. performance_audit.json aggregates completed valid raw, never discards negative points. Read continuation_state.json and actual originalhandles; do not duplicate or overwrite. Allstrict-gatefailures,SASS/resources,source snapshots,raw,cachepayload/Graphtests retained. restore_includes.py never overwrites different inputs. make_candidate.py/make_after.py/make_typed.py reproduce inputs. Original5304matrix untouched. H800queue03:18:48 stillfivepending,twowatchers,estimated11/05,no new submission/source change.
''')
print('REPORT_WRITTEN',len(a['rows']),len(a['pairs']))
