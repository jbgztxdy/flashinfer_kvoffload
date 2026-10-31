"""Freeze the failed address-precomputation control against both production and JQ."""
import hashlib,json,datetime
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
a=read(p/'performance_audit.json');assert len(a['rows'])==9 and len(a['pairs'])==6 and not a['incomplete'] and not a['pending_references']
idx={(r['setting'],r['variant']):r for r in a['rows']};triples=[]
table=['| 链路 | 生产µs | JQ控制µs | 地址表µs | 较JQ降低 | 较生产降低 | 候选Memcpyµs | 候选逐步maxµs |','|---|---:|---:|---:|---:|---:|---:|---:|']
for setting in ['g100_l0','g100_l20000','g50_l20000']:
 b,j,n=[idx[setting,v] for v in ['base','joint_qk','address_map']];reduction=1-n['ours_us']/j['ours_us'];assert reduction<0
 table.append(f"| {setting} | {b['ours_us']:.3f} | {j['ours_us']:.3f} | {n['ours_us']:.3f} | {100*reduction:+.3f}% | {100*(1-n['ours_us']/b['ours_us']):+.3f}% | {n['memcpy_us']:.3f} | {n['reference_us']:.3f} |")
 triples.append(dict(setting=setting,base_us=b['ours_us'],joint_qk_us=j['ours_us'],address_map_us=n['ours_us'],reduction_vs_joint_qk=reduction,reduction_vs_base=1-n['ours_us']/b['ours_us']))
text='''# A100：复用闲置partial工作区预存KV行地址（2026-11-01）

候选拒绝，生产保持0f82e9de / b6eea095…e72a。相对同轮JQ控制，100GB0/20µs慢2.812%/3.159%、50GB20µs慢0.493%，不继续扫地址表布局或增加完整矩阵。

原tile loader从int32编码页号解码near/staging基址和576元素偏移。本候选使用原whole-row闲置partial_o存uint64最终行地址：prepare写near/缓存命中地址，resolve在原staged remap写入处写对应staging地址，padding compaction复制与原int32页号相同的disjoint mover范围。原两次grid同步同时发布地址和remap，不增kernel、barrier、分配或API字段。原remap和ready flag acquire不变；新tile预取只加载完整指针，再加每线程feature/KPE偏移。保持JQ相同QK/MDO/PV、fetch64行双buffer/f24、精确跨步LRU/真实cache载荷，canonical math、模拟器、参考公式和原5304矩阵不变。区别于旧prebind的fetch缓存绑定及direct_cache改变attention数据源。

派发只限原cached fully-staged whole-row、至少4×SM行，且uint64(rows)×max_kv_len×8 ≤ uint64(partial_capacity)×64×512×sizeof(DTypeO)；容量不足保留原生产路径。108SM默认864partial约56.6MB，R128/MTP8/K2048地址表16.8MB，不额外分配；其他形状不能假设总能放入。此预算不保证性能，额外8B/NNZ写入和8B地址读取会增加工作。

105项原BF16/FP16测试、4项长序列/全cache载荷/Graph测试，以及1项容量回退/保护区测试全部通过，共110项新增测试。保护区测试保持整个原backing allocation为哨兵，只把对外partial_o/lse视图缩为1partial，重放带padding的Graph并核验全部backing不变、输出和缓存载荷正确。JQ控制复用原binary/109测试证据，reused_joint_control.json记录原件SHA，不计作本轮新增测试。

实际正常BF16 Graph只有一个BatchMLAAddressedKernel，108CTA、block[32,4,2]、dynamic147968B/static2304B、active1，最大绝对误差0.004128。新EMU/normal REG252/STACK0/LOCAL0，原34EMU/68normal实例逐一指令/资源精确相同；没有修改原helper，仅新专属traits/prepare/resolve/compact/loader使用地址表。资源通过不意味着收益。

GPU3为A100-PCIE/NUMA1，far为同卡独立HBM模拟，不是真实NVLink。MTP8/R128、LRU4096/请求、f24/off、256位置×3冷启动pass，生产→JQ→地址表，所有步/pass算术均值，两参考和所有正负差保留。纯device参考2556.960µs，逐步max按原每步compute与unique miss计算。

{table}

[审计](../temp/glm_trace/tune_address_map_20261101/performance_audit.json)含9格/6个对生产配对、1728计时launch/108segment；composition_comparisons.json另列3个对JQ控制的三方点。没有缺失参考或未完成raw。100GB20µs相对生产的正差也保留，不能用它掩盖相对JQ全部负差。停止此机制，不扩大full/native/跨形状，不更改容量或布局来逐格追最优。

所有runner已结束，候选未安装。JM/JQ的原SXM验证仍使用各自冻结源继续，不受本实验影响；全目标仍active，历史100GB主差距尚未解决。
'''.replace('{table}','\n'.join(table))
(root/'docs/mla_host_a100_address_map_20261101.md').write_text(text)
(p/'README.md').write_text('''# Precomputed KV row pointers in idle partial storage, 2026-11-01

Rejected. See [report](../../../docs/mla_host_a100_address_map_20261101.md), decisions.json and composition_comparisons.json. All3points slower than exactJQcontrol,100GB2.812/3.159%,50GB0.493%; productionpositive0.742%at100GB20us retained too. No full/native/crossshape expansion or layout/capacity scan.

Pointers written with originalprepare/resolve and copied with disjointpaddingmovers; existinggridbarrierspublish them. Originalremap/readyflags/fetch/LRU/math/singlekernel retained. Unusedwhole-rowpartial_o allocation bounds uint64table; insufficientcapacityfallsback toproduction. No newallocation/barrier/kernel/API.109usual tests plus1capacitycanary=110newtests,34EMU/68normal oldkernel exact,actualone108CTA/block32x4x2kernel,dynamic147968/static2304,active1,REG252STACK0LOCAL0. ReusedJQ109test/binaryevidence isnotnewtests. Guardtest truncates declaredview toonepartial with fullbackingcanary,checksGraph/padding/cachepayload.

9cells/6productionpairs,1728timedlaunches/108segments,3three-waypoints,two references/source snapshots complete. Baselines reuseW,controlJQ externalJIT/gates with exactSHA. make_candidate.py/restore_includes.py refuse overwrite. Allrunners terminal,rootb6/math/simulator/original5304matrixunchanged. FixedJM/JQvalidationcontinues initsowndirectory.
''')
(p/'composition_comparisons.json').write_text(json.dumps(dict(candidate_adopted=False,triples=triples),indent=2)+'\n')
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds');s=read(p/'continuation_state.json')
s.update(recorded_at=now,active_runners=[],candidate_adopted=False,completed_handles={str(i):0 for i in [10839,97365,62831,20361,3002,25750,2690]},results='110newtests,old34/68exact,actualnewkernel,9cells/6productionpairs+3threewaypoints,1728launches/108segments. AllslowerthanJQcontrol; rejected.',next_action='Stopthisaddress-tablemechanism,no layout/capacityscan. ContinueexistingJM/JQSXM/fullregression/retentiongates; do notoverwritefrozen sources.')
(p/'continuation_state.json').write_text(json.dumps(s,indent=2)+'\n')
(p/'decisions.json').write_text(json.dumps(dict(timestamp=now,goal_status='active',candidate_adopted=False,decision='Rejectallthreeaddress-tablepointsversusJQ;nofurtherlayoutsweep.',triples=triples,tests=110,timed_launches=1728,accepted_segments=108,audit_sha256=sha(p/'performance_audit.json'),manifest=read(p/'manifest.json')),indent=2)+'\n')
print('REPORT_WRITTEN')
