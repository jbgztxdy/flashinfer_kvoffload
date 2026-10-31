"""Preserve denominator-reduction evidence against both the frozen JQ control and b6."""
import datetime
import hashlib
import json
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
a = read(p / 'performance_audit.json')
assert len(a['rows']) == 9 and len(a['pairs']) == 6
assert not a['incomplete'] and not a['pending_references']
assert read(p / 'validation_state.json')['complete']
idx = {(r['setting'], r['variant']): r for r in a['rows']}
triples = []
table = ['| 链路 | b6基线 µs | JQ控制 µs | lane分母 µs | 较JQ降低 | 较b6降低 | Memcpy µs | 纯device µs | 逐步max µs |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for setting in ['g100_l0', 'g100_l20000', 'g50_l20000']:
    b, j, n = [idx[setting, v] for v in ['base', 'joint_qk', 'lane_d']]
    reduction = 1 - n['ours_us'] / j['ours_us']
    assert reduction < 0
    table.append(f"| {setting} | {b['ours_us']:.3f} | {j['ours_us']:.3f} | {n['ours_us']:.3f} | {100*reduction:+.3f}% | {100*(1-n['ours_us']/b['ours_us']):+.3f}% | {n['memcpy_us']:.3f} | {n['local_us']:.3f} | {n['reference_us']:.3f} |")
    triples.append(dict(setting=setting, base_us=b['ours_us'], joint_qk_us=j['ours_us'],
        lane_d_us=n['ours_us'], reduction_vs_joint_qk=reduction,
        reduction_vs_base=1-n['ours_us']/b['ours_us'],
        raw_paths=[r['raw'] for r in [b, j, n]]))
text = '''# A100：lane私有分母累加、末尾一次warp归约（2026-11-01）

候选拒绝。相对冻结JQ控制，100GB0/20µs慢0.275%/0.723%，50GB20µs慢0.039%。没有性能采用信号，不扩展full/native/跨形状，不扫描归约间隔或tile参数。相对b6的正收益全部保留，但不归为本候选收益。

JQ原生PV把BF16/FP16概率输入一次mma rowsum，得到每个16KV分片的两行分母，四个相邻lane持有重复结果。本候选使用完全相同的量化概率，逐lane累加其四个不同KV列，到整个attention item末尾才用xor2/xor1两次shuffle合并，随后保留原跨warpgroup归约及归一化。联合MDO的max/重缩放因子对同一个四lane组相同，因此实数下局部分母缩放与最终求和可交换，softmax方程不变；FP32求和和重缩放分组变化，不声称逐位一致。初始每lane d=1/4，保证空item最后合并后仍是原每warpgroup d=1。奇数尾tile也使用新的局部累加，避免混用两种d状态。

P量化、PV输出MMA次序、全部PV barrier、补数/等待、64head/256线程、32KV/2stage、108CTA、dynamic147968B/static2304B、fetch64行双buffer/f24、精确LRU/真实cache载荷均保持。canonical mla.cuh、模拟器和原5304格矩阵未改；新PV helper只在专属LaneDenominatorTraits实例中替换rowsum。原34EMU/68normal kernel指令/资源及全部机器字/控制字精确相同。新BF16 REG252、FP16 REG253，均STACK0/LOCAL0；实际正常BF16 Graph仅一个新kernel，108CTA、block[32,4,2]、active1，最大绝对误差0.004128。

正常105项原测试和4项长序列测试全通过，共109项新测试。长测试新增LSE核对，覆盖BF16/FP16、K2048、128个缓存池、容量512/4096、淘汰/零miss、Graph多次重放、空/单元素/有padding行，并核验所有有效cache槽的真实KV载荷。JQ控制复用其原109测试/binary证据，reused_joint_control.json记录SHA，不算新增测试。本轮没有source预检或GPU验证失败，四个原handles均terminal0。

初始还检查过“先加载两个K片段再做MMA”的源码重排。冻结JQ的实际SASS在CKV循环中已先加载K_A(0xe310)、Q(0xe340)、K_B(0xe360)，再执行第一A-MMA(0xe3c0)和B-MMA(0xe400)，并提前加载下一feature片段。因此未重复编译该重排；qk_load_schedule_review.json和冻结摘录记录完整源/反汇编SHA。这是局部静态调度证据，不代表所有shared等待均已隐藏。

分母归约替换会减少一条每warp/32KV tile的Tensor Core指令，同时增加量化概率解包和标量加法。寄存器无spill、MMA数量减少均不证明端到端更快。EMU目标静态HMMA299→296、FADD4→28、SHFL53→57、总指令6952→6992，LDSM198及BAR58相同，完整源/SASS SHA和可重跑统计见static_instruction_review.json/audit_static_instructions.py。这些静态数量不是动态work、stall或时间分解。本轮不把静态计数或上述调度摘录计为trace收益，也没有用诊断改参考或模拟器。

GPU3 A100-PCIE/NUMA1，同卡独立HBM模拟远端，不是真实NVLink。真实GLM-5.3 trace，MTP8/R128、LRU4096/请求、f24/off、256位置×3次冷启动，固定JQ→lane分母→b6顺序，全部步/pass算术均值。

{table}

[审计原件](../temp/glm_trace/tune_lane_denominator_20261101/performance_audit.json)含9格/6个对b6配对、1728次计时launch/108段有效队列，另列3个对JQ的三方点；两参考、每步/每pass/命中率/unique miss/guard/源码SHA/快照完整，没有未完成raw。候选未安装，生产保持b6。原JQ72075/1398继续独立等待SXM完整正向六格及保留门槛，不能重复启动。主目标继续active，大batch100GB距逐步max仍约25%–29%。
'''.replace('{table}', '\n'.join(table))
(root / 'docs/mla_host_a100_lane_denominator_20261101.md').write_text(text)
(p / 'composition_comparisons.json').write_text(json.dumps(dict(candidate_adopted=False, triples=triples), indent=2) + '\n')
now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
(p / 'decisions.json').write_text(json.dumps(dict(timestamp=now, goal_status='active',
    candidate_adopted=False, decision='Reject: all3points slower than frozenJQcontrol; no full/native/crossshape or reduction-interval sweep.',
    triples=triples, tests=109, audit_sha256=sha(p / 'performance_audit.json'),
    manifest=read(p / 'manifest.json')), indent=2) + '\n')
s = read(p / 'continuation_state.json')
s.update(recorded_at=now, candidate_adopted=False, active_runners=[],
    next_action='Stop this lane-denominator mechanism; all original handles terminal. No reduction-interval/tile/unroll sweep or full/native expansion. Continue original JQ72075/1398 when GPU01 users naturally leave; inspect live processes before any restart. Overall goal active.',
    results='Rejected,109newtests includinglongLSE/cachepayload/Graph,34/68machinewordsexact,actualonekernel,BF16REG252/FP16REG253STACK0.9cells6basepairs3threewaypoints,1728launch108segments,tworeferencescomplete.100GBvsJQ-0.275/-0.723%,50GB-0.039%.')
s['completed_handles'] = {str(i): 0 for i in [99321, 1359, 92142, 22672]}
(p / 'continuation_state.json').write_text(json.dumps(s, indent=2) + '\n')
(p / 'README.md').write_text('''# Lane-private denominator, 2026-11-01

Rejected.100GB0/20us slower0.275/0.723%than frozenJQcontrol,50GBslower0.039%. No full/native/crossshape or reduction-interval/tile/unroll sweep. See [report](../../../docs/mla_host_a100_lane_denominator_20261101.md), decisions.json and performance_audit.json.

Same quantized BF16/FP16 P; eachlane accumulates its fourKV columns, finalquad shuffles once before native cross-warpgroup normalization. JointMDO factor isquad-uniform, real equation equivalent, FP32grouping differs. d starts1/4 to preserve emptyitemstate, oddtile uses same newlocal reduction. PV/barriers/refill/32KV2stage/fetch/LRU unchanged.109newtests include longLSE/empty/single/padding/Graph/allcachepayload,old34/68machinewords exact,actualone108CTAkernel,REG252BF16/253FP16STACK0. ReusedJQ109testevidence notnewtests.

9cells6basepairs3threewaypoints,1728timedlaunch108segments,bothreferences/source snapshots complete. All99321/1359/92142/22672terminal0,no failures or retries. SimplegroupedKload proposal already present in frozenJQSASS,not prototyped;qk_load_schedule_review.json preserves localstatic evidence,notperformance. Sourcegenerator/restore refuse overwrites. Root unchanged;originalJQretention1398independent,goalactive.
''')
print('REJECTION_REPORT_WRITTEN', [(t['setting'], 100*t['reduction_vs_joint_qk']) for t in triples])
