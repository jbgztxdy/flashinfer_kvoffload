"""Freeze the allocation-matched64KV/one-stage negative control."""
import datetime,hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
a=read(p/'performance_audit.json')
assert len(a['rows'])==6 and len(a['pairs'])==3
assert not a['pending_references'] and not a['incomplete']
assert all(r['reduction']<0 for r in a['pairs'] if r['setting'].startswith('g100'))
table=['| 链路 | 同轮 base µs | 64KV/1stage µs | 耗时降低 | 候选 Memcpy µs | 候选逐步 max µs |','|---|---:|---:|---:|---:|---:|']
for s in ['g100_l0','g100_l20000','g50_l20000']:
 r=next(x for x in a['pairs'] if x['setting']==s)
 table.append(f"| {s} | {r['old_us']:.3f} | {r['new_us']:.3f} | {100*r['reduction']:+.3f}% | {r['memcpy_us']:.3f} | {r['reference_us']:.3f} |")
text='''# A100：相同 shared 预算的 64KV 单 stage（2026-11-01）

候选拒绝。100GB/s两点慢1.576%/1.795%，50GB/s近乎持平，不能替代已保留生产0f82e9de / b6eea095…e72a。

将原64head/256线程、32KV/2stage改为native sharded64KV/1stage，dynamic shared仍147968B。每64行一次在线softmax最大值/输出重缩放，减少tile更新和同步次数，并扩大原Qfragment的KV复用；单stage加载等待及额外score寄存器可能抵消收益。数学头保持canonical34af不变，直接使用原native64KV实现；数学方程相同但FP32在线归约分组变化，不承诺逐位相同。仍是单cooperative kernel，原fetch64行双缓冲/f24、metadata、精确LRU、真实cache载荷和退出协议不变。派发仅原cached whole-row、至少4×SM行、完整staging的SM80路径。

它区别于C的单stage/32KV控制（softmax更新数不减少），也区别于早期配对QK（仍为两次独立MDO/PV更新）。只实现这一种明确的原生tile组合，没有KV大小/stage/fetch/cadence扫描。

105项正常BF16/FP16原测试和4项长序列/全cache载荷/Graph测试全部通过，共109项。原34EMU/68normal库内实例（含纯device）指令/资源全等价；实际正常BF16 Graph只有一个新KV64HostBatchParams kernel，grid[1,108,1]、block[32,4,2]、dynamic147968B/static2304B、active1，最大绝对误差0.004128。新EMU/normal均REG255、STACK8/LOCAL0；EMU目标9320条静态指令、822 HMMA/528 LDSM，有STL/LDL访存，不能写成stack0，也不能据此单因解释性能。

GPU3为A100-PCIE/NUMA1，far是同卡独立HBM模拟，不是真实NVLink。MTP8/R128、LRU4096/请求、f24/off、256位置×3冷启动pass；先候选、后base，所有步/pass算术均值，保留冷启动和负差。纯device同硬件参考2556.960µs，逐步max按逐步compute及unique miss原式计算。

{table}

[审计](../temp/glm_trace/tune_kv64_20261101/performance_audit.json)含6格/3配对、1152计时launch/72segment，每步/每pass、命中率、unique miss、两参考及源码快照完整；没有缺失参考或未完成raw。未修改模拟器、参考数据/公式或原5304矩阵。所有runner已结束，不采用，不扩大full1024×5pass/native/全形状。

后续独立joint_mdo实验保持原32KV/2stage，两个score tile共用一次最大值和d/O重缩放，再逐tile做原PV，并在第二个PV期间补入第一个future stage；目标是保留双缓冲同时减少重缩放。它改变FP32更新分组，尚须独立原实例等价、实际路径、109测试及性能门槛，不能由本报告推断收益。见tune_joint_mdo_20261101的原handles/源码，禁止重复启动或覆盖。

本轮F/S/WR/KV64合计872项测试、48格/27配对、9216计时launch/576segment均无采用；100GB约32%–34%主差距尚未解决，goal active。
'''.replace('{table}','\n'.join(table))
(root/'docs/mla_host_a100_kv64_20261101.md').write_text(text)
(p/'README.md').write_text('''# Native64KV/one-stage allocation-matched control, 2026-11-01

Rejected; production b6eea095…e72a /0f82e9de, goal active. See [report](../../../docs/mla_host_a100_kv64_20261101.md), decisions.json and performance_audit.json.

Original64head/256thread math uses native sharded64KV/1stage with the same147968B dynamic shared as32KV/2stage. One online softmax update per64KV; FP32 grouping changes, canonical math header untouched. Original fetch64row double buffer/f24,metadata,LRU and payload retained. Distinct from1stage/32KV and paired-QK-only controls, no parameter sweep. All109tests pass;34EMU/68normal old instances exactly equal. Actual one108CTA/block32x4x2 kernel,active1,static2304B. NewREG255/STACK8/LOCAL0, actualSTL/LDL in SASS; no stack-zero claim.

Six cells/three pairs,1152 timed launches/72segments and both references complete. At100GB0/20us slow1.576%/1.795%;50GB+20us fast0.024% (effectively flat), preserved. No adoption/full-window/native-link/cross-shape expansion. All runners terminal. restore_includes.py validates or restores frozen host/math without overwrite; baselines reuse W libraries. Source snapshots/raw/test/codegen/actual-dispatch and audit records retained; original5304matrix untouched.

The separate joint_mdo experiment retains32KV/2stage but combines two tiles' maximum/output rescale, refilling one stage during the other PV. Its independent gates and handles are in its own directory; no benefit inferred from this rejected control.
''')
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
s=read(p/'continuation_state.json');s.update(recorded_at=now,active_runners=[],candidate_adopted=False,
 completed_handles={str(i):0 for i in [34389,54291,62484,40753,86019,21765,92528]},
 next_action='AllKV64runnersterminal/rejected. Continue independentjoint_mdo; do not sweep KV/stage/fetch/cadence.',results='109tests,6cells/3pairs,1152launches/72segments,two references complete;100GBslower,50GBflat.')
(p/'continuation_state.json').write_text(json.dumps(s,indent=2)+'\n')
d=dict(timestamp=now,goal_status='active',candidate_adopted=False,decision='Reject64KV/one-stage;100GBbothslower,50GBflat.',pairs=a['pairs'],tests=109,timed_launches=1152,accepted_segments=72,audit_sha256=sha(p/'performance_audit.json'),manifest=read(p/'manifest.json'))
(p/'decisions.json').write_text(json.dumps(d,indent=2)+'\n');print('REPORT_WRITTEN')
