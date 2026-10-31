"""Render the three completed rescale controls from audited measurements."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
a = read(p/'performance_audit.json')
assert len(a['rows']) == 18 and len(a['pairs']) == 9
assert not a['pending_references'] and not a['incomplete']
variants = ['unit_scale', 'stable_scale', 'early_stable']
table = ['| 版本 | 链路 | 同轮 base µs | 候选 µs | 耗时降低 | 候选 Memcpy µs | 候选逐步 max µs |',
         '|---|---|---:|---:|---:|---:|---:|']
for v in variants:
    for s in ['g100_l0', 'g100_l20000', 'g50_l20000']:
        r = next(r for r in a['pairs'] if r['variant'] == v and r['setting'] == s)
        table.append(f"| {v} | {s} | {r['old_us']:.3f} | {r['new_us']:.3f} | {100*r['reduction']:+.3f}% | {r['memcpy_us']:.3f} | {r['reference_us']:.3f} |")
text = '''# A100：在线 softmax 单位重缩放分支（2026-11-01）

三版均拒绝，生产保持 0f82e9de / b6eea095…e72a。跳过数学上单位重缩放的分支没有改善大 batch 100GB/s 的耗时。

只对已有 cached whole-row、至少 4×SM 行、完整 staging 的 SM80 路径增加独立 traits/params；保持 64 heads、256 线程、32 行 KV/2 stage、108 CTA、单 CTA 驻留、原 fetch24、QK/P/PV、tile 顺序和精确 LRU/真实缓存载荷。全部都是单 cooperative kernel，模拟器和原 5304 格矩阵未改。每版 105 项原正常测试和 4 项 BF16/FP16 长序列测试通过，共 327 项；长测试含 K2048、128 个池、容量512/4096、淘汰、零 miss、padding、Graph 重放及所有有效缓存载荷。

- unit_scale 保留原表达式计算 o_scale，仅当其精确等于1时跳过 d/O 乘法；score exponentials 不变。新 REG244。
- stable_scale 保留原 scale 计算，在最大值改变或 scale 非有限时才重缩放，跳过相同有限最大值下由 FMA 舍入产生的微小非单位 scale。新 REG242。
- early_stable 将检查移到 rescale 指数之前，在最大值不变且有限时直接跳过 rescale 指数和 d/O 乘法，非有限最大值仍计算原式。新 REG242。

后两版利用有限状态下 exp2(0)=1 的数学恒等式，改变 FP32 舍入，不能称为逐位相同，也不承诺维持有限数乘巨大 scale 溢出后的旧 NaN 伪影。没有改变 attention 方程或截断特征。原 traits 的检测恒为 false；每版全部原34个EMU/68个正常BF16/FP16库内实例（含纯device）指令和资源逐一相同。审计要求匹配每版私有 math/host SHA 和正常68实例等价证明才复用 canonical math 的同硬件参考，未更改参考数据或公式。

实际正常 BF16 Graph 每版只有一个对应新 traits kernel：grid[1,108,1]、block[32,4,2]、dynamic147968B/static2304B、active1。所有新实例 STACK0/LOCAL0，实际 probe 最大绝对误差0.004128。资源减少不能推断性能收益。

GPU3为A100-PCIE/NUMA1，far源是同卡独立HBM模拟，不是真实NVLink。MTP8/R128、LRU4096/请求、256位置×3次冷启动pass、f24/split off；所有步/pass算术均值，保留每步/每pass、命中率、unique miss及两种参考。纯device参考2556.960µs。三组各自独立同轮base，顺序分别候选→base、base→候选、候选→base；不能将跨组绝对时间拼作收益。

{table}

第一版100GB慢6.646%–7.651%，stable慢4.411%–5.069%，early仍慢4.137%–4.161%；50GB三版均略慢。因此停止本系列，不继续调整条件、cadence、unroll或几何，不扩大完整1024位置/5pass/物理链路确认，不采用。

为检查精确等于1的条件，保留了两项独立诊断。primary kernel的静态SASS依赖链为FMUL→FFMA→EX2→unit比较，并有真实BRA跳过约64条FMUL，不只是谓词乘法。独立CUDA表达式probe从两个数组加载65536对相同最大值，26440个算得scale不精确为1，最大误差3.57627868652e−7。它证明该浮点反例存在，不能说明真实attention的最大值稳定比例、实际分支频率，或将全部性能负差归因于舍入；诊断不计作性能或参考。源、编译日志、SASS和结果均保留。

[完整审计](../temp/glm_trace/tune_unit_scale_20261101/performance_audit.json)含18格/9配对、3456次计时launch/216个有效segment，参考无缺失、raw无未完成项。三个host/math SHA以及最终拒绝理由见decisions.json；所有源码快照可校验。与前一feature-QK实验合计545项测试、36格/21配对、6912次计时launch/432segment；两组均没有生产收益。100GB大batch约32%–34%主差距仍在，goal active。
'''.replace('{table}', '\n'.join(table))
(root/'docs/mla_host_a100_unit_scale_20261101.md').write_text(text)
(p/'README.md').write_text('''# Online-softmax unit rescale, 2026-11-01

All three candidates rejected; production remains b6eea095…e72a / 0f82e9de, goal active. See [report](../../../docs/mla_host_a100_unit_scale_20261101.md), decisions.json and performance_audit.json.

unit_scale checks the original computed scale against1; stable_scale checks unchanged maxima after computing scale; early_stable checks unchanged finite maxima before computing scale. QK/P/PV/layout/fetch/LRU stay native. The latter two use the mathematical unit-scale identity but change FP32 rescale rounding; no bitwise equivalence claim. All variants passed105 normal+4 BF16/FP16 long/cache-payload/Graph cases,327 total. Each has34EMU/68normal original instances exactly equal and one actual108CTA/256-thread kernel with dynamic147968B/static2304B and one resident CTA/SM. New REG244/242/242, STACK0/LOCAL0.

Three independent short same-parameter cohorts contain18cells/9pairs,3456 timed launches/216segments and both references. At100GB0/20us the candidates slow6.6–7.7%,4.4–5.1%,4.1–4.2%; all50GB results are retained too. No adoption, full-window/native-link/cross-shape expansion or further parameter sweep. Private math/host source snapshots and matching68-instance proofs are required by the auditor before reusing canonical same-hardware references.

scale_expression_sass_trace.json and probe_scale_roundoff.cu/.py show a static FMA dependency and an independent equal-max floating-point counterexample. They do not measure actual attention branch frequency or establish performance causality. The executable is ignored, with rebuild source/flags/logs/SASS/results retained. restore_includes.py validates or reconstructs all private host/math pairs without overwriting. Baselines reuse W/jit_cached_ids and W/jit_retained_normal. All runners are terminal; the early build handle96149 is no longer retained by the execution service, while its complete log/codegen/test/benchmark gates prove successful downstream use.
''')
now = datetime.now().astimezone().isoformat(timespec='seconds')
state = read(p/'continuation_state.json')
state.update(recorded_at=now, active_runners=[], candidate_adopted=False,
    next_action='All three rescale variants rejected. Consider spatial warp-role isolation only after reviewing history; do not repeat temporal cofetch/hybrid or scan rescale conditions.',
    tests='All three variants passed105+4 normal tests,327 total.',
    results='18 cells/9 pairs,3456 launches/216 segments; no pending references or incomplete raw. All100GB comparisons slower. See performance_audit.json.')
state['completed_handles'].update({'24991':0,'28346':0})
state['retired_build_handle'] = {'session_id':96149,'status':'Unknown process id on final query; complete compile log and all downstream gates successful'}
state['last_screen_processes'] = read(p/'early256_f24_gpu3_state.json')
state['terminal_process_check'] = 'No early_stable/compiler/probe/screen process remained on final ps check.'
(p/'continuation_state.json').write_text(json.dumps(state,indent=2)+'\n')
decision = dict(timestamp=now, goal_status='active', candidate_adopted=False,
    decision='Reject all three rescale guards; no further condition/parameter sweep.',
    reason='All same-parameter100GB cases slower; early guard removes the unused SFU dependency but does not outperform retained base.',
    candidates=[dict(variant=v,host_sha256=sha(p/(v+'.cuh')),math_sha256=sha(p/(v+'_mla.cuh'))) for v in variants],
    pairs=a['pairs'],normal_tests=315,stress_tests=12,timed_launches=a['timed_launches'],
    accepted_segments=a['accepted_segments'],audit_sha256=sha(p/'performance_audit.json'),
    production_header_sha256=a['production_header_sha256'])
(p/'decisions.json').write_text(json.dumps(decision,indent=2)+'\n')
print('REPORT_WRITTEN',len(a['rows']),len(a['pairs']))
