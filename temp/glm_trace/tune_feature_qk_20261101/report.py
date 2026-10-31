"""Render the completed negative controls directly from the audited raw summaries."""
import json
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
audit = json.loads((p/'performance_audit.json').read_text())
assert len(audit['rows']) == 18 and len(audit['pairs']) == 12
assert not audit['pending_references'] and not audit['incomplete']
rows = {(r['fetch_ctas'], r['setting'], r['variant']):r for r in audit['rows']}
table = ['| fetch CTA | 链路 | 同参 base µs | 重复 QK 控制 µs | 特征分工 µs | 特征分工耗时降低 | 其 Memcpy µs | 其逐步 max µs |',
         '|---:|---|---:|---:|---:|---:|---:|---:|']
for fetch in [24,48]:
    for setting in ['g100_l0','g100_l20000','g50_l20000']:
        base, control, new = [rows[fetch,setting,v] for v in ['base','duplicate_control','dim_split']]
        table.append(f"| {fetch} | {setting} | {base['ours_us']:.3f} | {control['ours_us']:.3f} | {new['ours_us']:.3f} | {100*(1-new['ours_us']/base['ours_us']):+.3f}% | {new['memcpy_us']:.3f} | {new['reference_us']:.3f} |")
text = '''# A100：按特征分工的 QK 与 16 行 KV 双 stage（2026-11-01）

候选和控制版均拒绝，生产保持 0f82e9de / b6eea095…e72a。减少重复 QK 的收益不足以优于现有推荐 f24 调度。

现有 16 行 KV 路径让两个 warp group 重复计算完整 QK。本候选使用两个 32-head item、128 线程/CTA、2 个 KV stage/16 行：每个 warp group 计算一半 PE 和 CKV 特征，交换 4096B 部分分数并相加，再运行原 nonsharded softmax/PV。每个 dot 覆盖全部 576 个特征，attention 方程不变；FP32 累加顺序从原 PE→CKV 改为两部分之和，不能称作逐位相同。所有头直接写各自输出，无 head 合并，仍是单 cooperative kernel。原 LRU、容量和实际缓存载荷、模拟器不变。

duplicate_control 保持相同 32-head/16-KV/双 stage 布局和 4096B padding，调用原重复 QK。它与旧的 32-head/32-KV/单 stage 候选、64-head/32-KV 重复 QK 候选不同，没有覆盖或重做旧版本。派发仅限原 cached whole-row、至少 4×SM 行、完整 staging 的 SM80 路径；无 LRU 和其他原路径继续使用原 kernel。

每版通过 105 项原测试和 4 项 BF16/FP16 长序列测试，共 218 项。新增测试覆盖 K2048、128 个池、容量 512/4096、淘汰、零 miss、padding、Graph 重放及所有有效缓存行的逐字节载荷。实际正常 BF16 Graph 均只有一个对应新 kernel，216 CTA × 128 线程、dynamic shared78080B / static2176B，2 resident CTA/SM；反汇编后的模板布尔值分别为 true/false，证明执行了所选算法。probe 最大绝对误差 0.004128。

特征分工实例 REG231，控制 REG255，均 STACK0/LOCAL0。EMU 静态 HMMA 数量 207 对 315、LDSM156 对 264，目标总指令 7176 对 7608；这些是编译证据，不能替代耗时。私有 math 头 e5968…289a3 只复用既有 32-head Q load 索引修正。全部 34 个 EMU / 68 个正常原库内实例（含纯 device）逐一指令/资源相同；审计必须验证这一证明及新 host/math SHA，才可沿用 canonical math 34af…83b3 的同硬件参考。没有改参考数据、公式或原 5304 格矩阵。

GPU3 为 A100-PCIE，far 源是同卡独立 HBM 模拟，不能称作 NVLink。MTP8/R128、LRU4096/请求、staged/split off、256 位置 × 3 次冷启动 pass，所有步及 pass 算术均值。纯 device 参考 2556.960µs。f24 同参数先 base、控制、候选；第二组按预先保存的 normalized_budget_plan，将 fetch 提到48以恢复原 f24 的在途页数：48×32×2 = 24×64×2 = 3072 行。第二组先候选、控制、base。只做这一组明确预算控制，没有继续扫数值。

{table}

f24 的分工版相对重复 QK 控制在两个100GB点快约19%/9%，但相对原 f24 base 慢9.605%/27.923%。f48在同参数100GB两点仍慢2.408%/3.276%，而约4.03/4.08ms也慢于原推荐 f24的约3.5ms。50GB正差及所有负差均保留。因此不扩大完整1024位置、多形状或正常物理链路确认，不采用，也不再单独扫描此机制的 fetch/cadence/unroll。

[审计](../temp/glm_trace/tune_feature_qk_20261101/performance_audit.json)保存18格/12配对、3456次计时launch/216个有效segment、两参考、每步/每pass、命中率、unique miss、guard及完整源码快照。源码 SHA：dim_split 510d9c7c7b9ed5f48010b091b18a0af4fd04136db04a9263a727e829e75ce257；control b1befa4eac63e4a1d9159e362e5b2a271cd990ffe6fc8f09d78c4151949343a0。初次生成器断言将 KV load 和 writeback 两个 skip 误计为一个，在生成任何候选前失败；原脚本/日志保留，修正后才构建，未混入 GPU 结果。

H80001:34:48+08查询仍为五个原作业Priority/Dependency、两个watcher存活，AGENTS/MEMORY的哈希与已完整阅读的远端快照一致。没有远端源码写入或重复提交。该实验所有runner已结束；100GB约32%–34%主差距未改善，goal继续active。后续独立单位重缩放实验见 tune_unit_scale_20261101，不能由此处正确性或资源结论外推其收益。
'''.replace('{table}','\n'.join(table))
(root/'docs/mla_host_a100_feature_qk_20261101.md').write_text(text)
(p/'README.md').write_text('''# Feature-partitioned QK with two-stage 16-KV tiles, 2026-11-01

Both candidates rejected. Production remains retained b6eea095…e72a / 0f82e9de; goal active. See [report](../../../docs/mla_host_a100_feature_qk_20261101.md), decisions.json and performance_audit.json.

dim_split uses two 32-head items, 128 threads, two 16-row KV stages and a 4096B partial-score exchange. Warp groups cover disjoint PE/CKV features, then sum scores and use original nonsharded softmax/PV. The equation is unchanged but FP32 dot reduction order differs. duplicate_control has the same layout/padding and native duplicated QK. Both only change existing large cached whole-row SM80 dispatch.

Each passed 105 normal tests plus four BF16/FP16 long-sequence/cache-payload cases. All original 34 EMU / 68 normal library instances exactly match, including pure device. Actual BF16 Graphs prove one corresponding kernel, 216 CTAs x128 threads, dynamic78080B / static2176B and two resident CTAs/SM; demangled feature booleans are verified. New REG231 versus control255, both STACK0/LOCAL0. The private e5968 math header only reuses three established 32-head Q-load index changes; benchmark provenance captures both executed private host and math headers. The auditor requires the complete old-instance equivalence proof before reusing canonical-math references.

All 18 cells /12 pairs, 3456 timed launches /216 segments and two references are complete: two short cohorts at f24 and an explicitly planned f48 in-flight-payload control (32-row fetch blocks versus native64). f24 dim_split slows100GB0/20us9.605%/27.923%; f48 still slows same-parameter base2.408%/3.276%, and does not replace retained f24. All positive50GB and negative cells remain. No full-window/native-link/cross-shape expansion.

restore_includes.py validates existing private trees or restores missing host/math pairs without overwrite. Baselines reuse exact b6 W/jit_cached_ids and W/jit_retained_normal; restore/build W first on a fresh checkout. Source, codegen, actual-dispatch, tests, raw data, report generator and audits are retained. The initial generator assertion failure is preserved separately and produced no candidate artifacts or GPU results. All runners here are terminal.

H800 queue/context-hash snapshots are read-only. Five original jobs remain pending and both watchers live; no queued source or submission changed. The next independent unit-rescale experiment has its own directory and gates.
''')
print('REPORT_WRITTEN',len(audit['rows']),len(audit['pairs']))
