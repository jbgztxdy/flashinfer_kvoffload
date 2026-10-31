"""Validate and optionally install the frozen shape-only dispatch correction.

Run both audits first. Every paired cell is retained, including negative differences.
The 1% unchanged-path tolerance is a measurement screen, not a claim of zero noise.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
load = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
args = argparse.ArgumentParser()
args.add_argument('--install', action='store_true')
args = args.parse_args()
variant = 'balanced_cached_sm80'
candidate = p / (variant + '.cuh')
base_sha = '81891229df298b90219410b5a4df1e6c20dc4cb1ee9c0233bd17ad383b29225b'
candidate_sha = '52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'
assert sha(p / 'base.cuh') == base_sha
assert sha(candidate) == candidate_sha
# Verify the complete candidate is exactly host-side includes, helper and predicate edits.
expected = (p / 'base.cuh').read_text().replace(
    '#include "mla.cuh"', '#include <queue>\n#include <vector>\n#include "mla.cuh"')
marker = '// partial_capacity bounds the item states of the tail rows, in 64-head tiles.'
expected = expected.replace(marker, (p / 'balanced_dispatch_cached.inc').read_text() + marker)
expected = expected.replace('params.num_rows >= 4u * num_sms &&', '''(params.num_rows >= 4u * num_sms ||
           (CTA_TILE_KV == 32 && params.num_rows >= 2u * num_sms &&
            host_batch_balanced_whole_rows(params.num_rows, max_kv_len, num_sms,
                                          params.fetch_ctas, partial_capacity))) &&''')
assert candidate.read_text() == expected
assert sha(root / 'temp/glm_trace/kernel_report_20261029/matrix.json') == (
    'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99')
s = load(p / 'audit_general.json')
g = load(p / 'gpu3/audit.json')
assert g['gpu3_planned_performance_coverage_complete']
affected = {(8, 28), (8, 32), (8, 40), (8, 48), (1, 256)}
expected_sxm = {(kind, v, m, b, setting, f)
    for kind, fs in [('lru', [None]), ('nocache', [0., .1, .3])]
    for v in ['base', variant] for m, b in affected
    for setting in ['g100_l0', 'g100_l20000', 'g50_l20000'] for f in fs}
sr = [r for r in s['rows'] if Path(r['raw']).name.startswith('balanced_sxm_full_')]
assert {(r['kind'], r['variant'], r['mtp'], r['requests'], r['setting'], r['fraction'])
        for r in sr} == expected_sxm and len(sr) == 120
nr = [r for r in s['rows'] if Path(r['raw']).name.startswith('balanced_sxm_native_')]
assert len(nr) == 4
assert {(r['variant'], r['setting']) for r in nr} == {
    (v, link) for v in ['base', variant] for link in ['nvlink', 'pcie']}
for r in sr + nr:
    assert r['full_trace'] and r['positions'] == 1024
    assert r['passes'] == (5 if r in nr else 3)
    assert r['fetch_ctas'] == 16 and not r['host_split']
    assert r['header_sha256'] == (candidate_sha if r['variant'] == variant else base_sha)
index = {(Path(r['raw']).name.split(r['variant'] + '_staged_', 1)[0], r['kind'],
          r['mtp'], r['requests'], r['setting'], r['fraction'], r['variant']): r
         for r in sr + nr}
pairs = []
for k, new in index.items():
    if k[-1] != variant:
        continue
    old = index[(*k[:-1], 'base')]
    pairs.append(dict(hardware='SXM0/1', cohort=k[0], kind=k[1], mtp=k[2], requests=k[3],
                      setting=k[4], fraction=k[5], old_us=old['ours_us'], new_us=new['ours_us'],
                      reduction=1-new['ours_us']/old['ours_us'], old_raw=old['raw'], new_raw=new['raw'],
                      memcpy_us=new['memcpy_us'], local_us=new['local_us'],
                      reference_us=new['reference_us'], over_reference=new['over_reference'],
                      reference_hit_rate=new['cache_reference_hit_rate']))
gp = [r for r in g['pairs'] if r['cohort'] == 'gpu3_balanced_native_']
assert len(gp) == 40
pairs += [dict(r, hardware='GPU3') for r in gp]
assert len(pairs) == 102
unchanged = [r for r in pairs if r['mtp'] == 8 and r['requests'] == 28]
changed = [r for r in pairs if r not in unchanged]
assert len(unchanged) == 20
assert all(r['reduction'] >= -.01 for r in unchanged)
assert all(r['reduction'] > 0 for r in changed)
# Historical controls are applicable because all nine GPU kernels are identical and the
# candidate's host predicate is unchanged outside the medium, CTA_TILE_KV=32 domain.
controls = [r for r in g['pairs'] if r['cohort'] == 'gpu3_controls_']
assert len(controls) == 9 and all(r['reduction'] > -.01 for r in controls)
t = load(p / 'tests_balanced_cached_sm80_gpu3.json')
assert t['normal_build'] and t['complete'] and t['compile_exit_code'] == t['exit_code'] == 0
assert t['header_sha256'] == candidate_sha
assert t['test_source_sha256'] == sha(root / 'tests/attention/test_mla_host_batch.py')
assert t['log_sha256'] == sha(p / 'pytest_balanced_cached_sm80_gpu3.log')
assert '105 passed' in (p / 'pytest_balanced_cached_sm80_gpu3.log').read_text()
assert t['resources_start']['foreign'] == t['resources_end']['foreign'] == []
emu = load(p / 'compiler_summary.json')['comparisons'][variant]
assert sum(map(len, emu.values())) == 9
assert all(x['instruction_text_equal'] and x['resource_equal'] for group in emu.values() for x in group)
normal = load(p / 'compiler_normal_balanced_cached_sm80_summary.json')
assert normal['all_18_kernel_instruction_text_and_resources_equal']
assert normal['retained_header_sha256'] == candidate_sha and len(normal['comparisons']) == 18
memo = load(p / 'dispatch_memo_check.json')
assert memo['memo_equivalence_checks'] == 360
header = root / 'include/flashinfer/attention/mla_host_batch.cuh'
assert sha(header) in [base_sha, candidate_sha], 'Refuse to overwrite an unknown dirty header'
record = dict(timestamp=time.strftime('%FT%T%z'), goal_complete=False, adopted=False,
    variant=variant, base_header_sha256=base_sha, retained_header_sha256=candidate_sha,
    scope='Shape-only host dispatch; medium CTA_TILE_KV=32, staged, split-off, 64 heads, full staging. Same GPU math/LRU and caller parameters.',
    pairs=pairs, final_paired_cells=len(pairs), changed_path_paired_cells=len(changed),
    minimum_changed_reduction=min(r['reduction'] for r in changed),
    unchanged_path_paired_cells=len(unchanged),
    maximum_unchanged_slowdown=max(0, -min(r['reduction'] for r in unchanged)),
    unchanged_tolerance=.01, normal_tests=105, emulated_equal_kernels=9, normal_equal_kernels=18,
    memo=memo, gate_script_sha256=sha(Path(__file__)),
    remaining='Large-batch 100 GB/s gap of 32–34% is unchanged; no new H800 evidence.')
if args.install:
    evidence = p / 'retention_evidence'
    evidence.mkdir(exist_ok=True)
    record['evidence'] = {}
    for name in ['audit_general.json', 'gpu3/audit.json', 'compiler_summary.json',
                 'compiler_normal_balanced_cached_sm80_summary.json',
                 'tests_balanced_cached_sm80_gpu3.json', 'dispatch_memo_check.json']:
        source = p / name
        target = evidence / (sha(source) + '.json')
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            shutil.copyfile(source, target)
        record['evidence'][name] = dict(path=str(target.relative_to(root)), sha256=sha(target))
    (p / 'balanced_retention_gate.json').write_text(json.dumps(record, indent=2) + '\n')
    header.write_bytes(candidate.read_bytes())
    record['adopted'] = True
    record['production_header_sha256'] = sha(header)
    (p / 'balanced_retention.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: v for k, v in record.items() if k not in ['pairs', 'evidence']}, indent=2))
