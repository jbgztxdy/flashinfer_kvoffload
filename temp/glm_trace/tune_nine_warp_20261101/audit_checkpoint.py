"""Check both implementation schemas without treating candidate rejection as goal completion."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
m = read(p / 'manifest.json')
v = m.get('variant', 'nine_warp')
assert sha(p / 'base.cuh') == m['base_sha256'] == 'a2f37fb985fcbc523e740c85e9cc6468afad74fd8716c11aef4221ebab7c4b7c'
assert sha(p / (v + '.cuh')) == m['candidate_sha256']
assert sha(p / (v + '_mla.cuh')) == m['math_sha256'] == sha(root / 'include/flashinfer/attention/mla.cuh')
assert m['math_sha256'] == '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(root / 'include/flashinfer/attention/far_link_emu.cuh') == '5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root / 'temp/glm_trace/kernel_report_20261029/matrix.json') == 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
production = sha(root / 'include/flashinfer/attention/mla_host_batch.cuh')
assert production != m['candidate_sha256']
if production != 'b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a':
    retention = read(p.parent / 'tune_joint_qk_20261101/joint_qk_retention.json')
    assert retention['candidate_adopted'] and retention['candidate_header_sha256'] == production
for suffix, count in [('', 36), ('_normal', 72)]:
    proof = read(p / ('compiler_' + v + suffix + '_summary.json'))
    assert proof['complete'] and proof['unchanged_original_kernels'] == count
    assert proof['base_header_sha256'] == m['base_sha256'] and proof['candidate_header_sha256'] == m['candidate_sha256']
    assert proof['math_header_sha256'] == m['math_sha256'] and not proof['all_new_stack_and_local_zero']
    assert all('REG:168 STACK:248 SHARED:1664 LOCAL:0 ' in r['resource'] for r in proof['new_kernels'])
words = read(p / 'machine_word_identity.json')
assert words['complete'] and [r['original_kernel_instances'] for r in words['builds']] == [36, 72]
assert all(r['all_words_identical'] for r in words['builds'])
kernel = 'BatchMLANineWarpKernel' if v == 'nine_warp' else 'BatchMLANineStaticWarpKernel'
for tag in ['', 'synccheck_', 'racecheck_']:
    probe = read(p / ('dispatch_probe_' + tag + v + '.json'))
    assert probe['header_sha256'] == m['candidate_sha256'] and probe['math_header_sha256'] == m['math_sha256']
    assert probe['graph_nodes'] == probe['kernel_nodes'] == 1 and probe['max_abs_error'] < .01
    launch = probe['launches'][0]
    assert launch['grid'] == [1, 108, 1] and launch['block'] == [32, 9, 1]
    assert launch['dynamic_shared_bytes'] == 158336 and launch['active_blocks_per_SM'] == 1
    assert kernel in launch['kernel_name']
for kind in ['synccheck', 'racecheck']:
    assert 'SUMMARY: 0' in (p / (kind + '_' + v + '.log')).read_text()
for record in read(p / 'reused_joint_control.json')['records']:
    assert sha(root / record['path']) == record['sha256']
state = read(p / 'validation_state.json')
assert state['complete'] and all(s['exit_code'] == 0 for s in state['steps'])
a = read(p / 'performance_audit.json')
cells, pairs, launches, segments = (10, 5, 1920, 120) if v == 'nine_warp' else (6, 3, 1152, 72)
assert len(a['rows']) == cells and len(a['pairs']) == pairs
assert a['timed_launches'] == launches and a['accepted_segments'] == segments
assert not a['pending_references'] and not a['incomplete']
assert all(r['reduction'] < 0 for r in a['pairs'])
assert sum(t['tests'] for t in a['tests']) + sum(t['tests'] for t in a['stress_tests']) == 109
decision = read(p / 'decisions.json')
assert not decision['candidate_adopted'] and decision['audit_sha256'] == sha(p / 'performance_audit.json')
s = read(p / 'continuation_state.json')
assert s['goal_status'] == 'active' and not s['candidate_adopted'] and not s['active_runners']
snapshots = jsons = 0
names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '--',
    str(p.relative_to(root))], cwd=root, text=True).splitlines()
for name in set(names):
    f = root / name
    if f.suffix == '.json': read(f); jsons += 1
    if f.parent.name == 'source_snapshots' and len(f.stem) == 64:
        assert sha(f) == f.stem
        snapshots += 1
for f in p.glob('*.py'): compile(f.read_text(), str(f), 'exec')
result = dict(complete=True, goal_status='active', candidate_adopted=False, variant=v, tests=109,
    cells=cells, pairs=pairs, timed_launches=launches, accepted_segments=segments,
    parsed_json=jsons, verified_source_snapshots=snapshots, production_header_sha256=production,
    candidate_header_sha256=m['candidate_sha256'], comparison_base_sha256=m['base_sha256'],
    canonical_math_sha256=m['math_sha256'])
(p / 'checkpoint_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
