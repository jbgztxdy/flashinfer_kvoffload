"""Verify the rejected candidate, original-kernel identity, diagnostics and all raw provenance."""
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
assert sha(p / 'base.cuh') == m['base_sha256']
assert sha(p / 'joint_qk.cuh') == m['joint_qk_control_sha256']
assert sha(p / 'joint_pv.cuh') == m['candidate_sha256']
assert sha(root / 'include/flashinfer/attention/mla.cuh') == m['math_sha256']
assert sha(root / 'include/flashinfer/attention/far_link_emu.cuh') == '5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root / 'temp/glm_trace/kernel_report_20261029/matrix.json') == 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
production = sha(root / 'include/flashinfer/attention/mla_host_batch.cuh')
assert production != m['candidate_sha256']
if production != m['base_sha256']:
    retention = read(p.parent / 'tune_joint_qk_20261101/joint_qk_retention.json')
    assert retention['candidate_adopted'] and retention['candidate_header_sha256'] == production
for variant in ['base', 'joint_qk', 'joint_pv']:
    assert sha(p / (variant + '_mla.cuh')) == m['math_sha256']
for suffix, count in [('', 34), ('_normal', 68)]:
    proof = read(p / ('compiler_joint_pv' + suffix + '_summary.json'))
    assert proof['complete'] and proof['unchanged_original_kernels'] == count
    assert proof['base_header_sha256'] == m['base_sha256']
    assert proof['candidate_header_sha256'] == m['candidate_sha256']
    assert proof['math_header_sha256'] == m['math_sha256']
    assert proof['all_new_stack_and_local_zero']
    assert all('REG:252 STACK:0 SHARED:2304 LOCAL:0 ' in r['resource'] for r in proof['new_kernels'])
words = read(p / 'machine_word_identity.json')
assert words['complete'] and [b['original_kernel_instances'] for b in words['builds']] == [34, 68]
assert all(b['all_words_identical'] for b in words['builds'])
for tag in ['', 'synccheck_', 'racecheck_']:
    probe = read(p / ('dispatch_probe_' + tag + 'joint_pv.json'))
    assert probe['header_sha256'] == m['candidate_sha256'] and probe['math_header_sha256'] == m['math_sha256']
    assert probe['kernel_nodes'] == probe['graph_nodes'] == 1 and probe['max_abs_error'] < .01
    launch = probe['launches'][0]
    assert launch['grid'] == [1, 108, 1] and launch['block'] == [32, 4, 2]
    assert launch['dynamic_shared_bytes'] == 147968 and launch['active_blocks_per_SM'] == 1
    assert re.search('BatchMLAJointPVKernel', launch['kernel_name'])
sanitizer = read(p / 'sanitizer_audit.json')
assert sanitizer['complete'] and sanitizer['header_sha256'] == m['candidate_sha256']
for kind in ['synccheck', 'racecheck']:
    record = sanitizer[kind]
    assert sha(p / record['log']) == record['sha256'] and record['errors'] == 0
    assert 'SUMMARY: 0' in (p / record['log']).read_text()
for record in read(p / 'reused_joint_control.json')['records']:
    assert sha(root / record['path']) == record['sha256']
a = read(p / 'performance_audit.json')
assert len(a['rows']) == 9 and len(a['pairs']) == 6
assert not a['pending_references'] and not a['incomplete']
assert a['timed_launches'] == 1728 and a['accepted_segments'] == 108
assert sum(t['tests'] for t in a['tests']) + sum(t['tests'] for t in a['stress_tests']) == 109
decision = read(p / 'decisions.json')
assert not decision['candidate_adopted'] and decision['audit_sha256'] == sha(p / 'performance_audit.json')
assert len(decision['triples']) == 3
assert all(t['reduction_vs_joint_qk'] < 0 for t in decision['triples'] if t['setting'].startswith('g100'))
s = read(p / 'continuation_state.json')
assert s['goal_status'] == 'active' and not s['active_runners'] and not s['candidate_adopted']
assert all(s['completed_handles'][str(handle)] == code for handle, code in
           [(50732, 0), (77642, 0), (88832, 1), (47636, 143), (96800, 0), (53029, 0), (60928, 0), (59627, 0)])
snapshots = count_json = 0
names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '--',
    str(p.relative_to(root))], cwd=root, text=True).splitlines()
for name in set(names):
    f = root / name
    if f.suffix == '.json':
        read(f)
        count_json += 1
    if f.parent.name == 'source_snapshots' and len(f.stem) == 64:
        assert sha(f) == f.stem
        snapshots += 1
for f in p.glob('*.py'):
    compile(f.read_text(), str(f), 'exec')
result = dict(complete=True, goal_status='active', candidate_adopted=False, tests=109,
    rows=9, base_pairs=6, threeway_points=3, timed_launches=1728, accepted_segments=108,
    verified_source_snapshots=snapshots, parsed_json=count_json, production_header_sha256=production,
    candidate_header_sha256=m['candidate_sha256'], canonical_math_sha256=m['math_sha256'])
(p / 'checkpoint_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
