"""Verify current progress without converting pending gates into completed evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
m = read(p / 'manifest.json')
assert sha(p / 'base.cuh') == sha(root / 'include/flashinfer/attention/mla_host_batch.cuh') == m['base_sha256']
assert m['base_sha256'] == 'a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
assert sha(p / 'sparse_prepare.cuh') == m['candidate_sha256'] == 'f998f8317eaf283d3c433f927aba959faf98034d576a0df9649bc35c713d37ec'
assert sha(root / 'include/flashinfer/attention/mla.cuh') == sha(p / 'sparse_prepare_mla.cuh') == m['math_sha256']
assert sha(root / 'include/flashinfer/attention/far_link_emu.cuh') == '5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root / 'temp/glm_trace/kernel_report_20261029/matrix.json') == 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
for v in ['base', 'sparse_prepare']:
    tree = p / ('include_' + v) / 'flashinfer/attention'
    if tree.exists():
        assert sha(tree / 'mla_host_batch.cuh') == sha(p / (v + '.cuh'))
        assert sha(tree / 'mla.cuh') == m['math_sha256']
reused = read(p / 'reused_cached_pe_control.json')
assert reused['complete'] and reused['additional_tests_counted'] == 0
for item in reused['proofs']:
    assert sha(root / item['path']) == item['sha256']
t = read(p / 'tests_sparse_prepare_gpu3.json')
assert t['complete'] and t['compile_exit_code'] == t['exit_code'] == 0
assert t['normal_build'] and t['header_sha256'] == m['candidate_sha256']
assert t['resources_start']['foreign'] == t['resources_end']['foreign'] == []
assert t['log_sha256'] == sha(p / 'pytest_sparse_prepare_gpu3.log')
assert '105 passed' in (p / 'pytest_sparse_prepare_gpu3.log').read_text()
assert 'COMPILED sparse_prepare' in (p / 'compile_sparse_prepare_emu.log').read_text()
assert 'COMPILED sparse_prepare' in (p / 'compile_sparse_prepare_normal.log').read_text()
pending = []
proofs = []
for suffix, count in [('', 38), ('_normal', 76)]:
    f = p / ('compiler_sparse_prepare' + suffix + '_summary.json')
    if not f.exists():
        pending.append('old_instance_identity' + (suffix or '_EMU'))
        continue
    d = read(f)
    assert d['complete'] and d['unchanged_original_kernels'] == count and d['all_original_instruction_and_control_words_identical']
    assert d['candidate_header_sha256'] == m['candidate_sha256']
    for v, libs in d['libraries'].items():
        folder = p.parent / 'tune_cached_pe_20261101' / ('jit_cached_pe' + suffix) if v == 'base' else p / ('jit_' + v + suffix)
        for lib in libs:
            paths = list(folder.rglob(lib['name']))
            assert len(paths) == 1 and sha(paths[0]) == lib['sha256']
    proofs.append(dict(file=f.name, sha256=sha(f), old_instances=count))
f = p / 'stress_sparse_prepare_gpu3.json'
stress = read(f) if f.exists() else {}
tests = 105
if stress.get('complete'):
    assert stress['tests'] == 10 and stress['exit_code'] == 0
    assert stress['stress_source_sha256'] == sha(p / 'test_cofetch_progress.py')
    assert stress['resources_start']['foreign'] == stress['resources_end']['foreign'] == []
    assert stress['log_sha256'] == sha(p / 'pytest_stress_sparse_prepare.log')
    assert '10 passed' in (p / 'pytest_stress_sparse_prepare.log').read_text()
    tests += 10
else:
    pending.append('10_stress_cases')
f = p / 'dispatch_probe_sparse_prepare.json'
if f.exists():
    d = read(f)
    assert d['kernel_nodes'] == 1 and d['header_sha256'] == m['candidate_sha256']
    launch = d['launches'][0]
    assert launch['grid'] == [1, 108, 1] and launch['block'] == [32, 4, 2]
    assert launch['dynamic_shared_bytes'] == 147968 and launch['active_blocks_per_SM'] == 1
    assert 'BatchMLASparsePrepareKernel' in launch['kernel_name']
else:
    pending.append('actual_new_Graph')
for kind in ['synccheck', 'racecheck', 'memcheck']:
    f = p / (kind + '_sparse_prepare.log')
    log = f.read_text() if f.exists() else ''
    if 'ERROR SUMMARY: 0 errors' not in log and 'RACECHECK SUMMARY: 0 hazards' not in log:
        pending.append(kind)
model = read(p / 'sparse_job_review.json')
assert model['complete'] and model['case_count'] == 72 and model['cpu_only']
assert model['source_sha256'] == sha(p / 'review_sparse_jobs.py')
assert len(read(p / 'source_review.json')['math_helpers_exactly_preserved']) == 8
pre = read(p / 'precompile_review.json')
assert not pre['old_compiled'] and sha(p / 'precompile_review/sparse_prepare_initial.cuh') == pre['old_header_sha256']
a = read(p / 'performance_audit.json') if (p / 'performance_audit.json').exists() else {}
if a:
    assert not a['incomplete'] and not a['pending_references']
    for item in a['files']:
        assert sha(root / item['path']) == item['sha256']
else:
    pending.append('fixed_short_performance_screen')
live = []
for name, script in [('validation_state.json', 'validate_candidate.py'), ('confirmation_state.json', 'confirm_if_promising.py')]:
    s = read(p / name)
    if not s['complete'] and s['phase'] != 'failed':
        ps = subprocess.run(['ps', '-p', str(s['pid']), '-o', 'pid,ppid,etimes,args'], text=True, capture_output=True)
        assert ps.returncode == 0 and ('tune_sparse_prepare_20261101/' + script) in ps.stdout
        live.append(dict(state=name, pid=s['pid'], phase=s['phase'], ps=ps.stdout.splitlines()[1:]))
result = dict(checkpoint_verified=True, timestamp=time.strftime('%FT%T%z'), goal_complete=False, candidate_adopted=False,
    production_unchanged_CPEa436=True, candidate_header_sha256=m['candidate_sha256'],
    tests_passed=tests, proofs=proofs, pending_core_gates=pending, core_validation_complete=not pending,
    performance_rows=len(a.get('rows', [])), pairs=len(a.get('pairs', [])), live_own_producers=live,
    cpu_structural_cases=72, broad_regression_complete=False,
    limitation='Progress checkpoint only. No adoption from105tests, CPUoracle, resources or a short screen. Need all normal/native/SXM/fulltrace regression gates before retention.')
confirmation = read(p / 'confirmation_state.json')
if confirmation['phase'].startswith('rejected'):
    assert confirmation['complete'] and not live
    assert tests == 115 and not pending
    assert len(a['rows']) == 6 and len(a['pairs']) == 3
    assert a['timed_launches'] == 1152 and a['accepted_segments'] == 72
    assert any(r['reduction'] < 0 for r in a['pairs'] if r['setting'].startswith('g100'))
    result.update(experiment_complete=True, outcome='Rejected short screen, no full/native/SXM expansion.',
                  limitation='Completed rejected experiment only; broad regression not run. Admission order changes actualunique traffic, so50GB speedup is not isolated pass-fusion benefit. Goalactive.')
(p / 'checkpoint_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print('PROGRESS_CHECKPOINT_VERIFIED', tests, pending)
