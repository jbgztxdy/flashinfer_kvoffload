"""Check the frozen rescale controls, numerical evidence and unchanged production."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

p = Path(__file__).resolve().parent
root = p.parents[2]
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
read = lambda f: json.loads(f.read_text())
production = 'b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
canonical = '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
matrix = 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh') == production
assert sha(root/'include/flashinfer/attention/mla.cuh') == canonical
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json') == matrix
assert not read(p/'continuation_state.json')['active_runners']
assert not read(p/'decisions.json')['candidate_adopted']
a = read(p/'performance_audit.json')
assert len(a['rows']) == 18 and len(a['pairs']) == 9
assert a['timed_launches'] == 3456 and a['accepted_segments'] == 216
assert not a['pending_references'] and not a['incomplete']
assert sum(t['tests'] for t in a['tests']) == 315
assert sum(t['tests'] for t in a['stress_tests']) == 12
assert read(p/'decisions.json')['audit_sha256'] == sha(p/'performance_audit.json')
for v, registers in [('unit_scale',244),('stable_scale',242),('early_stable',242)]:
    host, math = [sha(p/(v+suffix+'.cuh')) for suffix in ['', '_mla']]
    for suffix, originals in [('',34),('_normal',68)]:
        proof = read(p/('compiler_'+v+suffix+'_summary.json'))
        assert proof['complete'] and proof['unchanged_original_kernels'] == originals
        assert proof['base_header_sha256'] == production
        assert proof['candidate_header_sha256'] == host and proof['math_header_sha256'] == math
        assert proof['all_new_stack_and_local_zero']
        assert all(f'REG:{registers} ' in r['resource'] and 'SHARED:2304 ' in r['resource'] for r in proof['new_kernels'])
    probe = read(p/('dispatch_probe_'+v+'.json'))
    assert probe['header_sha256'] == host and probe['math_header_sha256'] == math
    assert probe['graph_nodes'] == probe['kernel_nodes'] == 1
    c = probe['launches'][0]
    assert c['grid'] == [1,108,1] and c['block'] == [32,4,2]
    assert c['dynamic_shared_bytes'] == 147968 and c['active_blocks_per_SM'] == 1
    assert probe['max_abs_error'] < .01
probe = read(p/'scale_roundoff.json')
assert probe['complete'] and not probe['performance_claim'] and not probe['reference_claim']
assert probe['resources_start']['foreign'] == probe['resources_end']['foreign'] == []
assert probe['source_sha256'] == sha(p/'probe_scale_roundoff.cu')
assert probe['values']['count'] == probe['values']['equal_max_count'] == 65536
assert probe['values']['nonunit_scale_count'] == 26440
trace = read(p/'scale_expression_sass_trace.json')
proof = read(p/'compiler_unit_scale_summary.json')
entry = next(x for x in proof['new_kernels'] if x['symbol'] == trace['primary_kernel'])
sass = gzip.open(p/('unit_scale_'+entry['library'].removesuffix('.so')+'_sass.txt.gz'),'rt').read()
body = next(s for s in re.split(r'\n\s*Function : ',sass)[1:] if s.splitlines()[0].strip() == entry['symbol'])
for x in trace['trace']:
    assert re.search(r'/\*'+x['address']+r'\*/\s+'+re.escape(x['instruction'])+r'\s*;',body)
names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--',str(p.relative_to(root))],cwd=root,text=True).splitlines()
snapshots = 0
for name in set(names):
    f = root/name
    if f.suffix == '.json': read(f)
    if f.parent.name == 'source_snapshots' and len(f.stem) == 64:
        assert sha(f) == f.stem
        snapshots += 1
for f in p.glob('*.py'): compile(f.read_text(),str(f),'exec')
result = dict(complete=True,goal_status='active',candidate_adopted=False,normal_tests=327,
    performance_rows=18,paired_comparisons=9,timed_launches=3456,accepted_segments=216,
    verified_source_snapshots=snapshots,production_header_sha256=production,
    canonical_math_sha256=canonical,original_matrix_sha256=matrix)
(p/'checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
