"""Verify private-math equivalence, actual two-CTA dispatch and all negative controls."""
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
private = 'e5968e7605bb657de30e5600c4a8fc09b977a3e2e7163f64e46718de519289a3'
matrix = 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh') == production
assert sha(root/'include/flashinfer/attention/mla.cuh') == canonical
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json') == matrix
assert not read(p/'continuation_state.json')['active_runners']
assert not read(p/'decisions.json')['candidate_adopted']
audit = read(p/'performance_audit.json')
assert len(audit['rows']) == 18 and len(audit['pairs']) == 12
assert audit['timed_launches'] == 3456 and audit['accepted_segments'] == 216
assert not audit['pending_references'] and not audit['incomplete']
assert sum(t['tests'] for t in audit['tests']) == 210
assert sum(t['tests'] for t in audit['stress_tests']) == 8
assert read(p/'decisions.json')['audit_sha256'] == sha(p/'performance_audit.json')
for variant, hmma, ldsm, registers, enabled in [
    ('dim_split',207,156,231,True), ('duplicate_control',315,264,255,False)]:
    host_sha = sha(p/(variant+'.cuh'))
    assert sha(p/(variant+'_mla.cuh')) == private
    for suffix, originals in [('',34),('_normal',68)]:
        proof = read(p/('compiler_'+variant+suffix+'_summary.json'))
        assert proof['complete'] and proof['unchanged_original_kernels'] == originals
        assert proof['base_header_sha256'] == production
        assert proof['candidate_header_sha256'] == host_sha and proof['math_header_sha256'] == private
        assert proof['all_new_stack_and_local_zero']
        assert all(f'REG:{registers} ' in r['resource'] and 'SHARED:2176 ' in r['resource']
                   for r in proof['new_kernels'])
        if not suffix:
            entry = proof['new_kernels'][0]
            text = gzip.open(p/(variant+'_'+entry['library'].removesuffix('.so')+'_sass.txt.gz'),'rt').read()
            body = next(s for s in re.split(r'\n\s*Function : ',text)[1:]
                        if s.splitlines()[0].strip() == entry['symbol'])
            ops = re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*',body)
            assert sum('HMMA.' in x for x in ops) == hmma
            assert sum('LDSM.' in x for x in ops) == ldsm
    probe = read(p/('dispatch_probe_'+variant+'.json'))
    assert probe['header_sha256'] == host_sha and probe['math_header_sha256'] == private
    assert probe['graph_nodes'] == probe['kernel_nodes'] == 1
    assert probe['qk_feature_partitioned'] is enabled
    config = probe['launches'][0]
    assert config['grid'] == [1,216,1] and config['block'] == [32,2,2]
    assert config['dynamic_shared_bytes'] == 78080 and config['active_blocks_per_SM'] == 2
names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard',
    '--',str(p.relative_to(root))],cwd=root,text=True).splitlines()
snapshots = 0
for name in set(names):
    f = root/name
    if f.suffix == '.json': read(f)
    if f.parent.name == 'source_snapshots' and len(f.stem) == 64:
        assert sha(f) == f.stem
        snapshots += 1
for f in p.glob('*.py'): compile(f.read_text(),str(f),'exec')
result = dict(complete=True,goal_status='active',candidate_adopted=False,normal_tests=218,
    performance_rows=18,paired_comparisons=12,timed_launches=3456,accepted_segments=216,
    verified_source_snapshots=snapshots,production_header_sha256=production,
    canonical_math_sha256=canonical,private_math_sha256=private,original_matrix_sha256=matrix)
(p/'checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
