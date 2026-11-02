"""Read-only focused checkpoint; no additional GPU tests."""
import hashlib,json,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
r=json.loads((p/'retention.json').read_text())
assert r['candidate_adopted'] and sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==r['production_sha256']==sha(p/'qk_cursor.cuh')
for e in r['evidence']:assert sha(p/e['file'])==e['sha256']==sha(p/e['snapshot'])
tests=0;rows=pairs=launches=segments=0
for v,n in [('cache_store',3),('query_copy',4),('qk_cursor',4)]:
 d=p.parent/f'tune_{v}_20261102';state=json.loads((d/'focused_state.json').read_text());a=json.loads((d/'performance_audit.json').read_text());assert state['complete']
 assert f'{n} passed' in (d/'targeted_correctness.log').read_text();assert not a['incomplete'] and not a['pending_references']
 for x in a['rows']:
  f=root/x['raw'];raw=json.loads(f.read_text());assert raw['source_sha256']['include/flashinfer/attention/mla_host_batch.cuh']==sha(d/(x['variant']+'.cuh'))
 tests+=n;rows+=len(a['rows']);pairs+=len(a['pairs']);launches+=a['timed_launches'];segments+=a['accepted_segments']
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')=='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
a=dict(timestamp=time.strftime('%FT%T%z'),complete=True,production_sha256=r['production_sha256'],new_targeted_tests=tests,rows=rows,pairs=pairs,timed_launches=launches,accepted_segments=segments,overall_goal_complete=False,active_runners=[],note='Scope-limited evidence; no fresh root build/FP16/SXM/H800/sanitizer/fullmatrix claim.')
(p/'checkpoint.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a,indent=2))
