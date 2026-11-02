"""Retire a fetch cohort early while a smaller cohort drains the remaining chunks."""
import difflib,hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda data:hashlib.sha256(data).hexdigest()
b=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()
assert sha(b)=='754706f8d8fba14a361ce6dcfc64a358446e5df36b225f392fa8a3e7756ded79'
assert not(p/'elastic_fetch.cuh').exists()
s=b.decode();math=(root/'include/flashinfer/attention/mla.cuh').read_bytes()
start=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits')
end=s.index('// partial_capacity bounds',start)
section=s[start:end].replace('StreamSource','ElasticFetch').replace('stream_source_','elastic_fetch_')
fstart=s.rindex('template <typename KTraits, typename Params>',0,s.index('__device__ __forceinline__ void fetch_stream_source_pages('))
fetch=s[fstart:start].replace('fetch_stream_source_pages','fetch_elastic_fetch_pages')
old='''  uint32_t chunk = fetch_id;
  if (chunk * rows < count) issue(chunk, 0);
#pragma unroll 1
  for (uint32_t k = 0; chunk * rows < count; ++k, chunk += params.fetch_ctas) {
    const uint32_t next = chunk + params.fetch_ctas;'''
new='''  const uint32_t jobs = ceil_div(count, rows);
  const uint32_t retirees = params.fetch_ctas / 3;
  const uint32_t persistent = params.fetch_ctas - retirees;
  // All fetch CTAs cover the first third, then the remaining cohort covers the tail.
  // The boundary is a whole initial-grid round, so every job has exactly one owner.
  const uint32_t boundary = min(jobs, round_up(ceil_div(jobs, 3u), params.fetch_ctas));
  auto next_chunk = [&](uint32_t at) {
    if (at >= boundary) return at + persistent;
    const uint32_t next = at + params.fetch_ctas;
    if (next < boundary) return next;
    return fetch_id < retirees ? jobs : boundary + fetch_id - retirees;
  };
  uint32_t chunk = fetch_id;
  if (chunk * rows < count) issue(chunk, 0);
#pragma unroll 1
  for (uint32_t k = 0; chunk * rows < count; ++k, chunk = next_chunk(chunk)) {
    const uint32_t next = next_chunk(chunk);'''
assert fetch.count(old)==1;fetch=fetch.replace(old,new)
at=s.index('    using StreamSourceKernelTraits =');dend=s.index('    using CachedPEKernelTraits =',at)
dispatch=s[at:dend].replace('StreamSource','ElasticFetch').replace('stream_source','elastic_fetch')
s=s[:end]+fetch+section+s[end:]
at=s.index('    using StreamSourceKernelTraits =');s=s[:at]+dispatch+s[at:]
for v,data in [('base',b),('elastic_fetch',s.encode())]:
 (p/(v+'.cuh')).write_bytes(data);(p/(v+'_mla.cuh')).write_bytes(math)
 tree=p/('include_'+v);shutil.copytree(root/'include',tree)
 (tree/'flashinfer/attention/mla_host_batch.cuh').write_bytes(data)
old=p.parent/'tune_qk_cursor_20261102'
for name in ['bench_gpu3_native.py','resources_gpu3.py','experiment.json','plans.cpp','compile.py','audit_performance.py']:
 shutil.copy2(old/name,p/name)
(p/'candidatepaths.py').write_text((old/'candidatepaths.py').read_text().replace('qk_cursor','elastic_fetch'))
test=(p.parent/'tune_global_query_20261102/test_global_query.py').read_text().replace('global_query','elastic_fetch')
(p/'test_elastic_fetch.py').write_text(test)
for name in ['run_focused.py','build_candidate.py']:
 (p/name).write_text((p.parent/'tune_global_query_20261102'/name).read_text().replace('global_query','elastic_fetch'))
(p/'candidate.diff').write_text(''.join(difflib.unified_diff(b.decode().splitlines(True),s.splitlines(True),fromfile='base.cuh',tofile='elastic_fetch.cuh')))
# CPU coverage proof for the queue transition, including short/tail rounds and small grids.
cases=0
for f in [1,2,3,8,16,24,32,107]:
 for jobs in list(range(100))+[127,128,255,256,1023,4096,32768]:
  boundary=min(jobs,((jobs+2)//3+f-1)//f*f);retirees=f//3;persistent=f-retirees
  visits=[]
  for fetch_id in range(f):
   at=fetch_id
   while at<jobs:
    visits.append(at)
    if at>=boundary:at+=persistent
    elif at+f<boundary:at+=f
    else:at=jobs if fetch_id<retirees else boundary+fetch_id-retirees
  assert sorted(visits)==list(range(jobs)),(f,jobs)
  cases+=1
record=dict(base_sha256=sha(b),candidate_sha256=sha(s.encode()),math_sha256=sha(math),cpu_queue_cases=cases,
 mechanism='Elastic dedicated fetch cohort: all f CTAs cover first ceil(jobs/3) rounded to f; first f/3 CTAs then retire into existing attention work walk, remaining CTAs drain tail. Same64row/double buffer and source/cache/ready ordering.',
 invariants='One cooperative kernel; exact original attention/QK/PV; original prepare/resolve/LRU/capacity/bytes/FAR hooks. No worker cofetch or extra buffer, no simulator/reference change.',adopted=False)
(p/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
