from pathlib import Path
import shutil,re,hashlib,json
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
a=s.index('struct CleanService {');b=s.index('// partial_capacity bounds',a)
section=s[a:b]
assert section.count('20')>=15
section=re.sub(r'(?<![A-Za-z0-9_])20u?(?=$|[^A-Za-z0-9_])','kCleanServiceRows',section)
assert '20' not in section
s=s[:a]+'constexpr uint32_t kCleanServiceRows = 22;'+chr(10)+section+s[b:]
a='static_assert(clean_service_smem == 163072);';b='static_assert(clean_service_smem == 165376);'
assert s.count(a)==1;s=s.replace(a,b)
a='ceil_div(params.staging_capacity,20u)+128*16ull';b='ceil_div(params.staging_capacity,kCleanServiceRows)+128*16ull'
assert s.count(a)==1;s=s.replace(a,b)
name='service22';assert not (p/(name+'.cuh')).exists()
(p/(name+'.cuh')).write_text(s)
shutil.copytree(p/'include_clean',p/('include_'+name))
(p/('include_'+name)/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
for kind in ['test','bench','run']:
    (p/(kind+'_'+name+'.py')).write_text((p/(kind+'_clean.py')).read_text().replace('clean',name))
f=p/('bench_'+name+'.py');t=f.read_text().replace("service_rows=20 if VARIANT in", "service_rows=22 if VARIANT in")
f.write_text(t)
(p/(name+'_source.json')).write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'clean.cuh').read_bytes()).hexdigest(),mechanism='max feasible22page shared packet on A100 with8 mathwarps, exact same roundrobin ownership/QK/PV/LRU/stage+cache stores; no additional planner; expected dynamic165376/static1152/total166528 below known166912 block cap'),indent=2)+chr(10))
print(name,hashlib.sha256(s.encode()).hexdigest())
