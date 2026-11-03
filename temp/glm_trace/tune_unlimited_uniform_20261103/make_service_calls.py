from pathlib import Path
import shutil,hashlib,json
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
for fn in ['clean_service_issue','clean_service_stores']:
 a='__device__ __forceinline__ void '+fn;b='__device__ __noinline__ void '+fn
 assert s.count(a)==1;s=s.replace(a,b)
assert not (p/'service_calls.cuh').exists();(p/'service_calls.cuh').write_text(s);shutil.copytree(p/'include_clean',p/'include_service_calls');(p/'include_service_calls/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for name in ['test','bench','run']:(p/f'{name}_service_calls.py').write_text((p/f'{name}_clean.py').read_text().replace('clean','service_calls'))
f=p/'bench_service_calls.py';f.write_text(f.read_text().replace('latency_ns=0,emulated=True','latency_ns=0,emulated=False'))
(p/'service_calls_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'base_sha256':hashlib.sha256((p/'clean.cuh').read_bytes()).hexdigest(),'mechanism':'outline actual payload issue/stores only; no arithmetic, ownership or memory order changes'},indent=2))
