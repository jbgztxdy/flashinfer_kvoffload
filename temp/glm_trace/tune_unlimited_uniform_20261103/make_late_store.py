from pathlib import Path
import shutil,hashlib,json
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
a='    if constexpr(ACTIVE)clean_service_stores<KTraits>(params,storage);\n    clean_service_compute_qk_pair<KTraits>(storage, hi%2, lo%2, qpe, scores, scores+1);'
b='    clean_service_compute_qk_pair<KTraits>(storage, hi%2, lo%2, qpe, scores, scores+1);\n    if constexpr(ACTIVE)clean_service_stores<KTraits>(params,storage);'
assert s.count(a)==1;s=s.replace(a,b)
assert not (p/'late_store.cuh').exists();(p/'late_store.cuh').write_text(s);shutil.copytree(p/'include_clean',p/'include_late_store');(p/'include_late_store/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for name in ['test','bench','run']:(p/f'{name}_late_store.py').write_text((p/f'{name}_clean.py').read_text().replace('clean','late_store'))
f=p/'bench_late_store.py';f.write_text(f.read_text().replace('latency_ns=0,emulated=True','latency_ns=0,emulated=False'))
(p/'late_store_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'move real stage/cache stores after QK before softmax; same release barrier; remote shared read complete before refill'},indent=2))
