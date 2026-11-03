from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'flat_payload.cuh').read_text()
a='#pragma unroll\n for(uint32_t v=t;v<20*72;v+=256)';assert s.count(a)==2;s=s.replace(a,'#pragma unroll 1\n for(uint32_t v=t;v<20*72;v+=256)')
assert not(p/'flat_payload_loop.cuh').exists();(p/'flat_payload_loop.cuh').write_text(s)
shutil.copytree(p/'include_clean',p/'include_flat_payload_loop');(p/'include_flat_payload_loop/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_flat_payload_loop.py').write_text((p/f'{n}_flat_payload.py').read_text().replace('flat_payload','flat_payload_loop'))
(p/'flat_payload_loop_source.json').write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'flat_payload.cuh').read_bytes()).hexdigest(),mechanism='same flattened vector partition and protocol; keep both payload loops rolled to avoid six duplicated address/conditional paths'),indent=2)+'\n')
