from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'poolbounds.cuh').read_text()
a='for(uint32_t row=0;row<params.num_rows;++row) {\n   if(params.row_pool[row]%gridDim.y!=blockIdx.y)continue;'
b='for(uint32_t row=blockIdx.y;row<params.num_rows;row+=gridDim.y) {'
assert s.count(a)==1;s=s.replace(a,b)
assert not (p/'poolrows.cuh').exists();(p/'poolrows.cuh').write_text(s);shutil.copytree(p/'include_poolbounds',p/'include_poolrows');(p/'include_poolrows/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_poolrows.py').write_text((p/f'{n}_poolbounds.py').read_text().replace('poolbounds','poolrows'))
(p/'poolrows_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'base_sha256':hashlib.sha256((p/'poolbounds.cuh').read_bytes()).hexdigest(),'mechanism':'pool-owned unique packets unchanged; restore balanced round-robin query rows; isolates indivisible compute pool tail'},indent=2))
