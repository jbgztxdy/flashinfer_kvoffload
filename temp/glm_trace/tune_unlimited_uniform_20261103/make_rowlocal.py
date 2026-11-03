from pathlib import Path
import shutil,hashlib,json
p=Path(__file__).resolve().parent
s=(p/'contiguous.cuh').read_text()
a='for(uint32_t row=blockIdx.y;row<params.num_rows;row+=gridDim.y) {'
b='for(uint32_t row=uint64_t(params.num_rows)*blockIdx.y/gridDim.y;row<uint64_t(params.num_rows)*(blockIdx.y+1)/gridDim.y;++row) {'
assert s.count(a)==1
s=s.replace(a,b)
assert not (p/'rowlocal.cuh').exists()
(p/'rowlocal.cuh').write_text(s)
shutil.copytree(p/'include_contiguous',p/'include_rowlocal')
(p/'include_rowlocal/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for name in ['test','bench','run']:
 (p/f'{name}_rowlocal.py').write_text((p/f'{name}_contiguous.py').read_text().replace('contiguous','rowlocal'))
(p/'rowlocal_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'base':hashlib.sha256((p/'contiguous.cuh').read_bytes()).hexdigest(),'mechanism':'contiguous row compute aligned approximately with row-prefix unique ownership; no reordered KV arithmetic'},indent=2))
