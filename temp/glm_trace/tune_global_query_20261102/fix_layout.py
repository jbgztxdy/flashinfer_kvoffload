"""Preserve the pre-GPU draft and correct its native64KV fragment coordinates."""
import difflib,hashlib,json,shutil,time
from pathlib import Path
p=Path(__file__).resolve().parent
archive=p/'pre_gpu_layout';archive.mkdir()
for name in ['global_query.cuh','candidate.diff','manifest.json','build_global_query.json','compile_global_query_emu.log','compile_global_query_normal.log']:
 shutil.copy2(p/name,archive/name)
for name in ['jit_global_query','jit_global_query_normal']:
 (p/name).rename(archive/name)
s=(p/'global_query.cuh').read_text()
assert 'group*16+sub*32' in s
s=s.replace('group*16+sub*32','group*32+sub*16')
s=s.replace('group*16+8*(lane/16)+lane%8,(lane%16)/8);\n  const uint32_t other=32*','group*32+8*(lane/16)+lane%8,(lane%16)/8);\n  const uint32_t other=16*')
assert 'const uint32_t other=16*' in s
(p/'global_query.cuh').write_text(s)
(p/'include_global_query/flashinfer/attention/mla_host_batch.cuh').write_text(s)
manifest=json.loads((p/'manifest.json').read_text());manifest['candidate_sha256']=hashlib.sha256(s.encode()).hexdigest()
manifest['pre_gpu_layout_fix']='Original draft interleaved warpgroup/subfragment coordinates. Fixed to native contiguous32KV per warpgroup before any GPU run. Old CPU builds archived.'
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(p/'candidate.diff').write_text(''.join(difflib.unified_diff((p/'base.cuh').read_text().splitlines(True),s.splitlines(True),fromfile='base.cuh',tofile='global_query.cuh')))
(archive/'state.json').write_text(json.dumps(dict(timestamp=time.strftime('%FT%T%z'),gpu_runs=0,reason=manifest['pre_gpu_layout_fix']),indent=2)+'\n')
print(manifest['candidate_sha256'])
