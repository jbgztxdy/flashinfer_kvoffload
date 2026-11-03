"""Prepare an unbuilt handoff fix; do not start GPU or compile work during migration."""
import difflib,hashlib,json,shutil,time
from pathlib import Path
p=Path(__file__).resolve().parent;archive=p/'failed_smem_dispatch';archive.mkdir()
for name in ['square_smem.cuh','smem_manifest.json','build_square_smem.json','compile_square_smem_emu.log','compile_square_smem_normal.log']:
 shutil.copy2(p/name,archive/name)
s=(p/'square_smem.cuh').read_text();assert s.count('const bool square_smem =')==1
s=s.replace('const bool square_smem =','const bool square_smem_enabled =').replace('if (square_smem)','if (square_smem_enabled)')
(p/'square_smem.cuh').write_text(s);(p/'include_square_smem/flashinfer/attention/mla_host_batch.cuh').write_text(s)
m=json.loads((p/'smem_manifest.json').read_text());m['candidate_sha256']=hashlib.sha256(s.encode()).hexdigest()
m['build_status']='Unbuilt after dispatch-name fix; prior build failed scope collision of new bool square_smem and old control constexpr size_t square_smem. No GPU run.'
(p/'smem_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(p/'smem_pe.diff').write_text(''.join(difflib.unified_diff((p/'square_p.cuh').read_text().splitlines(True),s.splitlines(True),fromfile='square_p.cuh',tofile='square_smem.cuh')))
print(m['candidate_sha256'])
