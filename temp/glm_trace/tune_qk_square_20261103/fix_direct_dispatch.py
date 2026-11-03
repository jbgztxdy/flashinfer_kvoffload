import difflib,hashlib,json,shutil,time
from pathlib import Path
p=Path(__file__).resolve().parent;archive=p/'failed_direct_dispatch';archive.mkdir()
for name in ['square_p.cuh','direct_manifest.json','build_square_p.json','compile_square_p_emu.log','compile_square_p_normal.log']:
 shutil.copy2(p/name,archive/name)
s=(p/'square_p.cuh').read_text();first=s.index('    using SquarePKernelTraits =');last=s.index('    using SquareKernelTraits =',first)
section=s[first:last];assert section.count('const bool square =')==1
section=section.replace('bool square =','bool square_p =').replace('if (square)','if (square_p)');s=s[:first]+section+s[last:]
(p/'square_p.cuh').write_text(s);(p/'include_square_p/flashinfer/attention/mla_host_batch.cuh').write_text(s)
m=json.loads((p/'direct_manifest.json').read_text());m['candidate_sha256']=hashlib.sha256(s.encode()).hexdigest();m['cpu_compile_fix']='Rename copied dispatch condition to avoid scope collision with retained square control. No GPU run of originaldraft.'
(p/'direct_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(p/'direct_p.diff').write_text(''.join(difflib.unified_diff((p/'square.cuh').read_text().splitlines(True),s.splitlines(True),fromfile='square.cuh',tofile='square_p.cuh')))
r=(p/'run_direct_p.py').read_text().replace("run('targeted_correctness'","run('direct_targeted_correctness'");(p/'run_direct_p.py').write_text(r)
print(m['candidate_sha256'])
