from pathlib import Path
import subprocess,json,hashlib,ast
p=Path(__file__).resolve().parent;root=p.parents[2]
files=[root/x for x in subprocess.check_output(['git','ls-files','--others','--exclude-standard',str(p.relative_to(root))],cwd=root,text=True).splitlines()];verified=[]
for f in files:
 if f.suffix=='.py':ast.parse(f.read_text())
 if f.suffix=='.json':
  r=json.loads(f.read_text())
  if isinstance(r,dict) and 'libraries' in r:
   assert r['complete'],f
   for lib in r['libraries']:assert hashlib.sha256((root/lib['path']).read_bytes()).hexdigest()==lib['sha256'],f
   verified.append(f.name)
production={name:hashlib.sha256((root/'include/flashinfer/attention'/name).read_bytes()).hexdigest() for name in ['mla_host_batch.cuh','mla.cuh','far_link_emu.cuh']}
assert production['mla_host_batch.cuh']=='81093ba5ab33cc5d84b2df04c702975697387b4054ebe8f1991f558f1bd6cdd2'
manifest=dict(files=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],actual_library_verified_raw=verified,production=production,scope='negative experiments and bounded evidence only; no new retained candidate or100GB benchmark')
(p/'checkpoint_review_20261104_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print({'files':len(files),'actual_library_verified_raw':len(verified),'bytes':sum(f.stat().st_size for f in files)})
