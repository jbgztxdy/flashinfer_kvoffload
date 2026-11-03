from pathlib import Path
import subprocess,json,hashlib,ast
p=Path(__file__).resolve().parent;root=p.parents[2]
files=[root/x for x in subprocess.check_output(['git','ls-files','--others','--exclude-standard',str(p.relative_to(root))],cwd=root,text=True).splitlines()]
raw=[];binaries=[]
for f in files:
 if f.suffix=='.json':
  r=json.loads(f.read_text())
  if isinstance(r,dict) and 'libraries' in r:
   assert r['complete'],f
   for lib in r['libraries']:assert hashlib.sha256((root/lib['path']).read_bytes()).hexdigest()==lib['sha256'],f
   raw.append(f.name)
 elif f.suffix=='.py':ast.parse(f.read_text())
 if f.name in ['demand_payload','demand_consume']:binaries.append(f.name)
manifest=dict(files=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],completed_raw_loaded_library_sha_verified=raw,probe_executables=binaries,production_sha256=hashlib.sha256((root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()).hexdigest(),scope='negative results and protocol evidence only; no new retained optimization or100GB claim')
(p/'checkpoint_later_20261104_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print({'files':len(files),'complete_raw':len(raw),'bytes':sum(f.stat().st_size for f in files)})
