from pathlib import Path
import subprocess,json,hashlib,ast
p=Path(__file__).resolve().parent;root=p.parents[2]
files=[root/x for x in subprocess.check_output(['git','ls-files','--others','--exclude-standard',str(p.relative_to(root))],cwd=root,text=True).splitlines()]
checks=[]
for f in files:
 if f.suffix=='.json':
  r=json.loads(f.read_text())
  if isinstance(r,dict) and 'libraries' in r:
   assert r['complete'],f
   for lib in r['libraries']:
    actual=root/lib['path'];assert hashlib.sha256(actual.read_bytes()).hexdigest()==lib['sha256'],actual
   checks.append(f.name)
 elif f.suffix=='.py':ast.parse(f.read_text())
manifest=dict(files=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],completed_raw_library_sha_verified=checks,production_sha256=hashlib.sha256((root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()).hexdigest(),scope='checkpoint only; no production candidate retained; no new 100GB success')
(p/'checkpoint_20261104_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('FILES',len(files),'RAW_LIBRARY_SHA_CHECKS',len(checks),'BYTES',sum(f.stat().st_size for f in files))
