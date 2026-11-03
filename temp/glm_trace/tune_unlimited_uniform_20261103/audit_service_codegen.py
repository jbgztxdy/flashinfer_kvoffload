from pathlib import Path
import json,subprocess,hashlib,re,collections
p=Path(__file__).resolve().parent;root=p.parents[2];out={}
for variant,raw,needle in [('clean','clean_dependency_native_unlimited_f24.json','BatchMLACleanServiceKernel'),('service_calls','service_calls_native_unlimited_f24.json','BatchMLACleanServiceKernel'),('retained','retained_native_unlimited_f24.json','BatchMLADirectPVKernel')]:
 r=json.loads((p/raw).read_text());lib=next(x for x in r['libraries'] if 'host_overlap' in x['path']);f=root/lib['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==lib['sha256']
 sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(f)],text=True)
 resources=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-resource-usage',str(f)],text=True)
 targets=[]
 for part in sass.split('Function : ')[1:]:
  name=part.splitlines()[0].strip()
  if needle not in name:continue
  inst=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)',part)
  hist=collections.Counter(inst)
  targets.append(dict(symbol=name,static_instructions=len(inst),histogram=dict(hist),local_instructions=sum(v for k,v in hist.items() if k.startswith(('LDL','STL'))),barriers=sum(v for k,v in hist.items() if k.startswith('BAR')),mma=sum(v for k,v in hist.items() if 'MMA' in k)))
 selected=[]
 for part in resources.split('Function ')[1:]:
  if needle in part.splitlines()[0]:selected.append(part.strip())
 out[variant]=dict(library=lib,targets=targets,resources=selected)
(p/'service_codegen_audit.json').write_text(json.dumps(out,indent=2)+'\n')
for v,d in out.items():
 print(v,d['resources'])
 for t in d['targets']:print({k:t[k] for k in ['static_instructions','local_instructions','barriers','mma']})
