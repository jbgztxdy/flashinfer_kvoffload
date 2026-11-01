"""Archive the exact prepare-only diffs and compiled GPU atomic operations."""
import collections,difflib,gzip,hashlib,json,re
from pathlib import Path
p=Path(__file__).resolve().parent
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
def function(s,name):
 pos=s.index('void '+name+'(');a=s.rfind('template <',0,pos);b=s.index('{',pos);depth=1;e=b+1
 while depth:depth+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
base=(p/'base.cuh').read_text();old_prepare=function(base,'prepare_host_batch_rows');old_kernel=function(base,'BatchMLAJointQKKernel');source=[];code=[]
for v,stem in [('claim_probe','ClaimProbe'),('claim_store','ClaimStore')]:
 s=(p/(v+'.cuh')).read_text();kernel=function(s,'BatchMLA'+stem+'Kernel').replace('BatchMLA'+stem+'Kernel','BatchMLAJointQKKernel').replace(v+'_prepare_rows','prepare_host_batch_rows');assert kernel==old_kernel
 prepare=function(s,v+'_prepare_rows').replace(v+'_prepare_rows','prepare_host_batch_rows')
 diff='\n'.join(difflib.unified_diff(old_prepare.splitlines(),prepare.splitlines(),fromfile='productionprepare',tofile=v,lineterm=''))
 (p/(v+'_prepare.diff')).write_text(diff+'\n');source.append(dict(variant=v,normalized_lifecycle_kernel_sha256=sha(kernel),prepare_diff=v+'_prepare.diff'))
for variant,target in [('base','BatchMLAJointQKKernel'),('claim_probe','BatchMLAClaimProbeKernel'),('claim_store','BatchMLAClaimStoreKernel')]:
 for f in sorted(p.glob(variant+'*_sass.txt.gz')):
  if variant=='base' and f.name.startswith('base_for_store'):continue
  for part in re.split(r'\n\s*Function : ',gzip.decompress(f.read_bytes()).decode())[1:]:
   symbol=part.splitlines()[0].strip()
   if target not in symbol:continue
   ins=re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*',part)
   selected=[i for i in ins if any(x in i for x in ['.CAS','.EXCH','LDG.E.STRONG.GPU','STG.E.STRONG.GPU'])]
   code.append(dict(variant=variant,archive=f.name,symbol=symbol,instructions=len(ins),CAS=sum('.CAS' in i for i in ins),EXCH=sum('.EXCH' in i for i in ins),strong_gpu_loads=sum('LDG.E.STRONG.GPU' in i for i in ins),strong_gpu_stores=sum('STG.E.STRONG.GPU' in i for i in ins),selected_instructions=selected))
(p/'atomic_source_review.json').write_text(json.dumps(dict(source=source,code=code,limitation='Source/lifecycleidentityandstaticopcodepresenceonly.Dynamicatomicavoidanceandperformance requireactualtimings;notisolatedcausalityproof.'),indent=2)+'\n')
print([(r['variant'],r['CAS'],r['EXCH'],r['strong_gpu_loads'],r['strong_gpu_stores']) for r in code])
