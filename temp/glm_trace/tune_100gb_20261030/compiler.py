"""Archive resource/instruction evidence for the frozen baseline and final builds."""
import gzip,hashlib,json,re,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
records={}
for v in ('base','final'):
 libs=list((p/('jit_'+v)).rglob('*host_overlap*.so'))
 assert len(libs)==1
 lib=libs[0]
 sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(lib)],text=True)
 res=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-resource-usage',str(lib)],text=True)
 (p/(v+'_resources.txt')).write_text(res)
 with gzip.open(p/(v+'_sass.txt.gz'),'wb') as f:f.write(sass.encode())
 rows=[]
 for part in re.split(r'\n\s*Function : ',sass)[1:]:
  symbol=part.splitlines()[0].strip()
  if 'BatchMLAHostBatchKernel' not in symbol:continue
  instructions=re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*',part)
  resource=re.search(r'Function '+re.escape(symbol)+r':?\s+([^\n]+)',res)
  assert resource
  rows.append(dict(symbol=symbol,instructions=len(instructions),instruction_sha256=hashlib.sha256('\n'.join(instructions).encode()).hexdigest(),resource=resource[1]))
 records[v]=rows
old={r['symbol']:r for r in records['base']}
comparisons=[]
for r in records['final']:
 if r['symbol'] in old:
  b=old[r['symbol']]
  comparisons.append(dict(symbol=r['symbol'],instruction_count_before=b['instructions'],instruction_count_after=r['instructions'],instruction_text_equal=b['instruction_sha256']==r['instruction_sha256'],resource_equal=b['resource']==r['resource']))
assert len(comparisons)==6
(p/'compiler_summary.json').write_text(json.dumps(dict(builds=records,generic_comparisons=comparisons),indent=2)+'\n')
print(json.dumps(dict(generic_compared=len(comparisons),generic_instruction_text_equal=sum(c['instruction_text_equal'] for c in comparisons))))
