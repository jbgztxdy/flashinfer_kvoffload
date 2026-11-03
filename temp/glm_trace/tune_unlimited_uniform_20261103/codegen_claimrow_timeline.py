from pathlib import Path
import json,subprocess,re,hashlib
p=Path(__file__).resolve().parent;out={}
for v in ['clean','claimrow_timeline']:
 lib=next((p/('jit_'+v)).rglob('*host_overlap_v2.so'))
 resource=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-resource-usage',str(lib)],text=True)
 sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(lib)],text=True)
 sections=sass.split('Function : ');section=next(s for s in sections if s.startswith('_ZN10flashinfer3mla26BatchMLACleanServiceKernel'))
 blocks=resource.split('Function ');res=next(s for s in blocks if s.startswith('_ZN10flashinfer3mla26BatchMLACleanServiceKernel'))
 ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\s',section)
 out[v]=dict(library=str(lib.relative_to(p.parents[2])),sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),resources=res.strip(),instructions=len(ops),opcodes={k:ops.count(k) for k in sorted(set(ops)) if k in ['BRA','BSSY','BSYNC','FMUL','FSETP.NEU.AND','FSETP.EQ.AND']})
(p/'claimrow_timeline_codegen.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
