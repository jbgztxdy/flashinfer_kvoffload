"""Archive the actual two target bodies; static schedule diagnostics are not timings."""
import collections,gzip,hashlib,json,re,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;records=[]
for mode in ['emu','normal']:
 lib,=list((p/('jit_pv_rescale'+('_normal' if mode=='normal' else ''))).rglob('*host_overlap_v2.so'))
 raw=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(lib)],text=True)
 for label,name in [('base','BatchMLAStreamSourceKernel'),('pv_rescale','BatchMLAPVRescaleKernel')]:
  parts=[z for z in re.split(r'\n\s*Function : ',raw)[1:] if name in z.splitlines()[0]];assert len(parts)==1
  part=parts[0];out=p/f'{label}_{mode}_target.sass.gz'
  with gzip.GzipFile(filename=str(out),mode='wb',mtime=0) as f:f.write(part.encode())
  ins=[(m[1],m[2]) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\* 0x[0-9a-f]{16}',part)]
  ops=[re.sub(r'^@!?P\d+\s+','',s).split()[0] for a,s in ins]
  fmuls=[i for i,x in enumerate(ops) if x.startswith('FMUL')];hmmas=[i for i,x in enumerate(ops) if x.startswith('HMMA')]
  # Window containing first tensor-core operation with an O rescale directly
  # nearby; preserve addresses to make diagnostics inspectable.
  fused=[i for i in hmmas if any(0<i-j<=10 for j in fmuls)]
  records.append(dict(mode=mode,variant=label,symbol=part.splitlines()[0],library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),archive=out.name,static_instruction_count=len(ops),static_FMUL_count=len(fmuls),static_HMMA_count=len(hmmas),HMMA_with_FMUL_in_previous10_instructions=len(fused),example_windows=[ins[max(0,i-10):i+3] for i in fused[:2]],limits='Static instructions and schedule windows only; not dynamic counts, stall fractions, or causal performance proof.'))
(p/'schedule_review.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps([{k:v for k,v in x.items() if k!='example_windows'} for x in records],indent=2))
