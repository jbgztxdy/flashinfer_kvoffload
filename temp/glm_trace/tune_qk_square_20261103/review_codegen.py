"""Small code-generation review; no repeated full original-kernel equivalence audit."""
import argparse,hashlib,json,re,subprocess,time
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--variants',nargs='+',choices=['square','square_p','square_smem'],default=['square','square_p'])
args=parser.parse_args()
p=Path(__file__).resolve().parent;r=dict(timestamp=time.strftime('%FT%T%z'),targets=[])
for v in args.variants:
 label={'square':'Square','square_p':'SquareP','square_smem':'SquareSmem'}[v]
 for mode in ['emu','normal']:
  libs=list((p/('jit_'+v+('_normal' if mode=='normal' else ''))).rglob('*host_overlap_v2.so'))
  assert len(libs)==1,libs
  lib=libs[0];raw=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-resource-usage',str(lib)],text=True)
  (p/f'{v}_{mode}_resources.txt').write_text(raw)
  symbols=[]
  for part in re.split(r'Function ',raw)[1:]:
   name=part.splitlines()[0].rstrip(':')
   if 'BatchMLA'+label+'Kernel' in name:
    symbols.append(dict(symbol=name,resources=part.splitlines()[1].strip()))
  assert len(symbols)==1,symbols
  r['targets'].append(dict(variant=v,mode=mode,header_sha256=hashlib.sha256((p/(v+'.cuh')).read_bytes()).hexdigest(),library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),**symbols[0]))
output='codegen_square_smem.json' if args.variants==['square_smem'] else 'codegen_review.json'
(p/output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
