"""Compare normal-build BF16/FP16 GPU kernels before and after the CPU dispatch predicate."""
import argparse,gzip,hashlib,json,re,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parents[2]
bases=p.parent/'tune_100gb_20261030/jit_final_normal'
parser=argparse.ArgumentParser();parser.add_argument('--variant',default='balanced_cached_sm80',choices=('threshold2_sm80','balanced_sm80','balanced_cached_sm80'));args=parser.parse_args()
variant=args.variant
new=p/('jit_'+variant+'_normal')
hashfile=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
oldlibs={f.name:f for f in bases.rglob('*host_overlap*.so')}
newlibs={f.name:f for f in new.rglob('*host_overlap*.so')}
assert oldlibs.keys()==newlibs.keys() and len(oldlibs)==2,(oldlibs.keys(),newlibs.keys())
records={};libs={}
for label,paths in (('before',oldlibs),('retained',newlibs)):
    rows=[];libs[label]=[]
    for name,f in paths.items():
        sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(f)],text=True)
        res=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-resource-usage',str(f)],text=True)
        short='bf16' if 'bf16' in name else 'fp16'
        with gzip.open(p/f'{label}_{variant}_{short}_normal_sass.txt.gz','wb') as z:z.write(sass.encode())
        (p/f'{label}_{variant}_{short}_normal_resources.txt').write_text(res)
        libs[label].append(dict(name=name,sha256=hashfile(f)))
        for part in re.split(r'\n\s*Function : ',sass)[1:]:
            symbol=part.splitlines()[0].strip()
            if 'BatchMLAHostBatchKernel' not in symbol:continue
            ins=re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*',part)
            resource=re.search(r'Function '+re.escape(symbol)+r':?\s+([^\n]+)',res)
            assert resource
            rows.append(dict(symbol=symbol,instructions=len(ins),instruction_sha256=hashlib.sha256('\n'.join(ins).encode()).hexdigest(),resource=resource[1]))
    records[label]=rows
old={x['symbol']:x for x in records['before']}
comparisons=[]
for x in records['retained']:
    b=old[x['symbol']]
    comparisons.append(dict(symbol=x['symbol'],instruction_text_equal=x['instruction_sha256']==b['instruction_sha256'],resource_equal=x['resource']==b['resource']))
assert len(comparisons)==18
assert all(x['instruction_text_equal'] and x['resource_equal'] for x in comparisons)
result=dict(build='normal SM80 BF16 and FP16',all_18_kernel_instruction_text_and_resources_equal=True,libraries=libs,comparisons=comparisons,records=records,base_header_sha256=hashfile(p/'base.cuh'),retained_header_sha256=hashfile(p/(variant+'.cuh')))
(p/('compiler_normal_'+variant+'_summary.json')).write_text(json.dumps(result,indent=2)+'\n')
print('NORMAL_KERNELS_EQUAL',len(comparisons))
