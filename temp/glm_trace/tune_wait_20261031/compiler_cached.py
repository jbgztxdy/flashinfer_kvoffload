"""Verify cached specialization retains every original normal/emulated GPU kernel."""
import argparse
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--normal',action='store_true');a=parser.parse_args()
suffix='_normal' if a.normal else ''
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
records={};libraries={}
for v in ['base','cached_ids']:
    libs=sorted((p/('jit_'+v+suffix)).rglob('*host_overlap*.so'))
    assert len(libs)==(2 if a.normal else 1),(v,libs)
    records[v]=[];libraries[v]=[]
    for lib in libs:
        dtype='bf16' if 'bf16' in lib.name else 'fp16'
        sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(lib)],text=True)
        res=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-resource-usage',str(lib)],text=True)
        name=v+'_'+dtype+suffix
        archive=p/(name+'_sass.txt.gz')
        if not archive.exists() or gzip.decompress(archive.read_bytes())!=sass.encode():
            with gzip.GzipFile(filename=str(archive),mode='wb',mtime=0) as f:f.write(sass.encode())
        (p/(name+'_resources.txt')).write_text(res)
        libraries[v].append(dict(name=lib.name,sha256=sha(lib)))
        for part in re.split(r'\n\s*Function : ',sass)[1:]:
            symbol=part.splitlines()[0].strip()
            if 'BatchMLAHostBatchKernel' not in symbol:continue
            ins=re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*',part)
            resource=re.search(r'Function '+re.escape(symbol)+r':?\s+([^\n]+)',res)
            assert resource
            records[v].append(dict(symbol=symbol,instructions=len(ins),instruction_sha256=hashlib.sha256('\n'.join(ins).encode()).hexdigest(),resource=resource[1]))
old={x['symbol']:x for x in records['base']}
new={x['symbol']:x for x in records['cached_ids']}
assert set(old)<=set(new)
equal=[]
for key,b in old.items():
    n=new[key]
    assert b['instruction_sha256']==n['instruction_sha256'] and b['resource']==n['resource'],key
    equal.append(key)
extra=[x for x in records['cached_ids'] if x['symbol'] not in old]
assert len(equal)==(18 if a.normal else 9)
assert len(extra)==(2 if a.normal else 1)
assert all('CachedSharedIdsHostBatchParams' in x['symbol'] and 'ELj32E' in x['symbol'] for x in extra)
assert all('STACK:0' in x['resource'] and 'LOCAL:0' in x['resource'] for x in records['cached_ids'])
result=dict(build='normal BF16/FP16' if a.normal else 'emulated BF16',base_header_sha256=sha(p/'base.cuh'),candidate_header_sha256=sha(p/'cached_ids.cuh'),
    unchanged_original_kernels=len(equal),new_cached_kernels=len(extra),all_stack_and_local_zero=True,
    libraries=libraries,records=records,new_cached=extra)
(p/('compiler_cached'+suffix+'_summary.json')).write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k not in ['libraries','records','new_cached']})
