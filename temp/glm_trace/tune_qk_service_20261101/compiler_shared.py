"""Archive both candidates and compare all38/76 frozen cached-PE instances literally."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--normal',action='store_true');parser.add_argument('--variants',nargs='+',default=['qks_shared']);a=parser.parse_args()
suffix='_normal' if a.normal else ''
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
records={};libraries={}
for variant in ['base',*a.variants]:
    directory=(p.parent/'tune_qk_pv_roles_20261101'/('jit_qkpv_roles'+suffix)) if variant=='base' else p/('jit_'+variant+suffix)
    libs=sorted(directory.rglob('*.so'));assert len(libs)==(4 if a.normal else 2),(variant,libs)
    records[variant]={};libraries[variant]=[]
    for lib in libs:
        sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(lib)],text=True)
        resource=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-resource-usage',str(lib)],text=True)
        name=('base_for_shared' if variant=='base' else variant)+suffix+'_'+lib.stem
        with gzip.GzipFile(filename=str(p/(name+'_sass.txt.gz')),mode='wb',mtime=0) as f:f.write(sass.encode())
        (p/(name+'_resources.txt')).write_text(resource)
        libraries[variant].append(dict(name=lib.name,sha256=sha(lib)))
        kernels={}
        for part in re.split(r'\n\s*Function : ',sass)[1:]:
            symbol=part.splitlines()[0].strip()
            ins=re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*',part)
            usage=re.search(r'Function '+re.escape(symbol)+r':?\s+([^\n]+)',resource);assert usage
            words=re.findall(r'0x[0-9a-f]{16}\b',part);assert len(words)==2*len(ins)
            kernels[symbol]=dict(symbol=symbol,instructions=len(ins),instruction_sha256=hashlib.sha256('\n'.join(ins).encode()).hexdigest(),
                machine_word_sha256=hashlib.sha256('\n'.join(words).encode()).hexdigest(),machine_words=len(words),resource=usage[1])
        records[variant][lib.name]=kernels
for variant in a.variants:
    typename={'qks_shared':'QKSSharedHostParams'}[variant]
    assert records['base'].keys()==records[variant].keys()
    unchanged=[];added=[]
    for lib,old in records['base'].items():
        new=records[variant][lib];assert old.keys()<=new.keys()
        for symbol,before in old.items():
            assert before==new[symbol],(variant,lib,symbol,before,new[symbol])
            unchanged.append(dict(library=lib,symbol=symbol,machine_word_sha256=before['machine_word_sha256']))
        added += [dict(library=lib,**new[s]) for s in new.keys()-old.keys()]
    assert len(unchanged)==(80 if a.normal else 40)
    assert len(added)==(4 if a.normal else 2) and all(typename in x['symbol'] for x in added)
    result=dict(complete=True,build='normal BF16/FP16' if a.normal else 'EMU BF16',
        base_header_sha256=sha(p/'control_qp.cuh'),candidate_header_sha256=sha(p/(variant+'.cuh')),
        math_header_sha256=sha(p/(variant+'_mla.cuh')),unchanged_original_kernels=len(unchanged),
        all_original_instruction_and_control_words_identical=True,unchanged=unchanged,new_kernels=added,
        all_new_stack_and_local_zero=all('STACK:0' in r['resource'] and 'LOCAL:0' in r['resource'] for r in added),
        libraries={v:libraries[v] for v in ['base',variant]},records={v:records[v] for v in ['base',variant]})
    (p/('compiler_'+variant+suffix+'_summary.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(variant,result['build'],len(unchanged),[x['resource'] for x in added],flush=True)
