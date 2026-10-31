"""Record strict equivalence failure without normalizing away instruction differences."""
import gzip,hashlib,json,re
from pathlib import Path
p=Path(__file__).resolve().parent
result=dict(complete=True,performance_authorized=False,builds={})
for suffix in ['', '_normal']:
    originals=[];added=[]
    for f in sorted(p.glob('base'+suffix+'_batch*_sass.txt.gz')):
        other=p/f.name.replace('base'+suffix+'_','warp_roles'+suffix+'_',1)
        def parse(path):
            text=gzip.open(path,'rt').read()
            return {s.splitlines()[0].strip():re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*',s) for s in re.split(r'\n\s*Function : ',text)[1:]}
        a,b=parse(f),parse(other)
        for symbol,old in a.items():
            new=b[symbol]
            diffs=[dict(index=i,old=x,new=y) for i,(x,y) in enumerate(zip(old,new)) if x!=y]
            originals.append(dict(archive=f.name,symbol=symbol,equal=old==new,
                old_count=len(old),new_count=len(new),different_positions=len(diffs),examples=diffs[:4],
                pure_device='BatchMLAPagedAttentionKernel' in symbol))
        added += [dict(archive=other.name,symbol=s) for s in b.keys()-a.keys()]
    assert len(originals)==(68 if suffix else 34)
    result['builds'][suffix or 'emu']=dict(original_count=len(originals),
        equal_count=sum(x['equal'] for x in originals),changed_count=sum(not x['equal'] for x in originals),
        pure_device_exact=all(x['equal'] for x in originals if x['pure_device']),
        original_records=originals,new_instances=added)
result['limitation']='Observed commuted logical operands and register-reuse hints do not establish equivalent performance. No v1 benchmark launch; strict original34/68 gate remains required for isolated correction.'
(p/'v1_codegen_review.json').write_text(json.dumps(result,indent=2)+'\n')
print({s:{k:b[k] for k in ['original_count','equal_count','changed_count','pure_device_exact']} for s,b in result['builds'].items()})
