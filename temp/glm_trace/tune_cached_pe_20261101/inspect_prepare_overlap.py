"""CPU-only trace evidence for a future prepare traversal; not a kernel benchmark."""
import hashlib
import json
from pathlib import Path
import random
import time
import torch

p=Path(__file__).resolve().parent
root=p.parents[2]
torch.set_num_threads(4)
meta=json.loads((p/'experiment.json').read_text())
caps=[]
for f in sorted((root/'temp/glm_trace/assistant64k').glob('doc*.pt')):
    d=torch.load(f,map_location='cpu',weights_only=False)
    assert d['replay_provenance']['assistant_only']
    for layer in range(len(d['layers'])):
        for group in range(len(d['group_starts'])):
            caps.append((d['selections'][layer,group],dict(file=str(f.relative_to(root)),
                layer=d['layers'][layer],window_start=int(d['group_starts'][group]),
                source_trace_sha256=d['replay_provenance']['source_trace_sha256'])))
random.Random(meta['seed']).shuffle(caps)
requests=list(range(0,128,8));steps=list(range(0,128,8))
stats={name:dict(warps=0,references=0,unique_warp_pages=0,histogram=[0]*33)
       for name in ['row32','rows2_cols16','rows8_cols4']}
bindings=[]
for r in requests:
    values,binding=caps[r%len(caps)]
    bindings.append(dict(request=r,**binding))
    for step in steps:
        tile=values[step*8:step*8+8]
        assert tile.shape==(8,2048) and tile.min()>=0 and tile.max()<65536
        layouts=dict(row32=tile.reshape(-1,32),
            rows2_cols16=tile.reshape(4,2,128,16).permute(0,2,1,3).reshape(-1,32),
            rows8_cols4=tile.reshape(8,512,4).permute(1,0,2).reshape(-1,32))
        for name,warps in layouts.items():
            sorted_pages=warps.sort(dim=1).values
            counts=1+(sorted_pages[:,1:]!=sorted_pages[:,:-1]).sum(1)
            z=stats[name]
            z['warps']+=len(warps);z['references']+=warps.numel();z['unique_warp_pages']+=int(counts.sum())
            for n,v in enumerate(torch.bincount(counts,minlength=33).tolist()):z['histogram'][n]+=v
for z in stats.values():z['duplicate_reference_fraction']=1-z['unique_warp_pages']/z['references']
result=dict(complete=True,cpu_only=True,kernel_performance_claim=False,recorded_at=time.strftime('%FT%T%z'),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    benchmark_input='Same seeded request/capture bindings as frozen GLM53 benchmark; MTP8,16sampled requests and16sampled epochs. No source page remapping or LRU history changes.',
    requests=requests,steps=steps,bindings=bindings,layouts=stats,
    transaction_geometry='row32: one128B contiguous index span; rows2_cols16: two64B spans (same four32B sectors if aligned); rows8_cols4: eight16B spans (at least eight32B sectors). No measured hardware transactions claimed.',
    limits='Counts include all selected page references, before resident/hit checks. They are an upper bound on warp-match reuse, not cache-hit counts, atomic savings or latency. Transposing traversal also multiplies per-row cursor groups and may lose memory parallelism; a new implementation would require independent correctness and trace validation.')
(p/'prepare_overlap_review.json').write_text(json.dumps(result,indent=2)+'\n')
print({name:stats[name]['duplicate_reference_fraction'] for name in stats})
