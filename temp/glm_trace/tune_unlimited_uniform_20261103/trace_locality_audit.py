import torch,sys,random,json
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];torch.set_num_threads(8);sys.path.insert(0,str(root/'temp/h800'));import tiers
caps=[]
for path in sorted((root/'temp/glm_trace/assistant64k').glob('doc*.pt')):
 d=torch.load(path,map_location='cpu',weights_only=False)
 for l in range(len(d['layers'])):
  for g in range(len(d['group_starts'])):caps.append(d['selections'][l,g])
meta=json.loads((p/'experiment.json').read_text());random.Random(meta['seed']).shuffle(caps)
new=torch.stack([caps[r%len(caps)][:1024] for r in range(128)]).reshape(128,128,8,2048).permute(1,0,2,3)
# unique logical (request,token) pages before placement/compaction
owner=torch.arange(128).view(128,1,1)*65536
n=[torch.unique(new[t].long()+owner).numel() for t in range(128)]
old=tiers.load_captures(str(root/'temp/mtp_overlap/overlap_ctx65536_*.pt'));o=[]
for i in range(2):
 req=tiers.Requests(old,128,65536,128*65536,128,0.,42+128000+i,'cpu');ids,_,_=req.rows(0,8);o.append(torch.unique(ids).numel())
out=dict(new_mean_unique_pages=sum(n)/len(n),new_min=min(n),new_max=max(n),old_unique_pages=o,total_references=128*8*2048)
(p/'trace_locality_audit.json').write_text(json.dumps(out,indent=2));print(out)
