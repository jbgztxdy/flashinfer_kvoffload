from pathlib import Path
import torch,json,hashlib,statistics
p=Path(__file__).resolve().parent;root=p.parents[2];records=[];inputs=[]
for f in sorted((root/'temp/glm_trace/assistant64k').glob('doc*.pt')):
 d=torch.load(f,map_location='cpu',weights_only=False);inputs.append(dict(path=str(f.relative_to(root)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 for layer in range(len(d['layers'])):
  for group in range(len(d['group_starts'])):
   x=d['selections'][layer,group][:256].reshape(32,8,2048)
   for step in range(32):
    for pair in range(4):
     a=x[step,pair*2].tolist();b=x[step,pair*2+1].tolist();sa=set(a);sb=set(b);sa.discard(-1);sb.discard(-1);common=len(sa&sb)
     aligned=sum(aa==bb and aa>=0 for aa,bb in zip(a,b))
     tiles=0
     for k in range(0,2048,32):tiles+=len((set(a[k:k+32])&set(b[k:k+32]))-{-1})
     records.append(dict(file=f.name,layer=layer,group=group,step=step,pair=pair,common_pages=common,aligned_pages=aligned,same_tile_common=tiles,union_pages=len(sa|sb)))
out=dict(scope='all captured assistant layer/windows first256positions, adjacent nonoverlapMTP8pairs, rawselectedKV before LRU/remap; no GPU timing or proposed arithmetic reorder',inputs=inputs,pairs=len(records),mean_common_fraction=statistics.mean(x['common_pages']/2048 for x in records),mean_position_aligned_fraction=statistics.mean(x['aligned_pages']/2048 for x in records),mean_same32tile_common_fraction=statistics.mean(x['same_tile_common']/2048 for x in records),mean_union_pages=statistics.mean(x['union_pages'] for x in records),records=records)
(p/'mtp_pair_reuse.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ['inputs','records']})
