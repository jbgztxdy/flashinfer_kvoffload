from pathlib import Path
import random,json
p=Path(__file__).resolve().parent;rng=random.Random(20261104);tests=0
# Mirrors thread-contiguous listing and packed scan order in run_host_batch_lru.
def listing(keys,threshold,rank,threads=256):
 groups=[list(range(t,len(keys),threads)) for t in range(threads)]
 less=[sum(keys[i]<threshold for i in g) for g in groups];equal=[sum(keys[i]==threshold for i in g) for g in groups];out=[None]*(sum(less)+rank);lp=0;ep=0
 for g,l,e in zip(groups,less,equal):
  take=min(e,max(0,rank-ep));pos=lp+min(ep,rank)
  for i in g:
   if keys[i]<threshold or (keys[i]==threshold and take>0):
    if keys[i]==threshold:take-=1
    out[pos]=i;pos+=1
  lp+=l;ep+=e
 return out
for capacity in [1,65,256,512,4096,8192]:
 for trial in range(80):
  keys=[0xffffffff if rng.random()<.3 else rng.choice([0,1,3,15,256,65537]) for _ in range(capacity)];eligible=[x for x in keys if x!=0xffffffff]
  if not eligible:continue
  victims=len(eligible);sortedkeys=sorted(eligible);threshold=sortedkeys[victims-1];rank=victims-sum(x<threshold for x in eligible)
  high=max(eligible);high_rank=sum(x==high for x in eligible)
  assert threshold==high and rank==high_rank
  assert listing(keys,threshold,rank)==listing(keys,high,high_rank);tests+=1
(p/'lru_select_all_model.json').write_text(json.dumps({'cases':tests,'same_threshold':True,'same_rank':True,'same_victim_slot_order':True,'scope':'CPU select-all special-case equivalence only; no GPU frequency or speed evidence','limitations':['when all stamps equal, existing radix already skips loop','extra high-count pass may negate savings','need actual select-all frequency before implementation']},indent=2)+'\n');print(tests)
