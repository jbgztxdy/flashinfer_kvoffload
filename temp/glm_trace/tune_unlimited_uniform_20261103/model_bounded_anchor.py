from pathlib import Path
import numpy as np,json
p=Path(__file__).resolve().parent;rng=np.random.default_rng(20261104);cases=[]
# Anchor is a normalization origin, not required to equal the latest maximum.
# Updating when current log2 max exceeds anchor by8 bounds every BF16 P by256.
for n in [1,31,32,65,128,2048]:
 for scale in [.05,1,4,32]:
  for jump in [0,1000,-1000]:
   logits=rng.normal(0,scale,(8,n));values=rng.normal(size=(n,17))
   logits[:,n//2:]+=jump
   anchor=np.full(8,-np.inf);den=np.zeros(8);out=np.zeros((8,17));updates=0;largest=0
   for end in range(n,0,-64):
    s=logits[:,max(0,end-64):end];v=values[max(0,end-64):end]
    current=s.max(1);change=(current-anchor>8)|~np.isfinite(anchor)
    new=np.where(change,current,anchor);factor=np.exp2(anchor-new)
    w=np.exp2(s-new[:,None]);largest=max(largest,float(w.max()));updates+=int(change.sum())
    den=den*factor+w.sum(1);out=out*factor[:,None]+w@v;anchor=new
   shifted=np.exp2(logits-logits.max(1)[:,None]);reference=shifted@values/shifted.sum(1)[:,None]
   assert largest<=256*(1+1e-12)
   err=float(np.abs(out/den[:,None]-reference).max());assert err<1e-11,err
   lse=anchor+np.log2(den);ref_lse=logits.max(1)+np.log2(shifted.sum(1));assert np.max(np.abs(lse-ref_lse))<1e-11
   cases.append(dict(n=n,scale=scale,jump=jump,error=err,anchor_updates=updates,max_probability_operand=largest))
(p/'bounded_anchor_model.json').write_text(json.dumps(dict(cases=cases,scope='float64 equation/bounded exponent check; notGPU BF16 numerical or performance evidence'),indent=2)+'\n');print(len(cases),'FP64 normalized-equation cases passed')
