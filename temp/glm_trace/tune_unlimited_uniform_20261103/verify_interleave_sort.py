from pathlib import Path
import json,random
p=Path(__file__).resolve().parent;r=random.Random(20261104);count=0
for width in [1,2,4,8,16,32,64,128]:
    for length in range(width//2+1,width+1):
        for repeat in range(5):
            source=[r.randrange(100000) for _ in range(length)]+[100001]*(width-length)
            original=source[:]
            k=2
            while k<=width:
                j=k//2
                while j:
                    target=[]
                    for t in range(width):
                        take_min=((t&k)==0)==((t&j)==0)
                        target.append(min(source[t],source[t^j]) if take_min else max(source[t],source[t^j]))
                    source=target;j//=2
                k*=2
            assert source==sorted(original)
            count+=1
(p/'claimrow_sort_model.json').write_text(json.dumps(dict(seed=20261104,cases=count,vector_widths=[1,2,4,8,16,32,64,128],verified='same comparator as device two-buffer bitonic sorting network; CPU only, not memory-order proof'),indent=2)+chr(10))
print(count,'sorted networks matched Python sorting')
