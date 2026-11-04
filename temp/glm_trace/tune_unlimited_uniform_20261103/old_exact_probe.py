import sys,runpy,json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sys.path[:0]=[str(root/'temp'),str(root/'temp/h800')]
import torch
import bench_glm52_host_batch as b
from nvlink_graph import inspect_graph
orig=torch.cuda.CUDAGraph
class Graph(orig):
 def __new__(cls,*a,**kw):kw['keep_graph']=True;return super().__new__(cls,*a,**kw)
 def __init__(self,*a,**kw):kw['keep_graph']=True;super().__init__(*a,**kw)
torch.cuda.CUDAGraph=Graph
old=b.replay
seen=False
def replay(g,n):
 global seen
 ans=old(g,n)
 if not seen:
  record={'launches':inspect_graph(g[0]),'libraries':[dict(path=f,sha256=hashlib.sha256(Path(f).read_bytes()).hexdigest()) for f in {z.split()[-1] for z in Path('/proc/self/maps').read_text().splitlines() if '.so' in z and 'jit_' in z}]}
  (p/'old_exact_dispatch.json').write_text(json.dumps(record,indent=2));seen=True
 return ans
b.replay=replay
sys.argv=[str(root/'temp/h800/bench_sweep.py'),'--tier','peer','--mtp','8','--requests','128','--fractions','0','0.3','--modes','attn','--rounds','3','--output',str(p/'floor_old_probe_20261104.json')]
runpy.run_path(sys.argv[0],run_name='__main__')
