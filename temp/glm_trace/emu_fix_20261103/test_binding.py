"""Exercise the public emu binding in its separate v2 workspace, including Graph replay."""
import fcntl,hashlib,json,os,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sys.path[:0]=[str(p),str(root/'temp/sim_link')]
from resources_gpu3 import snapshot
lock=(p.parent/'tune_general_20261031/gpu3_correctness.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
before=snapshot();assert before['available'],before
import torch,emu
from flashinfer.jit import env
env.FLASHINFER_INCLUDE_DIR=p/'include_v2'
torch.cuda.set_device(0);torch.manual_seed(20261103)
src=torch.randint(0,1000,(16384,576),device='cuda',dtype=torch.int16)
ids=torch.randperm(16384,device='cuda')[:8192+13];dst=torch.empty((ids.numel(),576),device='cuda',dtype=torch.int16)
emu.kernel_env(100,20000,src.data_ptr(),src.data_ptr()+src.numel()*2)
emu.gather(src,ids,dst);torch.cuda.synchronize();assert torch.equal(dst,src[ids])
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):emu.gather(src,ids,dst)
g.replay();g.replay();torch.cuda.synchronize();assert torch.equal(dst,src[ids])
r=dict(timestamp=time.strftime('%FT%T%z'),complete=True,rows=ids.numel(),bandwidth_GBps=100,latency_ns=20000,payload_ok=True,graph_replays=2,module_name='far_emu_gather_v2',build_directory=str(root/'temp/sim_link/build_v2'),resources_start=before,resources_end=snapshot(),sources={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'temp/sim_link/emu.py',root/'temp/sim_link/emu_gather.cu',root/'include/flashinfer/attention/far_link_emu.cuh']})
assert not r['resources_end']['foreign'];(p/'binding.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
