"""One synthetic1024x2048 launch for hardware counters, never a trace performance claim."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['base','joint_qk'],required=True);a=parser.parse_args()
from resources_gpu3 import snapshot,require_exclusive
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
assert 'membind: 1' in subprocess.check_output(['numactl','--show'],text=True)
lock=(p.parent/'tune_general_20261031/gpu3_correctness.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
while not snapshot()['available']:time.sleep(1)
before=require_exclusive()
import torch
sys.path[:0]=[str(root/'temp'),str(root/'tests/attention')]
from bench_glm52_host_batch import make_wrapper
from flashinfer.jit import env
env.FLASHINFER_INCLUDE_DIR=p/('include_'+a.variant)
torch.manual_seed(20261101)
requests,mtp,length,host_rows=128,8,2048,8192
rows=requests*mtp;pages=requests*host_rows;host_start=1024
kv=torch.randn(host_start+pages,1,576,device='cuda',dtype=torch.bfloat16)
q=torch.randn(rows,64,512,device='cuda',dtype=kv.dtype);qp=torch.randn(rows,64,64,device='cuda',dtype=kv.dtype)
selected=(host_start+torch.arange(requests,device='cuda',dtype=torch.int32)[:,None]*host_rows+torch.arange(length,device='cuda',dtype=torch.int32)[None,:])
indices=selected[:,None,:].expand(requests,mtp,length).reshape(-1).contiguous()
w=make_wrapper(rows,indices,torch.empty(256<<20,device='cuda',dtype=torch.uint8),length);w._host_prefetch_shape=False;w.host_stage=True;w.host_split=False;w.host_fetch_ctas=24;w.host_staging_rows=indices.numel()
w.set_host_cache(torch.arange(requests+1,dtype=torch.int32)*mtp,4096)
for _ in range(2):w.run(q,qp,kv[...,:512],kv[...,512:],host_page_start=host_start)
selected[:, -512:] += 512
indices.copy_(selected[:,None,:].expand(requests,mtp,length).reshape(-1))
torch.cuda.synchronize();rt=torch.cuda.cudart()
rt.cudaProfilerStart()
out=w.run(q,qp,kv[...,:512],kv[...,512:],host_page_start=host_start)
torch.cuda.synchronize();rt.cudaProfilerStop()
max_error=0.
for row in [0,rows-1]:
 values=kv[indices.view(rows,length)[row].long(),0].float();scores=(q[row].float()@values[:,:512].T+qp[row].float()@values[:,512:].T)/16
 expected=scores.softmax(-1)@values[:,:512];max_error=max(max_error,float((expected-out[row].float()).abs().max()));torch.testing.assert_close(out[row].float(),expected,rtol=.01,atol=.01)
resident,slot_page,_,store=w._host_cache_views();filled=(slot_page[:-1]>=0).nonzero().flatten();cached=slot_page[filled].long()
assert torch.equal(resident[cached-host_start],filled.int()) and torch.equal(store['kv'][filled],kv[cached,0])
record=dict(complete=True,variant=a.variant,synthetic=True,trace_performance_claim=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),header_sha256=hashlib.sha256((p/(a.variant+'.cuh')).read_bytes()).hexdigest(),configuration=dict(requests=requests,mtp=mtp,topk=length,capacity=4096,fetch_ctas=24,split=False,far='independentHBMonGPU3',changed_unique_fraction=.25),resources_start=before,resources_end=require_exclusive(),checked_cache_slots=int(filled.numel()),max_abs_error=max_error,limitation='One syntheticcounterlaunch,fullysharedMTPsetswith25%newpages;no trace performance/referenceclaim or exactcausality attribution.')
(p/('counters_'+a.variant+'_correctness.json')).write_text(json.dumps(record,indent=2)+'\n');print('COUNTER_APP_CORRECT',a.variant,max_error,flush=True)
