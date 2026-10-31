"""Measure same-device HBM payload copy bandwidth, explicitly not a physical remote link."""
import fcntl,hashlib,json,statistics,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parents[2]
sys.path.insert(0,str(root/'temp/h800'))
from resources_gpu3 import snapshot,require_exclusive
import torch
import tiers
lock=(p/'gpu3_correctness.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
while not snapshot()['available']:time.sleep(1)
before=require_exclusive()
torch.set_num_threads(8);torch.cuda.set_device(0);torch.manual_seed(42)
rows=1<<20;kv=tiers.TieredKV(rows,rows,rows,0,'peer',0)
main,alias=kv.main(),kv.alias()
main.normal_();src=main[rows:2*rows];target=alias[rows:2*rows]
ids=torch.randperm(rows,device='cuda',dtype=torch.int64)+rows
methods={}
for method in ('dma_contiguous','sm_gather_random'):
    def op():
        if method=='dma_contiguous':target.copy_(src)
        else:torch.index_select(main,0,ids,out=target)
    op();torch.cuda.synchronize();samples=[]
    for _ in range(9):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(4_000_000);start.record();op();end.record();end.synchronize()
        samples.append(start.elapsed_time(end)*1000)
    expected=src[:32] if method=='dma_contiguous' else main.index_select(0,ids[:32])
    torch.testing.assert_close(target[:32],expected,rtol=0,atol=0)
    methods[method]=dict(samples_us=samples,median_GBps=rows*1152/statistics.median(samples)/1000,payload_sample_check=True)
result=dict(timestamp=time.strftime('%FT%T%z'),physical_gpu=3,source='independent HBM allocation on the same GPU',resources_start=before,rows=rows,bytes=rows*1152,measurements=methods,reference_GBps=max(x['median_GBps'] for x in methods.values()),residency=kv.residency,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
del main,alias,src,target,ids,expected
torch.cuda.synchronize();kv.close();torch.cuda.empty_cache()
result['resources_end']=require_exclusive();result['complete']=True
(p/'gpu3/native_hbm_bandwidth.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
