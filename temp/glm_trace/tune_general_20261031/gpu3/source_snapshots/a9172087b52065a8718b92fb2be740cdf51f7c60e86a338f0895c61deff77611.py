"""Calibrate GPU3 PCIe on NUMA1, using the same bulk-reference definition as prior reports."""
import fcntl,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parents[2]
sys.path.insert(0,str(root/'temp/h800'))
from resources_gpu3 import require_exclusive,snapshot
import torch
import tiers
lock=(p/'gpu3_correctness.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
numa=subprocess.check_output(['numactl','--show'],text=True)
assert next(x.split(':',1)[1].strip() for x in numa.splitlines() if x.startswith('membind:'))=='1'
while not snapshot()['available']:
    time.sleep(1)
before=require_exclusive()
os.environ.pop('FLASHINFER_FAR_EMU',None)
torch.set_num_threads(8);torch.cuda.set_device(0);torch.manual_seed(42)
rows=1<<20
kv=tiers.TieredKV(rows,rows,rows,0,'host',0)
main,alias=kv.main(),kv.alias()
for start in range(0,2*rows,1<<18):main[start:start+(1<<18)].normal_()
torch.cuda.synchronize()
packed=torch.empty((rows,1,576),dtype=torch.bfloat16,pin_memory=True)
packed.copy_(kv.far_cpu())
target=alias[rows:2*rows]
ids=torch.randperm(rows,device='cuda',dtype=torch.int64)+rows
methods={}
for method in ('dma_contiguous','sm_gather_random'):
    def operation():
        if method=='dma_contiguous':target.copy_(packed,non_blocking=True)
        else:torch.index_select(main,0,ids,out=target)
    operation();torch.cuda.synchronize()
    samples=[]
    for _ in range(9):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(4_000_000)
        start.record();operation();end.record();end.synchronize()
        samples.append(start.elapsed_time(end)*1000)
    expected=packed[:32].to('cuda') if method=='dma_contiguous' else main.index_select(0,ids[:32])
    torch.testing.assert_close(target[:32],expected,rtol=0,atol=0)
    methods[method]=dict(samples_us=samples,median_GBps=rows*1152/statistics.median(samples)/1000,payload_sample_check=True)
result=dict(timestamp=time.strftime('%FT%T%z'),physical_gpu=3,numa_policy=numa,resources_start=before,settings=[dict(name='pcie',tier='host',rows=rows,bytes=rows*1152,measurements=methods,reference_GBps=max(x['median_GBps'] for x in methods.values()),residency=kv.residency,payload_sample_check=True)],source_sha256={str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (Path(__file__),root/'temp/h800/tiers.py')})
del main,alias,target,ids,packed,expected
torch.cuda.synchronize();kv.close();torch.cuda.empty_cache()
result['resources_end']=require_exclusive();result['complete']=True
(p/'gpu3/physical_bandwidth.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
