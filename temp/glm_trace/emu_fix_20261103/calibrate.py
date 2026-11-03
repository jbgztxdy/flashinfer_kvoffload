"""Standalone gather/chase calibration of the FIXED header, separate JIT and raw files."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sys.path.insert(0,str(p));from resources_gpu3 import snapshot,require_exclusive
lock=(p.parent/'tune_general_20261031/gpu3_correctness.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
before=snapshot();assert before['available'],before
import torch
from torch.utils.cpp_extension import load
build=p/'gather_build';build.mkdir(exist_ok=True)
mod=load('far_emu_gather_v2',sources=[str(root/'temp/sim_link/emu_gather.cu')],extra_include_paths=[str(p/'include_v2')],extra_cuda_cflags=['-O3','-arch=sm_80'],build_directory=str(build),verbose=False)
torch.cuda.set_device(0);torch.set_num_threads(8);torch.manual_seed(20261103)
rows=262144+13
src=torch.randint(0,1000,(rows*2,576),device='cuda',dtype=torch.int16)
ids=torch.randperm(rows*2,device='cuda')[:rows].to(torch.int64);dst=torch.empty((rows,576),device='cuda',dtype=torch.int16)
lo=src.data_ptr();hi=lo+src.numel()*src.element_size()
r=dict(start=time.strftime('%FT%T%z'),complete=False,physical_gpu=3,source='same-device HBM',resources_start=before,settings=[])
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
r['source_sha256']={str(f.relative_to(root)):sha(f) for f in [Path(__file__).resolve(),root/'temp/sim_link/emu_gather.cu',p/'include_v2/flashinfer/attention/far_link_emu.cuh']}
for bw in [0,100,50]:
 for lat in [0,2000,5000,10000,20000]:
  require_exclusive();os.environ['FLASHINFER_FAR_EMU']=f'{bw}:{lat}:{lo}:{hi}'
  mod.gather(src,ids,dst);torch.cuda.synchronize()
  s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);samples=[]
  for rep in range(3):
   s.record();mod.gather(src,ids,dst);e.record();e.synchronize();samples.append(s.elapsed_time(e)*1000)
  measured=rows*1152/(sum(samples)/len(samples))/1000
  chase=[mod.latency(src,ids[:256],256) for _ in range(3)]
  payload=bool(torch.equal(dst,src[ids]));assert payload
  setting=dict(bandwidth_GBps=bw,latency_ns=lat,measured_GBps=measured,samples_us=samples,dependent_read_samples_ns=chase,payload_ok=payload)
  r['settings'].append(setting);print(setting,flush=True)
  (p/'calibration.json').write_text(json.dumps(r,indent=2)+'\n')
r.update(complete=True,end=time.strftime('%FT%T%z'),resources_end=require_exclusive())
(p/'calibration.json').write_text(json.dumps(r,indent=2)+'\n')
