"""Inspect an actual captured launch, proving the stream_source path and one-kernel contract."""
import ctypes as C
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
assert 'membind: 1' in subprocess.check_output(['numactl','--show'],text=True)
from resources_gpu3 import snapshot,require_exclusive
assert json.loads((p/'focused_state.json').read_text())['complete']
lock=(p.parent/'tune_general_20261031/gpu3_correctness.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
while not snapshot()['available']:
    print('WAITING_GPU3',snapshot(),flush=True);time.sleep(10)
before=require_exclusive()
import torch
sys.path.insert(0,str(root/'temp'))
from bench_glm52_host_batch import make_wrapper,reference
from flashinfer.jit import env
env.FLASHINFER_INCLUDE_DIR=p/'include_qk_cursor'
torch.cuda.set_device(0);torch.manual_seed(1729)
sm=torch.cuda.get_device_properties(0).multi_processor_count
assert sm==108
rows=((4*sm+7)//8)*8;length=97;pages=8192;host_start=pages//2
kv=torch.randn(pages,1,576,device='cuda',dtype=torch.bfloat16)
idx=(torch.arange(rows*length,device='cuda',dtype=torch.int32)%pages).contiguous()
q=torch.randn(rows,64,512,device='cuda',dtype=torch.bfloat16);qr=torch.randn(rows,64,64,device='cuda',dtype=torch.bfloat16)
out=torch.empty_like(q)
w=make_wrapper(rows,idx,torch.empty(256<<20,device='cuda',dtype=torch.uint8),length)
w.host_batch=True;w.host_stage=True;w.host_split=False;w.host_fetch_ctas=32
w._host_prefetch_shape=False;w.host_staging_rows=rows*length
w.set_host_cache(torch.tensor([0,rows],dtype=torch.int32),128)
w.run(q,qr,kv[...,:512],kv[...,512:],out=out,host_page_start=host_start)
torch.cuda.synchronize()
g=torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(g):
    w.run(q,qr,kv[...,:512],kv[...,512:],out=out,host_page_start=host_start)
g.replay();torch.cuda.synchronize()
expected=reference(kv,q,qr,idx.view(rows,length),[0,rows-1])
torch.testing.assert_close(out[[0,rows-1]].float(),expected,rtol=.01,atol=.01)
libpath=next(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcudart.so' in line)
rt=C.CDLL(libpath)
rt.cudaGetErrorString.restype=C.c_char_p
rt.cudaFuncGetName.argtypes=[C.POINTER(C.c_char_p),C.c_void_p]
def check(code):
    if code:raise RuntimeError(rt.cudaGetErrorString(code).decode())
class Dim3(C.Structure):_fields_=[('x',C.c_uint),('y',C.c_uint),('z',C.c_uint)]
class KernelParams(C.Structure):
    _fields_=[('func',C.c_void_p),('grid',Dim3),('block',Dim3),('shared',C.c_uint),('params',C.c_void_p),('extra',C.c_void_p)]
rt.cudaGraphGetNodes.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)]
rt.cudaGraphNodeGetType.argtypes=[C.c_void_p,C.POINTER(C.c_int)]
rt.cudaGraphKernelNodeGetParams.argtypes=[C.c_void_p,C.POINTER(KernelParams)]
rt.cudaOccupancyMaxActiveBlocksPerMultiprocessor.argtypes=[C.POINTER(C.c_int),C.c_void_p,C.c_int,C.c_size_t]
graph=C.c_void_p(g.raw_cuda_graph());count=C.c_size_t()
check(rt.cudaGraphGetNodes(graph,None,C.byref(count)))
nodes=(C.c_void_p*count.value)();check(rt.cudaGraphGetNodes(graph,nodes,C.byref(count)))
launches=[]
for node in nodes:
    kind=C.c_int();check(rt.cudaGraphNodeGetType(node,C.byref(kind)))
    if kind.value!=0:continue
    cfg=KernelParams();check(rt.cudaGraphKernelNodeGetParams(node,C.byref(cfg)))
    name=C.c_char_p();check(rt.cudaFuncGetName(C.byref(name),cfg.func))
    threads=cfg.block.x*cfg.block.y*cfg.block.z;blocks=C.c_int()
    check(rt.cudaOccupancyMaxActiveBlocksPerMultiprocessor(C.byref(blocks),cfg.func,threads,cfg.shared))
    launches.append(dict(grid=[cfg.grid.x,cfg.grid.y,cfg.grid.z],block=[cfg.block.x,cfg.block.y,cfg.block.z],
        dynamic_shared_bytes=cfg.shared,active_blocks_per_SM=blocks.value,kernel_name=name.value.decode()))
record=dict(timestamp=time.strftime('%FT%T%z'),physical_gpu=3,resources_start=before,resources_end=require_exclusive(),
    header_sha256=hashlib.sha256((p/'qk_cursor.cuh').read_bytes()).hexdigest(),
    math_header_sha256=hashlib.sha256((p/'include_qk_cursor/flashinfer/attention/mla.cuh').read_bytes()).hexdigest(),
    graph_nodes=count.value,kernel_nodes=len(launches),launches=launches,
    checked_rows=[0,rows-1],max_abs_error=float((out[[0,rows-1]].float()-expected).abs().max()))
assert len(launches)==1 and launches[0]['grid']==[1,108,1]
assert launches[0]['block']==[32,4,2] and launches[0]['active_blocks_per_SM']==1
assert launches[0]['dynamic_shared_bytes']==147968
assert 'StreamSourceHostParams' in launches[0]['kernel_name'] and 'BatchMLAStreamSourceKernel' in launches[0]['kernel_name']
record['probe_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
archive=p/'dispatch_probes_stream_source';archive.mkdir(exist_ok=True)
(archive/(str(time.time_ns())+'.json')).write_text(json.dumps(record,indent=2)+'\n')
(p/'dispatch_probe_stream_source.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
