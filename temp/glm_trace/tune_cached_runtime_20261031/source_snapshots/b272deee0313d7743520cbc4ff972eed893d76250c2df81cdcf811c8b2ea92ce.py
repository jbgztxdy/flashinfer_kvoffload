"""Same-allocation full-window comparison; verify distinct captured kernel functions."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p=Path(__file__).resolve().parent;wdir=p.parent/'tune_wait_20261031'
bench=wdir/'bench.py';source=bench.read_text()
prefix,tail=source.rsplit('\nfor setting in SETTINGS:run_setting(setting)',1)
assert not tail.strip()
sys.argv=[str(bench),'--kind','lru','--mtp','8','--requests','128','--settings','g100_l0','g100_l20000',
          '--positions','1024','--passes','5','--fetch-ctas','24','--host-split','off']
ns={'__file__':str(bench),'__name__':'allocation_probe_bootstrap'}
exec(compile(prefix,str(bench),'exec'),ns)
torch,tiers=ns['torch'],ns['tiers']
import tvm_ffi
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
paths={'base':p.parent/'tune_heads_20261031/jit_base',
       'cached_ids':wdir/'jit_cached_ids','runtime_ids':p/'jit_runtime_ids'}
modules={};libraries={}
for variant,directory in paths.items():
    libs=list(directory.rglob('*q_bf16*host_overlap*.so'))
    assert len(libs)==1,libs
    modules[variant]=tvm_ffi.load_module(str(libs[0]))
    libraries[variant]=dict(path=str(libs[0]),sha256=sha(libs[0]))
R,M,K,steps=128,8,2048,128
kv=tiers.TieredKV(R*65536,R*65536,R*M*K,0,far_kind='peer',peer=1)
main=kv.main();main.normal_()
q=torch.randn(R*M,64,512,dtype=torch.bfloat16,device='cuda')
qr=torch.randn(R*M,64,64,dtype=torch.bfloat16,device='cuda')
idx=torch.empty(R*M*K,dtype=torch.int32,device='cuda')
wrapper=ns['make_wrapper'](R*M,idx,torch.empty(256<<20,dtype=torch.uint8,device='cuda'),K)
wrapper.host_fetch_ctas=24;wrapper.host_item_entries=64;wrapper.host_stage=True;wrapper.host_split=False
wrapper.host_overlap=True;wrapper._host_prefetch_shape=False;wrapper.host_staging_rows=R*M*K
wrapper.set_host_cache(torch.arange(R+1,dtype=torch.int32)*M,4096)
pages,_=ns['row_pages'](kv.near,1.);pages_gpu=pages.to('cuda')
out=torch.empty_like(q);pools=torch.arange(R)
check_rows=[0,R*M//2,R*M-1]
rt_path=next(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcudart.so' in line)
rt=C.CDLL(rt_path);rt.cudaGetErrorString.restype=C.c_char_p
def check(code):
    if code:raise RuntimeError(rt.cudaGetErrorString(code).decode())
class Dim3(C.Structure):_fields_=[('x',C.c_uint),('y',C.c_uint),('z',C.c_uint)]
class KernelParams(C.Structure):
    _fields_=[('func',C.c_void_p),('grid',Dim3),('block',Dim3),('shared',C.c_uint),('params',C.c_void_p),('extra',C.c_void_p)]
rt.cudaGraphGetNodes.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)]
rt.cudaGraphNodeGetType.argtypes=[C.c_void_p,C.POINTER(C.c_int)]
rt.cudaGraphKernelNodeGetParams.argtypes=[C.c_void_p,C.POINTER(KernelParams)]
def launches(graph):
    raw=C.c_void_p(graph.raw_cuda_graph());count=C.c_size_t()
    check(rt.cudaGraphGetNodes(raw,None,C.byref(count)))
    nodes=(C.c_void_p*count.value)();check(rt.cudaGraphGetNodes(raw,nodes,C.byref(count)))
    selected=[]
    for node in nodes:
        kind=C.c_int();check(rt.cudaGraphNodeGetType(node,C.byref(kind)))
        if kind.value!=0:continue
        cfg=KernelParams();check(rt.cudaGraphKernelNodeGetParams(node,C.byref(cfg)))
        if (cfg.block.x,cfg.block.y,cfg.block.z)!=(32,4,2):continue
        selected.append(dict(function=hex(cfg.func),grid=[cfg.grid.x,cfg.grid.y,cfg.grid.z],
                             block=[32,4,2],dynamic_shared_bytes=cfg.shared))
    assert len(selected)==steps and all(x==selected[0] for x in selected)
    return dict(total_graph_nodes=count.value,attention_kernel_nodes=len(selected),launch=selected[0])

result=dict(complete=False,diagnostic=True,hardware='SXM0/1',
    configuration=dict(mtp=M,requests=R,positions=1024,passes_each=5,capacity=4096,fetch_ctas=24,split=False),
    shared_query_indices_cache_workspace_output_and_KV=True,source_sha256=sha(Path(__file__)),
    benchmark_source_sha256=sha(bench),libraries=libraries,resources_start=ns['RESOURCE_START'],cases=[])
target=p/'allocation_probe.json'
def save():target.write_text(json.dumps(result,indent=2)+'\n')
save()
order=['base','cached_ids','runtime_ids','runtime_ids','cached_ids','base','base','cached_ids',
       'runtime_ids','runtime_ids','cached_ids','base','base','cached_ids','runtime_ids']
for latency in [0,20000]:
    ns['emu'].kernel_env(100,latency,main[kv.near].data_ptr(),main[kv.near].data_ptr()+kv.far*1152)
    graphs={};timers={};diagnostics={};dispatch={}
    for variant,module in modules.items():
        wrapper._host_prefetch_module=module
        idx.copy_(pages_gpu[0].view(-1))
        wrapper.run(q,qr,main[...,:512],main[...,512:],out=out,host_page_start=kv.near)
        wrapper.release_host_cache(pools);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph(keep_graph=True)
        pairs=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in range(steps)]
        diag=torch.empty(steps,dtype=torch.int32,device='cuda')
        with torch.cuda.graph(graph):
            for step in range(steps):
                idx.copy_(pages_gpu[step].view(-1));pairs[step][0].record()
                wrapper.run(q,qr,main[...,:512],main[...,512:],out=out,host_page_start=kv.near)
                pairs[step][1].record();diag[step]=wrapper._host_batch_buffers[5][1]
        graphs[variant]=graph;timers[variant]=pairs;diagnostics[variant]=diag
        dispatch[variant]=launches(graph)
    case=dict(setting=f'g100_l{latency}',dispatch=dispatch,records=[])
    result['cases'].append(case);save()
    assert len({d['launch']['function'] for d in dispatch.values()})==3, 'Cannot attribute variants to distinct loaded kernels'
    shapes=[{k:v for k,v in d['launch'].items() if k!='function'} for d in dispatch.values()]
    assert all(x==shapes[0] for x in shapes)
    for variant in order:
        wrapper.release_host_cache(pools);torch.cuda.synchronize()
        graphs[variant].replay();torch.cuda.synchronize()
        samples=[start.elapsed_time(end)*1000 for start,end in timers[variant]]
        ref=ns['reference'](main,q,qr,pages[-1].view(R*M,K),check_rows)
        got=out[check_rows].float();torch.testing.assert_close(got,ref,rtol=.01,atol=.01)
        resident,slot_page,_,store=wrapper._host_cache_views()
        filled=(slot_page[:-1]>=0).nonzero().flatten();selected=slot_page[filled].long()
        assert torch.equal(resident[selected-kv.near].long(),filled)
        assert torch.equal(store['kv'][filled],main.view(-1,576)[selected])
        case['records'].append(dict(variant=variant,per_step_us=samples,mean_us=sum(samples)/steps,
            unique_misses=diagnostics[variant].cpu().tolist(),
            max_abs_error=float((got-ref).abs().max()),cache_payload_checked_rows=len(filled)))
        save();print('SAME_ALLOCATION',case['setting'],variant,sum(samples)/steps,flush=True)
result['resources_end']=ns['require_exclusive']();result['complete']=True
result['timestamp']=time.strftime('%FT%T%z');save()
