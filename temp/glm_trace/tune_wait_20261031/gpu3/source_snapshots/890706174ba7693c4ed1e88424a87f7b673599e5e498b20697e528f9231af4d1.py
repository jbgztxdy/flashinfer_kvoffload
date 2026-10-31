"""Diagnose small PCIe differences with both binaries and exactly the same allocations.

Reuse the frozen benchmark's trace loader and mapping functions. This is a separate
diagnostic, not a replacement for the independent-process regression matrix.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p=Path(__file__).resolve().parent
bench=p/'bench_gpu3_native.py'
source=bench.read_text()
prefix,tail=source.rsplit('\nfor setting in SETTINGS:run_setting(setting)',1)
assert not tail.strip()
sys.argv=[str(bench),'--kind','lru','--mtp','1','--requests','1','--settings','pcie',
          '--positions','1024','--passes','5','--fetch-ctas','16','--host-split','off']
ns={'__file__':str(bench),'__name__':'same_process_benchmark_bootstrap'}
exec(compile(prefix,str(bench),'exec'),ns)
torch,tiers=ns['torch'],ns['tiers']
import tvm_ffi

sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
mods={};libs={}
for variant in ['base','cached_ids']:
    paths=list((p/('jit_'+variant+'_normal')).rglob('*q_bf16*host_overlap*.so'))
    assert len(paths)==1,paths
    mods[variant]=tvm_ffi.load_module(str(paths[0]))
    libs[variant]=dict(path=str(paths[0]),sha256=sha(paths[0]))
kv=tiers.TieredKV(65536,65536,2048,0,far_kind='host')
main=kv.main()
main.normal_()
q=torch.randn(1,64,512,dtype=torch.bfloat16,device='cuda')
qr=torch.randn(1,64,64,dtype=torch.bfloat16,device='cuda')
idx=torch.empty(2048,dtype=torch.int32,device='cuda')
w=ns['make_wrapper'](1,idx,torch.empty(256<<20,dtype=torch.uint8,device='cuda'),2048)
w.host_fetch_ctas=16;w.host_item_entries=64;w.host_stage=True;w.host_split=False
w._host_prefetch_shape=False;w.host_staging_rows=2048
w.set_host_cache(torch.tensor([0,1],dtype=torch.int32),4096)
pages,_=ns['row_pages'](kv.near,1.)
pages_gpu=pages.to('cuda')
out=torch.empty_like(q)
graphs={};events={};diagnostics={}
for variant in mods:
    w._host_prefetch_module=mods[variant]
    idx.copy_(pages_gpu[0].view(-1))
    w.run(q,qr,main[...,:512],main[...,512:],out=out,host_page_start=kv.near)
    w.release_host_cache(torch.tensor([0]))
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    pairs=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in range(1024)]
    diag=torch.empty(1024,dtype=torch.int32,device='cuda')
    with torch.cuda.graph(graph):
        for step in range(1024):
            idx.copy_(pages_gpu[step].view(-1))
            pairs[step][0].record()
            w.run(q,qr,main[...,:512],main[...,512:],out=out,host_page_start=kv.near)
            pairs[step][1].record()
            diag[step]=w._host_batch_buffers[5][1]
    graphs[variant]=graph;events[variant]=pairs;diagnostics[variant]=diag

records=[]
order=['base','cached_ids','cached_ids','base','base','cached_ids','cached_ids','base','base','cached_ids']
for variant in order:
    w.release_host_cache(torch.tensor([0]))
    torch.cuda.synchronize()
    graphs[variant].replay()
    torch.cuda.synchronize()
    samples=[start.elapsed_time(end)*1000 for start,end in events[variant]]
    misses=diagnostics[variant].cpu().tolist()
    ref=ns['reference'](main,q,qr,pages[-1].view(1,2048),[0])
    torch.testing.assert_close(out.float(),ref,rtol=.01,atol=.01)
    resident,slot_page,_,store=w._host_cache_views()
    filled=(slot_page[:-1]>=0).nonzero().flatten()
    selected=slot_page[filled].long()
    assert torch.equal(resident[selected-kv.near].long(),filled)
    assert torch.equal(store['kv'][filled],main.view(-1,576)[selected])
    record=dict(variant=variant,per_step_us=samples,mean_us=sum(samples)/1024,
                unique_misses=misses,mean_unique_misses=sum(misses)/1024,
                max_abs_error=float((out.float()-ref).abs().max()),cache_payload_checked_rows=len(filled))
    records.append(record)
    print('SAME_ALLOCATION',variant,record['mean_us'],record['mean_unique_misses'],flush=True)
result=dict(complete=True,diagnostic=True,hardware_group='gpu3_a100_pcie',
    shape=dict(mtp=1,requests=1,positions=1024,passes_each=5,cache_capacity=4096,fetch_ctas=16,split=False),
    shared_allocations=True,shared_query_indices_cache_and_workspace=True,
    source_sha256=sha(Path(__file__)),benchmark_source_sha256=sha(bench),libraries=libs,
    resources_start=ns['RESOURCE_START'],resources_end=ns['require_exclusive'](),
    timestamp=time.strftime('%FT%T%z'),records=records,
    numerical_limit=.01,production_changed=False)
(p/'same_process_control.json').write_text(json.dumps(result,indent=2)+'\n')
