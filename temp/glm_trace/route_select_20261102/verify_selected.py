"""Targeted numerical/cache-payload/Graph check of the measured route selector."""
import fcntl,hashlib,json,os,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
from resources import snapshot,require_exclusive
from select_route import configure_wrapper,HARDWARE
lock=(p.parent/'kernel_report_20261029/gpu01.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
while not snapshot()['available']:
    print('WAITING_GPU01',snapshot(),flush=True);time.sleep(10)
before=require_exclusive()
import torch
from flashinfer.jit import env
# Keep original paths and measured JIT; no rebuild or changes to frozen libraries.
from flashinfer.jit.attention.modules import gen_batch_mla_module
env.FLASHINFER_CSRC_DIR=root/'csrc';env.FLASHINFER_DATA=root
env.CUTLASS_INCLUDE_DIRS=[root/'3rdparty/cutlass/include',root/'3rdparty/cutlass/tools/util/include']
env.SPDLOG_INCLUDE_DIR=root/'3rdparty/spdlog/include';env.FLASHINFER_INCLUDE_DIR=p.parent/'tune_qk_cursor_20261102/include_qk_cursor'
import ast
from flashinfer.mla import BatchMLAPagedAttentionWrapper
fixture_source=(root/'tests/attention/test_mla_host_batch.py').read_text()
fixture_tree=ast.parse(fixture_source)
helpers=[x for x in fixture_tree.body if isinstance(x,ast.FunctionDef) and x.name in ['_plan','_make_indices']]
assert len(helpers)==2
exec(compile(ast.Module(body=helpers,type_ignores=[]),'<frozen-existing-fixture-helpers>','exec'),globals())
results=[];torch.manual_seed(20261102);torch.set_num_threads(8)
for requests in [16,128]:
    mtp=8;rows=requests*mtp;length=97;pages=requests*length*8;host_start=pages//2
    kv=torch.randn(pages,1,576,device='cuda',dtype=torch.bfloat16)
    q=torch.randn(rows,64,512,device='cuda',dtype=torch.bfloat16);qp=torch.randn(rows,64,64,device='cuda',dtype=torch.bfloat16)
    idx=torch.empty(rows*length,device='cuda',dtype=torch.int32)
    w=_plan(rows,length,idx);w.host_staging_rows=idx.numel();w.set_host_cache(torch.arange(requests+1,dtype=torch.int32)*mtp,128)
    # Route choice comes from the2048-length measured workload, not this short correctness fixture.
    choice=configure_wrapper(w,hardware=HARDWARE,requests=requests,mtp=mtp,capacity=4096,setting='nvlink')
    for step in range(3):
        selected=_make_indices(requests,mtp,length,pages,host_start,.75,.6,9191+step).cuda()
        selected[0]=-1;selected[1,1:]=-1;idx.copy_(selected.flatten())
        w.run(q,qp,kv[...,:512],kv[...,512:],host_page_start=host_start)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out,lse=w.run(q,qp,kv[...,:512],kv[...,512:],return_lse=True,host_page_start=host_start)
        graph.replay();graph.replay();torch.cuda.synchronize()
        for row in [0,1,rows//2,rows-1]:
            picked=selected[row][selected[row]>=0].long()
            if not picked.numel():assert torch.all(out[row]==0);continue
            values=kv[picked,0].float();scores=(q[row].float()@values[:,:512].T+qp[row].float()@values[:,512:].T)/16
            torch.testing.assert_close(out[row].float(),scores.softmax(-1)@values[:,:512],rtol=.01,atol=.01)
            torch.testing.assert_close(lse[row],torch.logsumexp(scores,-1)/torch.log(torch.tensor(2.0)),rtol=.001,atol=.001)
        resident,slot_page,_,store=w._host_cache_views();slots=(slot_page[:-1]>=0).nonzero().flatten();cached=slot_page[slots].long()
        assert torch.equal(resident[cached-host_start],slots.int());assert torch.equal(store['kv'][slots],kv[cached,0])
    results.append(dict(requests=requests,mtp=mtp,selected_route=choice['route'],epochs=3,graph_replays_per_epoch=2,all_valid_cache_payloads_equal=True,output_lse_checked=True))
    del w,kv,q,qp,idx,graph,out,lse;torch.cuda.empty_cache()
record=dict(timestamp=time.strftime('%FT%T%z'),complete=True,header_sha256=hashlib.sha256((p/'current.cuh').read_bytes()).hexdigest(),resources_start=before,resources_end=require_exclusive(),cases=results,scope='Short97fixture validates selector/Graph/payload on homogeneous device memory, not physical NVLink performance or full4096capacity.')
(p/'selected_correctness.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
