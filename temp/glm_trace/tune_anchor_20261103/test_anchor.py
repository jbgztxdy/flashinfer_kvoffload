import sys
from pathlib import Path
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tests/attention'))
from test_mla_host_batch import _plan,_make_indices
@pytest.mark.parametrize('length',[65,2048])
def test_anchor_payload_lse_graph(length):
    rows=512;requests=64;mtp=8;pages=max(8192,requests*length*8);host=pages//2
    torch.manual_seed(20261102)
    kv=torch.randn(pages,1,576,device='cuda',dtype=torch.bfloat16)
    q=torch.randn(rows,64,512,device='cuda',dtype=torch.bfloat16)
    qp=torch.randn(rows,64,80,device='cuda',dtype=torch.bfloat16)[...,:64]
    idx=torch.empty(rows*length,device='cuda',dtype=torch.int32)
    w=_plan(rows,length,idx);w.host_stage=True;w.host_split=False;w.host_fetch_ctas=24;w.host_staging_rows=idx.numel()
    w.set_host_cache(torch.arange(requests+1,dtype=torch.int32)*mtp,512)
    for step in range(3):
        selected=_make_indices(requests,mtp,length,pages,host,[1,.5,1][step],.75,731+step).cuda()
        selected[0]=-1;selected[1,1:]=-1;selected[3,-33:]=-1;idx.copy_(selected.flatten())
        w.run(q,qp,kv[...,:512],kv[...,512:],host_page_start=host)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out,lse=w.run(q,qp,kv[...,:512],kv[...,512:],return_lse=True,host_page_start=host)
        graph.replay();graph.replay();torch.cuda.synchronize()
        for row in [0,1,3,107,108,rows-1]:
            picked=selected[row][selected[row]>=0].long()
            if not picked.numel():assert torch.all(out[row]==0);continue
            values=kv[picked,0].float();s=(q[row].float()@values[:,:512].T+qp[row].float()@values[:,512:].T)/16
            torch.testing.assert_close(out[row].float(),s.softmax(-1)@values[:,:512],rtol=.01,atol=.01)
            torch.testing.assert_close(lse[row],torch.logsumexp(s,-1)/torch.log(torch.tensor(2.0)),rtol=.001,atol=.001)
        resident,slot_page,_,store=w._host_cache_views();slots=(slot_page[:-1]>=0).nonzero().flatten();cached=slot_page[slots].long()
        assert torch.equal(resident[cached-host],slots.int());assert torch.equal(store['kv'][slots],kv[cached,0])

def test_anchor_large_logits_and_rebase():
    rows=512;length=129;pages=8192;host=4096
    torch.manual_seed(20261103)
    kv=torch.randn(pages,1,576,device='cuda',dtype=torch.bfloat16)
    q=torch.randn(rows,64,512,device='cuda',dtype=torch.bfloat16)
    qp=torch.randn(rows,64,64,device='cuda',dtype=torch.bfloat16)
    idx=(torch.arange(rows*length,device='cuda',dtype=torch.int32)%pages).contiguous()
    w=_plan(rows,length,idx);w.host_stage=True;w.host_split=False;w.host_fetch_ctas=24;w.host_staging_rows=idx.numel()
    w.set_host_cache(torch.arange(65,dtype=torch.int32)*8,128)
    for scale in [.125,1.,8.]:
        queries=(q*scale).to(q.dtype);qpe=(qp*scale).to(qp.dtype)
        # Early/late positions with a large head-aligned logit challenge anchor initialization
        # and subsequent rebasing as the descending KV traversal reaches another maximum.
        selected=idx.view(rows,length).clone()
        selected[0,0]=17;selected[0,-1]=19
        kv[17,0,:512]=(queries[0,0]*4).to(kv.dtype)
        kv[19,0,:512]=(queries[0,1]*2).to(kv.dtype)
        idx.copy_(selected.flatten())
        out,lse=w.run(queries,qpe,kv[...,:512],kv[...,512:],return_lse=True,host_page_start=host)
        for row in [0,rows-1]:
            values=kv[selected[row].long(),0].float()
            scores=(queries[row].float()@values[:,:512].T+qpe[row].float()@values[:,512:].T)/16
            torch.testing.assert_close(out[row].float(),scores.softmax(-1)@values[:,:512],rtol=.01,atol=.015)
            torch.testing.assert_close(lse[row],torch.logsumexp(scores,-1)/torch.log(torch.tensor(2.0)),rtol=.001,atol=.003)
