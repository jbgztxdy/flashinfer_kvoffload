import sys
from pathlib import Path
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tests/attention'))
from test_mla_host_batch import _plan,_make_indices
@pytest.mark.parametrize('length,rows,staged',[(65,512,True),(2048,512,True)])
def test_pvpipe_output_lse_cache_graph(length,rows,staged):
    requests=rows//8;mtp=8;pages=max(8192,requests*length*8);host=pages//2
    torch.manual_seed(20261102)
    kv=torch.randn(pages,1,576,device='cuda',dtype=torch.bfloat16)
    q=torch.randn(rows,64,512,device='cuda',dtype=torch.bfloat16)
    qp=torch.randn(rows,64,80,device='cuda',dtype=torch.bfloat16)[...,:64]
    idx=torch.empty(rows*length,device='cuda',dtype=torch.int32)
    w=_plan(rows,length,idx);w.host_stage=staged;w.host_split=False;w.host_fetch_ctas=24;w.host_staging_rows=idx.numel()
    w.set_host_cache(torch.arange(requests+1,dtype=torch.int32)*mtp,512)
    import os
    lo=kv[host:].data_ptr();hi=lo+(pages-host)*1152
    os.environ['FLASHINFER_FAR_EMU']=f'100:20000:{lo}:{hi}'
    for step in range(3):
        selected=_make_indices(requests,mtp,length,pages,host,[1,.5,1][step],.75,731+step).cuda()
        selected[0]=-1;selected[1,1:]=-1;selected[3,-33:]=-1;idx.copy_(selected.flatten())
        w.run(q,qp,kv[...,:512],kv[...,512:],host_page_start=host)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out,lse=w.run(q,qp,kv[...,:512],kv[...,512:],return_lse=True,host_page_start=host)
        graph.replay();graph.replay();torch.cuda.synchronize()
        for row in [0,1,3,rows//2,rows-1]:
            picked=selected[row][selected[row]>=0].long()
            if not picked.numel():assert torch.all(out[row]==0);continue
            values=kv[picked,0].float();s=(q[row].float()@values[:,:512].T+qp[row].float()@values[:,512:].T)/16
            torch.testing.assert_close(out[row].float(),s.softmax(-1)@values[:,:512],rtol=.01,atol=.01)
            torch.testing.assert_close(lse[row],torch.logsumexp(s,-1)/torch.log(torch.tensor(2.0)),rtol=.001,atol=.001)
        resident,slot_page,_,store=w._host_cache_views();slots=(slot_page[:-1]>=0).nonzero().flatten();cached=slot_page[slots].long()
        assert torch.equal(resident[cached-host],slots.int());assert torch.equal(store['kv'][slots],kv[cached,0])

        stage_rows=store['kv'].shape[0]-store['slots']
        flag_words=((stage_rows+19)//20+1)//2*2
        records=w._host_batch_buffers[6].view(torch.int64).reshape(-1)[flag_words//2:flag_words//2+108*8].reshape(108,8).cpu()
        unique=int(w._host_batch_buffers[5][1]);chunks=(unique+19)//20
        assert int(records[:,0].sum())==chunks,(records[:,:4],unique)
        assert int(records[:,1].sum())==chunks
        assert int(records[:,3].sum())==unique
        print('UNIFORM_COUNTS',length,step,unique,chunks,int(records[:,2].sum()),flush=True)
