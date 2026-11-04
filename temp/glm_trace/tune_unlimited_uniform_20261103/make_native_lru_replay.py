from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'bench_nvlink_native_direct.py').read_text().split('def run_native_direct():')[0]
s+='''def run_replay():
    torch.manual_seed(META['seed']);near=R*C;slots=R*CAP;farstart=near+slots
    kv=tiers.TieredKV(farstart,R*C,R*CAP+NNZ,0,'peer',1);main=kv.main()
    for b in range(0,kv.near+kv.far,1<<20):main[b:b+(1<<20)].normal_()
    assert kv.near==farstart
    q=torch.randn(ROWS,64,512,dtype=torch.bfloat16,device=DEV);qr=torch.randn(ROWS,64,64,dtype=torch.bfloat16,device=DEV)
    idx=torch.empty(NNZ,dtype=torch.int32,device=DEV);donoridx=torch.empty_like(idx);o=torch.empty_like(q);donoro=torch.empty_like(q)
    w=make_wrapper(ROWS,idx,torch.empty(256<<20,dtype=torch.uint8,device=DEV),K)
    donor=make_wrapper(ROWS,donoridx,torch.empty(256<<20,dtype=torch.uint8,device=DEV),K)
    donor.host_fetch_ctas=24;donor.host_item_entries=64;donor.host_stage=False;donor.host_split=False;donor._host_prefetch_shape=False;donor.host_staging_rows=NNZ
    donor.set_host_cache(torch.arange(R+1,dtype=torch.int32)*M,CAP)
    pages,_=row_pages(farstart,1.)
    selected=[x.reshape(-1).to(DEV) for x in pages]
    def donorfull():donor.run(q,qr,main[...,:512],main[...,512:],out=donoro,host_page_start=farstart)
    donoridx.copy_(selected[0]);donorfull();torch.cuda.synchronize();donor.release_host_cache(torch.arange(R));torch.cuda.synchronize()
    meta=[];summaries=[]
    # Untimed oracle records actual production launch-LRU choices, avoiding CPU tie/admission mismatch.
    for t,ids in enumerate(selected):
        donoridx.copy_(ids);donorfull();torch.cuda.synchronize()
        remap=donor._host_batch_buffers[3].reshape(-1).clone()
        hit=remap<0;hit_slots=(-remap[hit].long()-1);hit_pages=ids[hit].long()
        unique_slots,inverse=torch.unique(hit_slots,return_inverse=True)
        src=torch.empty(len(unique_slots),dtype=torch.int64,device=DEV)
        src.scatter_(0,inverse,hit_pages)
        assert torch.equal(src[inverse],hit_pages)
        # Cache hits are protected from eviction in this launch; validate actual donor payloads.
        resident,sp,stamp,store=donor._host_cache_views()
        assert torch.equal(sp[unique_slots],src)
        for block in unique_slots.split(32768):assert torch.equal(store['kv'][block],main[sp[block].long(),0])
        mapped=torch.where(hit,near-remap.long()-1,(remap.long()&~(1<<30))).int()
        farrefs=int((mapped>=farstart).sum());expected_far=int(((remap>=0)&((remap&~(1<<30))>=farstart)).sum());assert farrefs==expected_far
        assert torch.equal(mapped[~hit],ids[~hit])
        meta.append(dict(indices=mapped,src=src,dst=unique_slots+near,selected=ids))
        summaries.append(dict(step=t,far_references=farrefs,hit_references=int(hit.sum()),unique_hit_slots=len(unique_slots),unique_misses=int(donor._host_batch_buffers[5][1])))
    del donor;torch.cuda.synchronize()
    packed=torch.empty((slots,1,576),dtype=torch.bfloat16,device=DEV)
    def prep(d):
        idx.copy_(d['indices'])
        n=d['src'].numel()
        if n:
            torch.index_select(main,0,d['src'],out=packed[:n])
            main.index_copy_(0,d['dst'],packed[:n])
    def full():w.run(q,qr,main[...,:512],main[...,512:],out=o)
    prep(meta[0]);full();torch.cuda.synchronize()
    result=dict(kind='native_FA2_direct_replay_exact_production_LRU',config=dict(requests=R,mtp=M,steps=STEPS,positions=A.positions,passes=PASSES,capacity=CAP,topk=K,precopy_misses=False,cache_hits_materialized_outside_timing=True,LRU_decisions='untimed production direct oracle then offline mapping/payload replay',simulator=False),bindings=bindings,residency=kv.residency,steps=summaries,passes=[],complete=False,resources_start=RESOURCE_START,hardware_start=hardware(),source_sha256=hashlib.sha256(EXECUTED_BENCH_SOURCE).hexdigest())
    target=DEST/'native_direct_lru_replay_20261104.json';assert not target.exists()
    checkids=torch.tensor([0,512,1023],device=DEV)
    for rep in range(PASSES):
        events=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in meta]
        graph=torch.cuda.CUDAGraph(keep_graph=True);expected=[]
        with torch.cuda.graph(graph):
            for t,d in enumerate(meta):
                prep(d);events[t][0].record();full();events[t][1].record()
                if t in [0,16,127]:expected.append((t,o.index_select(0,checkids).float().clone()))
        graph.replay();torch.cuda.synchronize()
        for t,got in expected:torch.testing.assert_close(got,reference(main,q,qr,meta[t]['selected'].view(ROWS,K),[0,512,1023]),atol=.01,rtol=.01)
        times=[a.elapsed_time(b)*1000 for a,b in events]
        if rep==0:
            from nvlink_graph import inspect_graph
            result['launches']=inspect_graph(graph)
            attention=[x for x in result['launches'] if 'BatchMLAPagedAttentionKernel' in x['kernel_name']]
            assert len(attention)==STEPS and all(x['grid']==[1,108,1] for x in attention)
            assert not any('BatchMLADirectPVKernel' in x['kernel_name'] for x in result['launches'])
        result['passes'].append(dict(us=times));save(target,result);print('NATIVE DIRECT EXACT LRU',rep,sum(times)/len(times),flush=True)
    result['libraries']=[dict(path=f,sha256=digest(f)) for f in {z.split()[-1] for z in Path('/proc/self/maps').read_text().splitlines() if '.so' in z and 'jit_' in z}];result['sources']=sources();result['math_checks']=dict(steps=[0,16,127],rows=[0,512,1023],atol=.01,rtol=.01);result['resources_end']=require_exclusive();result['complete']=True;save(target,result)
run_replay()
'''
(p/'bench_nvlink_native_lru_replay.py').write_text(s)
s=(p/'run_nvlink_native_direct.py').read_text().replace("['native_direct']","['native_lru_replay']").replace('native_direct_state.json','native_lru_replay_state.json').replace('native_direct_clocks.csv','native_lru_replay_clocks.csv')
(p/'run_nvlink_native_lru_replay.py').write_text(s)
