from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'bench_nvlink_retained.py').read_text().split('def run_perf(setting):')[0]
s+='''def run_floor():
    torch.manual_seed(META['seed']);near=min(R,128)*C
    kv=tiers.TieredKV(near,R*C,R*CAP+NNZ,0,'peer',1);main=kv.main();alias=kv.alias()
    for b in range(0,kv.near+kv.far,1<<20):main[b:b+(1<<20)].normal_()
    q=torch.randn(ROWS,64,512,dtype=torch.bfloat16,device=DEV);qr=torch.randn(ROWS,64,64,dtype=torch.bfloat16,device=DEV)
    idx=torch.empty(NNZ,dtype=torch.int32,device=DEV);o=torch.empty_like(q)
    w=make_wrapper(ROWS,idx,torch.empty(256<<20,dtype=torch.uint8,device=DEV),K)
    pages,_=row_pages(near,1.)
    meta=chunk_metadata(pages,None,0,STEPS,near,floor=True)
    def prep(d):
        idx.copy_(d['remap'])
        torch.index_select(main[near:],0,d['src'],out=alias[near:near+d['src'].numel()])
    def full():w.run(q,qr,alias[...,:512],alias[...,512:],out=o)
    prep(meta[0]);full();torch.cuda.synchronize()
    result=dict(kind='specified_device_reference',residency=kv.residency,passes=[],complete=False,resources_start=RESOURCE_START,hardware_start=hardware(),source_sha256=hashlib.sha256(EXECUTED_BENCH_SOURCE).hexdigest())
    target=DEST/'nvlink_20261104_floor.json';assert not target.exists()
    for rep in range(PASSES):
        times=[]
        for t,d in enumerate(meta):
            prep(d)
            g=torch.cuda.CUDAGraph()
            start=torch.cuda.Event(enable_timing=True,external=True);end=torch.cuda.Event(enable_timing=True,external=True)
            with torch.cuda.graph(g):start.record();full();end.record()
            g.replay();torch.cuda.synchronize();times.append(start.elapsed_time(end)*1000)
            if t in [0,16,31]:torch.testing.assert_close(o[[0,512,1023]].float(),reference(main,q,qr,d['selected'].view(ROWS,K),[0,512,1023]),atol=.01,rtol=.01)
        result['passes'].append(dict(us=times));save(target,result);print('FLOOR',rep,sum(times)/len(times),flush=True)
    result['sources']=sources();result['resources_end']=require_exclusive();result['complete']=True;save(target,result)
run_floor()
'''
(p/'bench_nvlink_floor.py').write_text(s)
