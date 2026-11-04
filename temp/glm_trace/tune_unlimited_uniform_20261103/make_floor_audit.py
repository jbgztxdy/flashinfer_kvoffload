from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'bench_nvlink_full_floor.py').read_text().split('def run_floor():')[0]
s+='''def audit():
    import statistics
    torch.manual_seed(META['seed']);near=R*C
    kv=tiers.TieredKV(near,R*C,R*CAP+NNZ,0,'peer',1);main=kv.main();alias=kv.alias()
    for b in range(0,kv.near+kv.far,1<<20):main[b:b+(1<<20)].normal_()
    q=torch.randn(ROWS,64,512,dtype=torch.bfloat16,device=DEV);qr=torch.randn(ROWS,64,64,dtype=torch.bfloat16,device=DEV)
    idx=torch.empty(NNZ,dtype=torch.int32,device=DEV);o=torch.empty_like(q)
    w=make_wrapper(ROWS,idx,torch.empty(512<<20,dtype=torch.uint8,device=DEV),K)
    eviction=torch.empty(128<<20,dtype=torch.uint8,device=DEV)
    def full():w.run(q,qr,alias[...,:512],alias[...,512:],out=o)
    result=dict(rows=ROWS,records=[],complete=False,resources_start=RESOURCE_START,residency=kv.residency)
    target=DEST/'floor_audit_20261104.json';assert not target.exists()
    def measure(label,pages,repeat=1):
        meta=chunk_metadata(pages,None,0,len(pages),near,floor=True)
        for t,d in enumerate(meta):
            idx.copy_(d['remap'])
            def prep():torch.index_select(main[near:],0,d['src'],out=alias[near:near+d['src'].numel()])
            prep();full();torch.cuda.synchronize()
            for policy in ['prepared','evicted']:
                vals=[]
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                st=torch.cuda.Event(enable_timing=True,external=True);en=torch.cuda.Event(enable_timing=True,external=True)
                with torch.cuda.graph(graph):
                    st.record()
                    for _ in range(repeat):full()
                    en.record()
                for rep in range(3):
                    prep()
                    if policy=='evicted':eviction.fill_(rep)
                    graph.replay();torch.cuda.synchronize();vals.append(st.elapsed_time(en)*1000/repeat)
                if t==0:
                    from nvlink_graph import inspect_graph
                    result.setdefault('launches',[]).append(dict(label=label,policy=policy,launches=inspect_graph(graph)))
                result['records'].append(dict(label=label,step=t,policy=policy,repeats=repeat,us=vals,unique_far_pages=d['src'].numel()))
                save(target,result)
        print(label,[(pol,statistics.mean(statistics.mean(r['us']) for r in result['records'] if r['label']==label and r['policy']==pol)) for pol in ['prepared','evicted']],flush=True)
    pages,_=row_pages(near,1.)
    measure('assistant_full',pages)
    captures=tiers.load_captures(str(ROOT/'temp/mtp_overlap/overlap_ctx65536_*.pt'))
    for fraction in [0.,.3]:
        sets=[]
        for setid in range(2):
            req=tiers.Requests(captures,R,C,near,128,fraction,42+1000*R+setid,'cpu')
            ids,_,_=req.rows(0,8);sets.append(ids)
        measure('old_fraction_'+str(fraction),torch.stack(sets),repeat=3)
    result['resources_end']=require_exclusive();result['libraries']=[dict(path=f,sha256=digest(f)) for f in {z.split()[-1] for z in Path('/proc/self/maps').read_text().splitlines() if '.so' in z and 'jit_' in z}]
    result['harness_sha256']=hashlib.sha256(EXECUTED_BENCH_SOURCE).hexdigest();result['complete']=True;save(target,result)
audit()
'''
(p/'bench_nvlink_floor_audit.py').write_text(s)
s=(p/'run_nvlink_floor.py').read_text().replace("['floor']","['floor_audit']").replace('nvlink_floor_retry_state.json','floor_audit_state.json').replace("'256'","'1024'")
(p/'run_nvlink_floor_audit.py').write_text(s)
