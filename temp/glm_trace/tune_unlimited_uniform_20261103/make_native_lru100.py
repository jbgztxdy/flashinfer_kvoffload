from pathlib import Path
import shutil
p=Path(__file__).resolve().parent
shutil.copytree(p/'include_retained',p/'include_native_lru100',symlinks=True)
s=(p/'bench_nvlink_native_lru_replay.py').read_text().replace('from resources_nvlink import','from resources_gpu3 import').replace("=='0', NUMA_POLICY","=='1', NUMA_POLICY").replace('kernel_report_20261029/gpu01.lock','tune_general_20261031/gpu3_correctness.lock').replace('"--id=0,1"','"--id=3"').replace("0,'peer',1)","0,'peer',0)")
s=s.replace("'native_direct_lru_replay_20261104.json'","'native_lru100_normal_gpu3.json'")
s=s.replace('    del donor;torch.cuda.synchronize()', '''    archive=DEST/'native_lru100_mapping.pt';assert not archive.exists()
    torch.save(dict(meta=[{k:v.cpu() for k,v in d.items() if k!='selected'} for d in meta],summaries=summaries,bindings=bindings,near=near,farstart=farstart,seed=META['seed']),archive)
    mapping_sha=digest(archive)
    del donor;torch.cuda.synchronize()''')
s=s.replace("kind='native_FA2_direct_replay_exact_production_LRU'","mapping_sha256=mapping_sha,kind='native_FA2_direct_replay_exact_production_LRU'")
needle="    result['libraries']=";i=s.index(needle,s.index('def run_replay'))
s=s[:i]+'''    # Fresh pure-device reference on the same GPU and full trace, with prep outside event.
    alias=kv.alias();localmeta=chunk_metadata(pages,None,0,STEPS,farstart,floor=True)
    for t,d in enumerate(localmeta):
        idx.copy_(d['remap']);torch.index_select(main[farstart:],0,d['src'],out=alias[farstart:farstart+d['src'].numel()])
        if t==0:w.run(q,qr,alias[...,:512],alias[...,512:],out=o)
    local_events=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in localmeta]
    local_graph=torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(local_graph):
        for t,d in enumerate(localmeta):
            idx.copy_(d['remap']);torch.index_select(main[farstart:],0,d['src'],out=alias[farstart:farstart+d['src'].numel()])
            local_events[t][0].record();w.run(q,qr,alias[...,:512],alias[...,512:],out=o);local_events[t][1].record()
    local_graph.replay();torch.cuda.synchronize()
    result['pure_device_us']=[a.elapsed_time(b)*1000 for a,b in local_events]
    torch.testing.assert_close(o.index_select(0,checkids).float(),reference(main,q,qr,selected[-1].view(ROWS,K),[0,512,1023]),atol=.01,rtol=.01)
    print('PURE DEVICE GPU3',sum(result['pure_device_us'])/STEPS,flush=True)
''' +s[i:]
(p/'bench_native_lru100_collect.py').write_text(s)
# EMU phase reuses precisely recorded normal-build mappings and random payload seed.
prefix=s.split('def run_replay():')[0]
body='''def run_emu():
    assert os.environ.get('FLASHINFER_EXTRA_CUDAFLAGS')=='-DFLASHINFER_FAR_EMU'
    torch.manual_seed(META['seed']);near=R*C;slots=R*CAP;farstart=near+slots
    kv=tiers.TieredKV(farstart,R*C,R*CAP+NNZ,0,'peer',0);main=kv.main()
    for b in range(0,kv.near+kv.far,1<<20):main[b:b+(1<<20)].normal_()
    q=torch.randn(ROWS,64,512,dtype=torch.bfloat16,device=DEV);qr=torch.randn(ROWS,64,64,dtype=torch.bfloat16,device=DEV)
    idx=torch.empty(NNZ,dtype=torch.int32,device=DEV);o=torch.empty_like(q)
    w=make_wrapper(ROWS,idx,torch.empty(256<<20,dtype=torch.uint8,device=DEV),K)
    archive=DEST/'native_lru100_mapping.pt';record=torch.load(archive,weights_only=False);assert record['seed']==META['seed'] and record['farstart']==farstart
    meta=[{k:v.to(DEV) for k,v in d.items()} for d in record['meta']]
    packed=torch.empty((slots,1,576),dtype=torch.bfloat16,device=DEV)
    pages,_=row_pages(farstart,1.);selected=[x.reshape(-1).to(DEV) for x in pages]
    checkids=torch.tensor([0,512,1023],device=DEV)
    def prep(d):
        idx.copy_(d['indices']);n=d['src'].numel()
        if n:torch.index_select(main,0,d['src'],out=packed[:n]);main.index_copy_(0,d['dst'],packed[:n])
    def full():w.run(q,qr,main[...,:512],main[...,512:],out=o)
    result=dict(mapping_sha256=digest(archive),steps=record['summaries'],config=dict(gpu=3,requests=R,mtp=M,positions=A.positions,capacity=CAP,model='production v2 CAS warp packets',passes=1),settings=[],resources_start=RESOURCE_START,complete=False,header_sha256=digest(VARIANT_INCLUDE/'flashinfer/attention/mla_host_batch.cuh'),far_sha256=digest(VARIANT_INCLUDE/'flashinfer/attention/far_link_emu.cuh'),harness_sha256=hashlib.sha256(EXECUTED_BENCH_SOURCE).hexdigest())
    target=DEST/'native_lru100_emu_gpu3.json';assert not target.exists()
    for label,bw,lat in [('disabled',None,0),('unlimited0',0,0),('g100_l0',100,0),('g100_l20000',100,20000)]:
        if bw is None:os.environ.pop('FLASHINFER_FAR_EMU',None)
        else:emu.kernel_env(bw,lat,main[farstart].data_ptr(),main[farstart].data_ptr()+kv.far*1152)
        prep(meta[0]);full();torch.cuda.synchronize()
        events=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in meta]
        graph=torch.cuda.CUDAGraph(keep_graph=True);expected=[]
        with torch.cuda.graph(graph):
            for t,d in enumerate(meta):
                prep(d);events[t][0].record();full();events[t][1].record()
                if t in [0,16,127]:expected.append((t,o.index_select(0,checkids).float().clone()))
        graph.replay();torch.cuda.synchronize()
        for t,got in expected:torch.testing.assert_close(got,reference(main,q,qr,selected[t].view(ROWS,K),[0,512,1023]),atol=.01,rtol=.01)
        from nvlink_graph import inspect_graph
        launches=inspect_graph(graph);attention=[x for x in launches if 'BatchMLAPagedAttentionKernel' in x['kernel_name']];assert len(attention)==STEPS
        times=[a.elapsed_time(b)*1000 for a,b in events]
        result['settings'].append(dict(name=label,bandwidth_GBps=bw,latency_ns=lat,us=times,launches=attention[:1]));save(target,result);print('NATIVE EMU',label,sum(times)/STEPS,flush=True)
        del graph,events,expected
    result['libraries']=[dict(path=f,sha256=digest(f)) for f in {z.split()[-1] for z in Path('/proc/self/maps').read_text().splitlines() if '.so' in z and 'jit_' in z}];result['resources_end']=require_exclusive();result['sources']=sources();result['complete']=True;save(target,result)
run_emu()
'''
(p/'bench_native_lru100_emu.py').write_text(prefix+body)
