from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'bench_clean.py').read_text().replace("latency_ns=0,emulated=True","latency_ns=0,emulated=False")
s=s.replace("VARIANT+'_'+setting['name']", "VARIANT+'_dependency_'+setting['name']")
needle="    resident,sp,stamp,store=w._host_cache_views();"
probe='''    # Event-external final-step dependency census; no timed kernel mutation.
    remap=w._host_batch_buffers[3].view(ROWS,K).cpu().to(torch.int64)
    stage_slot=-remap-1-R*CAP
    far=stage_slot>=0
    packet=stage_slot.clamp_min(0)//20
    depths=packet//108
    owner=packet%108
    census=[]
    for rowid in range(ROWS):
        mask=far[rowid,-64:]
        ds=depths[rowid,-64:][mask]
        owners=owner[rowid,-64:][mask]
        census.append(dict(row=rowid,first_pair_far=int(mask.sum()),first_pair_max_ticket=int(ds.max()) if ds.numel() else -1,own_far=int((owners==rowid%108).sum()),producer_ctas=int(owners.unique().numel())))
    result['final_step_dependency']=dict(rows=census,unique=int(state[1].cpu()),scope='final step only; first paired tiles indices1984..2047; normal trace K2048; counters event external')
'''
assert needle in s;s=s.replace(needle,probe+needle)
(p/'bench_dependency.py').write_text(s)
s=(p/'run_clean.py').read_text();s='\n'.join(x for x in s.splitlines() if not x.startswith("run('correctness"))+'\n'
s=s.replace("p/'clean_state.json'","p/'dependency_state.json'").replace("run('clean_native'","run('dependency_native'").replace("bench_clean.py","bench_dependency.py").replace("'--passes','3'","'--passes','1'")
(p/'run_dependency.py').write_text(s)
