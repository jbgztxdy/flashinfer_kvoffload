from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'bench_pooldependency.py').read_text().replace("VARIANT+'_pooldependency_'","VARIANT+'_claimrow_dependency_'")
a="    result['final_step_dependency']=dict(rows=census,unique=int(state[1].cpu()),scope='final step only; first paired tiles containing actual far entries; normal trace K2048; counters event external')"
b='''    rowbase=w._host_batch_buffers[8][2*NNZ:2*NNZ+ROWS+1].cpu().long()
    assert rowbase[0]==0 and rowbase[-1]==unique_count and bool((rowbase[1:]>=rowbase[:-1]).all())
    packet_first=torch.arange((unique_count+19)//20)*20
    claimrow=torch.searchsorted(rowbase[1:],packet_first,right=True)
    claimowner=claimrow%108
    load=torch.bincount(claimowner,minlength=108)
    new_owner=claimowner[packet[far]]
    old_owner=owner[far]
    row_owner=reference_rows%108
    result['claimrow_dependency']=dict(row_slot_base=rowbase.tolist(),packet_owner=claimowner.tolist(),packet_counts_per_cta=load.tolist(),unique=unique_count,far_references=int(far.sum()),old_self_references=int((old_owner==row_owner).sum()),new_self_references=int((new_owner==row_owner).sum()),compute_rows_per_cta=torch.bincount(torch.arange(ROWS)%108,minlength=108).tolist(),scope='last step only; arithmetic locality/load analysis from realpostlaunch rowbases/remap, noGPU producer reassignment or performance prediction')
    # Preserve pair dependencies under the proposed ownership, without changing the kernel.
    pair_dependencies=[]
    for rowid in range(ROWS):
      entry=census[rowid];end=entry['first_far_pair_end'];mask=far[rowid,end-64:end]
      pk=packet[rowid,end-64:end][mask];co=claimowner[pk]
      pair_dependencies.append(dict(row=rowid,producer_ctas=int(co.unique().numel()),self_references=int((co==rowid%108).sum())))
    result['claimrow_dependency']['first_far_pair_dependencies']=pair_dependencies
'''+a
assert s.count(a)==1;s=s.replace(a,b);(p/'bench_claimrow_dependency.py').write_text(s)
s=(p/'run_pooldependency.py').read_text().replace('pooldependency','claimrow_dependency');(p/'run_claimrow_dependency.py').write_text(s)
