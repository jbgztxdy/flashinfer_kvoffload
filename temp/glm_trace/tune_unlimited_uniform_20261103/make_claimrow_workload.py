from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'bench_claimrow_timeline.py').read_text().replace("VARIANT+'_'+setting['name']","VARIANT+'_workload_'+setting['name']")
a="    resident,sp,stamp,store=w._host_cache_views();slots=(sp[:store['slots']]>=0).nonzero().flatten()"
b='''    assert A.fetch_ctas==24
    victims=w._host_batch_buffers[8][:unique].cpu().long()
    per_packet_pages=torch.bincount(torch.arange(unique)//20,minlength=n)
    per_packet_admit=torch.bincount((torch.arange(unique)//20)[victims>=0],minlength=n)
    workload={}
    for policy,assignment in [('actual_quota',owners),('counterfactual_roundrobin',torch.arange(n)%grid)]:
      page_counts=torch.zeros(grid,dtype=torch.int64);admit_counts=torch.zeros(grid,dtype=torch.int64)
      page_counts.scatter_add_(0,assignment,per_packet_pages);admit_counts.scatter_add_(0,assignment,per_packet_admit)
      workload[policy]=dict(packet_counts=torch.bincount(assignment,minlength=grid).tolist(),unique_pages=page_counts.tolist(),cache_admitted_pages=admit_counts.tolist(),logical_payload_store_bytes=((page_counts+admit_counts)*1152).tolist())
    result['final_step_workload']=dict(per_cta=workload,total_unique=unique,total_admitted=int((victims>=0).sum()),last_step_timestamps_ns=result['passes'][-1]['phase_timestamps_ns'][-1],scope='actualGPU finalvictim/table metadata; RR is CPU counterfactualsamefinalstate only, notmeasuredRRtiming; logicalstoresnottransactioncounter')
'''+a
assert s.count(a)==1;s=s.replace(a,b);(p/'bench_claimrow_workload.py').write_text(s)
s=(p/'run_claimrow_timeline.py').read_text().replace("p/'claimrow_timeline_state.json'","p/'claimrow_workload_state.json'")
s='\n'.join(x for x in s.splitlines() if not x.startswith("run('correctness_claimrow_timeline'") and not x.startswith("run('claimrow_timeline_rr_native'"))+'\n'
s=s.replace("'claimrow_timeline_quota_native'","'claimrow_workload_native'").replace("bench_claimrow_timeline.py","bench_claimrow_workload.py")
(p/'run_claimrow_workload.py').write_text(s)
