from pathlib import Path
p=Path(__file__).resolve().parent;f=p/'bench_claimrow_quota.py';s=f.read_text()
a="    resident,sp,stamp,store=w._host_cache_views();slots=(sp[:store['slots']]>=0).nonzero().flatten()"
b='''    # Event-external exact inspection of the actual GPU-produced packet list.
    unique=int(state[1].cpu());n=(unique+19)//20;grid=108
    words=w._host_batch_buffers[6].view(torch.int32).flatten().cpu().long()
    offset=(((NNZ+19)//20+1)//2*2)+128*16
    actual=words[offset:offset+n].tolist()
    assert sorted(actual)==list(range(n))
    bases=w._host_batch_buffers[8][2*NNZ:2*NNZ+ROWS+1].cpu().long().tolist()
    capacities=[n//grid+(t<n%grid) for t in range(grid)]
    streams=[[] for _ in range(grid)]
    if unique:
      assert bases[0]==0 and bases[-1]==unique and all(a<=b for a,b in zip(bases,bases[1:]))
      for rowid in range(ROWS):streams[rowid%grid].extend(range((bases[rowid]+19)//20,(bases[rowid+1]+19)//20))
    keep=[min(len(a),b) for a,b in zip(streams,capacities)]
    overflow=[v for t,a in enumerate(streams) for v in a[keep[t]:]]
    expected=[];cursor=0;owners=torch.empty(n,dtype=torch.int64)
    for t in range(grid):
      need=capacities[t]-keep[t];part=streams[t][:keep[t]]+overflow[cursor:cursor+need];cursor+=need
      expected.extend(part)
      if part:owners[torch.tensor(part,dtype=torch.int64)]=t
    assert expected==actual and cursor==len(overflow)
    assert bool((words[:n]==int(state[3].cpu())).all())
    remap=w._host_batch_buffers[3].view(ROWS,K).cpu().long();slot=-remap-1-R*CAP;far=slot>=0
    packets=slot[far]//20;consumer=torch.arange(ROWS)[:,None].expand(ROWS,K)[far]%grid
    result['packet_plan_validation']=dict(exact_cpu_match=True,ready_epochs_match=True,total_packets=n,capacities=capacities,kept_packets=sum(keep),spill_packets=len(overflow),far_references=int(far.sum()),actual_self_references=int((owners[packets]==consumer).sum()),roundrobin_self_references=int((packets%grid==consumer).sum()),scope='actual final step GPU packettable +remap, eventexternal; nohardwarebyte counter')
'''+a
assert s.count(a)==1;s=s.replace(a,b);f.write_text(s)
