from pathlib import Path
import json, bisect, math, statistics
p=Path(__file__).resolve().parent
raw=json.loads((p/'claimrow_timeline_packet_weights_native_unlimited_f24.json').read_text())
d=raw['packet_weight_inputs']; rows=d['row_slot_base']; weights=d['per_packet_admit']; pages=d['per_packet_pages']
N=len(weights); G=108
streams=[[] for _ in range(G)]
for row in range(len(rows)-1):
    streams[row%G].extend(range((rows[row]+19)//20,(rows[row+1]+19)//20))
assert sorted(x for a in streams for x in a)==list(range(N))
def assignment(local_cap, interleave):
    capacity=[N//G+(i<N%G) for i in range(G)]
    keep=[min(len(a), capacity[i], local_cap) for i,a in enumerate(streams)]
    extra=[len(a)-keep[i] for i,a in enumerate(streams)]
    need=[capacity[i]-keep[i] for i in range(G)]
    ep=[0];dp=[0]
    for a,b in zip(extra,need):ep.append(ep[-1]+a);dp.append(dp[-1]+b)
    assert ep[-1]==dp[-1]
    E=ep[-1];stride=G+1
    if interleave:
        while math.gcd(stride,E)!=1:stride+=1
    else:stride=1
    owner=[-1]*N
    for dst in range(G):
        for packet in streams[dst][:keep[dst]]:owner[packet]=dst
        for j in range(need[dst]):
            g=dp[dst]+j
            perm=g*stride%E if E else 0
            src=bisect.bisect_right(ep,perm)-1
            packet=streams[src][keep[src]+perm-ep[src]]
            assert owner[packet]<0
            owner[packet]=dst
    assert all(0<=x<G for x in owner)
    wc=[0]*G;bc=[0]*G;pc=[0]*G
    for packet,dst in enumerate(owner):
        pc[dst]+=1;wc[dst]+=weights[packet];bc[dst]+=weights[packet]+pages[packet]
    assert max(pc)-min(pc)<=1
    return dict(local_cap=local_cap,interleave=interleave,stride=stride,kept=sum(keep),extra=E,
                admitted_cv=statistics.pstdev(wc)/statistics.mean(wc),admitted_range=[min(wc),max(wc)],
                payload_cv=statistics.pstdev(bc)/statistics.mean(bc),payload_range=[min(bc),max(bc)],
                owner=owner)
controls=[assignment(0,False),assignment(N,False)]
proposal=assignment((N//G)//2,True)
original=d['actual_packet_owner']
assert proposal['owner']!=original
assert controls[1]['owner']==original
rr_owner=[i%G for i in range(N)]
rr_admit=[0]*G;rr_payload=[0]*G
for packet,dst in enumerate(rr_owner):
    rr_admit[dst]+=weights[packet];rr_payload[dst]+=weights[packet]+pages[packet]
true_rr=dict(admitted_cv=statistics.pstdev(rr_admit)/statistics.mean(rr_admit),
             admitted_range=[min(rr_admit),max(rr_admit)],
             payload_cv=statistics.pstdev(rr_payload)/statistics.mean(rr_payload),
             payload_range=[min(rr_payload),max(rr_payload)])
summary=dict(scope='actual final-step row prefixes and LRU admitted pages; CPU counterfactual, not measured kernel performance or hardware transactions',
             unique=rows[-1],packets=N,true_roundrobin=true_rr,zero_local_sequential=controls[0],original_local_first=controls[1],half_local_interleaved=proposal)
(p/'claimrow_interleave_model.json').write_text(json.dumps(summary,indent=2)+chr(10))
for k in ['true_roundrobin','zero_local_sequential','original_local_first','half_local_interleaved']:
    x=summary[k];print(k,{z:x[z] for z in ['admitted_cv','admitted_range','payload_cv','payload_range']})
