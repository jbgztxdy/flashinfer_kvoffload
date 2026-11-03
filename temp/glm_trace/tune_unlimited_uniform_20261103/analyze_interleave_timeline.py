from pathlib import Path
import json,statistics,hashlib
p=Path(__file__).resolve().parent;root=p.parents[2]
r=json.loads((p/'claimrow_interleave_timeline_native_unlimited_f24.json').read_text())
assert r['complete'] and r['header_sha256']==hashlib.sha256((p/'claimrow_interleave_timeline.cuh').read_bytes()).hexdigest()
for lib in r['libraries']:assert hashlib.sha256((root/lib['path']).read_bytes()).hexdigest()==lib['sha256']
phases=['start','prepare_grid_done','resolve_done','plan_done','lru_grid_done','service_initialized','compute_drain_done','compute_drain_repeat']
steps=[]
for step,(event,stamps,unique) in enumerate(zip(r['passes'][0]['us'],r['passes'][0]['phase_timestamps_ns'],r['passes'][0]['unique_misses'])):
    assert len(stamps)==108 and all(len(a)==8 for a in stamps)
    assert all(all(0<a[j]<=a[j+1] for j in range(7)) for a in stamps),step
    origin=min(a[0] for a in stamps)
    latest=[max(a[i] for a in stamps) for i in range(8)]
    planner=[(a[3]-a[2])/1000 for a in stamps]
    compute=[(a[6]-a[5])/1000 for a in stamps]
    span=(latest[6]-origin)/1000
    assert span<=event+20,(step,span,event)
    steps.append(dict(step=step,event_us=event,unique=unique,latest_phase_from_start_us=dict(zip(phases,[(v-origin)/1000 for v in latest])),planner_cta_mean_us=statistics.mean(planner),planner_cta_max_us=max(planner),service_compute_cta_mean_us=statistics.mean(compute),observed_span_us=span))
summary=dict(event_us=statistics.mean(x['event_us'] for x in steps),unique=statistics.mean(x['unique'] for x in steps),phases_us={k:statistics.mean(x['latest_phase_from_start_us'][k] for x in steps) for k in phases},planner_cta_mean_us=statistics.mean(x['planner_cta_mean_us'] for x in steps),planner_cta_max_us=statistics.mean(x['planner_cta_max_us'] for x in steps),service_compute_cta_mean_us=statistics.mean(x['service_compute_cta_mean_us'] for x in steps))
result=dict(scope='normal GPU3 same-device HBM single pass full kernel with diagnostic stamps; previous quota comparison crossbinary, no strict causal subtraction or100GB prediction',source_actual_library_verified=True,summary=summary,steps=steps)
(p/'claimrow_interleave_timeline_analysis.json').write_text(json.dumps(result,indent=2)+chr(10))
print(json.dumps(summary,indent=2))
