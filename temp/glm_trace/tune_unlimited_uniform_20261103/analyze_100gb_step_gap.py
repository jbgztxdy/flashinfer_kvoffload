from pathlib import Path
import json,statistics,hashlib
p=Path(__file__).resolve().parent;root=p.parents[2];refpath=root/'temp/glm_trace/tune_scatter_20261031/gpu3/raw/reference_base_staged_f24_i64_soff_floor_floor_mtp8_b128.json';ref=json.loads(refpath.read_text());compute=[statistics.mean(x) for x in ref['cases'][0]['timings']['local']['per_step_samples_us'][:32]];out=[]
for name in ['base_g100_l0_f24','base_g100_l20000_f24','staged_g100_l0_f24','staged_g100_l20000_f24']:
 f=p.parent/'emu_packet_accuracy_20261103'/f'{name}.json';r=json.loads(f.read_text());assert r['complete'];steps=[];buckets={k:[] for k in ['compute','transfer']}
 for t,c in enumerate(compute):
  ns=[a['unique_misses'][t]*1152 for a in r['passes']];us=[a['us'][t] for a in r['passes']];ideal=statistics.mean(max(c,n/100000) for n in ns);kind='transfer' if statistics.mean(ns)/100000>c else 'compute';d=dict(step=t,mean_us=statistics.mean(us),compute_us=c,transfer_us=statistics.mean(ns)/100000,stepwise_ideal_us=ideal,excess_us=statistics.mean(us)-ideal,dominant=kind);steps.append(d);buckets[kind].append(d)
 result=dict(name=name,raw_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),mean_us=statistics.mean(x['mean_us'] for x in steps),reference_us=statistics.mean(x['stepwise_ideal_us'] for x in steps),buckets={k:dict(steps=len(v),mean_excess_us=statistics.mean(x['excess_us'] for x in v) if v else None,contribution_to_full_mean_excess_us=sum(x['excess_us'] for x in v)/32) for k,v in buckets.items()},steps=steps);out.append(result)
record=dict(reference_path=str(refpath.relative_to(root)),reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),scope='existing sameGPU reference first32steps; per-pass per-step max retained, no new timing or reference replacement',cases=out);(p/'step_gap_100gb.json').write_text(json.dumps(record,indent=2)+'\n')
for r in out:print(r['name'],r['buckets'])
