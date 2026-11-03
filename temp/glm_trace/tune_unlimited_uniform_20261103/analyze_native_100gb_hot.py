from pathlib import Path
import json,statistics,hashlib
p=Path(__file__).resolve().parent;root=p.parents[2];nativepath=p/'base_native_unlimited_f24.json';n=json.loads(nativepath.read_text());gappath=p/'step_gap_100gb.json';g=json.loads(gappath.read_text());out=[]
assert n['complete'] and n['config']['positions']==256
for name in ['base_g100_l0_f24','base_g100_l20000_f24']:
 case=next(x for x in g['cases'] if x['name']==name);steps=[]
 for a in case['steps']:
  t=a['step'];u=statistics.mean(x['us'][t] for x in n['passes']);steps.append(dict(step=t,dominant=a['dominant'],device_reference_us=a['compute_us'],native_us=u,g100_us=a['mean_us'],native_over_device_us=u-a['compute_us'],g100_minus_native_us=a['mean_us']-u,excess_over_ideal_us=a['excess_us']))
 hot=[x for x in steps if x['dominant']=='compute'];out.append(dict(name=name,compute_reference_dominant_steps=len(hot),hot_mean_native_over_device_us=statistics.mean(x['native_over_device_us'] for x in hot),hot_mean_g100_minus_native_us=statistics.mean(x['g100_minus_native_us'] for x in hot),hot_mean_excess_us=statistics.mean(x['excess_over_ideal_us'] for x in hot),steps=steps))
r=dict(native_path=str(nativepath.relative_to(root)),native_sha256=hashlib.sha256(nativepath.read_bytes()).hexdigest(),step_gap_sha256=hashlib.sha256(gappath.read_bytes()).hexdigest(),scope='cross-run diagnostic only: earlier normal native binary vs EMU binary, different cache admission ties/clocks possible; not causal subtraction or corrected timing',cases=out);(p/'native_100gb_hot_comparison.json').write_text(json.dumps(r,indent=2)+'\n')
for c in out:print({k:v for k,v in c.items() if k!='steps'})
