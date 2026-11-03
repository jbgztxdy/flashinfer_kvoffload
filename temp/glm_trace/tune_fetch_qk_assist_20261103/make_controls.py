"""Same-binary controls: required attention remains fully computed in all modes."""
from pathlib import Path
import hashlib,json,shutil
p=Path(__file__).resolve().parent;root=p.parents[2];assert not(p/'controls.cuh').exists()
s=(p/'fused.cuh').read_text();s=s.replace('FusedFetchAssist','ControlFetchAssist').replace('fused_fetch_assist_','control_fetch_assist_')
s=s.replace('if(host_batch_ld_acquire(&all->helper_done)>=p.fetch_ctas || !p.state[kHostBatchUnique])', 'if(p.host_item_entries==63 || host_batch_ld_acquire(&all->helper_done)>=p.fetch_ctas || !p.state[kHostBatchUnique])')
s=s.replace('if(run){uint32_t count=', 'if(run && p.host_item_entries>=64){uint32_t count=')
(p/'controls.cuh').write_text(s);shutil.copytree(root/'include',p/'include_controls');(p/'include_controls/flashinfer/attention/mla_host_batch.cuh').write_text(s)
m=json.loads((p/'manifest.json').read_text());m['controls_sha256']=hashlib.sha256(s.encode()).hexdigest();m['control_modes']={'63':'4warp fetch +all local worker math, no exchange','62':'same plus worker bookkeeping, no helper claim','64':'actual complete assist path'};(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
# Each mode computes all outputs once and preserves all necessary cache actions.
s=(p/'bench.py').read_text();s=s.replace('w.host_item_entries=64','w.host_item_entries=A.host_item');s=s.replace("VARIANT+'_'+setting['name']+'_f'+str(A.fetch_ctas)+'.json'", "VARIANT+'_'+setting['name']+'_f'+str(A.fetch_ctas)+'_i'+str(A.host_item)+'.json'")
s=s.replace("if VARIANT=='fused':", "if VARIANT=='controls':");(p/'bench_controls.py').write_text(s)
s=(p/'run_fused.py').read_text().split("m=json.loads((p/'manifest.json').read_text())")[0].replace("p/'fused_state.json'","p/'controls_state.json'")
s += '''save()
for item in [63,62,64]:
 run('controls_i'+str(item),'controls',['-u',str(p/'bench_controls.py'),'--kind','lru','--mtp','8','--requests','128','--positions','256','--passes','1','--settings','g100_l20000','--fetch-ctas','24','--host-split','off','--host-item',str(item)])
r['complete']=True;r['end']=time.strftime('%FT%T%z');save()
'''
(p/'run_controls.py').write_text(s)
