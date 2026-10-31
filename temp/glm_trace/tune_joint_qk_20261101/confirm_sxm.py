"""Full5pass SXM three-way confirmations, serial stages and resumable exact source checks."""
import hashlib,json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
probe=p/'dispatch_probe_joint_qk_sxm.json'
while not probe.exists():
 print('WAITING_ACTUAL_SXM_PROBE',flush=True);time.sleep(10)
r=read(probe);assert r['kernel_nodes']==1 and r['physical_gpu']==0
assert r['header_sha256']==sha(p/'joint_qk.cuh') and r['math_header_sha256']==sha(p/'joint_qk_mla.cuh')
assert r['launches'][0]['grid']==[1,108,1] and r['launches'][0]['block']==[32,4,2]
plan=dict(positions=1024,passes=5,mtp=8,requests=128,fetch_ctas=24,
 base_sha256=sha(p/'base.cuh'),joint_mdo_sha256=sha(p/'joint_mdo_typed.cuh'),joint_qk_sha256=sha(p/'joint_qk.cuh'),
 stages=[dict(tag='compositionsxmreverse1024_f24_',variants=['joint_qk','joint_mdo_typed','base'],settings=['g100_l0','g100_l20000','g50_l20000'],native=False),
 dict(tag='compositionsxmnative1024_f24_',variants=['base','joint_mdo_typed','joint_qk'],settings=['nvlink','pcie'],native=True),
 dict(tag='compositionsxmforward1024_f24_',variants=['base','joint_mdo_typed','joint_qk'],settings=['g100_l0','g100_l20000','g50_l20000'],native=False)])
f=p/'sxm_confirmation_plan.json'
if f.exists():assert read(f)==plan
else:f.write_text(json.dumps(plan,indent=2)+'\n')
statefile=p/'sxm_confirmation_state.json';state=read(statefile) if statefile.exists() else []
for stage in plan['stages']:
 complete=True
 for v in stage['variants']:
  for setting in stage['settings']:
   f=p/'raw'/f"{stage['tag']}{v}_staged_f24_i64_soff_lru_{setting}_mtp8_b128.json"
   if not f.exists() or not read(f).get('complete'):complete=False;continue
   d=read(f);assert d['configuration']['positions']==1024 and d['configuration']['passes']==5
   assert d['source_sha256']['include/flashinfer/attention/mla_host_batch.cuh']==sha(p/(v+'.cuh'))
 if complete:print('SKIP_COMPLETE_COHORT',stage['tag'],flush=True);continue
 cmd=['/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'screen.py'),'--hardware','sxm','--candidate','joint_qk','--variants',*stage['variants'],'--tag',stage['tag'],'--fetch-ctas','24','--positions','1024','--passes','5','--settings',*stage['settings']]
 if stage['native']:cmd+=['--native']
 record=dict(stage,start=time.strftime('%FT%T%z'),command=cmd);state.append(record)
 with (p/(stage['tag']+'screen.log')).open('a') as log:
  child=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT);record['pid']=child.pid;statefile.write_text(json.dumps(state,indent=2)+'\n');record['exit_code']=child.wait()
 record['end']=time.strftime('%FT%T%z');statefile.write_text(json.dumps(state,indent=2)+'\n')
 if record['exit_code']:raise SystemExit(record['exit_code'])
 print('CONFIRMATION_COHORT_COMPLETE',stage['tag'],flush=True)
print('SXM_CONFIRMATIONS_COMPLETE',flush=True)
