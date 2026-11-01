"""Complete full confirmations and frozen regression scope; no installation."""
import hashlib,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];py='/mnt/data/mhy-vllm-venv/bin/python'
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
out=p/'extension_state.json'
state=dict(start=time.strftime('%FT%T%z'),pid=os.getpid(),phase='full_forward_native',complete=False,candidate_adopted=False,goal_complete=False,steps=[])
def save():
 t=out.with_suffix('.json.new');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(out)
def run(cmd,log):
 item=dict(command=cmd,log=log,start=time.strftime('%FT%T%z'));state['steps'].append(item)
 with (p/log).open('a') as f:
  child=subprocess.Popen(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT);item['pid']=child.pid;save();item['exit_code']=child.wait()
 item['end']=time.strftime('%FT%T%z');save()
 if item['exit_code']:raise RuntimeError('Failed '+log)
def screen(task,hardware):
 cmd=[py,'-u',str(p/'screen_store.py'),'--hardware',hardware,'--candidate','claim_store','--variants',*task['variants'],
      '--positions','1024','--passes','5','--requests','128','--mtp','8','--fetch-ctas','24','--host-split','off','--settings',*task['settings'],'--tag',task['tag']]
 if task['native']:cmd+=['--native']
 run(cmd,task['tag']+'screen.log')
def audit():run([py,str(p/'audit_performance.py')],'audit_extension.log')
def refresh():
 audit();run([py,str(p/'audit_coverage.py')],'audit_coverage.log')
 if (p/'report.py').exists():run([py,str(p/'report.py')],'report.log')
def gate(cohort,n,positive100=False):
 rows=[r for r in read(p/'performance_audit.json')['pairs'] if r['cohort']==cohort];assert len(rows)==n,(cohort,len(rows))
 assert all(r['reduction']>-.01 for r in rows),(cohort,rows)
 if positive100:assert all(r['reduction']>0 for r in rows if r['setting'].startswith('g100')),(cohort,rows)
save()
try:
 assert sha(p/'claim_store.cuh')=='15aa954ac63ce52024439446ca9dde583875eaf5afa24fd8df343075ecd9c81e'
 assert read(p/'validation_store_state.json')['complete']
 audit();gate('claim_store_reverse1024_f24_',3,True)
 tasks=[dict(tag='claim_store_forward1024_f24_',variants=['base','claim_store'],native=False,settings=['g100_l0','g100_l20000','g50_l20000']),
        dict(tag='claim_store_native1024_f24_',variants=['base','claim_store'],native=True,settings=['native_hbm','pcie'])]
 (p/'extension_plan.json').write_text(json.dumps(dict(headers={v:sha(p/(v+'.cuh')) for v in ['base','claim_store']},tasks=tasks,positions=1024,passes=5,goal_complete=False),indent=2)+'\n')
 for task in tasks:screen(task,'gpu3')
 audit();gate('claim_store_forward1024_f24_',3,True);gate('claim_store_native1024_f24_',2)
 # Existing matrix drivers require this explicit full-confirmation gate.
 state['local_confirmation_gate_passed']=True;save()
 for cohort in ['limited','native']:
  run([py,str(p/'validate_shapes.py'),'--cohort',cohort,'--plan-only'],'plan_gpu3_'+cohort+'.log')
  run([py,str(p/'validate_sxm.py'),'--cohort',cohort,'--plan-only'],'plan_sxm_'+cohort+'.log')
 state['complete']=False;state['phase']='gpu3_matrix';save()
 for cohort in ['limited','native']:run([py,'-u',str(p/'validate_shapes.py'),'--cohort',cohort],'matrix_gpu3_'+cohort+'.log')
 refresh();state['gpu3_matrix_complete']=True;state['phase']='waiting_original_CPE';save()
 while True:
  f=p.parent/'tune_cached_pe_20261101/broader_validation_state.json';c=read(f)
  live=subprocess.run(['ps','-p','2084930','-o','pid=,args='],capture_output=True,text=True).stdout.strip()
  state['original_CPE_observation']=dict(timestamp=time.strftime('%FT%T%z'),phase=c['phase'],complete=c['complete'],live_process=live);save()
  if c['complete']:break
  if not live:
   if c['phase']=='failed':break
   raise RuntimeError('Original CPE producer absent without terminal state; inspect it,do not duplicate')
  print('WAIT_ORIGINAL_CPE',state['original_CPE_observation'],flush=True);time.sleep(30)
 state['phase']='sxm_full_confirmation';save()
 tasks=[dict(tag='claim_store_sxmreverse1024_f24_',variants=['claim_store','base'],native=False,settings=['g100_l0','g100_l20000','g50_l20000']),
        dict(tag='claim_store_sxmforward1024_f24_',variants=['base','claim_store'],native=False,settings=['g100_l0','g100_l20000','g50_l20000']),
        dict(tag='claim_store_sxmnative1024_f24_',variants=['base','claim_store'],native=True,settings=['nvlink','pcie'])]
 (p/'sxm_confirmation_plan.json').write_text(json.dumps(dict(tasks=tasks,positions=1024,passes=5),indent=2)+'\n')
 for task in tasks:screen(task,'sxm')
 refresh()
 gate('claim_store_sxmreverse1024_f24_',3,True);gate('claim_store_sxmforward1024_f24_',3,True);gate('claim_store_sxmnative1024_f24_',2)
 state['phase']='sxm_matrix';save()
 for cohort in ['limited','native']:run([py,'-u',str(p/'validate_sxm.py'),'--cohort',cohort],'matrix_sxm_'+cohort+'.log')
 refresh();state.update(phase='regression_measurements_complete',complete=True,end=time.strftime('%FT%T%z'));save()
 print('REGRESSION_COMPLETE_NO_INSTALL',flush=True)
except Exception as exc:
 state.update(phase='failed',complete=False,error=str(exc),end=time.strftime('%FT%T%z'));save();raise
