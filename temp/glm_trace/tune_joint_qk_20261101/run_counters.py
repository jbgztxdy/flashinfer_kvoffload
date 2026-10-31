"""Application replay reconstructs every mutable cache/state launch; archive failures too."""
import hashlib,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
metrics=['gpu__time_duration.sum','sm__cycles_active.avg.pct_of_peak_sustained_elapsed','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct','smsp__warp_issue_stalled_barrier_per_warp_active.pct','smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct','dram__bytes_read.sum','dram__bytes_write.sum','lts__t_bytes.sum','smsp__inst_executed_pipe_tensor.sum']
state=[]
for v,kernel in [('base','BatchMLAHostBatchKernel'),('joint_qk','BatchMLAJointQKKernel')]:
 workspace=(p.parent/'tune_wait_20261031/jit_retained_normal') if v=='base' else p/'jit_joint_qk_normal'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),FLASHINFER_WORKSPACE_BASE=str(workspace),FLASHINFER_EXTRA_CUDAFLAGS='')
 cmd=['/usr/local/cuda-13.1/bin/ncu','--profile-from-start','off','--replay-mode','application','--target-processes','all','--kernel-name','regex:'+kernel,'--launch-count','1','--metrics',','.join(metrics),'--csv','--log-file',str(p/('counters_'+v+'.csv')),'numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'diagnose_counters.py'),'--variant',v]
 record=dict(variant=v,start=time.strftime('%FT%T%z'),command=cmd,performance_claim=False);state.append(record)
 with (p/('counters_'+v+'.log')).open('w') as log:
  child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);record['pid']=child.pid;(p/'counters_state.json').write_text(json.dumps(state,indent=2)+'\n');record['exit_code']=child.wait()
 record['end']=time.strftime('%FT%T%z');(p/'counters_state.json').write_text(json.dumps(state,indent=2)+'\n')
 if record['exit_code']:raise SystemExit(record['exit_code'])
 print('COUNTER_CAPTURE_DONE',v,flush=True)
