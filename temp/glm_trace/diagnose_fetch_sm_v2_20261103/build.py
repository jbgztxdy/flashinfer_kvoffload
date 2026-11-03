import concurrent.futures,hashlib,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
def build(v):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT=v,FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v)),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU')
 r=dict(variant=v,start=time.strftime('%FT%T%z'),header_sha256=hashlib.sha256((p/(v+'.cuh')).read_bytes()).hexdigest())
 with (p/('compile_'+v+'.log')).open('w') as f:
  child=subprocess.Popen(['/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'compile.py')],env=env,cwd=root,stdout=f,stderr=subprocess.STDOUT);r['pid']=child.pid;print('BUILD_START',v,child.pid,flush=True);r['exit_code']=child.wait()
 r['end']=time.strftime('%FT%T%z');print('BUILD_END',v,r['exit_code'],flush=True);return r
with concurrent.futures.ThreadPoolExecutor(2) as pool:records=list(pool.map(build,['trace','capacity']))
(p/'build.json').write_text(json.dumps(records,indent=2)+'\n');assert all(r['exit_code']==0 for r in records)
