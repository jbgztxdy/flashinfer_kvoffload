import concurrent.futures,hashlib,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
def build(mode):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT='anchor',FLASHINFER_WORKSPACE_BASE=str(p/('jit_anchor'+('_normal' if mode=='normal' else ''))),FLASHINFER_EXTRA_CUDAFLAGS='' if mode=='normal' else '-DFLASHINFER_FAR_EMU')
 rec=dict(variant='anchor',mode=mode,start=time.strftime('%FT%T%z'),header_sha256=hashlib.sha256((p/'anchor.cuh').read_bytes()).hexdigest())
 with (p/f'compile_anchor_{mode}.log').open('w') as log:
  child=subprocess.Popen(['/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'compile.py')],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  rec['pid']=child.pid;print('BUILD_START',mode,child.pid,flush=True);rec['exit_code']=child.wait()
 rec['end']=time.strftime('%FT%T%z');print('BUILD_END',mode,rec['exit_code'],flush=True);return rec
with concurrent.futures.ThreadPoolExecutor(2) as pool:records=list(pool.map(build,['emu','normal']))
(p/'build_anchor.json').write_text(json.dumps(records,indent=2)+'\n')
assert all(r['exit_code']==0 for r in records)
