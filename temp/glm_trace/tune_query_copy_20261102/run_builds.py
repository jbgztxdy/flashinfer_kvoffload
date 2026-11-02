import concurrent.futures, hashlib, json, os, subprocess, time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
def build(v):
    records=[]
    for mode in ['emu','normal']:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),MAX_JOBS='2',TUNE_VARIANT=v,FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+v+('_normal' if mode=='normal' else ''))),FLASHINFER_EXTRA_CUDAFLAGS='' if mode=='normal' else '-DFLASHINFER_FAR_EMU')
        item=dict(variant=v,mode=mode,start=time.strftime('%FT%T%z'),header_sha256=hashlib.sha256((p/(v+'.cuh')).read_bytes()).hexdigest())
        with (p/f'compile_{v}_{mode}.log').open('w') as log:
            child=subprocess.Popen(['/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'compile.py')],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            item['pid']=child.pid;print('BUILD_START',v,mode,child.pid,flush=True);item['exit_code']=child.wait()
        item['end']=time.strftime('%FT%T%z');records.append(item)
        (p/f'build_{v}.json').write_text(json.dumps(records,indent=2)+'\n')
        print('BUILD_END',v,mode,item['exit_code'],flush=True)
        assert item['exit_code']==0
with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(build,['base','query_copy']))
