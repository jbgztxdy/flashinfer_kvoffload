"""CPU-build normal CUDA, then run correctness only on an exclusive selected local GPU."""
import argparse,csv,fcntl,hashlib,io,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--variant',required=True,choices=('base','joint_pv'))
parser.add_argument('--gpu',type=int,required=True)
parser.add_argument('--medium-only',action='store_true')
parser.add_argument('--main-only',action='store_true')
a=parser.parse_args()
assert not (a.medium_only and a.main_only)
assert a.gpu in (0,1,3)
variant=a.variant
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
env=dict(os.environ,CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root/'temp/pytest_shim')+':'+str(p)+':'+str(root),TMPDIR=str(root/'temp'),TUNE_VARIANT=variant,MAX_JOBS='4',FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+variant+'_normal')),FLASHINFER_EXTRA_CUDAFLAGS='')
state=dict(variant=variant,physical_gpu=a.gpu,normal_build=True,performance_claim=False,
    header_sha256=sha(p/(variant+'.cuh')),test_source_sha256=sha(root/'tests/attention/test_mla_host_batch.py'),
    start=time.strftime('%FT%T%z'),complete=False)
def save():
    out=p/f'tests_{variant}_gpu{a.gpu}.json'
    tmp=out.with_suffix('.json.new');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out)
def snapshot():
    gpu=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    r=next(csv.reader(io.StringIO(gpu)))
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
    foreign=[dict(uuid=z[0].strip(),pid=int(z[1]),process=z[2].strip(),used_mib=z[3].strip()) for z in csv.reader(io.StringIO(apps)) if len(z)>=4 and z[0].strip()==r[1].strip()]
    return dict(timestamp=time.strftime('%FT%T%z'),gpu=int(r[0]),uuid=r[1].strip(),name=r[2].strip(),used_mib=int(r[3]),total_mib=int(r[4]),utilization=int(r[5]),foreign=foreign,available=not foreign and int(r[3])<512 and int(r[5])<10)
compile_env=dict(env,CUDA_VISIBLE_DEVICES='')
with (p/f'compile_{variant}_normal.log').open('w') as log:
    child=subprocess.Popen(['/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'compile.py')],env=compile_env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    state['compile_pid']=child.pid;save()
    state['compile_exit_code']=child.wait();save()
assert state['compile_exit_code']==0
print('NORMAL_COMPILE_COMPLETE',variant,flush=True)
lock=(p.parent/'tune_general_20261031'/f'gpu{a.gpu}_correctness.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
while True:
    before=snapshot();state['last_resource_observation']=before;save()
    if before['available']:break
    print('WAITING_GPU',a.gpu,before,flush=True);time.sleep(30)
state['resources_start']=before
numa='1' if a.gpu==3 else '0'
cmd=['numactl','-N'+numa,'-m'+numa,'/mnt/data/mhy-vllm-venv/bin/python','-m','pytest','-q','-p','no:cacheprovider','-p','jitpaths','-p','candidatepaths','tests/attention/test_mla_host_batch.py']
if variant.startswith('threshold2') and not a.main_only:
    cmd.append(str(p/'test_medium_route.py'))
    state['extra_test_source_sha256']=sha(p/'test_medium_route.py')
    if a.medium_only:cmd.remove('tests/attention/test_mla_host_batch.py')
env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(a.gpu))
state['command']=cmd
print('TEST_START',variant,'physical GPU',a.gpu,flush=True)
with (p/f'pytest_{variant}_gpu{a.gpu}.log').open('w') as log:
    child=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    state['test_pid']=child.pid;save()
    state['exit_code']=child.wait()
state['resources_end']=snapshot()
state['log_sha256']=sha(p/f'pytest_{variant}_gpu{a.gpu}.log')
state['end']=time.strftime('%FT%T%z')
state['complete']=state['exit_code']==0 and not state['resources_end']['foreign']
save()
print('TEST_COMPLETE',state['complete'],state['exit_code'],flush=True)
raise SystemExit(0 if state['complete'] else 1)
