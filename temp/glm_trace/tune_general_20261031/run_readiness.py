"""Serial paired readiness screening; complete trace checks are a separate stage."""
import argparse,json,os,re,subprocess,time
from pathlib import Path
from resources import snapshot
p=Path(__file__).resolve().parent
root=p.parents[2]
base_env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUDA_HOME='/usr/local/cuda-13.1',FLASHINFER_CUDA_ARCH_LIST='8.0',PYTHONPATH=str(root),TMPDIR=str(root/'temp'),FLASHINFER_EXTRA_CUDAFLAGS='-DFLASHINFER_FAR_EMU')
parser=argparse.ArgumentParser()
parser.add_argument('--variants',nargs='+',choices=('base','warp_ready','row_ready','defer_cache'),default=['base','warp_ready'])
parser.add_argument('--tag',default='screen256_')
parser.add_argument('--positions',type=int,default=256)
parser.add_argument('--passes',type=int,default=3)
parser.add_argument('--mtp',type=int,default=8)
parser.add_argument('--requests',type=int,default=128)
parser.add_argument('--kind',choices=('lru','nocache'),default='lru')
parser.add_argument('--route',choices=('staged','inplace'),default='staged')
parser.add_argument('--fetch-ctas',type=int,default=24)
parser.add_argument('--host-split',choices=('on','off'),default='off')
parser.add_argument('--settings',nargs='+',default=['g100_l0','g100_l20000','g50_l20000','nvlink'])
args=parser.parse_args()
assert re.fullmatch(r'[A-Za-z0-9_]+_',args.tag)
state=[]
for variant in args.variants:
    while not snapshot()['available']:
        print('WAITING',snapshot(),flush=True)
        time.sleep(30)
    env=dict(base_env,TUNE_VARIANT=variant,TUNE_RUN_TAG=args.tag,FLASHINFER_WORKSPACE_BASE=str(p/('jit_'+variant)))
    cmd=['numactl','-N0','-m0','/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'bench.py'),'--kind',args.kind,'--mtp',str(args.mtp),'--requests',str(args.requests),'--settings',*args.settings,'--passes',str(args.passes),'--positions',str(args.positions),'--fetch-ctas',str(args.fetch_ctas),'--host-split',args.host_split,'--route',args.route]
    entry=dict(variant=variant,start=time.strftime('%FT%T%z'),command=cmd)
    state.append(entry)
    print('START',variant,entry['start'],flush=True)
    with (p/(args.tag+variant+f'_m{args.mtp}_b{args.requests}_{args.kind}_{args.route}.log')).open('w') as log:
        child=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        entry['pid']=child.pid
        (p/(args.tag+'state.json')).write_text(json.dumps(state,indent=2)+'\n')
        entry['exit_code']=child.wait()
    entry['end']=time.strftime('%FT%T%z')
    (p/(args.tag+'state.json')).write_text(json.dumps(state,indent=2)+'\n')
    if entry['exit_code']:raise SystemExit(entry['exit_code'])
    print('DONE',variant,entry['end'],flush=True)
print('SCREEN_COMPLETE',flush=True)
