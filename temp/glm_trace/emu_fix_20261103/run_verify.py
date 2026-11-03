import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sys.path.insert(0,str(p));from resources_gpu3 import snapshot
lock=(p.parent/'tune_general_20261031/gpu3_correctness.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
before=snapshot();assert before['available'],before
started=time.strftime('%FT%T%z')
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3')
r=subprocess.run(['numactl','-N1','-m1',str(p/'verify')],cwd=root,env=env,text=True,capture_output=True,timeout=60)
(p/'verify.stdout').write_text(r.stdout);(p/'verify.stderr').write_text(r.stderr)
assert r.returncode==0,(r.returncode,r.stderr,r.stdout[-1000:])
d=json.loads(r.stdout);d.update(start=started,end=time.strftime('%FT%T%z'),resources_start=before,resources_end=snapshot(),source_sha256=hashlib.sha256((p/'verify.cu').read_bytes()).hexdigest(),emulator_sha256=hashlib.sha256((root/'include/flashinfer/attention/far_link_emu.cuh').read_bytes()).hexdigest())
assert not d['resources_end']['foreign'],d['resources_end'];(p/'verification.json').write_text(json.dumps(d,indent=2)+'\n')
print('Reservation cases:',len(d['reservation_cases']),'bad:',sum(x['bad'] for x in d['reservation_cases']))
print('Arithmetic/stage cases:',d['arithmetic_cases'],d['stage_cases'])
for c in d['payload_cases']:
 c['mean_GBps']=c['bytes']/(sum(c['samples_us'])/len(c['samples_us']))/1000
 print(c)
(p/'verification.json').write_text(json.dumps(d,indent=2)+'\n')
