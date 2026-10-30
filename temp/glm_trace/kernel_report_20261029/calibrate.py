"""Verify physical residency and all emulated settings on the same two-GPU path."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
DEST=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/"temp/h800"),str(ROOT/"temp"),str(ROOT/"temp/sim_link")]
import torch
import tiers
import emu
torch.cuda.set_device(0);torch.set_num_threads(8)
rows=1<<20
kv=tiers.TieredKV(rows,2*rows,rows,0,"peer",1)
m,a=kv.main(),kv.alias()
for b in range(0,kv.near+kv.far,1<<20):m[b:b+(1<<20)].normal_()
ids=torch.randperm(kv.far,device="cuda")[:rows]+kv.near
dst=a[kv.near:kv.near+rows]
lo=m[kv.near].data_ptr();hi=lo+kv.far*1152
emu.kernel_env(0,0,lo,hi)
base=min(emu.latency_ns(m,ids[:4096]) for _ in range(3))
data=[]
for setting in json.loads((DEST/"experiment.json").read_text())["settings"]:
    if not setting.get("emulated"):continue
    bw,lat=setting["bandwidth_GBps"],setting["latency_ns"]
    emu.kernel_env(bw,lat,lo,hi)
    emu.gather(m,ids,dst);torch.cuda.synchronize()
    ts=[]
    for _ in range(5):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record();emu.gather(m,ids,dst);e.record();e.synchronize();ts.append(s.elapsed_time(e)*1000)
    measured=rows*1152/(sum(ts)/len(ts))/1000
    delay=min(emu.latency_ns(m,ids[:4096]) for _ in range(3))
    expected=max(lat,base)
    ok=(bw==0 or abs(measured-bw)<=.05*bw) and abs(delay-expected)<=.1*expected+600
    data.append(setting|dict(measured_GBps=measured,latency_ns_measured=delay,base_latency_ns=base,
                             samples_us=ts,ok=ok))
    print(data[-1],flush=True)
result=dict(settings=data,residency=kv.residency,complete=all(x["ok"] for x in data),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
(DEST/"calibration.json").write_text(json.dumps(result,indent=2)+"\n")
assert result["complete"]
