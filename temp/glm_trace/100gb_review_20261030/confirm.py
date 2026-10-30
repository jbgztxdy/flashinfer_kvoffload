"""Review existing scheduling parameters without changing the production kernel."""
import json
import os
from pathlib import Path
import subprocess
import time
from resources import snapshot

DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[2]
os.chdir(ROOT)
env=dict(os.environ,CUDA_DEVICE_ORDER="PCI_BUS_ID",CUDA_VISIBLE_DEVICES="0,1",
    CUDA_HOME="/usr/local/cuda-13.1",FLASHINFER_CUDA_ARCH_LIST="8.0",PYTHONPATH=str(ROOT),
    TMPDIR=str(ROOT/"temp"),FLASHINFER_WORKSPACE_BASE=str(ROOT/"temp/jit_report_emu"),
    FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_FAR_EMU")
tasks=[(32,64,"off"),(16,64,"on")]
progress=[]
for fetch,item,split in tasks:
    label=f"confirm_f{fetch}_i{item}_s{split}"
    outputs=[DEST/"raw"/f"{label}_lru_{s}_mtp8_b128.json" for s in ("g100_l0","g100_l20000")]
    if all(p.exists() and json.loads(p.read_text()).get("complete") for p in outputs):continue
    while True:
        state=snapshot()
        if state["available"]:break
        (DEST/"waiting.json").write_text(json.dumps(state,indent=2)+"\n")
        print("WAITING_FOR_GPU",state["timestamp"],flush=True)
        time.sleep(30)
    cmd=["numactl","-N0","-m0","/mnt/data/mhy-vllm-venv/bin/python","-u",str(DEST/"bench_confirmation.py"),
         "--kind","lru","--mtp","8","--requests","128","--settings","g100_l0","g100_l20000",
         "--passes","5","--fetch-ctas",str(fetch),"--host-item",str(item),"--host-split",split]
    print("START",label,time.strftime("%FT%T%z"),flush=True)
    start=time.time()
    with (DEST/(label+".log")).open("w") as stream:
        result=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
    progress.append(dict(fetch_ctas=fetch,item=item,split=split,exit_code=result.returncode,seconds=time.time()-start))
    (DEST/"confirmation_progress.json").write_text(json.dumps(progress,indent=2)+"\n")
    if result.returncode:raise SystemExit(result.returncode)
    print("DONE",label,round(time.time()-start,2),flush=True)
print("PARAMETER_CONFIRMATION_COMPLETE",flush=True)
