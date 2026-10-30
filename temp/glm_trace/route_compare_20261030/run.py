"""Run route comparisons sequentially after the existing GPU0/1 owner finishes."""
import json
import os
from pathlib import Path
import subprocess
import time
from resources import snapshot

ROOT=Path(__file__).resolve().parents[3]
DEST=Path(__file__).resolve().parent
os.chdir(ROOT)
env=dict(os.environ,CUDA_DEVICE_ORDER="PCI_BUS_ID",CUDA_VISIBLE_DEVICES="0,1",
    CUDA_HOME="/usr/local/cuda-13.1",FLASHINFER_CUDA_ARCH_LIST="8.0",
    PYTHONPATH=str(ROOT),TMPDIR=str(ROOT/"temp"))
progress=[]


def prior_owner_alive():
    # Observed profiler controllers intermittently launch GPU children; include
    # their CPU-only gaps and subsequent controllers from the same experiment.
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():continue
        try:argv=(path/"cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if not argv or b"/mozi-elastic-20261022" not in argv[0]:continue
        if any(a.endswith((b"/run_profiler_audit.py",b"/run_elastic_idle.py")) for a in argv[1:]):return True
    return False


def wait_for_pair():
    quiet_since=None
    while True:
        state=snapshot()
        state["prior_owner_alive"]=prior_owner_alive()
        if state["available"] and not state["prior_owner_alive"]:
            if quiet_since is None:quiet_since=time.monotonic()
            elif time.monotonic()-quiet_since>=25:return
        else:quiet_since=None
        (DEST/"waiting.json").write_text(json.dumps(state,indent=2)+"\n")
        print("WAITING_FOR_GPU",state["timestamp"],flush=True)
        time.sleep(30)


# First establish correctness of both paths with every short-replay step checked.
# Then compare the full trace, alternating route order across batch sizes.
tasks=[]
for route in ("staged","inplace"):
    tasks.append(dict(route=route,batch=16,settings=["g100_l20000"],smoke=True))
for batch in (128,16,64):
    for simulated,settings in ((True,["g100_l0","g100_l20000","g50_l20000"]),(False,["nvlink"])):
        routes=("staged","inplace") if batch!=16 else ("inplace","staged")
        for route in routes:
            tasks.append(dict(route=route,batch=batch,settings=settings,smoke=False))
for task in tasks:
    outputs=[DEST/"raw"/(("smoke_" if task["smoke"] else "")+
        f"lru_{task['route']}_{setting}_mtp8_b{task['batch']}.json") for setting in task["settings"]]
    if all(p.exists() and json.loads(p.read_text()).get("complete") for p in outputs):continue
    wait_for_pair()
    simulated=task["settings"][0]!="nvlink"
    runenv=dict(env,FLASHINFER_WORKSPACE_BASE=str(ROOT/("temp/jit_report_emu" if simulated else "temp/jit_report_normal")))
    if simulated:runenv["FLASHINFER_EXTRA_CUDAFLAGS"]="-DFLASHINFER_FAR_EMU"
    else:runenv.pop("FLASHINFER_EXTRA_CUDAFLAGS",None)
    command=["numactl","-N0","-m0","/mnt/data/mhy-vllm-venv/bin/python","-u",str(DEST/"bench_routes.py"),
        "--kind","lru","--mtp","8","--requests",str(task["batch"]),"--route",task["route"],
        "--settings",*task["settings"]]
    if task["smoke"]:command+=["--smoke"]
    label=("smoke_" if task["smoke"] else "")+task["route"]+"_b"+str(task["batch"])+"_"+task["settings"][0]
    print("START",label,time.strftime("%FT%T%z"),flush=True)
    started=time.time()
    with (DEST/(label+".log")).open("w") as stream:
        result=subprocess.run(command,env=runenv,stdout=stream,stderr=subprocess.STDOUT)
    progress.append(dict(task=task,exit_code=result.returncode,seconds=time.time()-started))
    (DEST/"progress.json").write_text(json.dumps(progress,indent=2)+"\n")
    if result.returncode:raise SystemExit(result.returncode)
    print("DONE",label,round(time.time()-started,2),flush=True)
print("ALL_ROUTES_COMPLETE",flush=True)
