"""Wait for the original A100 pair, repeat unstable groups, then finish the matrix."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from resources import snapshot

DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[2]
os.chdir(ROOT)
lock=(DEST/"resume.lock").open("w")
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)


def save_state(state,**extra):
    data=dict(state=state,pid=os.getpid(),timestamp=time.strftime("%FT%T%z"),**extra)
    target=DEST/"resume_state.json"
    temp=target.with_suffix(".new");temp.write_text(json.dumps(data,indent=2)+"\n");temp.replace(target)
    print(json.dumps(data),flush=True)


def wait_free():
    consecutive=0
    while consecutive<2:
        state=snapshot()
        consecutive=consecutive+1 if state["available"] else 0
        save_state("waiting_for_gpu" if not state["available"] else "confirming_gpu_idle",resources=state)
        if consecutive<2:time.sleep(30)


def invoke(cmd,log,env=None):
    save_state("running",command=cmd,log=str(log.relative_to(ROOT)))
    with log.open("a") as output:
        result=subprocess.run(cmd,env=env,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:
        save_state("failed",command=cmd,exit_code=result.returncode,log=str(log.relative_to(ROOT)))
        raise SystemExit(result.returncode)


env=dict(os.environ,CUDA_DEVICE_ORDER="PCI_BUS_ID",CUDA_VISIBLE_DEVICES="0,1",
         CUDA_HOME="/usr/local/cuda-13.1",FLASHINFER_CUDA_ARCH_LIST="8.0",
         PYTHONPATH=str(ROOT),TMPDIR=str(ROOT/"temp"))
stability_path=DEST/"stability_repeat.json"
stability=json.loads(stability_path.read_text())
stability.setdefault("repeated",[])
for filename in stability["files"]:
    if filename in {r["file"] for r in stability["repeated"]}:continue
    wait_free()
    path=ROOT/filename
    archive=DEST/"failures/stability_before_repeat";archive.mkdir(exist_ok=True)
    archived=archive/path.name
    old=json.loads((archived if archived.exists() else path).read_text())
    cfg=old["configuration"]
    if not archived.exists():path.replace(archived)
    else:path.unlink(missing_ok=True)
    simulated=cfg["setting"].get("emulated",False)
    runenv=dict(env,FLASHINFER_WORKSPACE_BASE=str(ROOT/("temp/jit_report_emu" if simulated else "temp/jit_report_normal")))
    runenv.pop("FLASHINFER_EXTRA_CUDAFLAGS",None)
    if simulated:runenv["FLASHINFER_EXTRA_CUDAFLAGS"]="-DFLASHINFER_FAR_EMU"
    cmd=["numactl","-N0","-m0","/mnt/data/mhy-vllm-venv/bin/python",str(DEST/"bench.py"),
         "--kind",cfg["kind"],"--mtp",str(cfg["mtp"]),"--requests",str(cfg["requests"]),
         "--settings",cfg["setting"]["name"]]
    invoke(cmd,DEST/"stability_repeat.log",runenv)
    fresh=json.loads(path.read_text());assert fresh["complete"]
    changes=[]
    for case in fresh["cases"]:
        prior=next(c for c in old["cases"] if c["host_fraction_requested"]==case["host_fraction_requested"])
        changes.append(dict(host_fraction=case["host_fraction_requested"],modes={
            mode:dict(old_mean_us=prior["timings"][mode]["mean_us"],new_mean_us=v["mean_us"],
                      new_pass_range_over_mean=(max(v["pass_mean_us"])-min(v["pass_mean_us"]))/v["mean_us"])
            for mode,v in case["timings"].items()}))
    stability["repeated"].append(dict(file=filename,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),changes=changes))
    stability_path.write_text(json.dumps(stability,indent=2)+"\n")
stability["complete"]=True;stability_path.write_text(json.dumps(stability,indent=2)+"\n")
wait_free()
invoke(["python3",str(DEST/"run.py")],DEST/"run_resume9.log")
invoke(["python3",str(DEST/"audit.py")],DEST/"finalize.log")
invoke(["python3",str(DEST/"report.py")],DEST/"finalize.log")
invoke(["python3",str(DEST/"plot.py")],DEST/"finalize.log")
invoke(["python3",str(DEST/"export_html.py")],DEST/"finalize.log")
save_state("complete")
