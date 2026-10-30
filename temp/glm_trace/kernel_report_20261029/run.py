"""Run the complete requested matrix, retaining logs and never replacing completed data."""
import json
import os
from pathlib import Path
import subprocess
import time
from resources import snapshot

ROOT=Path(__file__).resolve().parents[3]
DEST=Path(__file__).resolve().parent
os.chdir(ROOT)
meta=json.loads((DEST/"experiment.json").read_text())
normal=[s["name"] for s in meta["settings"] if not s.get("emulated")]
sim=[s["name"] for s in meta["settings"] if s.get("emulated")]
env=dict(os.environ,CUDA_DEVICE_ORDER="PCI_BUS_ID",CUDA_VISIBLE_DEVICES="0,1",
         CUDA_HOME="/usr/local/cuda-13.1",FLASHINFER_CUDA_ARCH_LIST="8.0",
         PYTHONPATH=str(ROOT),TMPDIR=str(ROOT/"temp"))
progress=[]
if not (DEST/"calibration.json").exists():
    runenv=dict(env,FLASHINFER_WORKSPACE_BASE=str(ROOT/"temp/jit_report_emu"),
                FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_FAR_EMU")
    with (DEST/"calibration.log").open("w") as stream:
        subprocess.run(["numactl","-N0","-m0","/mnt/data/mhy-vllm-venv/bin/python",str(DEST/"calibrate.py")],
                       env=runenv,stdout=stream,stderr=subprocess.STDOUT,check=True)
for kind in ("floor","snapshots","nocache","lru"):
    # Finish the shorter replay groups first; the same full matrix remains required.
    for mtp in ((8,1) if kind=="lru" else (1,8)):
        for batch in meta["batches"][str(mtp)]:
            groups=[(False,["floor"])] if kind in ("floor","snapshots") else [(False,normal),(True,sim)]
            for simulated,settings in groups:
                label=f"{kind}_{'sim' if simulated else 'normal'}_mtp{mtp}_b{batch}"
                outputs=[DEST/"raw"/f"{kind}_{s}_mtp{mtp}_b{batch}.json" for s in settings]
                def accepted(p):
                    if not p.exists():return False
                    data=json.loads(p.read_text())
                    return data.get("complete") and (kind!="nocache" or data["configuration"].get("memcpy_path_version",1)>=2)
                if all(accepted(p) for p in outputs):
                    print("SKIP_COMPLETE_GROUP",label,flush=True)
                    continue
                while True:
                    resources=snapshot()
                    if resources["available"]:break
                    (DEST/"waiting_for_gpu.json").write_text(json.dumps(resources,indent=2)+"\n")
                    print("WAITING_FOR_GPU",label,resources["timestamp"],flush=True)
                    time.sleep(30)
                log=DEST/(label+".log")
                runenv=dict(env,FLASHINFER_WORKSPACE_BASE=str(ROOT/"temp/jit_report_normal"))
                runenv.pop("FLASHINFER_EXTRA_CUDAFLAGS",None)
                if simulated:
                    runenv["FLASHINFER_EXTRA_CUDAFLAGS"]="-DFLASHINFER_FAR_EMU"
                    runenv["FLASHINFER_WORKSPACE_BASE"]=str(ROOT/"temp/jit_report_emu")
                cmd=["numactl","-N0","-m0","/mnt/data/mhy-vllm-venv/bin/python",str(DEST/"bench.py"),
                     "--kind",kind,"--mtp",str(mtp),"--requests",str(batch),"--settings",*settings]
                start=time.time()
                print("START",label,time.strftime("%Y-%m-%dT%H:%M:%S%z"),flush=True)
                for attempt in range(3):
                    if log.exists():
                        failed=DEST/"failures";failed.mkdir(exist_ok=True)
                        log.replace(failed/(log.stem+f"_{time.time_ns()}.log"))
                    runenv["REPORT_GUARD_CYCLES"]=str(16_000_000*(2**attempt))
                    with log.open("w") as stream:
                        result=subprocess.run(cmd,env=runenv,stdout=stream,stderr=subprocess.STDOUT)
                    if not result.returncode:break
                    if "CPU enqueue contaminated timing" not in log.read_text():break
                    print("RETRY_FULL_PROCESS_AFTER_INVALID_GUARD",label,attempt+1,flush=True)
                progress.append(dict(job=label,exit_code=result.returncode,seconds=time.time()-start))
                (DEST/"progress.json").write_text(json.dumps(progress,indent=2)+"\n")
                if result.returncode:
                    print("FAILED",label,"see",log,flush=True)
                    raise SystemExit(result.returncode)
                print("DONE",label,round(time.time()-start,2),flush=True)
print("ALL_MATRIX_COMPLETE",flush=True)
