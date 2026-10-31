"""Observe GPUs 0/1 without opening CUDA contexts or disturbing other jobs."""
import csv
import io
import os
import subprocess
import time


def snapshot(allowed_pid=None):
    allowed_pid=os.getpid() if allowed_pid is None else allowed_pid
    gpu_text=subprocess.check_output([
        "nvidia-smi","--id=0,1",
        "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"],text=True)
    gpus=[dict(index=int(r[0]),uuid=r[1].strip(),used_mib=int(r[2]),
               total_mib=int(r[3]),utilization=int(r[4]))
          for r in csv.reader(io.StringIO(gpu_text))]
    app_text=subprocess.check_output([
        "nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits"],text=True)
    uuids={g["uuid"] for g in gpus}
    foreign=[]
    for row in csv.reader(io.StringIO(app_text)):
        if len(row)<4 or row[0].strip() not in uuids:continue
        try:pid=int(row[1])
        except ValueError:continue
        if pid!=allowed_pid:
            foreign.append(dict(gpu_uuid=row[0].strip(),pid=pid,
                                process=row[2].strip(),used_mib=row[3].strip()))
    return dict(timestamp=time.strftime("%FT%T%z"),gpus=gpus,foreign=foreign,
                available=not foreign and all(g["used_mib"]<512 and g["utilization"]<10 for g in gpus))


def require_exclusive():
    state=snapshot()
    if state["foreign"]:
        raise RuntimeError("Other GPU0/1 processes appeared during the benchmark: "+str(state["foreign"]))
    return state
