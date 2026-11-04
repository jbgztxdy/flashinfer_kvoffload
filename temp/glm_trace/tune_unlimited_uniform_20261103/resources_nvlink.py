"""Observe physical GPU3 only; GPU0/1 results and other users stay independent."""
import csv,io,os,subprocess,time

def snapshot(allowed_pid=None):
    allowed_pid=os.getpid() if allowed_pid is None else allowed_pid
    raw=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=list(csv.reader(io.StringIO(raw)))
    r=rows[0]
    g=dict(index=int(r[0]),uuid=r[1].strip(),used_mib=int(r[2]),total_mib=int(r[3]),utilization=int(r[4]))
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
    foreign=[dict(gpu_uuid=z[0].strip(),pid=int(z[1]),process=z[2].strip(),used_mib=z[3].strip()) for z in csv.reader(io.StringIO(raw)) if len(z)>=4 and z[0].strip() in [r[1].strip() for r in rows] and int(z[1])!=allowed_pid]
    return dict(timestamp=time.strftime('%FT%T%z'),gpus=[dict(index=int(r[0]),uuid=r[1].strip(),used_mib=int(r[2]),total_mib=int(r[3]),utilization=int(r[4])) for r in rows],foreign=foreign,available=not foreign and all(int(r[2])<512 and int(r[4])<10 for r in rows))

def require_exclusive():
    state=snapshot()
    if state['foreign']:raise RuntimeError('Other physical GPU3 process: '+str(state['foreign']))
    return state
