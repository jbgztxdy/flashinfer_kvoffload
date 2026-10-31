"""Chain disjoint no-LRU shape gates after the confirmed live main GPU3 validator ends."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--parent-pid',type=int,required=True);a=parser.parse_args()
parent=a.parent_pid
state=dict(pid=os.getpid(),waiting_for_pid=parent,start=time.strftime('%FT%T%z'),status='waiting_for_main')
def save():(p/'gpu3_extra_supervisor.json').write_text(json.dumps(state,indent=2)+'\n')
save()
expected='temp/glm_trace/tune_general_20261031/validate_gpu3_medium_numa1_v2.py'
first=True
while True:
    try:
        stat=Path(f'/proc/{parent}/stat').read_text().split(') ',1)[1].split()
        if stat[0]=='Z':state['parent_terminal_evidence']='zombie';break
        cmd=Path(f'/proc/{parent}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
        assert cmd.endswith(expected),cmd
        if first:print('VERIFIED_MAIN_LIVE',parent,cmd,flush=True);first=False
    except FileNotFoundError:
        state['parent_terminal_evidence']='proc_handle_missing';break
    time.sleep(5)
main=json.loads((p/'gpu3_validation_numa1_v2_state.json').read_text())
assert len(main)==31 and all(x.get('exit_code')==0 for x in main),main[-1]
assert main[-1]['kind']=='nocache' and main[-1]['mtp']==1 and main[-1]['requests']==256 and main[-1]['variant']=='base'
state['status']='running_extra';save()
cmd=['/mnt/data/mhy-vllm-venv/bin/python','-u',str(p/'validate_gpu3_nocache_extra.py')]
with (p/'gpu3_nocache_extra_supervisor.log').open('a') as log:
    child=subprocess.Popen(cmd,cwd=p.parents[2],stdout=log,stderr=subprocess.STDOUT)
    state['child_pid']=child.pid;save();state['exit_code']=child.wait()
state['status']='complete' if state['exit_code']==0 else 'failed';state['end']=time.strftime('%FT%T%z');save()
print('EXTRA_VALIDATION_EXIT',state['exit_code'],flush=True)
raise SystemExit(state['exit_code'])
