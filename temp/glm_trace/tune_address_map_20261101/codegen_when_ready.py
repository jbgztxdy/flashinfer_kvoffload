"""Wait for the already-running CPU build; do not launch duplicate compilation."""
import argparse,json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--normal',action='store_true');parser.add_argument('--variant',default='address_map');a=parser.parse_args()
suffix='_normal' if a.normal else ''
while len(list((p/('jit_'+a.variant+suffix)).rglob('*.so'))) != (4 if a.normal else 2):
    gate=p/('tests_'+a.variant+'_gpu3.json')
    if gate.exists():
        t=json.loads(gate.read_text())
        if t.get('compile_exit_code') not in (None,0):raise SystemExit('CPU compile failed; inspect preserved logs')
    time.sleep(10)
cmd=['/mnt/data/mhy-vllm-venv/bin/python',str(p/('compiler_'+a.variant+'.py'))]+(['--normal'] if a.normal else [])
raise SystemExit(subprocess.call(cmd))
