"""One-time GPU2 harness adaptation; no candidate/source or existing raw mutation."""
from pathlib import Path

p=Path(__file__).resolve().parent
def write(name,text):
 target=p/name
 assert not target.exists(),target
 target.write_text(text)

s=(p/'resources_gpu3.py').read_text().replace('GPU3','GPU2').replace('--id=3','--id=2')
write('resources_gpu2.py',s)
s=(p/'bench_deferred_packet_normal.py').read_text()
s=s.replace('resources_gpu3','resources_gpu2').replace('gpu3_correctness.lock','gpu2_correctness.lock')
s=s.replace('"--id=3"','"--id=2"').replace('"gpu3/source_snapshots"','"gpu2/source_snapshots"')
s=s.replace("VARIANT not in ['base','clean','deferred_packet_normal']","VARIANT not in ['base','clean','deferred_packet_normal','next_packet_qk','retained']")
s=s.replace("target=DEST/(VARIANT+","target=DEST/('gpu2_'+VARIANT+")
assert "--id=3" not in s and "gpu3_correctness.lock" not in s
assert s.count("'retained']")==2
write('bench_gpu2_deferred.py',s)
s=(p/'run_deferred_packet_normal.py').read_text()
s=s.replace('resources_gpu3','resources_gpu2').replace('GPU3','GPU2')
s=s.replace('gpu3_correctness.lock','gpu2_correctness.lock').replace("CUDA_VISIBLE_DEVICES='3'","CUDA_VISIBLE_DEVICES='2'")
s=s.replace('deferred_packet_normal_state.json','gpu2_deferred_state.json')
s=s.replace("run('correctness_deferred_packet_normal'","run('gpu2_correctness_deferred_packet_normal'")
s=s.replace("run('deferred_packet_normal_native'","run('gpu2_deferred_packet_normal_native'")
s=s.replace("bench_deferred_packet_normal.py","bench_gpu2_deferred.py")
line=next(x for x in s.splitlines() if x.startswith("run('gpu2_deferred_packet_normal_native'"))
extra='\n'.join(line.replace("'gpu2_deferred_packet_normal_native','deferred_packet_normal'",f"'gpu2_{v}_native','{v}'") for v in ['next_packet_qk','retained'])
s=s.replace(line,line+'\n'+extra)
write('run_gpu2_deferred.py',s)
print('prepared resources, locked GPU2 runner, isolated output-prefix benchmark')
