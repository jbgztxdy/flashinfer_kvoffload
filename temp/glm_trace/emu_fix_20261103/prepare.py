"""Freeze v2 include tree and create isolated correctness/build drivers."""
from pathlib import Path
import hashlib,json,shutil
p=Path(__file__).resolve().parent;root=p.parents[2]
assert not (p/'include_v2').exists()
shutil.copytree(root/'include',p/'include_v2')
old=p.parent/'tune_pv_feature_20261103'
for name in ['compile.py','resources_gpu3.py','plans.cpp']:
 shutil.copy2(old/name,p/name)
(p/'candidatepaths.py').write_text('''import os\nfrom pathlib import Path\nfrom flashinfer.jit import env\np=Path(__file__).resolve().parent\nenv.FLASHINFER_INCLUDE_DIR=p/'include_v2'\n''')
s=(old/'build.py').read_text().replace('pv_feature','v2');(p/'build.py').write_text(s)
s=(old/'test_pv_feature.py').read_text().replace('test_pv_feature_payload_lse_graph','test_emulator_v2_payload_lse_graph')
# Test different geometry too: rows64 uses an older direct kernel with warp-issued packets.
s=s.replace("@pytest.mark.parametrize('length',[65,2048])","@pytest.mark.parametrize('length,rows,staged',[(65,512,True),(2048,512,True),(65,64,False)])")
s=s.replace('def test_emulator_v2_payload_lse_graph(length):','def test_emulator_v2_payload_lse_graph(length,rows,staged):')
s=s.replace('rows=512;requests=64;mtp=8;','requests=rows//8;mtp=8;')
s=s.replace('w.host_stage=True;','w.host_stage=staged;')
s=s.replace('    for step in range(3):','''    import os
    lo=kv[host:].data_ptr();hi=lo+(pages-host)*1152
    os.environ['FLASHINFER_FAR_EMU']=f'100:20000:{lo}:{hi}'
    for step in range(3):''')
s=s.replace('[0,1,3,107,108,rows-1]','[0,1,3,rows//2,rows-1]')
(p/'test_v2.py').write_text(s)
for v in ['v2']:
 (p/(v+'.cuh')).write_bytes((root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes())
(p/'manifest.json').write_text(json.dumps(dict(version=2,files={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'include/flashinfer/attention/far_link_emu.cuh',root/'include/flashinfer/attention/mla.cuh',root/'include/flashinfer/attention/mla_host_batch.cuh',root/'temp/sim_link/emu_gather.cu']},model='Issue-time linearizable max-plus reservation,ceil(ns)service,latency excludedfromserializer; crosswarp prewaitbarrier.'),indent=2)+'\n')
