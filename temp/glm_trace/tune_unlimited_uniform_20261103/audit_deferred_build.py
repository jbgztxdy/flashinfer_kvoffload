"""CPU audit of frozen headers and compiled target; no loaded-GPU claim."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

p=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
header=sha(p/'deferred_packet.cuh')
out=[]
for variant in ['deferred_packet_normal','deferred_packet']:
 inc=p/('include_'+variant)/'flashinfer/attention'
 assert sha(inc/'mla_host_batch.cuh')==header
 libs=list((p/('jit_'+variant)).rglob('*_host_overlap_v2.so'))
 assert len(libs)==1
 lib=libs[0]
 sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(lib)],text=True)
 targets=[section for section in re.split(r'Function : ',sass) if section.startswith('_ZN10flashinfer3mla26BatchMLACleanServiceKernel')]
 assert len(targets)==1
 target=targets[0]
 waits=re.findall(r'DEPBAR[^;]+;',target)
 assert any(re.search(r',\s*0x1\s*;',x) for x in waits),waits
 assert any(re.search(r',\s*0x0\s*;',x) for x in waits),waits
 out.append(dict(variant=variant,header_sha256=header,math_sha256=sha(inc/'mla.cuh'),
                 far_sha256=sha(inc/'far_link_emu.cuh'),built_library=str(lib),built_library_sha256=sha(lib),
                 depbar_forms=sorted(set(waits)),cta_barriers=target.count('BAR.SYNC'),
                 target_symbol=target.splitlines()[0]))
record=dict(complete=True,scope='CPU compiled binary only; not actual GPU dispatch or correctness',variants=out)
(p/'deferred_build_audit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
