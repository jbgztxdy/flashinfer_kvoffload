from pathlib import Path
import subprocess,json,hashlib,re
p=Path(__file__).resolve().parent;root=p.parents[2];result={}
for name in ['clean','refill_current']:
    r=json.loads((p/'refill_current_codegen.json').read_text())[name]
    lib=root/r['library'];assert hashlib.sha256(lib.read_bytes()).hexdigest()==r['sha256']
    sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(lib)],text=True)
    part=next(x for x in sass.split('Function : ') if x.startswith('_ZN10flashinfer3mla26BatchMLACleanServiceKernel'))
    opcodes=[]
    for line in part.splitlines():
        if '*/' not in line:continue
        tail=line.split('*/',1)[1].strip()
        if tail.startswith('@'):tail=tail.split(None,1)[1]
        if tail:opcodes.append(tail.split()[0])
    result[name]=dict(library_sha256=r['sha256'],static_instructions=len(opcodes),barrier_ops={k:sum(op.startswith(k) for op in opcodes) for k in ['BAR.SYNC','BAR.RED','MEMBAR']},cp_async_waits=sum('CP.ASYNC.WAIT' in op for op in opcodes))
(p/'refill_barrier_codegen.json').write_text(json.dumps(result,indent=2)+chr(10))
print(result)
