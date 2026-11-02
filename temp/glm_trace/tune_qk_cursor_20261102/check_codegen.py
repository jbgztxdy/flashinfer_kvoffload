import gzip,hashlib,json,re,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
record={}
for mode in ['emu','normal']:
    variants={};libs={}
    for v in ['base','qk_cursor']:
        kernels={};libs[v]=[]
        for lib in sorted((p/('jit_'+v+('_normal' if mode=='normal' else ''))).rglob('*.so')):
            libs[v].append(dict(path=str(lib.relative_to(p)),sha256=sha(lib)))
            sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(lib)],text=True)
            resources=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-resource-usage',str(lib)],text=True)
            with gzip.GzipFile(filename=str(p/f'{v}_{mode}_{lib.stem}.sass.gz'),mode='wb',mtime=0) as f:f.write(sass.encode())
            for part in re.split(r'\n\s*Function : ',sass)[1:]:
                symbol=part.splitlines()[0].strip();words=re.findall(r'0x[0-9a-f]{16}\b',part)
                kernels[lib.name,symbol]=dict(sha256=hashlib.sha256('\n'.join(words).encode()).hexdigest(),resource=re.search(r'Function '+re.escape(symbol)+r':?\s+([^\n]+)',resources)[1],streaming_stores=len(re.findall(r'STG[^;\n]*?\.EF',part)))
        assert len(libs[v])==2,libs[v];variants[v]=kernels
    assert variants['base'].keys()==variants['qk_cursor'].keys()
    changed=[];same=[]
    for k,b in variants['base'].items():
        c=variants['qk_cursor'][k]
        if b==c:same.append(k[1])
        else:
            assert 'BatchMLAStreamSourceKernel' in k[1],k
            changed.append(dict(library=k[0],kernel=k[1],base=b,candidate=c))
    record[mode]=dict(unchanged=len(same),changed=changed,libraries=libs)
(p/'codegen.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
