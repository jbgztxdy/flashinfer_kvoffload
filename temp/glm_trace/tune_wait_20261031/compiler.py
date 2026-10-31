"""Archive resources and compare unchanged generic code for each isolated candidate."""
import gzip,hashlib,json,re,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
hashfile=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
records={};libraries={}
for v in ('base','prebind','fetch_ids','direct_cache'):
    libs=list((p/('jit_'+v)).rglob('*host_overlap*.so'))
    if v!='base' and not libs:
        print('BUILD_PENDING',v)
        continue
    assert len(libs)==1,(v,libs)
    lib=libs[0]
    sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(lib)],text=True)
    res=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-resource-usage',str(lib)],text=True)
    (p/(v+'_resources.txt')).write_text(res)
    archive=p/(v+'_sass.txt.gz')
    if not archive.exists() or gzip.decompress(archive.read_bytes())!=sass.encode():
        with gzip.GzipFile(filename=str(archive),mode='wb',mtime=0) as f:f.write(sass.encode())
    libraries[v]=dict(library_sha256=hashfile(lib),header_sha256=hashfile(p/(v+'.cuh')))
    rows=[]
    for part in re.split(r'\n\s*Function : ',sass)[1:]:
        symbol=part.splitlines()[0].strip()
        if 'BatchMLAHostBatchKernel' not in symbol:continue
        instructions=re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*',part)
        resource=re.search(r'Function '+re.escape(symbol)+r':?\s+([^\n]+)',res)
        assert resource
        rows.append(dict(symbol=symbol,whole_rows='WholeRowHostBatchParams' in symbol,
            tail_wave='TailWaveHostBatchParams' in symbol,
            instructions=len(instructions),instruction_sha256=hashlib.sha256('\n'.join(instructions).encode()).hexdigest(),resource=resource[1]))
    records[v]=rows
old={r['symbol']:r for r in records['base']}
comparisons={}
for v in (v for v in records if v!='base'):
    generic=[];whole=[]
    for r in records[v]:
        if r['tail_wave']:continue
        b=old[r['symbol']]
        c=dict(symbol=r['symbol'],instruction_count_before=b['instructions'],instruction_count_after=r['instructions'],
            instruction_text_equal=b['instruction_sha256']==r['instruction_sha256'],resource_equal=b['resource']==r['resource'])
        (whole if r['whole_rows'] else generic).append(c)
    assert len(generic)==6 and len(whole)==3
    assert all(c['instruction_text_equal'] and c['resource_equal'] for c in generic)
    if v.startswith('threshold2') or v.startswith('balanced'):
        assert all(c['instruction_text_equal'] and c['resource_equal'] for c in whole)
    comparisons[v]=dict(generic=generic,whole_rows=whole)
(p/'compiler_summary.json').write_text(json.dumps(dict(build='emulated SM80 BF16',builds=records,libraries=libraries,comparisons=comparisons),indent=2)+'\n')
print(json.dumps({v:dict(generic_compared=6,generic_instruction_and_resource_equal=6,
    all_stack_and_local_zero=all('STACK:0' in r['resource'] and 'LOCAL:0' in r['resource'] for r in records[v])) for v in comparisons}))
