"""Archive the exact normal BF16/FP16 kernels; enforce unaffected configurations."""
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path
p = Path(__file__).resolve().parent
previous = p.parent / 'tune_general_20261031'
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
variant = 'whole_pipe4'
oldlibs = {f.name: f for f in (previous / 'jit_balanced_cached_sm80_normal').rglob('*host_overlap*.so')}
newlibs = {f.name: f for f in (p / ('jit_' + variant + '_normal')).rglob('*host_overlap*.so')}
assert oldlibs.keys() == newlibs.keys() and len(oldlibs) == 2
prior = json.loads((previous / 'compiler_normal_balanced_cached_sm80_summary.json').read_text())
assert {x['name']: x['sha256'] for x in prior['libraries']['retained']} == {n: sha(f) for n, f in oldlibs.items()}
records, libraries = {}, {}
for label, paths in [('before', oldlibs), ('candidate', newlibs)]:
    records[label], libraries[label] = [], []
    for name, f in paths.items():
        sass = subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump', '--dump-sass', str(f)], text=True)
        resource = subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump', '--dump-resource-usage', str(f)], text=True)
        dtype = 'bf16' if 'bf16' in name else 'fp16'
        archive = p / (label + '_' + variant + '_' + dtype + '_normal_sass.txt.gz')
        if not archive.exists() or gzip.decompress(archive.read_bytes()) != sass.encode():
            with gzip.GzipFile(filename=str(archive), mode='wb', mtime=0) as z:
                z.write(sass.encode())
        (p / (label + '_' + variant + '_' + dtype + '_normal_resources.txt')).write_text(resource)
        libraries[label].append(dict(name=name, sha256=sha(f)))
        for part in re.split(r'\n\s*Function : ', sass)[1:]:
            symbol = part.splitlines()[0].strip()
            if 'BatchMLAHostBatchKernel' not in symbol:
                continue
            ins = re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*', part)
            res = re.search(r'Function ' + re.escape(symbol) + r':?\s+([^\n]+)', resource)
            assert res
            records[label].append(dict(symbol=symbol, instructions=len(ins),
                instruction_sha256=hashlib.sha256('\n'.join(ins).encode()).hexdigest(), resource=res[1]))
old = {x['symbol']: x for x in records['before']}
comparisons = []
for new in records['candidate']:
    before = old[new['symbol']]
    targeted = 'WholeRowHostBatchParams' in new['symbol'] and 'ELj32E' in new['symbol']
    equal = new['instruction_sha256'] == before['instruction_sha256'] and new['resource'] == before['resource']
    if not targeted:
        assert equal, new
    assert 'STACK:0' in new['resource'] and 'LOCAL:0' in new['resource'], new
    comparisons.append(dict(symbol=new['symbol'], targeted=targeted,
        instruction_text_equal=new['instruction_sha256'] == before['instruction_sha256'],
        resource_equal=new['resource'] == before['resource']))
assert len(comparisons) == 18 and sum(x['targeted'] for x in comparisons) == 2
result = dict(build='normal SM80 BF16/FP16', unaffected_equal_kernels=16, targeted_changed_kernels=2,
    all_stack_and_local_zero=True, comparisons=comparisons, records=records, libraries=libraries,
    base_header_sha256=sha(p / 'base.cuh'), candidate_header_sha256=sha(p / (variant + '.cuh')))
(p / 'compiler_normal_summary.json').write_text(json.dumps(result, indent=2) + '\n')
print({k: v for k, v in result.items() if k not in ['comparisons', 'records', 'libraries']})
