"""Archive and compare every original device kernel, including pure-device math."""
import argparse
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

p = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--normal', action='store_true')
a = parser.parse_args()
suffix = '_normal' if a.normal else ''
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
records, libraries = {}, {}
for variant in ['base', 'half_heads']:
    libs = sorted((p / ('jit_' + variant + suffix)).rglob('*.so'))
    assert len(libs) == (4 if a.normal else 2), (variant, libs)
    records[variant], libraries[variant] = {}, []
    for lib in libs:
        sass = subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump', '--dump-sass', str(lib)], text=True)
        resource = subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump', '--dump-resource-usage', str(lib)], text=True)
        name = variant + suffix + '_' + lib.stem
        with gzip.GzipFile(filename=str(p / (name + '_sass.txt.gz')), mode='wb', mtime=0) as f:
            f.write(sass.encode())
        (p / (name + '_resources.txt')).write_text(resource)
        libraries[variant].append(dict(name=lib.name, sha256=sha(lib)))
        kernels = {}
        for part in re.split(r'\n\s*Function : ', sass)[1:]:
            symbol = part.splitlines()[0].strip()
            instructions = re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*', part)
            usage = re.search(r'Function ' + re.escape(symbol) + r':?\s+([^\n]+)', resource)
            assert usage, symbol
            kernels[symbol] = dict(symbol=symbol, instructions=len(instructions),
                instruction_sha256=hashlib.sha256('\n'.join(instructions).encode()).hexdigest(), resource=usage[1])
        records[variant][lib.name] = kernels
assert records['base'].keys() == records['half_heads'].keys()
unchanged, added = [], []
for library, old in records['base'].items():
    new = records['half_heads'][library]
    assert old.keys() <= new.keys(), library
    for symbol, baseline in old.items():
        assert baseline == new[symbol], (library, symbol, baseline, new[symbol])
        unchanged.append(dict(library=library, symbol=symbol))
    added.extend(dict(library=library, **new[symbol]) for symbol in new.keys() - old.keys())
assert len(added) == (4 if a.normal else 2), added
assert all('HalfHeadHostTraits' in r['symbol'] and 'STACK:0' in r['resource'] and 'LOCAL:0' in r['resource'] for r in added)
result = dict(build='normal BF16/FP16' if a.normal else 'emulated BF16',
    base_header_sha256=sha(p/'base.cuh'), candidate_header_sha256=sha(p/'half_heads.cuh'),
    base_math_header_sha256=sha(p/'base_mla.cuh'), candidate_math_header_sha256=sha(p/'half_heads_mla.cuh'),
    unchanged_original_kernels=len(unchanged), unchanged=unchanged, new_kernels=added,
    unique_new_kernel_symbols=len({r['symbol'] for r in added}),
    libraries=libraries, records=records, complete=True)
(p / ('compiler_heads' + suffix + '_summary.json')).write_text(json.dumps(result, indent=2) + '\n')
print(dict(build=result['build'], unchanged_original_kernels=len(unchanged), new_kernels=len(added), complete=True))
