"""Compare every fresh production instruction, control word and resource to measured CPE."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

p = Path(__file__).resolve().parent
root = p.parents[2]
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
assert (root / 'include/flashinfer/attention/mla_host_batch.cuh').read_bytes() == (p / 'cached_pe.cuh').read_bytes()
old = json.loads((p / 'compiler_cached_pe_normal_summary.json').read_text())
records, libraries = {}, []
folder = p / 'jit_production_cached_pe_normal'
libs = sorted(folder.rglob('*.so'))
assert len(libs) == 4
for lib in libs:
    sass = subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump', '--dump-sass', str(lib)], text=True)
    resource = subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump', '--dump-resource-usage', str(lib)], text=True)
    prefix = p / ('production_cached_pe_normal_' + lib.stem)
    with gzip.GzipFile(filename=str(prefix) + '_sass.txt.gz', mode='wb', mtime=0) as f:
        f.write(sass.encode())
    Path(str(prefix) + '_resources.txt').write_text(resource)
    reference = gzip.open(p / ('cached_pe_normal_' + lib.stem + '_sass.txt.gz'), 'rt').read()
    words = {b.splitlines()[0].strip(): re.findall(r'0x[0-9a-f]{16}\b', b)
             for b in re.split(r'\n\s*Function : ', reference)[1:]}
    kernels = {}
    for body in re.split(r'\n\s*Function : ', sass)[1:]:
        symbol = body.splitlines()[0].strip()
        instructions = re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*', body)
        machine = re.findall(r'0x[0-9a-f]{16}\b', body)
        usage = re.search(r'Function ' + re.escape(symbol) + r':?\s+([^\n]+)', resource)
        assert usage and machine == words[symbol] and len(machine) == 2 * len(instructions)
        record = dict(symbol=symbol, instructions=len(instructions),
                      instruction_sha256=hashlib.sha256('\n'.join(instructions).encode()).hexdigest(), resource=usage[1])
        assert record == old['records']['cached_pe'][lib.name][symbol]
        kernels[symbol] = dict(**record, machine_word_sha256=hashlib.sha256('\n'.join(machine).encode()).hexdigest())
    assert kernels.keys() == old['records']['cached_pe'][lib.name].keys()
    records[lib.name] = kernels
    libraries.append(dict(name=lib.name, sha256=sha(lib)))
assert records.keys() == old['records']['cached_pe'].keys()
assert sum(map(len, records.values())) == 76
result = dict(complete=True, build='normal BF16/FP16', kernel_instances=76,
              all_original_and_new_instruction_control_words_and_resources_identical=True,
              header_sha256=sha(p / 'cached_pe.cuh'), math_sha256=sha(root / 'include/flashinfer/attention/mla.cuh'),
              libraries=libraries, records=records)
(p / 'production_codegen_normal_summary.json').write_text(json.dumps(result, indent=2) + '\n')
print('FRESH_PRODUCTION_76_KERNEL_INSTANCES_IDENTICAL')
