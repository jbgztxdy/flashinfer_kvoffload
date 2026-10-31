"""Compare original Ampere instruction and control words; no instruction normalization."""
import gzip
import hashlib
import json
from pathlib import Path
import re

p = Path(__file__).resolve().parent
builds = []
for suffix, expected in [('', 36), ('_normal', 72)]:
    proof = json.loads((p / ('compiler_nine_warp_static' + suffix + '_summary.json')).read_text())
    entries = []
    for lib in proof['records']['base']:
        tables = {}
        for v in ['base', 'nine_warp_static']:
            f = p / (v + suffix + '_' + lib.removesuffix('.so') + '_sass.txt.gz')
            tables[v] = {s.splitlines()[0].strip(): re.findall(r'0x[0-9a-f]{16}\b', s)
                for s in re.split(r'\n\s*Function : ', gzip.open(f, 'rt').read())[1:]}
        for symbol, old in tables['base'].items():
            new = tables['nine_warp_static'][symbol]
            assert old and old == new, (suffix, lib, symbol)
            assert len(new) == 2 * proof['records']['nine_warp_static'][lib][symbol]['instructions']
            entries.append(dict(library=lib, symbol=symbol, machine_words=len(old),
                machine_word_sha256=hashlib.sha256('\n'.join(old).encode()).hexdigest()))
    assert len(entries) == expected
    builds.append(dict(build=suffix or 'EMU', original_kernel_instances=expected,
                       all_words_identical=True, records=entries))
(p / 'machine_word_identity.json').write_text(json.dumps(dict(complete=True, builds=builds), indent=2) + '\n')
print([(b['build'], b['original_kernel_instances'], b['all_words_identical']) for b in builds])
