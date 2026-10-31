"""Review the compiled reuse tradeoff; static instruction counts are not performance."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

p = Path(__file__).resolve().parent
proof = json.loads((p / 'compiler_cached_pe_summary.json').read_text())
lib = next(n for n in proof['records']['base'] if 'host_overlap' in n)
rows = []
for variant, kernel in [('base', 'BatchMLAJointQKKernel'), ('cached_pe', 'BatchMLACachedPEKernel')]:
    f = p / (variant + '_' + lib.removesuffix('.so') + '_sass.txt.gz')
    parts = re.split(r'\n\s*Function : ', gzip.open(f, 'rt').read())[1:]
    part = next(s for s in parts if kernel in s.splitlines()[0])
    symbol = part.splitlines()[0].strip()
    instructions = re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*', part)
    opcodes = Counter()
    for instruction in instructions:
        tokens = instruction.split()
        opcode = tokens[1] if tokens[0].startswith('@') else tokens[0]
        opcodes[opcode.split('.')[0]] += 1
    assert len(instructions) == proof['records'][variant][lib][symbol]['instructions']
    rows.append(dict(variant=variant, symbol=symbol, sass_archive_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
        instruction_count=len(instructions), resource=proof['records'][variant][lib][symbol]['resource'],
        opcodes=dict(sorted(opcodes.items())),
        instruction_sha256=hashlib.sha256('\n'.join(instructions).encode()).hexdigest()))
    (p / (variant + '_reuse_excerpt.txt')).write_text('\n'.join(line for line in part.splitlines()
        if re.search(r'\b(LDG|LDSM|LDL|STL|BRA|HMMA)', line)) + '\n')
addressed = [(int(m[1],16),m[2]) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\*', part)]
qloads = [(addr, ins) for addr, ins in addressed if 0xc240 <= addr <= 0xc420 and 'LDG.E' in ins]
assert sorted(int(re.search(r'LDG.E R(\d+)', ins)[1]) for _,ins in qloads) == list(range(12,28))
loop = [(addr,ins) for addr,ins in addressed if 0xdec0 <= addr <= 0x116e0]
assert any('BRA 0xdec0' in ins for _,ins in loop)
assert not any(re.search(r'\b(?:LDL|STL)\b',ins) for _,ins in loop)
for _, ins in loop:
    write = re.match(r'(?:@!?P\d+\s+)?\S+\s+R(\d+)\b',ins)
    assert not write or int(write[1]) not in range(12,28), ins
assert all(any(re.search(r'HMMA\S* R\d+, R'+str(reg)+r'(?:\.reuse)?,' ,ins) for _,ins in loop) for reg in [12,16,20,24])
cache_liveness = dict(verified=True,compiler='CUDA13.1 sm80',registers=list(range(12,28)),
    load_region=['0xc240','0xc420'], paired_loop_region=['0xdec0','0x116e0'],
    qpe_loads=[dict(address=hex(addr),instruction=ins) for addr,ins in qloads],
    no_writes_to_query_registers_in_pair_loop=True,no_local_memory_in_pair_loop=True,
    detail='All16 QPE words load before the paired-tile backedge and stay inR12-R27. The kernel still uses8B stack elsewhere; no spill-free-kernel claim. Register reuse is verified, but is not a performance result.')
result = dict(complete=True, build='EMU BF16', records=rows,cache_liveness=cache_liveness,
    selected={k:{r['variant']:r['opcodes'].get(k,0) for r in rows} for k in ['LDSM','LDG','LDL','STL','HMMA','BAR']},
    limits='Static counts include loops and odd tail, not dynamic executed counts or timing. The QPE register witness covers the paired loop; STACK8 elsewhere means the whole kernel is not spill-free. Extra loads, register pressure and scheduling can offset LDSM removal. Full SASS/control/resource archives remain authoritative.')
(p / 'static_instruction_review.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'}, indent=2))
