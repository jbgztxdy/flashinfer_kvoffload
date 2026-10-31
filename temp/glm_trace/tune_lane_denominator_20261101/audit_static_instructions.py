"""Count static target-kernel instructions; keep these separate from dynamic trace work."""
import gzip
import hashlib
import json
from pathlib import Path
import re

p = Path(__file__).resolve().parent
q = p.parent / 'tune_joint_qk_20261101'
records = []
for variant, folder, symbol in [('joint_qk', q, 'BatchMLAJointQKKernel'),
                              ('lane_d', p, 'BatchMLALaneDenominatorKernel')]:
    f = next(folder.glob(variant + '_batch*host_overlap_v2_sass.txt.gz'))
    part = next(s for s in re.split(r'\n\s*Function : ', gzip.open(f, 'rt').read())[1:]
                if symbol in s.splitlines()[0])
    instructions = re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*', part)
    counts = {op: sum(bool(re.search(r'\b' + re.escape(op) + r'(?:\.|\b)', s)) for s in instructions)
              for op in ['HMMA', 'LDSM', 'BAR', 'FADD', 'F2F', 'SHFL', 'FMUL', 'FFMA', 'STL', 'LDL']}
    records.append(dict(variant=variant, source=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
        symbol=part.splitlines()[0], static_instructions=len(instructions), static_opcode_counts=counts))
old, new = [r['static_opcode_counts'] for r in records]
assert new['HMMA'] == old['HMMA'] - 3
assert new['LDSM'] == old['LDSM'] and new['BAR'] == old['BAR']
assert new['SHFL'] == old['SHFL'] + 4
(p / 'static_instruction_review.json').write_text(json.dumps(dict(complete=True, performance_claim=False,
    scope='EMU BF16 targetkernel staticSASS,notdynamicwork/stall/time decomposition', kernels=records), indent=2) + '\n')
print([(r['variant'], r['static_instructions'], r['static_opcode_counts']) for r in records])
