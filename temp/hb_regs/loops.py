# Purpose: Count HMMA, LDGSTS and spill reloads (LDL) in each inner SASS loop of a cubin dump.
# Usage: cuobjdump -sass x.cubin > x.sass; python3 temp/hb_regs/loops.py x.sass
import re
import sys

ins = []
for line in open(sys.argv[1]).read().splitlines():
    m = re.match(r'\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;', re.sub(r'/\* 0x[0-9a-f]+ \*/', '', line))
    if m:
        ins.append((int(m.group(1), 16), m.group(2).strip()))
idx = {a: i for i, (a, _) in enumerate(ins)}
for i, (a, t) in enumerate(ins):
    m = re.search(r'BRA\s+(?:`\(\.L_x_\d+\)\s*)?(0x[0-9a-f]+)', t)
    if m:
        tgt = int(m.group(1), 16)
        if tgt < a and tgt in idx and i - idx[tgt] < 900:
            body = [x for _, x in ins[idx[tgt]:i + 1]]
            if any('HMMA' in x for x in body):
                print(f"loop {tgt:#x}: HMMA={sum('HMMA' in x for x in body)} "
                      f"LDGSTS={sum('LDGSTS' in x for x in body)} LDL={sum('LDL' in x for x in body)}")
