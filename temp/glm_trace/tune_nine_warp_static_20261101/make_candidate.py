"""Keep the nine-warp protocol, make all register-buffer accesses statically named."""
import hashlib
import json
from pathlib import Path
import re
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
old = p.parent / 'tune_nine_warp_20261101'
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
host = (old / 'nine_warp.cuh').read_text()
original = (old / 'service.inc').read_text()
assert sha(host) == '36fde9d70bdaa6a1f61aff87d6dba41be1a4402afed3461b8761a73049ecd22b'
def block(s, start):
    a = s.index('{', start); b = a + 1; depth = 1
    while depth:
        depth += (s[b] == '{') - (s[b] == '}'); b += 1
    return a, b
start = original.index('  auto issue =')
a, b = block(original, start)
issue_body = original[a+1:b-1]
initial = original.index('#pragma unroll\n  for (uint32_t job = 0; job < 3;', b)
loop = original.index('  for (uint32_t job = 0; base(job) < count; ++job)', initial)
la, lb = block(original, loop)
consume_body = original[la+1:lb-1]
prefix = original[:start].replace('  uint4 register_stage[WORDS];',
    '\n'.join('  uint4 register_stage_' + str(i) + ';' for i in range(18)))
def expand_payload_loops(s):
    while True:
        marker = '#pragma unroll\n    for (uint32_t i = 0; i < WORDS; ++i)'
        at = s.find(marker)
        if at == -1:
            marker = '#pragma unroll\n      for (uint32_t i = 0; i < WORDS; ++i)'
            at = s.find(marker)
        if at == -1: break
        x, y = block(s, at)
        body = s[x+1:y-1]
        expanded = []
        for i in range(18):
            v = body.replace('register_stage[i]', 'register_stage_' + str(i))
            v = re.sub(r'\bi\b', str(i), v)
            expanded.append('    {' + v + '    }')
        s = s[:at] + '\n'.join(expanded) + s[y:]
    return s
issue_body = expand_payload_loops(issue_body)
consume_body = expand_payload_loops(consume_body)
consume_body = consume_body.replace('const uint32_t first = base(job), stage = job % 3, chunk = first / 64;',
    'const uint32_t first = base(job), chunk = first / 64;')
refill = '    if (base(job + 3) < count) issue(job + 3, stage);'
assert refill in consume_body
def issue(stage, expr):
    s = re.sub(r'\bjob\b', 'issue_job', issue_body)
    s = re.sub(r'\bstage\b', str(stage), s)
    return '{\n    const uint32_t issue_job = ' + expr + ';\n' + s + '\n  }'
body = prefix
for stage in range(3):
    body += '  if (base(' + str(stage) + ') < count) ' + issue(stage, str(stage)) + '\n'
body += '  bool chunk_deferred = false;\n#pragma unroll 1\n  for (uint32_t cycle = 0; base(cycle) < count; cycle += 3) {\n'
for stage in range(3):
    s = consume_body.replace(refill, '    if (base(job + 3) < count) ' + issue(stage, 'job + 3'))
    s = re.sub(r'\bstage\b', str(stage), s)
    body += '    if (base(cycle + ' + str(stage) + ') < count) {\n      const uint32_t job = cycle + ' + str(stage) + ';\n' + s + '\n    }\n'
body += '  }\n' + original[lb:]
assert 'register_stage[' not in body and 'auto issue' not in body
assert host.count(original) == 1
candidate = host.replace(original, body).replace('nine_', 'nine_static_').replace('Nine', 'NineStatic')
for name, value in [('service_static.inc', body), ('nine_warp_static.cuh', candidate),
                    ('base.cuh', (old/'base.cuh').read_text()),
                    ('nine_warp_static_mla.cuh', (old/'nine_warp_mla.cuh').read_text()),
                    ('base_mla.cuh', (old/'base_mla.cuh').read_text())]:
    f = p / name
    if f.exists(): assert f.read_text() == value
    else: f.write_text(value)
for variant in ['base', 'nine_warp_static']:
    d = p / ('include_' + variant)
    if not d.exists():
        shutil.copytree(root/'include', d)
        (d/'flashinfer/attention/mla_host_batch.cuh').write_bytes((p/(variant+'.cuh')).read_bytes())
    assert (d/'flashinfer/attention/mla_host_batch.cuh').read_bytes() == (p/(variant+'.cuh')).read_bytes()
    assert (d/'flashinfer/attention/mla.cuh').read_bytes() == (p/(variant+'_mla.cuh')).read_bytes()
m = json.loads((old/'manifest.json').read_text())
m.update(variant='nine_warp_static', predecessor_sha256=sha(host), candidate_sha256=sha(candidate),
    correction='OriginalREG168STACK248withLDL/STL. Replacepayloadarraywith18nameduint4scalars,explicitfixedstageissue/consume/refillwithoutcaptureofpayloadaddress. Samejobs,3buffers(2shared+1registerintent),FARpost-readhook,64rowflags,cachedefer/QPE/math/288threads. No buffersize/warpcount/unroll/pacing sweep.',
    candidate_adopted=False,goal_complete=False)
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps(m,indent=2))
