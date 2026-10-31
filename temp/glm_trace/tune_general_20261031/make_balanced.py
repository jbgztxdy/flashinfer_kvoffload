"""Freeze the compute-wave dispatch candidate before building it."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent
base=(p/'base.cuh').read_text()
assert base.count('#include "mla.cuh"')==1
s=base.replace('#include "mla.cuh"','#include <queue>\n#include <vector>\n#include "mla.cuh"')
mark='// partial_capacity bounds the item states of the tail rows, in 64-head tiles.'
assert s.count(mark)==1
s=s.replace(mark,(p/'balanced_dispatch.inc').read_text()+mark)
assert s.count('params.num_rows >= 4u * num_sms &&')==1
s=s.replace('params.num_rows >= 4u * num_sms &&','''(params.num_rows >= 4u * num_sms ||
           (CTA_TILE_KV == 32 && params.num_rows >= 2u * num_sms &&
            host_batch_balanced_whole_rows(params.num_rows, max_kv_len, num_sms,
                                          params.fetch_ctas, partial_capacity))) &&''')
v='balanced_sm80'
assert not (p/(v+'.cuh')).exists()
(p/(v+'.cuh')).write_text(s)
shutil.copytree(p/'include_base',p/('include_'+v))
(p/('include_'+v)/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
(p/('manifest_'+v+'.json')).write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),change='Shape-only host model compares direct-row and original split-plan cyclic makespans. Medium A100 rows use whole only when predicted compute cost is no worse. >=4 SM waves/other shared-memory configurations retain original behavior. No measured times, link setting or hit state in the policy.'),indent=2)+'\n')
for name in ('candidatepaths.py','test_candidate.py','run_readiness.py'):
    q=p/name;t=q.read_text().replace("'threshold2','threshold2_sm80')", "'threshold2','threshold2_sm80','balanced_sm80')");q.write_text(t)
q=p/'compiler.py';t=q.read_text().replace("'threshold2','threshold2_sm80')", "'threshold2','threshold2_sm80','balanced_sm80')").replace("if v.startswith('threshold2'):", "if v.startswith('threshold2') or v=='balanced_sm80':");q.write_text(t)
