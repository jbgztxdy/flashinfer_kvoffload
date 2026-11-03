"""Small interleaving model: not a CUDA memory-model proof."""
from pathlib import Path
import itertools,json,hashlib
p=Path(__file__).resolve().parent
# After refill's metadata barrier, warp0 can finish refill and enter the next step.
# A delayed warp still needs to snapshot current in refill. Without the trailing
# CTA barrier, leader's next-step current update can overtake that snapshot.
a=['leader_refill_reads_current','leader_updates_done','leader_next_updates_current']
b=['delayed_refill_reads_current']
bad=[];valid=0
for seq in itertools.permutations(a+b):
 if [x for x in seq if x in a]!=a:continue
 current=0;done=0;reads={}
 for e in seq:
  if e=='leader_refill_reads_current':reads['leader']=current
  elif e=='leader_updates_done':done=reads['leader']+1
  elif e=='leader_next_updates_current':current=done
  else:reads['delayed']=current
 valid+=1
 if reads['delayed']!=reads['leader']:bad.append(seq)
assert bad
r=dict(source_sha256=hashlib.sha256((p/'clean.cuh').read_bytes()).hexdigest(),interleavings=valid,counterexamples_without_trailing_barrier=bad,conclusion='Do not remove trailing service_step barrier without changing ticket ownership. Explicit immutable per-invocation ticket is a possible redesign, not validated here.')
(p/'step_barrier_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
