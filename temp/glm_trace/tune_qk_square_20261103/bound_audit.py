"""Read-only interpretation of existing same-GPU references; do not rewrite any reference."""
import hashlib,json,statistics
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
raw=p/'gpu3/raw/focused256_base_staged_f24_i64_soff_lru_g100_l0_mtp8_b128.json'
d=json.loads(raw.read_text());case=d['cases'][0]
audit=json.loads((p/'performance_audit.json').read_text());row=next(x for x in audit['rows'] if x['raw']==str(raw.relative_to(root)))
ref=root/row['reference_raw'];floor=json.loads(ref.read_text());fc=floor['cases'][0]
lookup={t:statistics.mean(z) for t,z in zip(fc['step_indices'],fc['timings']['local']['per_step_samples_us'])}
compute=[lookup[t] for t in case['step_indices']];transfer=[n*1152/100000 for n in case['ours_unique_misses']]
bound=statistics.mean(max(c,b) for c,b in zip(compute,transfer));assert abs(bound-row['reference_us'])<1e-7
compute_dominated=sum(c>=b for c,b in zip(compute,transfer));total=len(compute)
r=dict(raw=str(raw.relative_to(root)),raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
 reference=str(ref.relative_to(root)),reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
 actual_us=case['timings']['ours']['mean_us'],stepwise_reference_us=bound,goal110_us=1.1*bound,
 remaining_reduction_to110=1-1.1*bound/case['timings']['ours']['mean_us'],
 steps=total,compute_dominated_steps=compute_dominated,transfer_dominated_steps=total-compute_dominated,
 mean_compute_us=statistics.mean(compute),mean_unique_transfer_us=statistics.mean(transfer),
 hypothetical_compute10pct_faster_stepwise_reference_us=statistics.mean(max(.9*c,b) for c,b in zip(compute,transfer)),
 hypothetical_compute20pct_faster_stepwise_reference_us=statistics.mean(max(.8*c,b) for c,b in zip(compute,transfer)),
 interpretation='The original stepwise reference is max of a measured device implementation and a unique-transfer budget. Faster computation can fall below its compute-dominated steps; the mandatory physical byte budget is still binding on transfer-dominated steps. Hypothetical numbers are algebra, not measured kernels or replacement references.',
 scope='GPU3/A100-PCIE same-devicefarHBM EMU100GB0us,MTP8/R128/LRU4096,256positions x3coldpasses. Do not mixSXM absolute times.')
(p/'bound_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
