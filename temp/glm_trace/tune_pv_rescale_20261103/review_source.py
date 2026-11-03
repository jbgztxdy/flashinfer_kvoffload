"""Check that the schedule experiment retains every unrelated helper byte-for-byte."""
import hashlib,json,re
from pathlib import Path
p=Path(__file__).resolve().parent
base=(p/'base.cuh').read_text();candidate=(p/'pv_rescale.cuh').read_text()
normalize=lambda s:s.replace('PVRescale','StreamSource').replace('pv_rescale_','stream_source_')
start=candidate.index('template <typename Q, typename KV, typename O, typename Id>\nstruct PVRescaleTraits')
end=candidate.index('// partial_capacity bounds',start)
new=normalize(candidate[start:end])
a=base.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits')
b=base.index('// partial_capacity bounds',a)
old=base[a:b]
def get(s,name):
 a=s.index(name+'(');b=s.index('{',a);level=1;i=b+1
 while level:
  level+=(s[i]=='{')-(s[i]=='}');i+=1
 return s[a:i]
names=['stream_source_qk_pair_fragments','stream_source_load_query','stream_source_compute_pe','stream_source_compute_one','stream_source_compute_qk_pair','stream_source_run_joint_host_item','stream_source_run_joint_host_items','BatchMLAStreamSourceKernel']
checks={name:get(old,name)==get(new,name) for name in names};assert all(checks.values()),checks
assert candidate[:base.index('// partial_capacity bounds')]==base[:base.index('// partial_capacity bounds')]
# The native PV differs only in its parameter list, an invariant static_assert and
# the once-per-feature eight FMULs immediately before the first MMA.
# The generator's asserted exact replacements are retained with the full source diff.
r=dict(candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),unchanged_helpers_after_type_rename=checks,unchanged_existing_host_header_prefix=True,math_unchanged=(p/'base_mla.cuh').read_bytes()==(p/'pv_rescale_mla.cuh').read_bytes(),schedule_equation='For each O feature: O <- factor*O followed by original first PV MMA(s), then original second PV MMA(s). d and P arithmetic/order unchanged. KPE->P barriers unchanged.',limits='Source review only; compiled schedule/resources and end-to-end timing reviewed separately.')
(p/'source_review.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
