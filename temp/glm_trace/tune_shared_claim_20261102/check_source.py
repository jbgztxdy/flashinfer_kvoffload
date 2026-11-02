"""Bound scratch lifetime and preserve literal attention/fetch helpers."""
import hashlib,json,re,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
base=(p/'base.cuh').read_text();candidate=(p/'shared_claim.cuh').read_text()
begin=base.index('// Two different KV tiles share each Q fragment.',base.index('struct StreamSourceTraits'))
end=base.index('template <typename KTraits, typename Params, bool ROW_MERGE>\n__global__',begin)
old=base[begin:end];newbegin=candidate.index('// Two different KV tiles share each Q fragment.',candidate.index('struct SharedClaimTraits'))
newend=candidate.index('template <typename KTraits, typename Params, bool ROW_MERGE>\n__global__',newbegin)
new=candidate[newbegin:newend].replace('shared_claim_','stream_source_')
assert old==new
start=base.index('__device__ __forceinline__ void fetch_stream_source_pages(')
end=base.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits',start)
fetch=base[start:end];assert candidate[start:start+len(fetch)]==fetch
assert 'params.claim_entry' not in (p/'prepare.inc').read_text()
assert 'max_kv_len <= 2048' in candidate
ffi=(root/'csrc/batch_mla_run.cu').read_text()
assert 'params.entry_victim = scratch;' in ffi and 'params.slot_pool = scratch + misses;' in ffi
record=dict(timestamp=time.strftime('%FT%T%z'),candidate_sha256=sha(p/'shared_claim.cuh'),
 canonical_math_sha256=sha(root/'include/flashinfer/attention/mla.cuh'),
 attention_helper_section_literal_equal_after_renaming=True,fetch_source_helper_literal_equal=True,
 scratch='entry_victim/victim_of: [0,staging_capacity); slot_pool: [staging_capacity,2*staging_capacity). Temp group page IDs live at original kv_indptr[groupfirst]+rank until resolve; original post-resolve grid barrier precedes LRU writes. slot_pool writes cannot clobber temporary page IDs.',
 bounds='Dispatch requires staging_capacity>=numrows*maxK and maxK<=2048. A same-pool subgroup has <=4rows and <=8192unique pages; hash has16384slots, owner rank<2^16. Group source ranges disjoint, global CAS remains only crossgroup owner election.',
 numerical_scope='Two BF16 targeted GPU tests passed; not an exhaustive proof of timing-dependent state or all LRU tie decisions.',
 performance='Candidate raw incomplete due foreign GPU process; no valid performance pair or speedup claim.')
(p/'source_review.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
