"""Use the victim buffer until resolve finishes; slot_pool must stay a separate array."""
import difflib,hashlib,json,shutil,time
from pathlib import Path
p=Path(__file__).resolve().parent;archive=p/'pre_gpu_alias';archive.mkdir()
for name in ['shared_claim.cuh','candidate.diff','manifest.json','build_shared_claim.json','compile_shared_claim_emu.log','compile_shared_claim_normal.log']:
 shutil.copy2(p/name,archive/name)
for name in ['jit_shared_claim','jit_shared_claim_normal']:(p/name).rename(archive/name)
s=(p/'shared_claim.cuh').read_text()
first=s.index('// Deduplicate a bounded group in shared memory');end=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct SharedClaimTraits',first)
old=s[first:end];assert old.count('params.claim_entry')==2
new=old.replace('params.claim_entry','params.entry_victim').replace('unused claim_entry','unused entry_victim')
s=s[:first]+new+s[end:]
(p/'shared_claim.cuh').write_text(s);(p/'include_shared_claim/flashinfer/attention/mla_host_batch.cuh').write_text(s)
m=json.loads((p/'manifest.json').read_text());m['candidate_sha256']=hashlib.sha256(s.encode()).hexdigest()
m['mechanism']=m['mechanism'].replace('claim_entry ranges','entry_victim ranges before LRU')
m['pre_gpu_scratch_fix']='claim_entry aliases slot_pool in the FFI. Use entry_victim (victim_of) for temporary page IDs, disjoint from slot_pool; resolve ends before LRU overwrites victims. No GPU run of old draft.'
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(p/'candidate.diff').write_text(''.join(difflib.unified_diff((p/'base.cuh').read_text().splitlines(True),s.splitlines(True),fromfile='base.cuh',tofile='shared_claim.cuh')))
(archive/'state.json').write_text(json.dumps(dict(timestamp=time.strftime('%FT%T%z'),gpu_runs=0,reason=m['pre_gpu_scratch_fix']),indent=2)+'\n')
print(m['candidate_sha256'])
