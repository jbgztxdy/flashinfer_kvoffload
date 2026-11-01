"""Restore the original two-buffer fetch policy after removing unused shared QPE."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
old=p/'no_copy_pe.cuh'
assert sha(old)=='a05122ada68293e0968d9622b15a0d67900668aa9a303c558dcfe47600a4c086'
s=old.read_text()
start=s.index('struct NoCopyPEStorage {');end=s.index('struct NoCopyPETraits',start)
part=s[start:end]
assert part.count('    alignas(16) O o_smem[64 * 512];')==1
part=part.replace('    alignas(16) O o_smem[64 * 512];',
    '    alignas(16) O o_smem[64 * 512];\n'
    '    // Keep the original fetch double-buffer policy and launch budget.\n'
    '    alignas(16) uint8_t fetch_budget[147968];')
s=s[:start]+part+s[end:]
assert s.count('static_assert(no_copy_pe_smem == 139904);')==1
s=s.replace('static_assert(no_copy_pe_smem == 139904);','static_assert(no_copy_pe_smem == 147968);')
for name,value in [('no_copy_pe_fixed.cuh',s),('no_copy_pe_fixed_mla.cuh',(p/'base_mla.cuh').read_text())]:
    f=p/name
    if f.exists():assert f.read_text()==value
    else:f.write_text(value)
d=p/'include_no_copy_pe_fixed'
if not d.exists():
    shutil.copytree(p/'include_no_copy_pe',d)
    (d/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
assert sha(d/'flashinfer/attention/mla_host_batch.cuh')==sha(p/'no_copy_pe_fixed.cuh')
m=dict(variant='no_copy_pe_fixed',header_sha256=sha(p/'no_copy_pe_fixed.cuh'),
    math_sha256=sha(p/'base_mla.cuh'),original_control_header_sha256=sha(old),
    compute_layout_bytes=139904,storage_bytes=147968,dynamic_shared_bytes=147968,
    fetch_payload_bytes=2*64*576*2,fetch_buffers=2,original_control_fetch_buffers=1,
    original_control_is_out_of_bounds=False,candidate_adopted=False,goal_complete=False,
    change='Add union fetch-budget padding, restoring sizeof(SharedStorage)>=147456 and original double-buffer fetch. Original139904 storage caused a safe one-buffer specialization; allocation-only change would not restore double buffering.')
(p/'fixed_control_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(m)
