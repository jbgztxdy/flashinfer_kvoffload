"""Reviewable host-dispatch naming cleanup; preserve every measured device definition literally."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
source=(p/'joint_qk.cuh').read_text();sha=lambda t:hashlib.sha256(t.encode()).hexdigest()
assert sha(source)=='a2f37fb985fcbc523e740c85e9cc6468afad74fd8716c11aef4221ebab7c4b7c'
# All replacements occur in the new host dispatcher; no device/helper/parameter definition changes.
start=source.index('    using KV64Traits = JointQKTraits')
end=source.index('    DISPATCH_SMEM_CONFIG(',start)
old=source[start:end];new=old.replace('KV64Traits','JointTraits').replace('KV64Params','JointParams').replace('kv64_smem','joint_smem').replace('kv64_kernel','joint_kernel').replace('kv64_blocks','joint_blocks').replace('kv64_grid','joint_grid').replace('const bool kv64 =','const bool joint_rows =').replace('if (kv64)','if (joint_rows)')
assert new!=old
retained=source[:start]+new+source[end:]
assert source[:start]==retained[:start] and source[end:]==retained[start+len(new):]
for filename,value in [('retained_joint_qk.cuh',retained),('retained_joint_qk_mla.cuh',(p/'joint_qk_mla.cuh').read_text())]:
 f=p/filename
 if f.exists():assert f.read_text()==value
 else:f.write_text(value)
for suffix in ['', '_normal']:
 d=p/('include_retained_joint_qk'+suffix)
 if not d.exists():shutil.copytree(root/'include',d);(d/'flashinfer/attention/mla_host_batch.cuh').write_text(retained)
 assert sha((d/'flashinfer/attention/mla_host_batch.cuh').read_text())==sha(retained)
m=dict(measured_header_sha256=sha(source),reviewable_header_sha256=sha(retained),math_header_sha256=sha((p/'joint_qk_mla.cuh').read_text()),
 change='Hostlocalaliases/variables renamedfromKV64 toJoint; device/helpers/parameterdefinitionsliteralunchanged. Requirefulloriginal+targetkernelinstruction/resourceidentity inbothEMUandnormalbuilds beforebridginganymeasurement.',candidate_adopted=False)
(p/'retained_source_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
