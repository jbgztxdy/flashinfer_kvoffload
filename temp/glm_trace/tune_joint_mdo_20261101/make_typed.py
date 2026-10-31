"""Use a dedicated native-equivalent traits type to isolate helper instantiation usage."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
s=(p/'joint_mdo_after.cuh').read_text()
key='template <typename Params>\nstruct JointMDOHostParams'
assert s.count(key)==1
s=s.replace(key,'''template <typename Q, typename KV, typename O, typename Id>
struct JointMDOTraits : KernelTraits<false, 2, true, 512, 64, 64, 32, Q, KV, O, Id> {};

template <typename Params>
struct JointMDOHostParams''',1)
key='using KV64Traits = KernelTraits<false, 2, true, 512, 64, 64, 32, DTypeQ, DTypeKV, DTypeO, IdType>;'
assert s.count(key)==1
s=s.replace(key,'using KV64Traits = JointMDOTraits<DTypeQ, DTypeKV, DTypeO, IdType>;',1)
sha=lambda s:hashlib.sha256(s.encode()).hexdigest();math=(p/'base_mla.cuh').read_text();v='joint_mdo_typed'
for name,value in [(v+'.cuh',s),(v+'_mla.cuh',math)]:
 f=p/name
 if f.exists():assert f.read_text()==value
 else:f.write_text(value)
for suffix in ['', '_normal']:
 d=p/('include_'+v+suffix)
 if not d.exists():shutil.copytree(p/'include_base',d);(d/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
 assert sha((d/'flashinfer/attention/mla_host_batch.cuh').read_text())==sha(s)
m=json.loads((p/'after_manifest.json').read_text());m.update(candidate_sha256=sha(s),predecessor_host_sha256=sha((p/'joint_mdo_after.cuh').read_text()),correction='Definition order alone reproduces same original instruction changes. Give the new kernel a dedicated inherited native-equivalent traits type, isolating template helper instantiation usage. No algorithm, fields, shared memory or math header change; keep both failed gates without normalization.')
(p/'typed_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
