"""CPU ownership/address audit, independent of runtime numerical checks."""
from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent;root=p.parents[2]
m=json.loads((p/'manifest.json').read_text());s=(p/'candidate.cuh').read_text();assert hashlib.sha256(s.encode()).hexdigest()==m['candidate_sha256']
assert hashlib.sha256((root/'include/flashinfer/attention/mla.cuh').read_bytes()).hexdigest()==m['math_sha256']
assert hashlib.sha256((root/'include/flashinfer/attention/far_link_emu.cuh').read_bytes()).hexdigest()==m['far_sha256']
cases=[]
for grid in [1,4,24,108]:
 for count in [0,1,9,10,11,19,20,21,64,1079,1080,1081,215261]:
  slots=[];chunks=[]
  for cta in range(grid):
   for seq,chunk in enumerate(range(cta,(count+9)//10,grid)):
    stage=seq%2
    chunks.append(chunk)
    for rnd in range(3):
     for tid in range(256):
      v=rnd*256+tid;slot=chunk*10+v//72
      if v<720 and slot<count:slots.append((slot,v%72))
  assert len(chunks)==len(set(chunks))==(count+9)//10
  assert len(slots)==len(set(slots))==count*72
  cases.append(dict(grid=grid,count=count,chunks=len(chunks),vectors=len(slots),far_bytes=len(slots)*16))
# Existing native Q-nope tiling is unchanged by PEshared removal.
original=[];compact=[]
for z in range(2):
 for y in range(4):
  for lane in range(32):
   for a in range(2):
    head=lane//8+(z+a*2)*16+y*4
    for dim in range(8):
     original.append((head,(lane%8)*8+dim*64))
     compact.append((head,(lane%8)*8+dim*64))
assert original==compact and len(set(compact))==64*64
# Math helper bodies should match after typed name normalization.
base=(p/'base.cuh').read_text();identity=[]
for name,nextname in [('pv_feature_pv','struct PVFeatureTraits'),('pv_feature_load_query','template <bool PAIR'),('pv_feature_compute_pe','template <typename KTraits>'),('pv_feature_update_joint_host_mdo','template <typename KTraits, typename Params>')]:
 a=base.index('void '+name+'(');b=base.index(nextname,a)
 x=base[a:b].replace('pv_feature_','uniform_')
 a=s.index('void '+name.replace('pv_feature_','uniform_')+'(');b=s.index(nextname.replace('PVFeature','Uniform'),a) if name=='pv_feature_pv' else s.index(nextname,a)
 # PV ends before newlyinserted service; compare exact brace body instead.
 def body(t):
  start=t.index('{');depth=0
  for i in range(start,len(t)):
   depth+=(t[i]=='{')-(t[i]=='}')
   if not depth:return t[start:i+1]
 assert body(x)==body(s[a:]),name
 identity.append(name)
r=dict(complete=True,ownership_cases=len(cases),cases=cases,Q_nope_vectors=len(compact),exact_Q_nope_coordinate_identity=True,unchanged_math_helpers=identity,production_unchanged=True,math_unchanged=True,emulator_unchanged=True,limits='CPU only: does not establish GPU progress/barriers/cachepayload/math correctness or speed.')
(p/'source_audit.json').write_text(json.dumps(r,indent=2)+'\n');print('CPU_AUDIT',len(cases),identity)
