"""Compose the validated joint-MDO grouping with the frozen paired-QK fragment helper."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text();math=(root/'include/flashinfer/attention/mla.cuh').read_text()
control=(p.parent/'tune_joint_mdo_20261101/joint_mdo_typed.cuh').read_text()
assert sha(base)=='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
assert sha(math)=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(control)=='1190d4fd5f85d38a489ae67f0fc7e7ad5fdd567f39829eb218e1fb26ef52aa59'
helper=(p.parent/'tune_qk_pair_20261031/pair_helpers.inc').read_text().split('template <typename KTraits,typename Params>')[0]
assert 'host_batch_compute_qk_pair' in helper and '#pragma unroll 4' in helper
s=control.replace('JointMDOTraits','JointQKTraits').replace('JointMDOHostParams','JointQKHostParams').replace('BatchMLAJointMDOKernel','BatchMLAJointQKKernel')
key='// Two native32-row score tiles'
assert s.count(key)==1
s=s.replace(key,helper+'\n'+key,1)
key='''    // The second current tile may still be in flight while the first native QK runs.
    cp_async::wait_group<1>();
    __syncthreads();
    compute_mla_qk<KTraits>(&storage, hi%2, scores);
    cp_async::wait_group<0>();
    __syncthreads();
    compute_mla_qk<KTraits>(&storage, lo%2, scores+1);'''
assert s.count(key)==1
s=s.replace(key,'''    // The frozen pair helper reuses Q fragments for the two current tiles. It requires
    // both stages ready, trading the control's first-QK/load overlap for fewer Q loads.
    cp_async::wait_group<0>();
    __syncthreads();
    host_batch_compute_qk_pair<KTraits>(storage, hi%2, lo%2, scores, scores+1);''',1)
for v,host in [('base',base),('joint_mdo_typed',control),('joint_qk',s)]:
 for filename,value in [(v+'.cuh',host),(v+'_mla.cuh',math)]:
  f=p/filename
  if f.exists():assert f.read_text()==value
  else:f.write_text(value)
 for suffix in ['', '_normal']:
  d=p/('include_'+v+suffix)
  if not d.exists():shutil.copytree(root/'include',d);(d/'flashinfer/attention/mla_host_batch.cuh').write_text(host)
  assert sha((d/'flashinfer/attention/mla_host_batch.cuh').read_text())==sha(host)
  assert sha((d/'flashinfer/attention/mla.cuh').read_text())==sha(math)
m=dict(base_sha256=sha(base),joint_mdo_control_sha256=sha(control),candidate_sha256=sha(s),math_sha256=sha(math),paired_helper_sha256=sha(helper),
 mechanism='JointMDOalreadyholds2scorefragments. Reuse each Qfragment forbothnative32KVtiles, then same jointmax/d/Orescale and descendingnativePV/refill. Frozen pairedQKhelper with originalunroll4, no numeric sweep. Requireswait_group0beforeQKpair versuscontrol firstQKoverlappingsecondload. PEthenCKV and per-score MMA order preserved, FP32grouping same asjointMDO.',
 distinction='Earlier pairedQK-only prototype used2separateMDOupdates and was rejected; current jointMDOhas independentlyconfirmed~2%100GBbenefit. This isolates their composition against both production and exactjointMDOcontrol. No fixedJMsource or runningplan changes.',
 candidate_adopted=False,goal_complete=False)
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
