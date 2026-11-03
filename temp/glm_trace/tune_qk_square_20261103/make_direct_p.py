import difflib,hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()
s=(p/'square.cuh').read_text();assert not(p/'square_p.cuh').exists()
begin=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct SquareTraits');end=s.index('// partial_capacity bounds',begin)
section=s[begin:end].replace('Square','SquareP').replace('square_','square_p_')
section=section.replace('float score_exchange[2][64*32];','float square_sum[2][2][64];\n    alignas(16) float square_anchor[64];')
first=section.index('template <typename KTraits>\n__device__ __forceinline__ void square_p_compute_qk_pair(')
last=section.index('// Two native32-row score tiles',first)
section=section[:first]+(p/'direct_p.inc').read_text().replace('square_dot<','square_p_dot<')+'\n'+section[last:]
first=section.index('    square_p_compute_qk_pair<KTraits>(storage, hi%2, lo%2, qpe, scores, scores+1);')
last=section.index('    if (next_hi >= 0) {',first)
section=section[:first]+'''    square_p_prepare<KTraits>(storage, variant, hi%2, lo%2, hi*32, lo*32, kv_len, qpe, o, m, d);
    square_p_pv<KTraits>(storage, hi%2, o);
'''+section[last:]
section=section.replace('compute_mla_pv<KTraits>(&storage, lo%2, scores+1, d, o);','square_p_pv<KTraits>(storage, lo%2, o);')
# Only clone the new candidate block; retain the full control square path for evidence.
s=s[:end]+section+s[end:]
at=s.index('    using SquareKernelTraits =');dend=s.index('    using StreamSourceKernelTraits =',at)
dispatch=s[at:dend].replace('Square','SquareP').replace('square_','square_p_')
dispatch=dispatch.replace('bool square =','bool square_p =').replace('if (square)','if (square_p)')
dispatch=dispatch.replace('square_p_smem == 164352','square_p_smem == 149248')
s=s[:at]+dispatch+s[at:]
math=(p/'base_mla.cuh').read_bytes();(p/'square_p.cuh').write_text(s);(p/'square_p_mla.cuh').write_bytes(math)
tree=p/'include_square_p';shutil.copytree(root/'include',tree);(tree/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
for name in ['candidatepaths.py','build_candidate.py','run_focused.py','test_square.py']:
 value=(p/name).read_text().replace('square','square_p')
 (p/(name.replace('test_square','test_square_p').replace('build_candidate','build_direct_p').replace('run_focused','run_direct_p').replace('candidatepaths','directpaths'))).write_text(value)
# The new runner uses a separate plugin and keeps the already-complete control cohort intact.
runner=(p/'run_direct_p.py').read_text().replace("'candidatepaths'","'directpaths'").replace("p/'focused_state.json'","p/'direct_state.json'").replace("'focused256_'","'direct256_'").replace("'focused256_'+v","'direct256_'+v")
runner=runner.replace("run('targeted_correctness'","run('direct_targeted_correctness'")
(p/'run_direct_p.py').write_text(runner)
(p/'direct_p.diff').write_text(''.join(difflib.unified_diff((p/'square.cuh').read_text().splitlines(True),s.splitlines(True),fromfile='square.cuh',tofile='square_p.cuh')))
m=dict(base_sha256=sha((p/'base.cuh').read_bytes()),control_sha256=sha((p/'square.cuh').read_bytes()),candidate_sha256=sha(s.encode()),math_sha256=sha(math),
 mechanism='Square QK publishes BF16P directly to the original KPE/P stages. Producers reduce perheadmax before originalOrescale, native quantizedP rowsum is relayed to allPVwarps. Removes16KiBFP32score exchange and nativeP republishing; originalPV MMAorder/refill maintained. Shared149248B.',
 invariants='Same fullQK/PEthenCKV, jointmaximum, BF16quantization/nativeP rowsum, exactlaunchLRU/sourcefetch/FAR/1152Bunique; FP32reduction grouping differs. Onecooperative kernel.',adopted=False)
(p/'direct_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
