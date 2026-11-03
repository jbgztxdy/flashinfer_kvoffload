import difflib,hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];old=p.parent/'tune_elastic_fetch_20261102'
sha=lambda x:hashlib.sha256(x).hexdigest()
b=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes();math=(root/'include/flashinfer/attention/mla.cuh').read_bytes()
assert sha(b)=='754706f8d8fba14a361ce6dcfc64a358446e5df36b225f392fa8a3e7756ded79'
assert not(p/'anchor.cuh').exists()
s=b.decode();start=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits');end=s.index('// partial_capacity bounds',start)
section=s[start:end].replace('StreamSource','Anchor').replace('stream_source_','anchor_')
section=section.replace('fetch_anchor_pages','fetch_stream_source_pages')
begin=section.index('// Two native32-row score tiles share one maximum')
finish=section.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void anchor_attend_joint_host_tiles',begin)
section=section[:begin]+(p/'anchor.inc').read_text()+'\n'+section[finish:]
section=section.replace('anchor_update_joint_host_mdo<KTraits>','anchor_update_mdo<true,KTraits>')
section=section.replace('update_mdo_states_<KTraits>(&storage, 0, variant, scores, o, m, d);','anchor_update_mdo<false,KTraits>(&storage, variant, scores, scores, o, m, d);')
section=section.replace('  finalize_m_<KTraits>(variant, m);','  // The anchor is already in log2 exponent units, including odd-tail items.')
at=s.index('    using StreamSourceKernelTraits =');dend=s.index('    using CachedPEKernelTraits =',at)
dispatch=s[at:dend].replace('StreamSource','Anchor').replace('stream_source','anchor')
dispatch=dispatch.replace('const bool anchor = params.stage','const bool anchor = std::is_same_v<DTypeKV,nv_bfloat16> && params.sm_scale > 0.f && params.stage')
s=s[:end]+section+s[end:];at=s.index('    using StreamSourceKernelTraits =');s=s[:at]+dispatch+s[at:]
for v,data in [('base',b),('anchor',s.encode())]:
 (p/(v+'.cuh')).write_bytes(data);(p/(v+'_mla.cuh')).write_bytes(math)
 tree=p/('include_'+v);shutil.copytree(root/'include',tree);(tree/'flashinfer/attention/mla_host_batch.cuh').write_bytes(data)
for name in ['bench_gpu3_native.py','resources_gpu3.py','experiment.json','plans.cpp','compile.py','audit_performance.py']:
 shutil.copy2(old/name,p/name)
for name in ['candidatepaths.py','build_candidate.py']:
 (p/name).write_text((old/name).read_text().replace('elastic_fetch','anchor'))
runner=(p.parent/'tune_global_query_20261102/run_focused.py').read_text().replace('global_query','anchor')
(p/'run_focused.py').write_text(runner)
test=(p.parent/'tune_global_query_20261102/test_global_query.py').read_text().replace('global_query','anchor')
(p/'test_anchor.py').write_text(test)
(p/'candidate.diff').write_text(''.join(difflib.unified_diff(b.decode().splitlines(True),s.splitlines(True),fromfile='base.cuh',tofile='anchor.cuh')))
m=dict(base_sha256=sha(b),candidate_sha256=sha(s.encode()),math_sha256=sha(math),
 mechanism='BF16 bounded log2 exponent anchor: round upward to8exponent units, retain until exceeded; warp-uniform skip of O/d rescale while anchors stable. Unlike prior unit-scale branch, anchor stability is extended. No secondpass/extraKV/featuretruncation.',
 invariants='One cooperative kernel, original QK/PV order, actualLRU/payload/fetch/sourcehint/FAR/unique bytes unchanged. Softmax equation invariant to common additive exponent anchor; BF16 P rounding and FP32 scale grouping differ.',
 risks='Extra headmax/ceil/branch may cost more than saved rescale; up to8log2 extra exponent range may underflow subnormal BF16 probabilities. Large/extreme logits and late-max adversarial tests required.',adopted=False)
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
