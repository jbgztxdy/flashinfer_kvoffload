"""Preserve two attention stages, add small worker assistance to dedicated fetchers."""
import hashlib
import json
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
previous = p.parent/'tune_cofetch_20261101'
source = (previous/'cofetch.cuh').read_text()
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
assert sha(source) == 'af61098d178108843357bee2218d6f6719a2f3355591a5efe6f77dc6eebbbf6f'
begin = source.index('// A CTA keeps its attention state while advancing a disjoint cyclic fetch queue.')
end = source.index('// Warp 0 waits until every staged entry of a KV tile has landed;', begin)
s = source[:begin]+(p/'hybrid_helpers.inc').read_text()+'\n'+source[end:]
s = s.replace('CoFetchHostBatchParams', 'HybridCoFetchHostBatchParams')
s = s.replace('using OneStageTraits = KernelTraits<false, 1, true,',
              'using OneStageTraits = KernelTraits<false, 2, true,')
key = '''      cg::this_grid().sync();
    }
    init_host_batch_cofetch<KTraits>(params, unique, epoch);'''
assert s.count(key) == 1
s = s.replace(key, '''      // The compaction gap flag is dead after prepare; reset and reuse it as
      // a job cursor until the existing exit path resets it before the next launch.
      if (blockIdx.y == 0 && tid == 0) params.state[kHostBatchGaps] = 0;
      cg::this_grid().sync();
    }
    init_host_batch_cofetch<KTraits>(params, unique, epoch, smem);
    if (from_back) finish_host_batch_cofetch<KTraits>(params);''')
for variant, text in [('base', (previous/'base.cuh').read_text()), ('hybrid', s)]:
    assert not (p/(variant+'.cuh')).exists()
    (p/(variant+'.cuh')).write_text(text)
    shutil.copytree(root/'include', p/('include_'+variant))
    (p/('include_'+variant)/'flashinfer/attention/mla_host_batch.cuh').write_text(text)
for name in ['.gitignore', 'compile.py', 'resources.py', 'resources_gpu3.py', 'experiment.json',
             'plans.cpp', 'bench.py', 'bench_gpu3_native.py', 'test_candidate.py', 'candidatepaths.py',
             'screen.py', 'audit_performance.py', 'test_cofetch_progress.py', 'stress_candidate.py']:
    text = (previous/name).read_text().replace("'cofetch','stage1_control'", "'hybrid'")
    text = text.replace("['cofetch', 'stage1_control']", "['hybrid']")
    text = text.replace("default=['cofetch', 'base']", "default=['hybrid', 'base']")
    text = text.replace("default='cofetch'", "default='hybrid'")
    if name == 'experiment.json':
        obj = json.loads(text)
        obj.update(candidate='hybrid', status='candidate_in_progress')
        text = json.dumps(obj, indent=2)+'\n'
    (p/name).write_text(text)
text = (previous/'compiler_cofetch.py').read_text().replace('cofetch', 'hybrid')
text = text.replace('CoFetchHostBatchParams', 'HybridCoFetchHostBatchParams')
(p/'compiler_hybrid.py').write_text(text)
text = (previous/'probe_cofetch.py').read_text().replace('cofetch', 'hybrid')
text = text.replace('CoFetchHostBatchParams', 'HybridCoFetchHostBatchParams').replace('111104', '147968')
(p/'probe_hybrid.py').write_text(text)
manifest = dict(predecessor_sha256=sha(source), candidate_sha256=sha(s),
    base_sha256=sha((previous/'base.cuh').read_text()),
    mechanism='Keep original two-stage attention. Dedicated CTAs fetch 64-row jobs from 2x64 buffers then join attention; workers service 2x4-row buffers once each eight KV tiles and during waits. Atomic job cursor assigns disjoint16-row jobs; every16rows gets its own ready flag. Same math/LRU/payload/unchanged emulator hooks.',
    progress='Worker16-row jobs publish only after all four4-row subchunks. Queues drain before exit, wait loops service local jobs; dedicated queues drain before joining attention. LRU selections finish before issue. Gap flag reused only after prepare/compaction and reset at existing last-CTA exit.',
    predecessor_evidence='One-stage control 100GB slowed11.8-12.3%; all-CTA16-row prototype slowed38.9-53.0%. Preserve original attention stages and dedicated transfer channel rather than cadence/unroll scan.',
    worker_payload_bytes=9216, attention_dynamic_bytes=147968,
    candidate_adopted=False, goal_complete=False)
(p/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(manifest)
