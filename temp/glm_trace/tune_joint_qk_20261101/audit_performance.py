"""Audit scatter candidates; keep GPU3 references and SXM references separate."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
p = Path(__file__).resolve().parent
root = p.parents[2]
load = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
rows, files, incomplete = [], [], []
pending_references=[]
launches = segments = 0
gpu3_floors = {}
sxm_floors={}
for refpath in sorted(p.glob('raw/*floor*.json')):
    ref=load(refpath);cfg=ref['configuration']
    if not ref.get('complete') or cfg['kind'] not in ['floor','snapshots']:
        continue
    assert ref['extra_cuda_flags']=='' and cfg['positions']==1024
    assert ref['resources_start']['foreign']==ref['resources_end']['foreign']==[]
    assert ref['source_sha256']['include/flashinfer/attention/mla.cuh']=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
    for case in ref['cases']:
        sxm_floors[cfg['kind'],cfg['mtp'],cfg['requests'],case['host_fraction_requested']]=(refpath,ref,case)
for directory in [p.parent/'tune_general_20261031/gpu3/raw',p.parent/'tune_scatter_20261031/gpu3/raw',p/'gpu3/raw']:
    for refpath in sorted(set(directory.glob('*floor*.json')) | set(directory.glob('*snapshots*.json'))):
        ref = load(refpath)
        cfg = ref['configuration']
        if not ref.get('complete') or cfg['kind'] not in ['floor','snapshots']:
            continue
        assert ref['extra_cuda_flags']=='' and cfg['positions']==1024
        assert ref['resources_start']['foreign']==ref['resources_end']['foreign']==[]
        for case in ref['cases']:
            gpu3_floors[cfg['kind'],cfg['mtp'],cfg['requests'],case['host_fraction_requested']] = (refpath,ref,case)
# Fresh retained-production small references fill only missing keys; existing experiment
# reference choices stay unchanged. No timings/formula are rewritten or normalized.
for refpath in sorted((p.parent/'tune_wait_20261031/gpu3/raw').glob('*.json')):
    if 'floor' not in refpath.name and 'snapshots' not in refpath.name:
        continue
    ref=load(refpath);cfg=ref['configuration']
    if not ref.get('complete') or cfg['kind'] not in ['floor','snapshots']:
        continue
    assert ref['extra_cuda_flags']=='' and cfg['positions']==1024
    assert ref['resources_start']['foreign']==ref['resources_end']['foreign']==[]
    assert ref['source_sha256']['include/flashinfer/attention/mla.cuh']=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
    for case in ref['cases']:
        key=(cfg['kind'],cfg['mtp'],cfg['requests'],case['host_fraction_requested'])
        if key not in gpu3_floors:
            gpu3_floors[key]=(refpath,ref,case)
# The already-audited fresh small SXM reference fills a missing self reference only.
for refpath in sorted((p.parent/'tune_wait_20261031/raw').glob('*floor*.json')):
    ref=load(refpath);cfg=ref['configuration']
    if not ref.get('complete') or cfg['kind'] not in ['floor','snapshots']:
        continue
    assert ref['extra_cuda_flags']=='' and cfg['positions']==1024
    assert ref['resources_start']['foreign']==ref['resources_end']['foreign']==[]
    assert ref['source_sha256']['include/flashinfer/attention/mla.cuh']=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
    for case in ref['cases']:
        sxm_floors.setdefault((cfg['kind'],cfg['mtp'],cfg['requests'],case['host_fraction_requested']),(refpath,ref,case))
for folder in [p / 'gpu3/raw', p / 'raw']:
    for f in sorted(folder.glob('*.json')):
        d = load(f)
        if not d.get('complete'):
            incomplete.append(dict(path=str(f.relative_to(root)), sha256=sha(f)))
            continue
        cfg = d['configuration']
        if cfg['variant'] == 'profile_wait':
            continue
        gpu3 = cfg.get('hardware_group') == 'gpu3_a100_pcie'
        hardware = 'GPU3' if gpu3 else 'SXM0/1'
        snapdir = p / ('gpu3/source_snapshots' if gpu3 else 'source_snapshots')
        assert d['resources_start']['foreign'] == d['resources_end']['foreign'] == []
        assert cfg['positions'] == cfg['steps'] * cfg['mtp']
        assert cfg['memcpy_path_version'] == 2
        assert d['source_sha256']['include/flashinfer/attention/far_link_emu.cuh'] == '5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
        for name, h in d['source_sha256'].items():
            assert sha(snapdir / (h + Path(name).suffix)) == h
        if gpu3:
            assert cfg['far_peer_is_same_device'] and cfg['far_peer_physical_device'] == 3
            assert 'membind: 1' in d['numa_policy']
        for c in d['cases']:
            ticks = c['step_indices']
            assert max(c['numerical_checks']['max_abs_error'].values()) < .01
            for mode, t in c['timings'].items():
                samples = t['per_step_samples_us']
                assert len(samples) == len(ticks)
                assert all(len(x) == cfg['passes'] for x in samples)
                assert all(math.isfinite(x) and x > 0 for z in samples for x in z)
                assert abs(statistics.mean(x for z in samples for x in z) - t['mean_us']) < 1e-7
                for repeat in range(cfg['passes']):
                    assert abs(statistics.mean(z[repeat] for z in samples) - t['pass_mean_us'][repeat]) < 1e-7
                    end = 0
                    for guard in [g for g in c['guards'] if g['mode'] == mode and g['pass_index'] == repeat]:
                        assert guard['valid'] and guard['first'] == end
                        if guard['timing_queue'] != 'cuda_graph':
                            assert guard['guard_ms'] >= guard['enqueue_ms'] + max(1, .05 * guard['enqueue_ms'])
                        end = guard['last']
                        segments += 1
                    assert end == len(ticks)
                launches += len(ticks) * cfg['passes']
            if cfg['kind'] in ['floor','snapshots']:
                assert d['extra_cuda_flags'] == '' and cfg['positions'] == 1024
                continue
            fraction=c['host_fraction_requested']
            if cfg['kind']=='lru':
                assert cfg['capacity'] == 4096
                assert ticks==list(range(cfg['steps']))
                assert all(x == c['far_unique'][0] for x in c['ours_unique_miss_samples'][0])
                assert all(x == c['far_refs'][0] for x in c['ours_reference_miss_samples'][0])
                hit = 1 - sum(map(sum, c['ours_reference_miss_samples'])) / cfg['passes'] / sum(c['far_refs'])
                assert abs(hit - c['ours_reference_cache_hit_rate']) < 1e-12
            else:
                assert cfg['kind']=='nocache' and cfg['capacity']==0
                assert ticks==[x//cfg['mtp'] for x in [0,256,512,768]]
                hit=0
            refkind='floor' if cfg['kind']=='lru' else 'snapshots'
            if gpu3:
                floorpath,floor,fc=gpu3_floors[refkind,cfg['mtp'],cfg['requests'],fraction]
            else:
                fresh=sxm_floors.get((refkind,cfg['mtp'],cfg['requests'],fraction))
                if fresh:
                    floorpath,floor,fc=fresh
                else:
                    floorpath = root / f"temp/glm_trace/kernel_report_20261029/raw/{refkind}_floor_mtp{cfg['mtp']}_b{cfg['requests']}.json"
                    floor = load(floorpath)
                    fc = next(x for x in floor['cases'] if x['host_fraction_requested']==fraction)
            assert floorpath.exists(), 'Missing same-hardware normal compute reference: ' + str(floorpath)
            assert floor['complete'] and floor['bindings'] == d['bindings']
            if 'include/flashinfer/attention/mla.cuh' not in floor['source_sha256']:
                pending_references.append(dict(raw=str(f.relative_to(root)),fraction=fraction,
                    cohort=f.stem.split(cfg['variant']+'_staged_',1)[0],variant=cfg['variant'],
                    kind=cfg['kind'],mtp=cfg['mtp'],requests=cfg['requests'],fetch_ctas=cfg['fetch_ctas'],
                    setting=cfg['setting']['name'],positions=cfg['positions'],passes=cfg['passes'],
                    ours_us=c['timings']['ours']['mean_us'],memcpy_us=c['timings']['memcpy']['mean_us'],
                    reference_hit_rate=hit,pass_mean_us=c['timings']['ours']['pass_mean_us'],
                    reference_raw=str(floorpath.relative_to(root)),reference_sha256=sha(floorpath),
                    reason='Legacy reference lacks math-header SHA; fresh same-hardware normal reference queued. Timings/guards/numerics were validated but reference row is withheld.'))
                continue
            reference_math = floor['source_sha256']['include/flashinfer/attention/mla.cuh']
            executed_math = d['source_sha256']['include/flashinfer/attention/mla.cuh']
            if reference_math != executed_math:
                assert reference_math == '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
                assert executed_math == sha(p/(cfg['variant']+'_mla.cuh'))
                proof = load(p/('compiler_'+cfg['variant']+'_normal_summary.json'))
                assert proof['complete'] and proof['unchanged_original_kernels'] == 68
                assert proof['math_header_sha256'] == executed_math
                assert proof['candidate_header_sha256'] == d['source_sha256']['include/flashinfer/attention/mla_host_batch.cuh']
                assert proof['base_header_sha256'] == 'b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'

            fullticks=fc['step_indices']
            compute = [statistics.mean(fc['timings']['local']['per_step_samples_us'][fullticks.index(i)]) for i in ticks]
            bw = cfg['setting'].get('bandwidth_GBps')
            if not bw and not gpu3:
                calibration = load(root / 'temp/glm_trace/briefing_report_20261030/physical_bandwidth.json')
                bw = next(x['reference_GBps'] for x in calibration['settings'] if x['name'] == cfg['setting']['name'])
            if not bw and gpu3:
                calibration=p.parent/'tune_general_20261031/gpu3'
                if cfg['setting']['name']=='native_hbm':
                    bw=load(calibration/'native_hbm_bandwidth.json')['reference_GBps']
                    assert d['extra_cuda_flags']==''
                else:
                    bw=next(x['reference_GBps'] for x in load(calibration/'physical_bandwidth.json')['settings'] if x['name']=='pcie')
            assert bw
            reference = statistics.mean(max(t, u * 1152 / (1000 * bw)) for t, u in zip(compute, c['ours_unique_misses']))
            variant = cfg['variant']
            rows.append(dict(hardware=hardware, cohort=f.stem.split(variant + '_staged_', 1)[0],
                variant=variant, kind=cfg['kind'], fraction=fraction, mtp=cfg['mtp'], requests=cfg['requests'], positions=cfg['positions'],
                passes=cfg['passes'], fetch_ctas=cfg['fetch_ctas'], split=cfg['host_split'], setting=cfg['setting']['name'],
                ours_us=c['timings']['ours']['mean_us'], memcpy_us=c['timings']['memcpy']['mean_us'],
                local_us=statistics.mean(compute), reference_us=reference,
                over_reference=c['timings']['ours']['mean_us'] / reference - 1,
                reference_hit_rate=hit, unique_miss_MB=statistics.mean(c['ours_unique_misses']) * 1152 / 1e6,
                pass_mean_us=c['timings']['ours']['pass_mean_us'], max_abs_error=max(c['numerical_checks']['max_abs_error'].values()),
                raw=str(f.relative_to(root)), reference_raw=str(floorpath.relative_to(root)),
                header_sha256=d['source_sha256']['include/flashinfer/attention/mla_host_batch.cuh']))
        files.append(dict(path=str(f.relative_to(root)), sha256=sha(f)))
index = {(r['hardware'], r['cohort'], r['kind'], r['fraction'], r['mtp'], r['requests'], r['positions'], r['passes'], r['fetch_ctas'], r['split'], r['setting'], r['variant']): r for r in rows}
pairs = []
for k, new in index.items():
    if k[-1] == 'base':
        continue
    old = index.get((*k[:-1], 'base'))
    if old:
        pairs.append(dict(**{key: new[key] for key in ['hardware', 'cohort', 'variant', 'kind', 'fraction', 'mtp', 'requests', 'fetch_ctas', 'setting', 'positions', 'passes']},
            old_us=old['ours_us'], new_us=new['ours_us'], reduction=1-new['ours_us']/old['ours_us'],
            old_raw=old['raw'], new_raw=new['raw'], memcpy_us=new['memcpy_us'], local_us=new['local_us'],
            reference_us=new['reference_us'], over_reference=new['over_reference']))
tests = []
for f in sorted(p.glob('tests_*_gpu3.json')):
    t = load(f)
    if not t.get('complete'):
        continue
    if f.name=='tests_production_gpu3.json':
        assert t['normal_build'] and t['exit_code']==0
        assert t['resources_start']['foreign']==t['resources_end']['foreign']==[]
        assert t['production_header_sha256']==sha(p/'cached_ids.cuh')
        assert t['math_header_sha256']=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
        assert t['test_source_sha256']==sha(root/'tests/attention/test_mla_host_batch.py')
        log=p/'pytest_production_gpu3.log'
        assert sha(log)==t['log_sha256'] and '105 passed' in log.read_text()
        assert len(t['libraries'])==4
        retained=load(p/'cached_ids_retention.json')
        assert retained['production_validation']['sha256']==sha(f)
        tests.append(dict(variant='retained_production',tests=105,sha256=sha(f),fresh_root_include=True))
        continue
    assert t['normal_build'] and t['exit_code'] == t['compile_exit_code'] == 0
    assert t['resources_start']['foreign'] == t['resources_end']['foreign'] == []
    assert t['header_sha256'] == sha(p / (t['variant'] + '.cuh'))
    test_snapshot = p / 'source_snapshots' / (t['test_source_sha256'] + '.py')
    if not test_snapshot.exists():
        test_source = root / 'tests/attention/test_mla_host_batch.py'
        assert t['test_source_sha256'] == sha(test_source)
        test_snapshot.parent.mkdir(exist_ok=True)
        test_snapshot.write_bytes(test_source.read_bytes())
    assert sha(test_snapshot) == t['test_source_sha256']
    log = p / ('pytest_' + t['variant'] + '_gpu3.log')
    assert sha(log) == t['log_sha256'] and '105 passed' in log.read_text()
    tests.append(dict(variant=t['variant'], tests=105, sha256=sha(f)))
stress_tests = []
for f in sorted(p.glob('stress_*_gpu3.json')):
    t = load(f)
    if not t.get('complete'):
        continue
    assert t['exit_code'] == 0 and t['tests'] == 4
    assert t['resources_start']['foreign'] == t['resources_end']['foreign'] == []
    assert t['header_sha256'] == sha(p/(t['variant']+'.cuh'))
    assert t['helper_source_sha256'] == sha(root/'tests/attention/test_mla_host_batch.py')
    source = p/'test_cofetch_progress.py'
    assert sha(source) == t['stress_source_sha256']
    snapshot = p/'source_snapshots'/(t['stress_source_sha256']+'.py')
    if not snapshot.exists():
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    assert sha(snapshot) == t['stress_source_sha256']
    log = p/('pytest_stress_'+t['variant']+'.log')
    assert sha(log) == t['log_sha256'] and '4 passed' in log.read_text()
    stress_tests.append(dict(variant=t['variant'], tests=4, sha256=sha(f),
        scope='BF16/FP16, 128 pools >108 SMs, K2048, capacities512/4096, eviction/zero-miss/Graph, all filled cache payloads'))
pending_index={(r['cohort'],r['kind'],r['mtp'],r['requests'],r['setting'],r['fraction'],r['variant']):r for r in pending_references}
pending_pairs=[]
for key,new in pending_index.items():
    old=pending_index.get((*key[:-1],'base'))
    if key[-1]!='base' and old:
        pending_pairs.append(dict(cohort=new['cohort'],mtp=new['mtp'],requests=new['requests'],setting=new['setting'],
            old_us=old['ours_us'],new_us=new['ours_us'],reduction=1-new['ours_us']/old['ours_us'],
            old_raw=old['raw'],new_raw=new['raw'],reference_state='pending math-header provenance'))
audit = dict(goal_complete=False, candidate_adopted=False, rows=rows, pairs=pairs, tests=tests, stress_tests=stress_tests,
    pending_references=pending_references,
    pending_reference_pairs=pending_pairs,
    timed_launches=launches, accepted_segments=segments, files=files, incomplete=incomplete,
    production_header_sha256=sha(root / 'include/flashinfer/attention/mla_host_batch.cuh'),
    original_matrix_sha256=sha(root / 'temp/glm_trace/kernel_report_20261029/matrix.json'))
retention=p/'cached_ids_retention.json'
if retention.exists():
    retained=load(retention)
    audit['candidate_adopted']=bool(retained['candidate_adopted'] and retained['candidate_header_sha256']==audit['production_header_sha256'])
    audit['retained_variant']='cached_ids' if audit['candidate_adopted'] else None
assert audit['original_matrix_sha256'] == 'e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
(p / 'performance_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
pending_fields=['cohort','variant','kind','mtp','requests','fetch_ctas','setting','fraction','positions','passes',
                'ours_us','memcpy_us','reference_hit_rate','raw','reference_raw','reference_sha256','reason']
with (p/'pending_reference_timings.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=pending_fields,extrasaction='ignore',lineterminator='\n')
    writer.writeheader();writer.writerows(pending_references)
if rows:
    with (p / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({k: v for k, v in audit.items() if k not in ['rows', 'files']}, indent=2))
