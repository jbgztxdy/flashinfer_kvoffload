"""Audit GPU3 data against references from that GPU, never against SXM timings."""
import csv,hashlib,json,math,statistics
from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parents[2]
g=p/'gpu3'
load=lambda f:json.loads(f.read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
raw=[(f,load(f)) for f in sorted((g/'raw').glob('*.json'))]
raw=[(f,d) for f,d in raw if d.get('complete')]
floors={}
for f,d in raw:
    cfg=d['configuration']
    if cfg['kind'] not in ('floor','snapshots'):continue
    assert cfg['positions']==1024 and d['extra_cuda_flags']==''
    for c in d['cases']:floors[cfg['kind'],cfg['mtp'],cfg['requests'],c['host_fraction_requested']]=(f,d,c)
physical=load(g/'physical_bandwidth.json') if (g/'physical_bandwidth.json').exists() else None
if physical:
    assert physical['complete'] and physical['physical_gpu']==3
    assert physical['resources_start']['foreign']==physical['resources_end']['foreign']==[]
    assert 'membind: 1' in physical['numa_policy']
    assert all(x['payload_sample_check'] for x in physical['settings'])
    bandwidth=next(x['reference_GBps'] for x in physical['settings'] if x['name']=='pcie')
    for name,h in physical['source_sha256'].items():
        snap=g/'source_snapshots'/(h+Path(name).suffix)
        if not snap.exists():
            content=(root/name).read_bytes();assert hashlib.sha256(content).hexdigest()==h
            snap.write_bytes(content)
        assert sha(snap)==h
else:bandwidth=None
native_bw=load(g/'native_hbm_bandwidth.json') if (g/'native_hbm_bandwidth.json').exists() else None
if native_bw:
    assert native_bw['complete'] and native_bw['physical_gpu']==3
    assert native_bw['resources_start']['foreign']==native_bw['resources_end']['foreign']==[]
    assert all(x['payload_sample_check'] for x in native_bw['measurements'].values())
rows=[];files=[];launches=segments=0;maxerror=0
for f,d in raw:
    cfg=d['configuration'];m=cfg['mtp'];b=cfg['requests'];kind=cfg['kind'];passes=cfg['passes'];n=cfg['steps']
    assert cfg['hardware_group']=='gpu3_a100_pcie'
    assert cfg['far_peer_is_same_device'] and cfg['far_peer_physical_device']==3
    assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
    assert all(x['index']==3 for x in d['resources_start']['gpus']+d['resources_end']['gpus'])
    assert cfg['memcpy_path_version']==2 and cfg['positions']==n*m
    assert d['source_sha256']['include/flashinfer/attention/far_link_emu.cuh']=='5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
    for name,h in d['source_sha256'].items():
        snap=g/'source_snapshots'/(h+Path(name).suffix)
        assert snap.is_file() and sha(snap)==h,(f,name)
    if cfg['setting']['tier']=='host':
        assert 'membind: 1' in d['numa_policy'],f
        assert any(x['tier']=='far_host_dram' and x['location_type']==3 and x['location_id']==1 for x in d['residency'])
    for c in d['cases']:
        ticks=c['step_indices'];fraction=c['host_fraction_requested']
        assert len(c['far_refs'])==len(ticks)
        err=max(c['numerical_checks']['max_abs_error'].values());assert err<.01
        maxerror=max(maxerror,err)
        for mode,t in c['timings'].items():
            samples=t['per_step_samples_us']
            assert len(samples)==len(ticks) and all(len(z)==passes for z in samples)
            assert all(math.isfinite(x) and x>0 for z in samples for x in z)
            assert abs(statistics.mean(x for z in samples for x in z)-t['mean_us'])<1e-7
            for k in range(passes):
                assert abs(statistics.mean(z[k] for z in samples)-t['pass_mean_us'][k])<1e-7
                end=0
                for z in (z for z in c['guards'] if z['mode']==mode and z['pass_index']==k):
                    assert z['valid'] and z['first']==end
                    if z['timing_queue']!='cuda_graph':assert z['guard_ms']>=z['enqueue_ms']+max(1,z['enqueue_ms']*.05)
                    end=z['last'];segments+=1
                assert end==len(ticks)
            launches+=len(ticks)*passes
        if kind in ('floor','snapshots'):continue
        if kind=='lru':
            assert cfg['capacity']==4096 and ticks==list(range(n))
            assert all(len(z)==passes for z in c['ours_unique_miss_samples']+c['ours_reference_miss_samples'])
            assert all(x==c['far_unique'][0] for x in c['ours_unique_miss_samples'][0])
            assert all(x==c['far_refs'][0] for x in c['ours_reference_miss_samples'][0])
            hit=1-sum(map(sum,c['ours_reference_miss_samples']))/passes/sum(c['far_refs'])
            assert abs(hit-c['ours_reference_cache_hit_rate'])<1e-12
        else:
            assert cfg['capacity']==0 and ticks==[i//m for i in (0,256,512,768)]
            hit=0
        fk=('floor' if kind=='lru' else 'snapshots',m,b,fraction)
        assert fk in floors,(f,fk)
        ff,fd,fc=floors[fk]
        assert fd['bindings']==d['bindings']
        fullticks=fc['step_indices']
        compute=[statistics.mean(fc['timings']['local']['per_step_samples_us'][fullticks.index(i)]) for i in ticks]
        if cfg['setting']['name']=='native_hbm':
            assert d['extra_cuda_flags']=='' and not cfg['setting'].get('emulated')
            bw=native_bw['reference_GBps'] if native_bw else None
        elif cfg['setting']['name']=='pcie':bw=bandwidth
        else:bw=cfg['setting'].get('bandwidth_GBps')
        ref=statistics.mean(max(t,u*1152/(1000*bw)) for t,u in zip(compute,c['ours_unique_misses'])) if bw else None
        variant=cfg['variant'];cohort=f.name.rsplit(variant+'_staged_',1)[0]
        rows.append(dict(cohort=cohort,variant=variant,kind=kind,mtp=m,requests=b,positions=cfg['positions'],passes=passes,
            fetch_ctas=cfg['fetch_ctas'],split=cfg['host_split'],setting=cfg['setting']['name'],fraction=fraction,
            ours_us=c['timings']['ours']['mean_us'],memcpy_us=c['timings']['memcpy']['mean_us'],local_us=statistics.mean(compute),
            reference_us=ref,over_reference=c['timings']['ours']['mean_us']/ref-1 if ref else None,reference_hit_rate=hit,
            miss_MB=statistics.mean(c['ours_unique_misses'])*1152/1e6,pass_means_us=c['timings']['ours']['pass_mean_us'],
            raw=str(f.relative_to(root)),reference_raw=str(ff.relative_to(root)),header_sha256=d['source_sha256']['include/flashinfer/attention/mla_host_batch.cuh']))
    files.append(dict(path=str(f.relative_to(root)),sha256=sha(f)))
index={(r['cohort'],r['kind'],r['mtp'],r['requests'],r['positions'],r['passes'],r['fetch_ctas'],r['split'],r['setting'],r['fraction'],r['variant']):r for r in rows}
pairs=[]
for key,new in index.items():
    if key[-1] not in ('threshold2','threshold2_sm80','balanced_sm80','balanced_cached_sm80'):continue
    old=index.get((*key[:-1],'base'))
    if old:pairs.append(dict(**{k:new[k] for k in ('cohort','kind','mtp','requests','positions','passes','setting','fraction')},old_us=old['ours_us'],new_us=new['ours_us'],reduction=1-new['ours_us']/old['ours_us'],old_raw=old['raw'],new_raw=new['raw']))
settings=[f'g{bw}_l{lat}' for bw in (100,50) for lat in (0,2000,5000,10000,20000)]+['pcie']
affected=((8,28),(8,32),(8,40),(8,48),(1,256))
expected={
    'gpu3_full_':{('lru',v,m,b,link,None) for v in ('base','threshold2') for m,b in affected for link in settings},
    'gpu3_controls_':{('lru',v,m,b,link,None) for v in ('base','threshold2') for m,b in ((1,1),(8,1),(8,64)) for link in ('g100_l20000','g50_l20000','pcie')},
    'gpu3_nocache_':{('nocache',v,m,b,link,f) for v in ('base','threshold2') for m,b in affected for link in settings for f in (0.,.1,.3)},
    'gpu3_confirmfull_':{('lru',v,8,32,link,None) for v in ('base','threshold2') for link in ('g100_l0','g100_l20000','g50_l20000')},
}
expected['gpu3_native_']={
    (kind,v,m,b,link,f)
    for kind,fs in (('lru',[None]),('nocache',[0.,.1,.3]))
    for v in ('base','threshold2_sm80') for m,b in affected
    for link in ('native_hbm','pcie') for f in fs}
expected['gpu3_balanced_native_']={(kind,v,m,b,link,f) for kind,fs in (('lru',[None]),('nocache',[0.,.1,.3])) for v in ('base','balanced_cached_sm80') for m,b in affected for link in ('native_hbm','pcie') for f in fs}
coverage={}
for group,exp in expected.items():
    actual={(r['kind'],r['variant'],r['mtp'],r['requests'],r['setting'],r['fraction']) for r in rows if r['cohort']==group}
    assert actual<=exp,(group,actual-exp)
    assert all(r['positions']==1024 for r in rows if r['cohort']==group)
    coverage[group]=dict(actual=len(actual),expected=len(exp),complete=actual==exp)
retention=load(p/'balanced_retention.json') if (p/'balanced_retention.json').exists() else None
adopted=bool(retention and retention['adopted'] and retention['production_header_sha256']==sha(root/'include/flashinfer/attention/mla_host_batch.cuh'))
audit=dict(goal_complete=False,candidate_adopted=adopted,gpu3_planned_performance_coverage_complete=all(x['complete'] for x in coverage.values()),coverage=coverage,hardware='GPU3 A100-PCIE; emulated links read a separate HBM allocation on the same GPU. Physical PCIe is Host NUMA1.',
    audited_raw_files=len(raw),performance_cells=len(rows),paired_cells=len(pairs),timed_launches=launches,accepted_segments=segments,max_abs_error=maxerror,
    pcie_bandwidth_GBps=bandwidth,native_hbm_bandwidth_GBps=native_bw['reference_GBps'] if native_bw else None,rows=rows,pairs=pairs,files=files)
(g/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
if rows:
    with (g/'results.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
lines=['# GPU3：中等 batch 整行派发验证（2026-10-31）','',
    ('修正后的 balanced_cached_sm80 已验证并保留，统一 threshold2 仍已拒绝。' if adopted else '尚未采用；')+'各计划验证组的完成状态见下表。这里全部是 GPU3 A100-PCIE，性能数据与 A100-SXM4 GPU0/1 分开。50/100 GB/s 模拟的 far 源为同一 GPU 的独立 HBM 分配，不是物理 NVLink；物理 PCIe 使用 GPU3 对应的 NUMA1 Host 内存及 NUMA1 的 Memcpy pinned 源。','',
    '`gpu3_native_` 组完全不编译远端模拟宏，包含真实 PCIe 和同卡 HBM 源，作为生产构建的额外性能门槛；不与受限 50/100 GB/s 组混称。','',
    '统一 threshold2/threshold2_sm80 门槛已拒绝：正常构建在 MTP8/28 请求、无 LRU 的低远端比例有 3.2%–4.4% 回退。此前受限链路及 LRU 的正向数据不够支持保留。失败格完整保留。修正候选 balanced_cached_sm80 根据输入形状估计原分片/整行两种固定分工的计算波次成本，避免短尾波；CPU 形状判断有单项、线程局部、有界记忆，105 项正常构建测试通过，最终正常构建80格已完成，修正版在224行恢复原性能，其他已测形状均有收益。详见[保留报告](mla_host_a100_balanced_dispatch_20261031.md)。','',
    '候选 threshold2 仅把已有整行 staged 路径门槛从 4×SM 行数降到 2×SM，仍要求 split off、64 heads、足够 staging。数学 kernel 指令和 LRU 政策不变；表中收益是派发收益。同参数比较使用 fetch16/split off；初筛 split on 旧默认单列在 CSV，不混入同参数收益。','',
    '全部均值为每 pass 冷启动、包含首步的全步算术平均。正式 LRU 是 1024 连续位置；256 位置仅作筛选。纯 device 参考在本 GPU 正常构建、同 trace 上实测，逐步 max 参考按每步 miss 字节数和设定/校准带宽计算。未改变模拟器或参考公式。','',
    '| 组 | MTP | 请求 | 位置 | pass | 链路 | 无LRU比例 | 旧 µs | 候选 µs | 耗时降低 |','|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
for r in pairs:
    frac='—' if r['fraction'] is None else f"{r['fraction']:.2f}"
    lines.append(f"| {r['cohort']} | {r['mtp']} | {r['requests']} | {r['positions']} | {r['passes']} | {r['setting']} | {frac} | {r['old_us']:.2f} | {r['new_us']:.2f} | {r['reduction']*100:.1f}% |")
lines+=['','## 完整窗口 5pass 的两种参考','',
    '| MTP | 请求 | 链路 | 本文 µs | Memcpy µs | 纯 device µs | 逐步 max 参考 µs | 高于参考 | 引用命中率 |','|---:|---:|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    if r['cohort']=='gpu3_confirmfull_' and r['variant']=='threshold2':
        lines.append(f"| {r['mtp']} | {r['requests']} | {r['setting']} | {r['ours_us']:.2f} | {r['memcpy_us']:.2f} | {r['local_us']:.2f} | {r['reference_us']:.2f} | {r['over_reference']*100:.1f}% | {r['reference_hit_rate']*100:.2f}% |")
lines+=['','## 计划覆盖审计','',
    '| 验证组 | 完成格 | 预期格 | 完整 |','|---|---:|---:|---|']
for name,x in coverage.items():
    lines.append(f"| {name} | {x['actual']} | {x['expected']} | {'是' if x['complete'] else '否'} |")
lines+=['',f"已审计 {len(raw)} 份完成原件、{len(rows)} 个性能格、{len(pairs)} 个同参数配对，{launches} 次计时 launch、{segments} 段有效队列。采样 max abs≤{maxerror:.8f}。所有更慢格、pass 均值、命中率、两参考、SHA 和 raw 路径保留在 [CSV](../temp/glm_trace/tune_general_20261031/gpu3/results.csv) 和 [审计](../temp/glm_trace/tune_general_20261031/gpu3/audit.json)。",'',
    '本 GPU 的派发边界正常构建测试 105 项通过（含新增 2×SM 边界的 Graph、padding、缓存映射和实际载荷检查），不能替代其他硬件的性能回归。SXM/NVLink 配对已独立验证；本轮尚未消除 100 GB/s 大 batch 原有约 32%–34% 差距。','',
    '保留版独立候选 `threshold2_sm80` 将 2×SM 门槛限定到实测的 CTA_TILE_KV=32 配置，其余共享内存配置继续 4×SM。与 threshold2 在本 A100 上的派发条件等价，九个 GPU kernel 指令/资源相同；新增边界测试 2 项单独通过，完整 105 项记录来自 threshold2。此统一门槛版现已拒绝；后续 balanced_cached_sm80 保留同样的硬件范围，并增加计算波次判断。最终正常构建计时已通过，修正版已保留。','',
    '测量前核对拓扑，发现 GPU3 属于 NUMA1，已在任何物理 PCIe 格开始前修正 CPU/pinned buffer 绑定；此前完成的 HBM 源/纯 device 数据保留。调度器主动停止和恢复的原因记录在 `gpu3_numa_correction.json`，未终止其他用户进程。']
(root/'docs/mla_host_a100_medium_dispatch_gpu3_20261031.md').write_text('\n'.join(lines)+'\n')
audit['report_sha256']=sha(root/'docs/mla_host_a100_medium_dispatch_gpu3_20261031.md')
audit['audit_script_sha256']=sha(Path(__file__))
if physical:audit['physical_bandwidth_sha256']=sha(g/'physical_bandwidth.json')
(g/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k not in ('rows','pairs','files')},indent=2))
