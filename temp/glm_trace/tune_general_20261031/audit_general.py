"""Audit isolated experiments, keeping diagnostic and shortened-trace results explicit."""
import hashlib,json,math,statistics
from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parents[2]
original=root/'temp/glm_trace/kernel_report_20261029'
load=lambda f:json.loads(f.read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
physical={r['name']:r['reference_GBps'] for r in load(root/'temp/glm_trace/briefing_report_20261030/physical_bandwidth.json')['settings']}
phases=['start','prepare_done','prepare_barrier_done','resolve_done','before_fetch_barrier','fetch_start','fetch_done','attention_done','before_merge','merge_done','last_lru_start','last_lru_done']
rows=[];diagnostics=[];files=[];launches=0;segments=0
for path in sorted((p/'raw').glob('*.json')):
    d=load(path)
    if not d.get('complete'):continue
    cfg=d['configuration'];m=cfg['mtp'];b=cfg['requests'];n=cfg['steps'];passes=cfg['passes']
    assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
    assert n>0 and passes>0 and cfg['memcpy_path_version']==2
    assert d['source_sha256']['include/flashinfer/attention/far_link_emu.cuh']=='5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
    for name,h in d['source_sha256'].items():
        snap=p/'source_snapshots'/(h+Path(name).suffix)
        assert snap.is_file() and sha(snap)==h,(path,name)
    kind=cfg['kind'];is_profile=cfg['variant'].startswith('profile')
    if kind not in ('lru','nocache'):continue
    refkind='floor' if kind=='lru' else 'snapshots'
    floor=load(original/'raw'/f'{refkind}_floor_mtp{m}_b{b}.json')
    assert d['bindings']==floor['bindings']
    for c in d['cases']:
        ticks=c['step_indices'];fraction=c['host_fraction_requested']
        assert len(ticks)==(n if kind=='lru' else len(c['far_refs']))
        fc=next(x for x in floor['cases'] if x['host_fraction_requested']==fraction)
        if kind=='lru':
            compute=[statistics.mean(fc['timings']['local']['per_step_samples_us'][i]) for i in ticks]
            assert cfg['capacity']==4096
            assert len(c['ours_unique_miss_samples'])==len(ticks)
            assert all(len(z)==passes for z in c['ours_unique_miss_samples']+c['ours_reference_miss_samples'])
            assert all(x==c['far_unique'][0] for x in c['ours_unique_miss_samples'][0])
            assert all(x==c['far_refs'][0] for x in c['ours_reference_miss_samples'][0])
            hit=1-sum(map(sum,c['ours_reference_miss_samples']))/passes/sum(c['far_refs'])
            assert abs(hit-c['ours_reference_cache_hit_rate'])<1e-12
        else:
            assert cfg['capacity']==0
            fullticks=fc['step_indices']
            compute=[statistics.mean(fc['timings']['local']['per_step_samples_us'][fullticks.index(i)]) for i in ticks]
            hit=0
        assert max(c['numerical_checks']['max_abs_error'].values())<.01
        for mode,t in c['timings'].items():
            samples=t['per_step_samples_us']
            assert len(samples)==len(ticks) and all(len(z)==passes for z in samples)
            assert all(math.isfinite(x) and x>0 for z in samples for x in z)
            assert abs(statistics.mean(x for z in samples for x in z)-t['mean_us'])<1e-7
            for k in range(passes):
                assert abs(statistics.mean(z[k] for z in samples)-t['pass_mean_us'][k])<1e-7
                end=0
                for g in (g for g in c['guards'] if g['mode']==mode and g['pass_index']==k):
                    assert g['valid'] and g['first']==end
                    if g['timing_queue']!='cuda_graph':assert g['guard_ms']>=g['enqueue_ms']+max(1,g['enqueue_ms']*.05)
                    end=g['last'];segments+=1
                assert end==len(ticks)
            launches+=len(ticks)*passes
        bw=cfg['setting'].get('bandwidth_GBps') or physical.get(cfg['setting']['name'])
        reference=statistics.mean(max(t,u*1152/(1000*bw)) for t,u in zip(compute,c['ours_unique_misses'])) if bw else None
        rows.append(dict(raw=str(path.relative_to(root)),variant=cfg['variant'],diagnostic=is_profile,full_trace=n*m==1024,
            positions=n*m,passes=passes,kind=kind,mtp=m,requests=b,setting=cfg['setting']['name'],fraction=fraction,
            fetch_ctas=cfg['fetch_ctas'],host_split=cfg['host_split'],header_sha256=d['source_sha256']['include/flashinfer/attention/mla_host_batch.cuh'],
            ours_us=c['timings']['ours']['mean_us'],memcpy_us=c['timings']['memcpy']['mean_us'],
            pass_means_us=c['timings']['ours']['pass_mean_us'],local_us=statistics.mean(compute),reference_us=reference,
            over_reference=c['timings']['ours']['mean_us']/reference-1 if reference else None,
            cache_reference_hit_rate=hit,unique_miss_MB=statistics.mean(c['ours_unique_misses'])*1152/1e6,
            max_abs_error=max(c['numerical_checks']['max_abs_error'].values())))
    if is_profile:
        flat=d['phase_timestamps_ns_flat'];assert len(flat)==128*128*16
        steps=[]
        for epochslot in range(128):
            active=[flat[(epochslot*128+cta)*16:(epochslot*128+cta+1)*16] for cta in range(128)]
            active=[z for z in active if z[0]]
            if not active:continue
            assert len({z[12] for z in active})==1
            assert all(z[12]%128==epochslot and z[13]==len(active) and z[14]==m*b and z[15]==1 for z in active)
            origin=min(z[0] for z in active)
            measures={}
            for i,name in enumerate(phases):
                vals=[(z[i]-origin)/1000 for z in active if z[i]]
                assert vals
                measures[name]=dict(min_us=min(vals),max_us=max(vals))
            steps.append(dict(epoch=active[0][12],phases=measures))
        assert len(steps)==128 and n==128 and m==8 and passes==1
        means={name:{a:statistics.mean(s['phases'][name][a] for s in steps) for a in ('min_us','max_us')} for name in phases}
        diagnostics.append(dict(raw=str(path.relative_to(root)),setting=cfg['setting']['name'],event_mean_us=d['cases'][0]['timings']['ours']['mean_us'],phase_means=means))
    files.append(dict(path=str(path.relative_to(root)),sha256=sha(path)))
retention=load(p/'balanced_retention.json') if (p/'balanced_retention.json').exists() else None
adopted=bool(retention and retention['adopted'] and retention['production_header_sha256']==sha(root/'include/flashinfer/attention/mla_host_batch.cuh'))
audit=dict(status='ongoing',goal_complete=False,production_candidate_adopted=adopted,rows=rows,diagnostics=diagnostics,
    timed_launches=launches,accepted_segments=segments,files=files,
    original_matrix_sha256=sha(original/'matrix.json'),production_header_sha256=sha(root/'include/flashinfer/attention/mla_host_batch.cuh'))
assert audit['original_matrix_sha256']=='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
(p/'audit_general.json').write_text(json.dumps(audit,indent=2)+'\n')
test_records=[]
for tp in sorted(p.glob('tests_*_gpu*.json')):
    td=load(tp)
    if not td.get('complete'):continue
    assert td['normal_build'] and td['compile_exit_code']==td['exit_code']==0
    assert td['resources_start']['foreign']==td['resources_end']['foreign']==[]
    assert td['header_sha256']==sha(p/(td['variant']+'.cuh'))
    assert td['resources_start']['gpu']==td['resources_end']['gpu']==td['physical_gpu']
    log=p/f"pytest_{td['variant']}_gpu{td['physical_gpu']}.log"
    assert td['log_sha256']==sha(log)
    count=__import__('re').search(r'(\d+) passed',log.read_text())
    assert count
    # Freeze the test source before later production test extensions.
    snap=p/'source_snapshots'/(td['test_source_sha256']+'.py')
    if not snap.exists():
        content=(root/'tests/attention/test_mla_host_batch.py').read_bytes()
        assert hashlib.sha256(content).hexdigest()==td['test_source_sha256']
        snap.write_bytes(content)
    assert sha(snap)==td['test_source_sha256']
    if td.get('extra_test_source_sha256'):
        extra=p/'test_medium_route.py'
        assert sha(extra)==td['extra_test_source_sha256']
        (p/'source_snapshots'/(sha(extra)+'.py')).write_bytes(extra.read_bytes())
    test_records.append(dict(variant=td['variant'],physical_gpu=td['physical_gpu'],device=td['resources_start']['name'],
        passed=int(count[1]),record=str(tp.relative_to(root)),record_sha256=sha(tp),log_sha256=sha(log),header_sha256=td['header_sha256']))
audit['normal_gpu_correctness']=test_records
lines=['# A100：阶段诊断与 staging 就绪协议优化（2026-10-31）','',
    '本轮进行中。基线为已验证的整行 staged 实现（SHA256 `81891229df298b90219410b5a4df1e6c20dc4cb1ee9c0233bd17ad383b29225b`）；修正后的中等 batch 派发已通过最终验证并保留，见[保留报告](mla_host_a100_balanced_dispatch_20261031.md)。大 batch 主要差距尚未改善。原 5304 格矩阵保持原件。' if adopted else '本轮进行中，候选尚未进入生产；原 5304 格矩阵保持原件。','',
    '## 已完成阶段诊断','',
    '三格均为 MTP8、128 请求、完整 1024 位置、staged、LRU4096 槽/请求、fetch24、split off、1 次冷启动 pass。插桩只用于诊断，不能用作正式新旧收益。时间相对每步最早 CTA 进入；完成值先逐步取 CTA 最大值再取算术平均。','',
    '| 链路 | event µs | fetch 开始 µs | 最晚 fetch 完成 µs | 最晚 attention 完成 µs | 最晚 LRU 项完成 µs |','|---|---:|---:|---:|---:|---:|']
for d in diagnostics:
    z=d['phase_means']
    lines.append(f"| {d['setting']} | {d['event_mean_us']:.2f} | {z['fetch_start']['max_us']:.2f} | {z['fetch_done']['max_us']:.2f} | {z['attention_done']['max_us']:.2f} | {z['last_lru_done']['max_us']:.2f} |")
lines+=['','fetch、LRU 与 attention 重叠，表中阶段不能相加作为开销分摊。LRU 时间戳是各 CTA 最后处理的一项，不是完整 LRU 独占耗时。100 GB/s 下最晚 fetch 后约 0.41 ms attention 尾部，物理 NVLink 约 0.96 ms；准备/解析约 0.19 ms。支持继续检查计算效率、fetch 占用 SM 和任务调度，不能把剩余差距全部归因 LRU。','',
    '## 新候选 warp_ready','',
    '仅整行 staged 分支改变就绪协议：8 lane 的同页装载组复用已读 page ID，由组长 acquire staging flag，syncwarp 后发出 cp.async。删除 warp0 额外读取同 tile 的 remap 与重复等待。保留原 CTA 共享内存复用 barrier、fetch 的远端模拟以及单 cooperative kernel。未改变计算、LRU 或容量。通用分支须再用反汇编与回归核对。','',
    '候选在独立 include/JIT 中；生成器、SHA 和全部原始数据位于 [实验目录](../temp/glm_trace/tune_general_20261031/)。短窗口筛选使用前 256 位置、3 次冷启动 pass；只有完整 1024 位置、多 pass 对照和跨场景回归完成后才可考虑保留。','',
    '| 实现 | MTP | 请求 | 位置 | pass | 链路 | 本文 µs | Memcpy µs | 纯 device µs | 逐步 max 参考 µs | 高于参考 | 引用命中率 |','|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    if r['diagnostic']:continue
    lines.append(f"| {r['variant']} | {r['mtp']} | {r['requests']} | {r['positions']} | {r['passes']} | {r['setting']} | {r['ours_us']:.2f} | {r['memcpy_us']:.2f} | {r['local_us']:.2f} | {r['reference_us']:.2f} | {r['over_reference']*100:.1f}% | {r['cache_reference_hit_rate']*100:.2f}% |")
lines+=['','## 正常构建正确性','',
    '以下为本轮实际执行的正常构建 GPU 测试；GPU3 A100-PCIE 的正确性结果不拼接到 GPU0/1 A100-SXM4 的性能数据。','',
    '| 候选 | 物理 GPU | 硬件 | 通过项数 |','|---|---:|---|---:|']
for t in test_records:
    lines.append(f"| {t['variant']} | {t['physical_gpu']} | {t['device']} | {t['passed']} |")
lines+=['','两端资源检查无其他进程，实际 header SHA、测试源码 SHA、原始日志 SHA 和命令在独立 `tests_*_gpu*.json` 中保存。正确性通过不证明性能或跨场景收益。','',
    '## 候选筛选结论与待验证项','',
    '`warp_ready` 的 256 位置、3pass 同参数筛选中，100 GB/s 的 0/20 µs 两格均慢约 1.8%，物理 NVLink 慢约 1.3%，50 GB/s＋20 µs 约持平。当前没有保留依据，不进入生产；原始样本及慢格已保存到 `raw/` 和 `decisions.json`。未对未采用候选扩展全场景回归，不能把采样数学检查称作完整正确性验证。','',
    '`row_ready` 独立候选在进入 attention 循环前等待该行全部 Host tile，就绪后的计算循环不再逐 tile 读取 flag。它可能损失 resident 前缀计算与搬运的重叠，须用实际对照决定。模拟/正常构建均已 CPU 编译；模拟 SM80 的 2-stage/32-KV 整行实例从 7784 条指令变为 7624 条、寄存器从 244 变为 240，无 spill。指令减少不是性能收益证据。','',
    '`row_ready` 的四格筛选也已完成：100 GB/s 0/20 µs 分别慢约 2.0%/2.6%，50 GB/s＋20 µs 慢约 2.1%，NVLink 慢约 1.1%，不采用。两项候选没有改善证据，不能宣称就绪等待已经解释或消除了主要差距。','',
    '第三项独立候选 `defer_cache` 已完成 fetch16 的同参数 SXM 筛选：四种链路慢约 29%–35%，不采用。它把缓存绑定/scatter 移到 fetch 后，从 staging 重读 HBM，虽然正常构建 103 项正确性通过，实际额外工作和尾部没有收益。全部命中率、miss 量、慢格及原始样本保留，不继续重复此机制。','',
    '修正的 balanced_cached_sm80 用原分片计划及固定计算成本比较调度波次，在 224 行保留原路径；不读取实测时间、命中率或模拟设置。最终同参数验证为 SXM 120 格受限链路＋4 格正常构建物理链路，以及 GPU3 80 格正常构建。102 个配对中改变路径的 82 个均快 1.29%–28.78%；224 行原路径的 20 个配对最大负差 0.31%。105 项正常构建测试、9 个模拟/18 个正常 kernel 指令和资源等价检查通过。已保留于生产，详细收益、慢格、来源 SHA 及剩余范围见[独立报告](mla_host_a100_balanced_dispatch_20261031.md)。' if adopted else '修正的 balanced_cached_sm80 正在验证；未采用候选不得据筛选收益宣称完成。','',
    '中等 batch 派发候选 `threshold2` 只把整行路径的门槛从 4×SM 行数降至 2×SM，仍需 staged、split off、64 heads 和足够 staging。kernel 数学及两端分工不变；单独记录派发收益，同参数对照后再判断。105 项正常构建正确性通过（含新增 2×SM 边界），九个 GPU kernel 指令与资源和 base 相同。SXM 完整窗口 5pass 及 GPU3 全范围受限链路/LRU 验证均有收益，但额外正常构建测出 224 行无 LRU/低远端比例慢 3.2%–4.4%，因此不采用统一 2×SM 门槛。详见 [GPU3 独立报告](mla_host_a100_medium_dispatch_gpu3_20261031.md)，不与 SXM 数据拼接。','',
    '上轮成果已只读复核并提交为 `5157ff83`：577 审计格、944064 次计时 launch、59760 段有效队列、103 项 pytest 记录；未改变原 5304 格矩阵或他人文件。`previous_reaudit.json` 保存复核证据。','',
    '两种参考沿用原矩阵同 trace 的纯 device 逐步数据。重叠参考是 `mean_t(max(mean_pass(C_t), mean_pass(unique_miss_t)*1152/(1000*BW)))`；物理链路使用已保存的校准带宽，不计附加延迟和共享资源争用。短窗口只与相同位置范围的新旧对照，不与旧完整窗口绝对时间比较。','',
    f'当前审计：{len(rows)} 格（含插桩/短窗口），{launches} 次计时 launch，{segments} 段计时检查；所有已完成数据的 source snapshot、冷启动 miss、采样数学误差和 CUDA event/Graph 队列检查通过。审计只证明已有数据，完整性能目标尚未完成。','',
    '模拟构建的 `compiler_summary.json` 对比已完成编译的候选各自六个通用 BF16 kernel 的指令文本与资源均相同，整行候选无 spill。正常构建 GPU 正确性以本轮实际执行表为准；全场景性能验证尚未完成。','',
    'H800 五个已有作业仍处于 Priority/Dependency，两个原 watcher 存活；本轮没有提交新集群作业、改动排队源码或生成 H800 性能结论。']
(root/'docs/mla_host_a100_readiness_20261031.md').write_text('\n'.join(lines)+'\n')
audit['artifact_sha256']={name:sha(p/name) for name in ('compiler_summary.json','decisions.json','previous_reaudit.json')}
audit['report_sha256']=sha(root/'docs/mla_host_a100_readiness_20261031.md')
audit['audit_script_sha256']=sha(Path(__file__))
(p/'audit_general.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k not in ('rows','diagnostics','files')},indent=2))
