"""Report the measured CTA budgets and independently screened prebind candidate."""
import json
from pathlib import Path
p = Path(__file__).resolve().parent
root = p.parents[2]
a = json.loads((p / 'audit.json').read_text())
retained_path=p/'cached_ids_retention.json'
retained=json.loads(retained_path.read_text()) if retained_path.exists() else None
lead=('缓存限定共享页号版已保留，生产头SHA为`b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a`，基线为ce341a02/52919…ff2c5。保持单cooperative kernel、原数学/LRU/模拟器和原5304矩阵。整体100GB大batch目标仍未完成。'
      if retained and retained['candidate_adopted'] else '整体目标仍未完成。生产保持ce341a02/52919…ff2c5，本页候选尚未安装；原5304格矩阵保持原件。')
lines = ['# A100：CTA等待预算与取数元数据流水（2026-10-31）', '',lead, '',
    '## 新诊断：tile等待与每CTA工作量', '',
    '此前阶段计时已经定位到约0.19ms准备和100GB/s约0.41ms的fetch后attention尾部。本次增加每CTA的tile等待调用总时长、实际阻塞调用数、attention/LRU工作项数及时间，以判断剩余工作应集中在哪里；没有重复启动旧插桩。', '',
    '三格均为SXM0/1、MTP8/128请求、完整1024位置、LRU4096、staged、fetch24/split off、1次冷启动插桩pass。固定128步×128CTA容量，只用于实际108CTA的形状。每步核对所有CTA合计恰好1024个attention和128个LRU工作项，128个epoch全部保留。', '',
    '| 链路 | 插桩event µs | fetch开始 µs | 最晚fetch完成 µs | 最晚attention完成 µs | 等待调用/attention时间 | 阻塞调用比例 | attention/worker | attention/fetcher |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in a['records']:
    s, z = r['summaries'], r['phase_means']
    lines.append(f"| {r['setting']} | {r['diagnostic_event_mean_us']:.2f} | {z['fetch_start_us']:.2f} | {z['fetch_end_us']:.2f} | {z['attention_end_us']:.2f} | {s['all']['wait_fraction_of_attention']*100:.2f}% | {s['all']['blocked_call_fraction']*100:.2f}% | {s['workers']['mean_attention_items_per_CTA']:.2f} | {s['fetchers']['mean_attention_items_per_CTA']:.2f} |")
lines += ['', '等待占比是所有CTA累计等待调用时间除以累计attention工作项时间，阻塞比例是至少一次看到未就绪flag的调用数占全部等待调用数。调用时间含flag/remap读取和计时器成本；不能把它视作纯等待净开销，或将跨CTA时间相加当作launch分摊。timer和额外scratch也可能改变调度，本表只用于诊断，不能当作新优化收益。', '',
    '100GB/s及NVLink下记录的等待占比较小，而fetch CTA大部分时间无法参与attention；50GB/s的较长传输则使等待占比上升。这些观察支持优先检查取数流水和SM资源占用，而不是再重复已拒绝的warp/整行就绪等待简化。它们不证明全部差距都能由某一项解释。', '',
    '| 链路 | worker平均等待 µs | worker平均attention µs | fetcher平均attention µs | 关键CTA平均等待 µs | 关键CTA平均attention µs |',
    '|---|---:|---:|---:|---:|---:|']
for r in a['records']:
    s = r['summaries']
    lines.append(f"| {r['setting']} | {s['workers']['mean_wait_us_per_CTA']:.2f} | {s['workers']['mean_attention_us_per_CTA']:.2f} | {s['fetchers']['mean_attention_us_per_CTA']:.2f} | {s['critical_CTA']['wait_us']:.2f} | {s['critical_CTA']['attention_us']:.2f} |")
lines += ['', '关键CTA逐步按attention结束最晚者确定，再取均值；它有时是fetcher，不能直接与worker均值相减归因。全部逐步/逐CTA数据及SHA在[诊断审计](../temp/glm_trace/tune_wait_20261031/audit.json)。采样attention误差、冷首步miss、队列guard、128epoch工作项总数和源码快照均已复核。', '',
    '## 新候选prebind', '',
    '只在整行CTA_TILE_KV=32路径，把已选好的LRU victim的检查和映射绑定提前到当前cp.async远端读取的完成等待之前，让元数据操作与在途读取重叠。原模拟器等待、staging写入/发布及从同一共享缓冲执行一次cache scatter的顺序保留；早期pool未就绪的chunk仍按原机制defer。', '',
    'prepare在第二个grid barrier前已完成所有resident lookup/stamp；本步命中的槽不可被LRU选择为victim，因此提前绑定不覆盖本步缓存读取。下次launch仍等所有CTA结束后才开始。候选没有晚期HBM重读，没有迁移到attention写回，没有改数学、LRU策略、模拟器或参考。是否有性能收益要由独立正常测试和同参数对照决定。', '']
perf = p / 'performance_audit.json'
if perf.exists():
    d = json.loads(perf.read_text())
    lines += ['下表单列筛选和完整确认；回归矩阵的全部格另存于CSV。硬件、位置/pass和LRU分别标明，不能拼接绝对时间。', '',
              '| 硬件 | 候选/组 | LRU/比例 | 位置/pass | MTP/请求 | fetch CTA | 链路 | base µs | 候选 µs | 耗时降低 | Memcpy µs | 纯device µs | 逐步max µs |',
              '|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in d['pairs']:
        if '_matrix_' in r['cohort']:
            continue
        label='LRU4096' if r['kind']=='lru' else f"无/{r['fraction']:.1f}"
        lines.append(f"| {r['hardware']} | {r['variant']}/{r['cohort']} | {label} | {r['positions']}/{r['passes']} | {r['mtp']}/{r['requests']} | {r['fetch_ctas']} | {r['setting']} | {r['old_us']:.2f} | {r['new_us']:.2f} | {r['reduction']*100:.2f}% | {r['memcpy_us']:.2f} | {r['local_us']:.2f} | {r['reference_us']:.2f} |")
    lines += ['', f"性能审计目前{len(d['rows'])}格/{len(d['pairs'])}配对，正常构建测试记录{d['tests']}。每步/每pass、命中、两参考及SHA另存于[性能审计](../temp/glm_trace/tune_wait_20261031/performance_audit.json)和CSV。", '']
    lines += ['## 原共享页号版的真实回退与修正', '',
        '原fetch_ids在大batch完整5pass得到约1.4%的100GB/s和2.9%的正常NVLink收益，但GPU3正常构建、MTP8/32请求、无LRU、PCIe远端比例0.1初测慢18.5%。反向5pass复核：base830.57µs，fetch_ids923.90µs（慢11.24%），前后base为830.57/830.98µs。该回退不能用大batch正收益掩盖，原版不采用。最初0.3比例的8%负差在复核中未复现，两个记录都保留。', '',
        'cached_ids新增独立的缓存实例，只在开启LRU且走整行CTA32时共享页号；全部原generic/whole kernel使用shared_fetch_ids=false。模拟9个、正常BF16/FP16的18个原kernel指令及资源完全一致，新增1/2个缓存kernel，全部无stack/local使用。105正常测试通过。原PCIe0.1格复核为835.17µs，相对830.57µs慢0.55%；无LRU调用确实走原kernel。此为修正证据，采用仍需新的完整回归。', '',
        '已拒绝的原矩阵在发现回退后停止扩展：只暂停自己的CPU父调度器，等待当前benchmark子进程自然结束，再终止父调度器；没有给GPU benchmark或其他用户任务发信号。原计划、全部完成/失败raw及停止记录保留。正常SXM矩阵还发生一次旧自有CUDA context退出尾声导致的预检失败，GPU计时未开始。修正版使用新cohort，不覆盖这些证据。', '']
    cover=p/'coverage.json'
    if cover.exists():
        coverage=json.loads(cover.read_text())['coverage']
        lines += ['| 回归组 | 已测量格 | 已审计格 | 预期格 | 审计完整 |','|---|---:|---:|---:|---|']
        for name,c in coverage.items():
            lines.append(f"| {name} | {c.get('measured',c['actual'])} | {c['actual']} | {c['expected']} | {'是' if c['complete'] else '否'} |")
        lines += ['', '原ids_matrix组已因回退停止，预期格数保留，未伪称完成；cached_ids_matrix组才是当前候选的验证计划。所有更慢格仍在审计和CSV中。', '']
    if d.get('pending_references'):
        lines += ['完整GPU性能矩阵896格已测完，但4个SXM单请求MTP1控制格使用的旧纯device参考未记录mla.cuh的SHA。因此这4格的时间/guard/数值已验证，纯device/逐步max引用暂不采用；补测同硬件正常参考已排队，原参考不覆盖。原始时间及配对也保存在performance_audit的pending_reference_pairs和pending_reference_timings.csv，未隐藏。当前可带完整参考审计892格。初次缺字段导致的审计失败日志另存audit_missing_floor_math_sha_failure.log。', '']
        for r in d.get('pending_reference_pairs',[]):
            lines += [f"待补参考的MTP{r['mtp']}/R{r['requests']} {r['setting']}：base{r['old_us']:.3f}µs，候选{r['new_us']:.3f}µs，耗时降低{r['reduction']*100:.3f}%；此处未计算缺来源的两参考比率。", '']
else:
    lines += ['当前仍在构建/正常测试/筛选，尚无已审计的性能收益结论。', '']
controls=p/'controls_audit.json'
if controls.exists():
    c=json.loads(controls.read_text())
    lines += ['## 慢格夹测与同分配诊断', '',
        'GPU3正常120格已完成；原矩阵的负差继续保留。对超过1%的原路径负差使用完整1024位置×5pass、base→cached_ids→base夹测，结果如下。无LRU仍是原四快照。', '',
        '| 场景 | MTP/请求 | 链路 | 比例 | base前 µs | 候选 µs | base后 µs | 较两base均值降低 |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for r in c['independent_process_controls']:
        fraction='LRU4096' if r['fraction'] is None else str(r['fraction'])
        lines.append(f"| {r['kind']} | {r['mtp']}/{r['requests']} | {r['setting']} | {fraction} | {r['base_before_us']:.3f} | {r['candidate_us']:.3f} | {r['base_after_us']:.3f} | {r['reduction_vs_base_mean']*100:.3f}% |")
    z=c['shared_allocation_diagnostic']
    lines += ['', f"R32无LRU的PCIe及R28 LRU的HBM负差没有复现；单请求MTP1 PCIe仍约慢3.9%。另在同一进程载入两份独立二进制，共用同一套query/indices、Host KV、cache、workspace及输出，交替执行各5次完整1024位置冷启动。base均值{z['means_us']['base']:.6f}µs，cached_ids为{z['means_us']['cached_ids']:.6f}µs，降低{z['reduction']*100:.4f}%。10次均校验输出和最终4096行实际cache载荷。", '',
        '该诊断支持内存分配或环境因素的解释，但不能确定因果；独立进程的负差未删除，也没有据此放宽采用门槛。逐步miss因同一步stamp下的并行入槽顺序略有变化，不假称每个pass命中完全一致。所有逐步时间、miss和原件SHA保存在[controls审计](../temp/glm_trace/tune_wait_20261031/controls_audit.json)及同目录原件。', '',
        '四个SXM旧runner已exit1：完整确认和正常矩阵的预检遇到前一自有CUDA context退出尾声，模拟矩阵后段遇到外部进程进入，失败日志保留。现已在取得文件锁后等待显存/进程空闲，再打开CUDA；仅恢复缺项，计划和完成raw不覆盖。恢复器及完整确认使用新handle，见continuation_state.json。', '']
if retained and retained['candidate_adopted']:
    lines += ['## 最终保留与验证', '',
        '896格/448配对的计划已全部测量并带两参考审计。96个改变的LRU whole CTA32配对，原矩阵未加权耗时降低均值2.028%、最大11.603%；不是跨负载总加速。原矩阵最差−1.753%仍保留，其四个R128/100GB负差经完整1024位置×5pass前后base夹测后，较前后base均值最差−0.357%、最好+0.712%。R40无LRU PCIe0.3原−2.849%在夹测中+0.004%；无LRU及其他原kernel352配对的指令/资源不变，全部测量负差继续保留，不承诺每次独立分配都更快。', '',
        'SXM正常构建、MTP8/R128、完整1024位置×5次冷启动、fetch24/off：真实NVLink3349.277→3229.662µs（3.571%），PCIe9937.678→9957.666µs（慢0.201%）。完整EMU100GB0/20µs仅快0.383%/0.288%，同分配诊断也显示该大形状收益小，没有以它声称解决32%–34%历史差距。', '',
        '独立候选105项正常测试、模拟9/正常18个原kernel指令资源等价及新缓存实例零stack/local已通过。安装逐字节相同原件后，生产root/include、独立新JIT的正常BF16/FP16编译及105项测试再次通过。新单请求正常纯device参考已补齐math头SHA，原参考未改写，pending_reference为0。保留门槛、完整审计快照及全部慢格理由见[cached_ids_retention.json](../temp/glm_trace/tune_wait_20261031/cached_ids_retention.json)。', '']
lines += ['短窗口只用于筛选，正式保留依据完整窗口多pass及跨场景数据。模拟构建的真实NVLink关闭runtime hooks，不能替代正常构建。整体100GB/s大batch约32%–34%历史差距尚未解决。', '',
    '32-head CTA的独立资源方案及失败筛选见[新报告](mla_host_a100_half_heads_20261031.md)。重复QK两版本也已拒绝。缓存运行期判断及三版同分配诊断见[完整报告](mla_host_a100_cached_runtime_20261031.md)，没有稳定100GB大形状优势；所有旧负差保留，生产没有改变。', '',
    'H800五个原作业在23:13查询仍为Priority/Dependency，两个原watcher存活。未提交新任务或覆盖排队源码。', '']
(root / 'docs/mla_host_a100_wait_budget_20261031.md').write_text('\n'.join(lines))
print('diagnostic_records', len(a['records']))
