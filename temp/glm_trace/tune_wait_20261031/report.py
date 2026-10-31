"""Report the measured CTA budgets and independently screened prebind candidate."""
import json
from pathlib import Path
p = Path(__file__).resolve().parent
root = p.parents[2]
a = json.loads((p / 'audit.json').read_text())
lines = ['# A100：CTA等待预算与取数元数据流水（2026-10-31）', '',
    '整体目标仍未完成。生产保持 `ce341a02` 的中等 batch 派发版，头 SHA 为 `52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5`。本页诊断和候选均没有替换生产或改写原5304格矩阵。', '',
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
        lines += ['| 回归组 | 已审计格 | 预期格 | 完整 |','|---|---:|---:|---|']
        for name,c in coverage.items():
            lines.append(f"| {name} | {c['actual']} | {c['expected']} | {'是' if c['complete'] else '否'} |")
        lines += ['', '原ids_matrix组已因回退停止，预期格数保留，未伪称完成；cached_ids_matrix组才是当前候选的验证计划。所有更慢格仍在审计和CSV中。', '']
else:
    lines += ['当前仍在构建/正常测试/筛选，尚无已审计的性能收益结论。', '']
lines += ['短窗口只用于筛选；采用仍需完整1024位置、正反向多pass和全场景回归。模拟构建的真实NVLink关闭runtime hooks，不能替代最终正常构建。整体100GB/s大batch约32%–34%差距仍未改善。', '',
    'H800五个原作业在20:42查询仍为Priority/Dependency，两个原watcher存活。未提交新任务或覆盖排队源码。', '']
(root / 'docs/mla_host_a100_wait_budget_20261031.md').write_text('\n'.join(lines))
print('diagnostic_records', len(a['records']))
