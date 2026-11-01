"""Render measured evidence when available, distinguishing it from live validation."""
import json
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
m = read(p / 'manifest.json')
state = read(p / 'validation_state.json')
a = read(p / 'performance_audit.json') if (p / 'performance_audit.json').exists() else {}
confirmation = read(p / 'confirmation_state.json') if (p / 'confirmation_state.json').exists() else {}
rejected = confirmation.get('phase', '').startswith('rejected')
lines = ['# A100：按pool分配稀疏staging并融合prepare/resolve（2026-11-01）', '',
    '候选已拒绝，不扩展该机制。生产仍为已保留的CPEa436。' if rejected else
    '候选正在独立验证，尚未安装；当前生产保持已完整保留验证的CPEa436。', '',
    '每个临时batch pool的staging区间从pool_row_start×maxK开始，warp一次为全部新claim分配rank，'
    '先发布自己全部八个候选中的owner页，再等待其他owner的物理slot。prepare直接写最终负staging索引；'
    '后续只扫描pool计数，替代对全部远端reference的resolve遍历。dense fetch job通过有界二分映射到稀疏物理槽。'
    '原LRU与取数重叠、shared→staging→cache双写及原16/64行选择保持。', '',
    '只在原CPE大cached fully-staged派发上增加maxK为正且可被256整除的条件，使prepare_parts=1、'
    'prepare_segment=maxK、pool区间及16/64行readyflag均对齐。原workspace不增分配、kernel或grid barrier；'
    'host_items暂存pool计数，row_slot_base暂存job前缀，在原claim清空前复制并经原grid同步发布。', '',
    '保持原stamp精确LRU选择和本launch hit保护，但pool内miss admission顺序由row rank改为warp到达顺序，'
    '不声明独立launch的完整cache状态完全相同。inline owner等待、pool原子分配争用、新job索引和局部性变化'
    '可能抵消省掉的遍历，需要实际miss/载荷/计时验证。八个attention数学helper规范化命名后逐字相同；'
    'canonical数学头、真实字节/FAR模型、参考公式和原5304矩阵保持。', '',
    f"冻结候选SHA：`{m['candidate_sha256']}`，直接基线CPEa436。当前原验证父phase：`{state['phase']}`。", '',
    '| 证据 | 当前结果 |', '|---|---|']
for name, label in [('tests_sparse_prepare_gpu3.json', '正常105项测试'), ('stress_sparse_prepare_gpu3.json', '长队列/不等长pool等10项测试')]:
    d = read(p / name) if (p / name).exists() else {}
    lines.append(f"| {label} | {'通过' if d.get('complete') else '待完成'} |")
for suffix, count in [('', 38), ('_normal', 76)]:
    f = p / ('compiler_sparse_prepare' + suffix + '_summary.json')
    d = read(f) if f.exists() else {}
    lines.append(f"| 原{count}实例指令/资源/完整机器字与控制字 | {'全部相同' if d.get('complete') else '待完成'} |")
lines += ['', 'EMU及正常BF16/FP16资源预检均为REG255/STACK8/static2304/LOCAL0；8B栈如实保留，'
          '并非零栈。实际Graph与sanitizer结果以原件为准，资源预检不替代运行路径检查。', '']
if a:
    lines += ['| cohort/链路 | CPE µs | 候选 µs | 耗时降低 | Memcpy µs | 纯device µs | 逐步max µs |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in a['pairs']:
        lines.append(f"| {r['cohort']}/{r['setting']} | {r['old_us']:.3f} | {r['new_us']:.3f} | "
                     f"{r['reduction']:+.3%} | {r['memcpy_us']:.3f} | {r['local_us']:.3f} | {r['reference_us']:.3f} |")
    lines += ['', f"已审计{len(a['rows'])}格/{len(a['pairs'])}配对、{a['timed_launches']}计时launch/"
              f"{a['accepted_segments']}有效段。逐步逐pass、全部正负差、实际miss/命中率、guard及源码SHA保留。", '']
else:
    lines += ['尚无本候选有效性能结果；不能从省遍历、CPU结构oracle或编译资源推断收益。', '']
lines += ['CPU结构oracle72case只核验给定pool计数的区间、dense job二分映射、16/64行边界及FAR行数，'
          '不替代GPU活性、同步、LRU/数学或性能。首个未编译稿误把job行数固定64，编译前修正为原16/64选择；'
          '旧header/generator/manifest保存在precompile_review/。一次CPU观察脚本相对目录错误已经纠正，无GPU影响。', '',
          'GPU3为A100-PCIE/NUMA1，同卡far HBM不能称NVLink。完整1024位置多pass、真实NVLink/SXM、'
          '全形状及正常链路和负差审计均是保留门槛，部分结果不能替代完整验证。整体目标继续active。', '',
          '[实现、原件及状态](../temp/glm_trace/tune_sparse_prepare_20261101/README.md)。', '']
if rejected:
    lines[-2:-2] = [
        '固定短测两100GB耗时降低分别+0.0109%和−0.7015%，50GB+3.0985%；预设两100GB均须正收益的门槛未过，'
        '原验证父54327与确认父10845均terminal0，后者明确rejected_short_screen，没有启动full/native/SXM。'
        '基线mean unique约215243–215268/步，候选约207692–207714/步，实际搬运约247.96–247.99MB→239.26–239.29MB（约少3.5%）；'
        'reference命中率约76.73%→78.38%。50GB收益包含admission顺序改变后的流量影响，不能全部归因省resolve。'
        '两参考使用原同硬件计算时间及各自实际unique miss，公式/模拟器/原件均未修改。', '',
        '实际Graph为单108CTA/block32x4x2/dynamic147968/active1，新路径用K256验证；首匹配BF16/K256的'
        'synccheck/racecheck/memcheck全部零错（memcheck关闭API报告、设备检查保留），不声称穷尽全部trace。'
        '115项测试与完整代码等价并不替代完整性能回归，本候选没有生产收益。', '',
    ]
(root / 'docs/mla_host_a100_sparse_prepare_20261101.md').write_text('\n'.join(lines))
