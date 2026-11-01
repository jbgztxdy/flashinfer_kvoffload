"""Report local suffix ordering, keeping the locality proxy separate from timings."""
import json
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
m = read(p / 'manifest.json')
review = read(p / 'local_order_review.json')
a = read(p / 'performance_audit.json') if (p / 'performance_audit.json').exists() else {}
state = read(p / 'validation_state.json')
confirmation = read(p / 'confirmation_state.json')
rejected = confirmation['phase'].startswith('rejected')
lines = ['# A100：重排resident/near尾部以改善跨MTP局部性（2026-11-01）', '',
    '候选拒绝，不扩大该机制。生产保持已保留验证的CPEa436。' if rejected else
    '候选正在独立验证，尚未安装；生产保持CPEa436。', '',
    '原whole-row attention item只由一个成功claim的CTA执行。它在load_q前复用动态shared区域，'
    '按signed物理行索引bitonic排序已经在HBM/LRU的尾部，再用原attention计算。'
    'miss前缀、prepare/resolve、LRU admission/eviction、farfetch、真实payload/flags、claim及grid协议保持。'
    '同一query的KV多重集合保持，在线FP32缩放/量化P的分组随遍历次序改变，不能声称逐位相同。', '',
    '在原CPE大cached fully-staged派发上加1<maxK≤2048条件；原定义及其派发保留。'
    '保持单108CTA/256线程/32KV双stage/dynamic147968B/f24/64行双fetch缓冲，'
    '无新增workspace或grid barrier。排序最多66轮CTA比较同步，另有读写/同步，'
    '且把尾部写回global；这些成本可能超过局部性收益。', '',
    f"冻结候选SHA：`{m['candidate_sha256']}`；直接基线CPEa436。当前验证phase：`{state['phase']}`。", '',
    '真实trace观察使用CPE正常构建、GPU3同卡独立HBM、MTP8/R128/LRU4096，连续16个epoch，'
    '采样epoch0/7/15和每8个请求中的一个，共48组实际remap。'
    'output采样及所有有效cache载荷检查通过；原物理索引快照gzip及源码SHA保留。'
    f"按8query相同进度的32页窗口合并，页集合由{review['summary']['original']['window_unique_pages']}"
    f"降为{review['summary']['sorted']['window_unique_pages']}（{review['relative_window_footprint_reduction']:.2%}），"
    f"总reference {review['summary']['original']['references']}保持。"
    '这是CPU局部性代理，不能当作实际L2事务/命中率、stall、并发或kernel加速；观察本身未测性能。', '',
    '| 验证 | 当前结果 |', '|---|---|']
for name, label in [('tests_local_order_gpu3.json', '正常105项'), ('stress_local_order_gpu3.json', '长队列/重排等12项')]:
    d = read(p / name) if (p / name).exists() else {}
    lines.append(f"| {label} | {'通过' if d.get('complete') else '待完成'} |")
for suffix, n in [('', 38), ('_normal', 76)]:
    f = p / ('compiler_local_order' + suffix + '_summary.json')
    d = read(f) if f.exists() else {}
    lines.append(f"| 原{n}实例指令/资源/全部机器字和控制字 | {'全部相同' if d.get('complete') else '待完成'} |")
lines += ['',
    'EMU及正常BF16/FP16新目标均REG255/STACK8/static2304/LOCAL0，8B栈如实记录；'
    '正常CPE控制目标原为STACK0。新增K129测试对全部行解码cache/staging索引，核对页多重集合和尾部有序，'
    '检查重复页、padding、Graph、LSE、命中槽保护和全部cache载荷。另保留长K2048、奇尾、'
    '128池/不等长池、跨CTA共享/互斥页、K65/QPE stride80等覆盖。', '']
if a:
    lines += ['| cohort/链路 | CPE µs | 候选 µs | 耗时降低 | Memcpy µs | 纯device µs | 逐步max µs |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in a['pairs']:
        lines.append(f"| {r['cohort']}/{r['setting']} | {r['old_us']:.3f} | {r['new_us']:.3f} | {r['reduction']:+.3%} | "
                     f"{r['memcpy_us']:.3f} | {r['local_us']:.3f} | {r['reference_us']:.3f} |")
    lines += ['', f"当前{len(a['rows'])}格/{len(a['pairs'])}配对、{a['timed_launches']}计时launch/"
              f"{a['accepted_segments']}有效段，两参考、逐步逐pass/实际命中与unique、源码SHA及全部正负格保留。", '']
else:
    lines += ['尚无本候选完成审计的性能结果，不能从局部性代理或编译资源推断加速。', '']
lines += ['实际Graph与首匹配BF16/K97 launch的三sanitizer结果以原件为准，'
          'memcheck关闭API报告、保留设备检查；不称全部trace穷尽证明。'
          'GPU3独立far HBM不是NVLink。完整1024位置多pass、正常物理链路、实际SXM/NVLink和全形状回归'
          '仍是任何保留的前提。数学头、模拟器、两参考公式及原5304格矩阵均保持，整体目标未完成。', '',
          '[实现、原件与当前状态](../temp/glm_trace/tune_local_order_20261101/README.md)。', '']
(root / 'docs/mla_host_a100_local_order_20261101.md').write_text('\n'.join(lines))
