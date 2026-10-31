"""Render the measured small-batch histogram result without modifying raw timings."""
import json
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda name: json.loads((p / name).read_text())
m, a = read('manifest.json'), read('performance_audit.json')
decision = read('decisions.json')
lines = [
    '# 小 batch 每 warp LRU 直方图实验（2026-11-01）', '',
    decision['conclusion'], '',
    '对照为冻结 JQ `a2f37f…`，不是仍在等待原保留门槛的生产 b6。'
    '只改变原 cached/staged、rows×headtiles≤32 的 ROW_MERGE 路径；'
    'A100 上保持 108 CTA、256 线程、2 stage、32 KV、fetch16 和原小 batch planner。', '',
    '原 helper 的八个 warp 向同一 256-bin 直方图写入已经 warp 聚合的整数计数。'
    '候选为每个 warp 分配独立的 256-bin 行，warp0 将八行求和写回各 lane 原来负责的八个 bin，'
    '再执行相同 radix rank/prefix scan。阈值、thread-major 同龄淘汰顺序、stamp、payload 和 attention 数学均保持；'
    '不增加 CTA/grid barrier。额外静态共享内存 8192 B，归并增加共享读取。', '',
    '| 构建/实例 | 检查结果 |', '|---|---|',
]
for suffix in ['', '_normal']:
    proof = read('compiler_warp_hist' + suffix + '_summary.json')
    resources = sorted(set(r['resource'] for r in proof['new_kernels']))
    lines.append(f"| {'正常 BF16/FP16' if suffix else '受限链路 BF16'} | 原 JQ {proof['unchanged_original_kernels']} 实例指令和资源相同；新增实例 {' / '.join(resources)} |")
lines += [
    '', '机器字审计另外逐项比较原 36/72 实例的指令和控制字。'
    '正常构建原 105 测试及新增 4 测试通过；新增测试覆盖 BF16/FP16、capacity512/4096，'
    '相同 stamp、单字节/四字节 radix、精确淘汰槽位、受保护命中、全部缓存载荷、'
    'output/LSE、empty/single/padding 和零 miss Graph 重放。实际 Graph 和 synccheck/racecheck 证据单列保存。', '',
    '| cohort | MTP / 请求数 | 链路 | JQ µs | 候选 µs | 耗时降低 |',
    '|---|---:|---|---:|---:|---:|',
]
for r in a['pairs']:
    lines.append(f"| {r['cohort']} | {r['mtp']} / {r['requests']} | {r['setting']} | {r['old_us']:.3f} | {r['new_us']:.3f} | {r['reduction']:+.3%} |")
lines += [
    '', f"共 {len(a['rows'])} 格 / {len(a['pairs'])} 配对，"
    f"{a['timed_launches']} 计时 launch / {a['accepted_segments']} 段。"
    '使用各步所有有效 pass 的算术均值；每 pass 冷启动，warm pass 排除，保留正确 Memcpy、'
    '同硬件正常纯 device 与逐步 max(计算,unique_miss字节/带宽) 两参考、命中率及源码 SHA。', '',
    decision['next_scope'], '',
    f"候选 SHA256：`{m['candidate_sha256']}`。canonical math、模拟器和原5304格矩阵均不变。"
    '整体目标继续 active；本实验不能证明大 batch 100GB 与重叠参考的剩余差距已解决。', '',
    '[原始记录与复现脚本](../temp/glm_trace/tune_warp_histogram_20261101/README.md)。', '',
]
(root / 'docs/mla_host_a100_warp_histogram_20261101.md').write_text('\n'.join(lines))
