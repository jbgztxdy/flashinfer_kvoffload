"""Write the retained dispatch report from the fixed paired-cell retention record."""
import json
from pathlib import Path
p = Path(__file__).resolve().parent
root = p.parents[2]
r = json.loads((p / 'balanced_retention.json').read_text())
assert r['adopted']
pairs = r['pairs']
changed = [x for x in pairs if (x['mtp'], x['requests']) != (8, 28)]
lines = ['# A100：中等 batch 派发修正（2026-10-31）', '',
    '修正后的 `balanced_cached_sm80` 已保留到生产头文件。它根据形状比较整行和原分片的计算波次，保留较合适的已有路径；GPU attention、单 cooperative kernel、跨步 LRU 和远端模拟均未改变。', '',
    f"最终 102 个同参数配对中，改变路径的 82 个耗时降低 {min(x['reduction'] for x in changed)*100:.2f}%–{max(x['reduction'] for x in changed)*100:.2f}%。224 行保留原分片的 20 个配对最大负差 {r['maximum_unchanged_slowdown']*100:.2f}%，全部原始负差保留。验收允许原路径 1% 的测量波动；这不是对零开销的证明。", '',
    '**大 batch 的 100 GB/s 约 32%–34% 差距没有改善，整体目标仍进行中。** 本项只改善中等 batch 的派发，不能替代大 batch 主目标，也没有新的 H800 性能结论。', '',
    '## 派发规则', '',
    '仍要求 staged、split off、64 heads、全量 staging。行数≥4×SM 沿用已有整行规则；在 2×SM 到 4×SM 之间且 CTA_TILE_KV=32 时，用原 tail planner、固定 tile 读写成本以及既有前后两端循环分工估计计算完成波次，整行成本不高才选整行。其他配置保持原门槛。', '',
    '模型只读取 rows、max_kv_len、SM/grid、fetch_ctas、partial_capacity 五个标量，不读取实测时间、链路、命中率或 KV 内容。108SM、fetch16、K2048 下，216 行选整行，224/232/240/248 行保留分片，256/320/384 行选整行。单项 thread-local 记忆覆盖五个键，360 次键变化和等价检查通过；独立 C++ 探针 cold≈47.7µs、warm≈0.002µs，不是端到端 Python CPU 时延。', '',
    '统一 2×SM 门槛曾在 224 行、无 LRU、低远端比例的正常构建中慢 3.2%–4.4%，已拒绝。修正规则根据计算波次恢复该形状的原路径，没有按链路或逐格测试结果选择实现。旧失败记录在 [GPU3 报告](mla_host_a100_medium_dispatch_gpu3_20261031.md)。', '',
    '## 最终性能验证', '',
    '全部 fetch16/split off，同参数 base 对照。LRU4096 每 pass 冷启动，包含首步，完整 1024 连续位置；无 LRU 回放原 trace 的 0/256/512/768 四快照和 0/.1/.3 三种远端比例。', '',
    '| 硬件与构建 | 覆盖 | 性能格 | 配对 |',
    '|---|---|---:|---:|',
    '| SXM0/1，模拟构建 | 5 形状，LRU/无 LRU，100GB/s＋0/20µs、50GB/s＋20µs，3pass | 120 | 60 |',
    '| SXM0/1，正常构建 | MTP8/32 请求，LRU，真实 NVLink/PCIe，5pass | 4 | 2 |',
    '| GPU3，正常构建 | 5 形状，LRU/无 LRU，同卡独立 HBM 源/真实 PCIe，3pass | 80 | 40 |', '',
    '5 形状为 MTP8/28、32、40、48 请求和 MTP1/256 请求。GPU3 是 A100-PCIE/NUMA1，SXM0/1 是 A100-SXM4/NUMA0，绝对时间不拼接；同卡独立 HBM 源不是 NVLink。', '',
    'SXM 的 LRU 完整窗口结果：', '',
    '| MTP | 请求 | 链路 | base µs | 新 µs | 耗时降低 | Memcpy µs | 纯 device µs | 逐步 max µs | 高于 max |',
    '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for x in sorted((x for x in pairs if x['hardware'] == 'SXM0/1' and x['kind'] == 'lru'),
                key=lambda x: (x['mtp'], x['requests'], x['setting'])):
    lines.append(f"| {x['mtp']} | {x['requests']} | {x['setting']} | {x['old_us']:.2f} | {x['new_us']:.2f} | {x['reduction']*100:.2f}% | {x['memcpy_us']:.2f} | {x['local_us']:.2f} | {x['reference_us']:.2f} | {x['over_reference']*100:.1f}% |")
lines += ['', '两种参考均使用同硬件、同 trace 的纯 device 逐步计算时间；重叠参考为 `mean_t(max(C_t, unique_miss_t*1152/(1000*BW)))`，物理链路用保存的校准带宽，不含附加延迟和共享资源争用。没有用 max of means，也没有修改原 5304 格矩阵。', '',
    'CUDA event/Graph、队列 guard、首步冷 miss、跨 pass 均值、采样数学误差、源码快照均由审计重新核对。GPU3 最后一项曾因其他用户进程进入而末检失败，该次未完成原件保存在 `gpu3/rejected/`，仅补跑缺失项；已完成格没有覆盖。', '',
    '## 正确性与保留证据', '',
    '正常构建 105 项 GPU 测试通过，包含 2/4×SM 边界、Graph replay、padding、映射及实际 cache 载荷。模拟构建 9 个、正常 BF16/FP16 构建 18 个 GPU kernel 的指令文本和资源与 base 相同。生产文件与独立测试的完整候选逐字节相同。', '',
    f"- 原头 SHA256：`{r['base_header_sha256']}`。",
    f"- 保留头 SHA256：`{r['retained_header_sha256']}`。",
    '- [保留记录与全部102配对](../temp/glm_trace/tune_general_20261031/balanced_retention.json)保存负差、原件路径和冻结审计 SHA；[保留脚本](../temp/glm_trace/tune_general_20261031/retain_balanced.py)核对完整源码、覆盖、测试、代码生成和原矩阵。',
    '- [SXM审计](../temp/glm_trace/tune_general_20261031/audit_general.json)、[GPU3审计](../temp/glm_trace/tune_general_20261031/gpu3/audit.json)及[GPU3 CSV](../temp/glm_trace/tune_general_20261031/gpu3/results.csv)保存每步/每pass、命中率、两参考、raw及SHA。', '',
    '原 50/100GB/s、0/2/5/10/20µs 的 440 格验证和大小 batch 控制保留在 GPU3 审计中，来自已拒绝的统一门槛版本；它们证明已有路径的行为，不能冒称修正版逐格重跑。最终修正版在三个代表性受限链路与正常物理链路完成上述独立覆盖。大/小形状派发及 GPU 指令均未改变。', '']
(root / 'docs/mla_host_a100_balanced_dispatch_20261031.md').write_text('\n'.join(lines))
print({'paired_cells': len(pairs), 'changed_min': min(x['reduction'] for x in changed),
       'changed_max': max(x['reduction'] for x in changed)})
