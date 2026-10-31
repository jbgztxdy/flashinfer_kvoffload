"""Generate the ongoing fetch-resource report, with failed screens kept visible."""
import json
from pathlib import Path
p = Path(__file__).resolve().parent
root = p.parents[2]
a = json.loads((p / 'audit.json').read_text())
lines = ['# A100：缓存写回与 fetch 资源实验（2026-10-31）', '',
    '本轮仍在验证，下面的候选均未进入生产。生产是 `ce341a02` 保留的中等 batch 派发版，完整头 SHA 为 `52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5`。大 batch 100GB/s 约32%–34%差距没有被这些筛选消除。', '',
    '## 新机制及当前边界', '',
    '- `attention_cache`：fetch 只写 staging/publish，attention 从已经载入的共享 KV tile 填充 LRU。每个重复引用用原子认领唯一 victim。105 项正常构建测试通过，但 GPU3 同参数短窗口100GB/s两格慢约12%、50GB/s＋20µs慢约1.2%，无采用依据。',
    '- `owner_cache`：prepare 的成功 CAS 所有者经 remap 标记传递到 attention，避免每个重复引用再做原子认领。105 项正常构建测试通过，但 CPU 审查发现临时 bit30 与合法的大 Host 页号可能冲突，已被后续编码修正取代。它的两项完成筛选原件保留，第三项因外部进程进入而末检失败；没有完整配对收益结论。',
    '- `owner_complement`：临时 Host 页号用按位取反标记所有者，resolve 恢复完整页号，再给受容量边界保护的负 staging 索引附加标记。唯一所有者从共享 tile 写回，没有额外 HBM 重读。105 项正常构建测试通过；默认 fetch24 未见收益，另测 fetch16/8 评估减少保留 fetch SM 是否足以偿还 attention 的元数据负担。',
    '- `whole_pipe4`：保留 fetch 内即时 cache fill，仅整行 CTA_TILE_KV=32 改为32行/四缓冲取数，其他 GPU 路径和数学不变。历史 fastpipe4 源码没有 WholeRowHostBatchParams，所以旧记录没有测试这一组合；本候选与旧实验独立。状态以测试、编译和调度原件为准。', '',
    '保持单 cooperative kernel、原 attention 数学、跨步 LRU 容量和策略；远端模拟文件及两种参考未改。所有派发候选基于冻结生产头，独立 include/JIT，不覆盖前轮源或集群排队源码。', '',
    '## 已完成同参数对照', '',
    '当前均为 GPU3 A100-PCIE/NUMA1。模拟 far 源是同卡独立 HBM 分配，不能当作真实 NVLink。MTP8、128请求、LRU4096、staged/split off、前256位置、3pass冷启动全部步骤平均；这只是筛选，不是完整1024位置的收益确认。', '',
    '| 候选 | fetch CTA | 链路 | base µs | 新 µs | 耗时降低 | Memcpy µs | 纯 device µs | 逐步 max µs | 高于 max |',
    '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for x in a['pairs']:
    lines.append(f"| {x['variant']} | {x['fetch_ctas']} | {x['setting']} | {x['old_us']:.2f} | {x['new_us']:.2f} | {x['reduction']*100:.2f}% | {x['memcpy_us']:.2f} | {x['local_us']:.2f} | {x['reference_us']:.2f} | {x['over_reference']*100:.1f}% |")
lines += ['', '同参数 base 对照区分代码收益；fetch CTA 变化另作为参数收益比较，不能用新参数与旧默认混称代码收益。纯 device 参考在同一 GPU 正常构建、相同完整 trace 上独立实测，筛选使用其相同前缀；重叠参考逐步取 `max(C_t, unique_miss_t*1152/(1000*BW))` 后求均值。', '',
    '## 审计与失败记录', '',
    f"已审计 {len(a['rows'])} 性能格、{len(a['pairs'])} 配对、{a['timed_launches']} 次计时 launch、{a['accepted_segments']} 段有效队列；{len(a['incomplete'])} 个未完成原件排除于有效计时。每步/每pass、hit/miss、两参考、guard和执行源码SHA保存在 [CSV](../temp/glm_trace/tune_scatter_20261031/results.csv) 与[审计](../temp/glm_trace/tune_scatter_20261031/audit.json)。", '',
    '正常构建已完成的测试记录：', '',
    '| 候选 | 通过项数 |', '|---|---:|']
for t in a['tests']:
    lines.append(f"| {t['variant']} | {t['tests']} |")
lines += ['', '全部六个原通用 BF16 kernel 的指令文本与资源逐项对比；整行实例单列，未满足零 stack 的实例也原样保留。`attention_cache`/owner 系列 CTA64 有 STACK:8、LOCAL:0，不能把它写成全配置零 stack。A100 CTA32 实例为248寄存器、STACK:0、LOCAL:0。完整结果在 `compiler_summary.json`、压缩SASS和资源文本中。', '',
    '源生成器的重复锚点断言、随后缺头文件编译、单项 tuple 审计错误、保守 stack 门槛失败，以及源页号标记的静态问题，均保存在 `development_failures.json` 和原始失败日志；未将失败计时算作收益。自己的取消前提检查发现测试已启动，因此没有向任何进程发信号。其他用户任务只触发等待或末检排除。', '',
    'H800五个原作业在20:18查询仍为Priority/Dependency，两个原watcher存活；本轮无新提交、无排队源覆盖、无新H800性能结论。原5304格矩阵SHA保持不变。', '']
(root / 'docs/mla_host_a100_fetch_resources_20261031.md').write_text('\n'.join(lines))
print({'rows': len(a['rows']), 'pairs': len(a['pairs']), 'tests': a['tests']})
