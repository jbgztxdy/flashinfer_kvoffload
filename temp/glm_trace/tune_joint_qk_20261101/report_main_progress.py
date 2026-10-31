"""Add a dated candidate checkpoint, preserving the original 5304-cell report and figures."""
import hashlib
import json
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
a = json.loads((p / 'performance_audit.json').read_text())
assert hashlib.sha256((root / 'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()).hexdigest() == 'b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
rows = []
for cohort, mtp, requests, setting in [
    ('joint_qk_shapes_sxm_limited_', 1, 512, 'g100_l20000'),
    ('joint_qk_shapes_sxm_limited_', 8, 64, 'g100_l20000'),
    ('compositionsxmreverse1024_f24_', 8, 128, 'g100_l0'),
    ('compositionsxmreverse1024_f24_', 8, 128, 'g100_l20000'),
    ('compositionsxmreverse1024_f24_', 8, 128, 'g50_l20000'),
    ('compositionsxmnative1024_f24_', 8, 128, 'nvlink'),
    ('compositionsxmnative1024_f24_', 8, 128, 'pcie')]:
    pair = next(r for r in a['pairs'] if r['cohort'] == cohort and r['mtp'] == mtp
                and r['requests'] == requests and r['setting'] == setting and r['variant'] == 'joint_qk')
    rows.append(pair)
table = ['| MTP/请求 | 冷pass | 链路 | b6基线 µs | JQ候选 µs | 降低 | Memcpy µs | 纯device µs | 逐步max µs | 高于max |',
         '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    table.append(f"| {r['mtp']}/{r['requests']} | {r['passes']} | {r['setting']} | {r['old_us']:.3f} | {r['new_us']:.3f} | {100*r['reduction']:+.3f}% | {r['memcpy_us']:.3f} | {r['local_us']:.3f} | {r['reference_us']:.3f} | {100*r['over_reference']:.2f}% |")
section = '''### 7.9 最新候选：联合softmax重缩放与Q片段复用

11月1日06:13检查点：**尚未安装，生产仍是§7.8的b6版本。** JQ候选让两个32KV tile共享一次在线softmax的max和d/O重缩放，并共用Q片段读取；保持完整576特征、各score的PE→CKV累加顺序、精确LRU/真实cache载荷和单cooperative kernel。在线归约分组变化，数学方程不变；canonical数学头、链路模拟器、两参考公式及原5304格矩阵均保持原件。

新增GPU3/SXM的156格矩阵已经完整审计；60个改变路径配对的未加权耗时降低均值1.602%、范围−0.188%至+5.253%。小/中batch、有/无LRU原路径控制及所有负差保留。两项超过1%的原路径测量负差经完整5pass前/候选/后夹测未复现，原样本不删除。下面均为A100-SXM4/NUMA0、真实GLM-5.3 trace完整1024位置、LRU4096/请求、fetch24/split off，与同轮b6同参数比较。3pass矩阵点和5pass确认分别标注，不能混为同一测量轮次。

{table}

对生产的改善包含联合重缩放和Q复用，不能全部归于Q复用。三方控制中，正常NVLink的JM单独联合重缩放为3136.016µs，JQ为3183.982µs，组合较JM慢1.530%；此项负差完整列在[组合实验报告](mla_host_a100_joint_qk_20261101.md)。100GB的完整反向三方确认中，JQ较生产快约4.9%–5.2%，但距逐步max仍约25%–27%，未达到10%以内的目标。

JQ已通过105＋4正常BF16/FP16与长cache载荷/Graph测试；仅清理host局部名称的保留源码又独立通过105＋4。原34EMU/68normal实例机器字含控制字完全相同，清理源全部36/72实例亦完全相同；实际仍单108CTA/256线程kernel。现有完整反向确认、正常NVLink/PCIe三方确认和正向b6与JM六格均完成，最后JQ三格因新外部进程占用GPU0/1等待，原自动保留流程继续等待门槛；通过全部确认后仍须从生产root独立新JIT编译和109测试，不能把候选短测或矩阵完成写成已安装。

独立追加的“联合发布两个P tile以减少PV barrier”候选通过109测试、机器码等价和同步/race诊断，但在100GB0/20µs较JQ慢0.182%/1.115%，已拒绝；50GB约0.05%正差也保留。[实现、失败与审计](mla_host_a100_joint_pv_20261101.md)。另一个lane私有分母候选使用相同量化P、只在item末归约，通过109项测试（含长序列LSE），但100GB0/20µs较JQ慢0.275%/0.723%、50GB慢0.039%，也已拒绝。[分母实验与审计](mla_host_a100_lane_denominator_20261101.md)。另测试完整8个计算warp＋1个服务warp的九warp方案及固定stage控制，两版共218项测试和同步/race诊断通过，但100GB慢54%–56%，均已拒绝；[架构、两参考及全部负差](mla_host_a100_nine_warp_20261101.md)。整体优化目标继续active；本节补充结果没有改写主矩阵或图1–6。

'''.replace('{table}', '\n'.join(table))
f = root / 'docs/mla_host_glm53_report.md'
s = f.read_text()
start = s.find('### 7.9 最新候选：联合softmax重缩放与Q片段复用')
end = s.index('## 8. 全参数汇总、计时口径与结论边界')
if start != -1:
    s = s[:start] + section + s[end:]
else:
    s = s[:end] + section + s[end:]
s = s.replace('若采用最新整行实现，应以§7.6的确认数据评价改进和剩余差距。',
    '整行及后续生产改进见§7.6–7.8；最新未安装候选及约25%–27%的剩余差距单列于§7.9，不能与原矩阵参数核查混用。')
f.write_text(s)
print('MAIN_REPORT_CANDIDATE_CHECKPOINT_WRITTEN')
