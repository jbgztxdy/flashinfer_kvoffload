"""Render current evidence, keeping planned cells distinct from completed measurements."""
import json
from pathlib import Path

p=Path(__file__).resolve().parent
root=p.parents[2]
read=lambda name:json.loads((p/name).read_text())
m=read('manifest.json');a=read('performance_audit.json');c=read('coverage.json')
jq=p.parent/'tune_joint_qk_20261101'
retention=json.loads((jq/'completion_state.json').read_text())
prior_status=('原JQ保留及生产root109项验证已完成，当前生产为7193255e…d469；'
              '本候选继续使用与该清理源全部36/72实例机器字等价的冻结a2f37f控制。'
              if retention.get('complete') else '原JQ72075/1398先完成自己的确认及生产root验证；本候选不绕过该流程安装。')
d=read('decisions.json') if (p/'decisions.json').exists() else dict(
    status='pending',conclusion='QPE片段跨item内KV tile复用候选仍在验证，尚未安装。')
lines=['# A100：在整个 attention item 内复用 QPE MMA 片段（2026-11-01）','',d['conclusion'],'',
    '对照是冻结 JQ `a2f37f…`，不是原生产 b6。候选从global按原ldmatrix.x4的MMA布局读取64个QPE特征，'
    '每lane保存16个uint32 packed word，在该item所有成对及奇尾KV tile中复用；PE→CKV MMA累加次序、'
    '联合max/d/O缩放、量化P/PV/refill和全部576特征保持。原QPE共享分配和初始copy也保留，以隔离复用机制。', '',
    '保持原256线程、64head、32KV/2stage、108CTA、fetch24专用CTA及精确launch级LRU/真实cache载荷。'
    '只在原大batch cached fully-staged JQ条件下派发新traits/params/kernel；没有tile/unroll/fetch参数扫描。'
    '这不同于N9中每tile重读global QPE的288线程服务warp方案。canonical math、模拟器、参考和原5304格矩阵不变。','',
    '| 构建 | 新资源 | 原JQ实例代码证明 |','|---|---|---|']
for suffix,label in [('', 'EMU BF16'),('_normal','正常BF16/FP16')]:
    proof=read('compiler_cached_pe'+suffix+'_summary.json')
    usage=' / '.join(sorted(set(r['resource'] for r in proof['new_kernels'])))
    lines.append(f"| {label} | {usage} | {proof['unchanged_original_kernels']}实例指令/资源精确相同 |")
lines+=['','另逐项验证36EMU/72normal实例的Ampere机器字和控制字。正常105项测试和新增6项均通过，共111项。'
    '新增覆盖BF16/FP16、K2048/63tile奇尾、128池、容量512/4096、空/单元素/padding、output/LSE、'
    '淘汰及全部cache载荷、Graph多次重放，以及大形状K65/QPE head stride80。实际Graph仍为单108CTA/'
    'block[32,4,2]/dynamic147968B/static2304B/active1；首个匹配launch的synccheck和racecheck均零错误。','',
    '静态EMU反汇编为LDSM198→190、LDG85→101、HMMA299不变、BAR58不变；总6952→6968条。'
    '固定CUDA13.1/sm80原件中，16个QPE word在成对tile循环前读入R12–R27，循环内不覆写它们，也无LDL/STL。'
    '8B stack对应的LDL/STL位于该循环范围外，因此整kernel仍有stack使用。此指令/活跃区间证明不等于动态work、'
    'stall分摊或性能收益，详见static_instruction_review.json及完整SASS。','',
    '| 硬件/cohort | 位置/pass | 链路 | JQ µs | 候选 µs | 耗时降低 | Memcpy µs | 纯device µs | 逐步max µs |',
    '|---|---:|---|---:|---:|---:|---:|---:|---:|']
for r in a['pairs']:
    if 'shapes_' in r['cohort']:continue
    lines.append(f"| {r['hardware']}/{r['cohort']} | {r['positions']}/{r['passes']} | {r['setting']} | {r['old_us']:.3f} | {r['new_us']:.3f} | {r['reduction']:+.3%} | {r['memcpy_us']:.3f} | {r['local_us']:.3f} | {r['reference_us']:.3f} |")
lines+=['',f"当前审计{len(a['rows'])}格/{len(a['pairs'])}配对，{a['timed_launches']}计时launch/{a['accepted_segments']}有效段；"
    f"待参考{len(a['pending_references'])}项、未完成raw{len(a['incomplete'])}项。每步/每pass、全部正负差、命中率、"
    'unique miss、队列guard、源码SHA与原件均保留。使用所有有效样本的算术均值，每个pass冷启动。', '',
    '| 回归计划 | 已审计 / 计划格数 | 状态 |','|---|---:|---|']
for g in c['groups']:
    lines.append(f"| {g['hardware']}/{g['cohort']} | {g['completed_audited_cells']} / {g['expected_cells']} | {'完整' if g['complete'] else '待完成'} |")
if (p/'controls_audit.json').exists():
    controls=read('controls_audit.json')
    if controls.get('complete'):
        lines+=['','两项未改路径原负差（MTP1/R1 HBM慢1.133%、R256 HBM慢1.078%）均保留；'
            '追加同参数完整1024位置×5冷pass的前/候选/后夹测如下。代码等价和未复现不证明唯一因果，也不承诺每次独立分配都更快。','',
            '| MTP/请求 | 前基线 µs | 候选 µs | 后基线 µs | 较两端均值降低 |','|---|---:|---:|---:|---:|']
        for x in controls['controls']:
            v=x['raws'];o=x['original_point']
            lines.append(f"| {o['mtp']}/{o['requests']} | {v[0]['mean_us']:.3f} | {v[1]['mean_us']:.3f} | {v[2]['mean_us']:.3f} | {x['reduction']:+.3%} |")
lines+=['','GPU3为A100-PCIE/NUMA1，受限far是同卡独立HBM，不能称为NVLink；SXM0/1数据来自实际NV12/NUMA0。'
    'MTP1/R512受限链路必须在SXM完成，不能拿GPU3代替。'+prior_status+
    '完整回归、正常物理链路、SXM完整确认和所有负差门槛均通过后，才可考虑冻结保留和root新编译/测试。', '',
    '初始generator因签名空格差异触发断言，在创建候选前失败；随后编译前检查发现新helper的const数组引用与原MMA签名不兼容，'
    '改为仅供MMA读取的非const引用。失败源/说明及未编译28d149版本均保留，未测量或混入其时间。','',
    f"当前已测候选SHA256：`{m['candidate_sha256']}`。整体目标继续active，100GB大batch剩余差距未解决。",'',
    '[原始记录、状态与复现脚本](../temp/glm_trace/tune_cached_pe_20261101/README.md)。','']
(root/'docs/mla_host_a100_cached_pe_20261101.md').write_text('\n'.join(lines))
