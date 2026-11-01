"""Render source-streaming cache hint measurements without claiming adoption."""
import json
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
m = read(p / 'manifest.json')
a = read(p / 'performance_audit.json')
state = read(p / 'confirmation_state.json')
extension = read(p / 'extension_state.json') if (p / 'extension_state.json').exists() else {}
rejected = state['phase'].startswith('rejected')
retained = (p / 'focused_retention.json').exists() and read(p / 'focused_retention.json')['candidate_adopted']
lines = ['# A100：去重远端源读取使用L2 evict-first提示（2026-11-01）', '',
    '已按用户要求的关键场景验证保留实测源码，当前生产为48c2a724…86f73。' if retained else
    '候选因完整确认门槛失败而拒绝，生产保持CPEa436。' if rejected else
    '候选仍在跨场景验证，尚未安装；生产保持已完整验证的CPEa436。', '',
    '仅专用fetch中的去重far-source cp.async增加L2 evict_first策略，固定fraction1.0。'
    '每个源page在本launch去重后只读取一次，attention读取staging/cache，源数据与计算数据的复用方式不同。'
    '测试其硬件cache竞争假说；它只是性能提示，不能推断缓存命中或收益。'
    'attention、staging/cache写入和元数据读取策略、数学计算、精确LRU、ready/claim协议及FAR模型均保持。', '',
    '新typed派发条件与CPE相同：单108CTA/256线程、32KV双stage、dynamic147968/static2304B，'
    '原64行双fetch缓冲和f24、1152字节/unique页、原commit/wait/barrier/FAR post-read hook全部保持。'
    '原38EMU/76normal实例指令、资源、机器字和控制字精确相同；九个attention helper规范化命名后逐字相同。'
    '新REG255，EMU STACK8、正常BF16/FP16 STACK0、LOCAL0，与CPE资源级别相同。', '',
    '105+10=115项正常数值/长队列/LSE/全cache-payload/Graph测试通过；'
    '实际Graph仍单108CTA/block32x4x2/dynamic147968/active1。首匹配BF16/K97 launch的'
    'synccheck/racecheck/memcheck全零错误；memcheck关闭API报告、设备检查保留，不作穷尽证明。'
    '三种构建静态目标均有36条带desc的LDGSTS，策略描述符高字为0x12f00000、低字为0；'
    '静态指令确认提示进入二进制，不是动态访问次数、命中率或stall因果份额。', '',
    '| 硬件/cohort | 位置/pass | 链路 | CPE µs | 候选 µs | 耗时降低 | Memcpy µs | 纯device µs | 逐步max µs |',
    '|---|---:|---|---:|---:|---:|---:|---:|---:|']
for r in a['pairs']:
    if 'shapes_' in r['cohort']:
        continue
    lines.append(f"| {r['hardware']}/{r['cohort']} | {r['positions']}/{r['passes']} | {r['setting']} | "
                 f"{r['old_us']:.3f} | {r['new_us']:.3f} | {r['reduction']:+.3%} | {r['memcpy_us']:.3f} | "
                 f"{r['local_us']:.3f} | {r['reference_us']:.3f} |")
lines += ['', f"当前审计{len(a['rows'])}格/{len(a['pairs'])}配对、{a['timed_launches']}计时launch/"
          f"{a['accepted_segments']}有效段，待参考{len(a['pending_references'])}项、未完成raw{len(a['incomplete'])}项。"
          '全部正负格、逐步逐pass、实际miss/命中率、guard、源码和binary SHA保持；使用全样本算术均值，'
          '每pass冷启动，不取最好样本。两参考仍为原同硬件纯device及逐步max公式。', '',
          f"完整确认状态：`{state['phase']}`；扩展状态：`{extension.get('phase', '尚未启动')}`。", '']
if (p / 'coverage.json').exists():
    c = read(p / 'coverage.json')
    lines += ['| 回归组 | 已审计/计划格 |', '|---|---:|']
    for g in c['groups']:
        lines.append(f"| {g['hardware']}/{g['cohort']} | {g['completed_audited_cells']}/{g['expected_cells']} |")
    lines += ['']
lines += ['GPU3为A100-PCIE/NUMA1，受限far与native_hbm均为同卡独立HBM，不能称NVLink。'
          'SXM0/1若有结果来自真实NV12/NUMA0，绝对时间不混拼。完整156矩阵、SXM完整两顺序确认、正常物理链路及'
          '大范围回归尚未执行。用户18:32明确要求减少逐轮验证，已停止尚未启动任何矩阵的CPU扩展父；'
          '已有数值/代码/实际路径及完整本地关键对照用于本轮机制判断。实测原件逐字节安装，'
          '未重复生产root新编译/整套测试，后续按具体疑点补测，不声称所有形状/SXM均已验证。', '',
          f"当前实测源码SHA：`{m['candidate_sha256']}`。canonical数学34af、FAR5e409及原5304矩阵e3fd保持。"
          '整体目标仍未完成。', '',
          '[原始记录与复现入口](../temp/glm_trace/tune_stream_source_20261101/README.md)。', '']
(root / 'docs/mla_host_a100_stream_source_20261101.md').write_text('\n'.join(lines))
