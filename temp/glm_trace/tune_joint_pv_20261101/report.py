"""Freeze the probability-publication experiment, including slower points and diagnostics."""
import datetime
import hashlib
import json
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
a = read(p / 'performance_audit.json')
assert len(a['rows']) == 9 and len(a['pairs']) == 6
assert not a['incomplete'] and not a['pending_references']
idx = {(r['setting'], r['variant']): r for r in a['rows']}
table = ['| 链路 | b6基线 µs | JQ控制 µs | 联合发布P µs | 较JQ降低 | 较b6降低 | Memcpy µs | 纯device µs | 逐步max µs |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
triples = []
for setting in ['g100_l0', 'g100_l20000', 'g50_l20000']:
    b, j, n = [idx[setting, v] for v in ['base', 'joint_qk', 'joint_pv']]
    reduction = 1 - n['ours_us'] / j['ours_us']
    if setting.startswith('g100'):
        assert reduction < 0
    table.append(f"| {setting} | {b['ours_us']:.3f} | {j['ours_us']:.3f} | {n['ours_us']:.3f} | {100*reduction:+.3f}% | {100*(1-n['ours_us']/b['ours_us']):+.3f}% | {n['memcpy_us']:.3f} | {n['local_us']:.3f} | {n['reference_us']:.3f} |")
    triples.append(dict(setting=setting, base_us=b['ours_us'], joint_qk_us=j['ours_us'],
        joint_pv_us=n['ours_us'], reduction_vs_joint_qk=reduction,
        reduction_vs_base=1-n['ours_us']/b['ours_us'],
        raw_paths=[r['raw'] for r in [b, j, n]]))
text = '''# A100：两个概率tile联合发布、减少PV同步（2026-11-01）

候选拒绝。相对冻结JQ控制，100GB0/20µs分别慢0.182%/1.115%，50GB20µs仅快0.048%。没有证据支持进一步扩大完整窗口/跨形状/物理链路矩阵；停止此概率联合发布机制，不扫描tile、unroll、fetch或同步间隔。相对b6基线的正收益也保留，但不能将JQ已有改进计为本候选收益。

JQ已先算完两个32KV tile的QK并共同完成MDO重缩放，随后仍两次调用原生PV，各有一次KPE→P覆写保护和一次P发布barrier。本候选把两次P量化、rowsum及共享写入提前合并，用一个CTA barrier发布两份P，再按原hi-PV→hi-refill→lo-PV→lo-refill顺序消费。每对tile的4个原生PV barrier变为1个；原补数前barrier、奇数尾tile的原生PV和全部ready等待保留。原联合MDO barrier已经覆盖两个KPE stage的QK读取完成；两份P写入不同stage，和m_wg互不重叠；hi补数不修改lo的P。浮点在线softmax方程、P量化/rowsum表达式及各输出PV的MMA次序相同，仅把第二次d更新移到第一次PV之前，d和PV输出累加独立。此依赖论证不等于性能收益。

固定原64head/256线程、32KV/2stage、dynamic147968B/static2304B、108CTA/active1、fetch64行双buffer/f24、精确LRU/真实cache载荷。canonical mla.cuh、模拟器及原5304格矩阵完全不改。新helper从canonical原生P写入/PV消费表达式逐字提取，只实例化专属JointPVTraits；原34EMU/68normal kernel指令与资源精确相同，机器字和控制字也全部相同。新REG252/STACK0/LOCAL0；JQ控制为REG251/STACK0，资源数字不用于推断因果。

105项原正常BF16/FP16测试和4项K2048长队列/容量512及4096/淘汰/零miss/Graph/完整cache payload测试均通过，共109项新测试。实际正常BF16 Graph只有一个BatchMLAJointPVKernel，108CTA、block[32,4,2]、dynamic147968、active1，最大绝对误差0.004128。冻结JQ控制复用其原109测试和binary证据，reused_joint_control.json记录原件SHA，不算新增测试。

另行Compute Sanitizer诊断：正常BF16、432行、K97的实际probe，synccheck零错误；racecheck仅检查第一个匹配BatchMLAJointPVKernel的launch，零hazard、零错误、零warning。独立dispatch记录确认名称匹配。这不是完整trace或穷尽race证明，工具时间不计性能。首个synccheck在等待正确性时触发附加超时，尚未创建GPU context，仅结束已核对的自有collector/子进程，原log及stop JSON保留；第二次在测试就绪后使用launch-timeout=0并成功。短测首次因漏拷贝plans.cpp在源码预检退出，无有效计时；补齐与JQ逐字相同的plan源码、加入CPU输入存在检查后，只重跑尚未完成的短测，日志/state追加、全部原失败保留，没有重跑通过的测试。

GPU3 A100-PCIE、NUMA1、far为独立同卡HBM模拟，不是真实NVLink。MTP8/R128、LRU4096/请求、f24/off、256位置×3冷启动pass，JQ→候选→b6固定顺序；全部步/pass算术平均，两参考按原同硬件数据与公式。

{table}

[完整审计](../temp/glm_trace/tune_joint_pv_20261101/performance_audit.json)含9格/6个对b6配对、1728计时launch/108有效segment，另列3个对JQ的三方比较；每步/每pass/命中/unique miss/guard/源码SHA/快照齐全，两参考无缺失，没有未完成raw。全部runner已terminal，候选未安装。JQ原SXM确认/保留1398在独立目录等待原门槛，不受本候选影响。主目标仍active，100GB完整trace距参考约25%–28%仍未解决。
'''.replace('{table}', '\n'.join(table))
(root / 'docs/mla_host_a100_joint_pv_20261101.md').write_text(text)
(p / 'composition_comparisons.json').write_text(json.dumps(dict(candidate_adopted=False, triples=triples), indent=2) + '\n')
now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
(p / 'decisions.json').write_text(json.dumps(dict(timestamp=now, goal_status='active',
    candidate_adopted=False, decision='Reject: both100GBpoints slower than exactJQcontrol; no parameter sweep/full/native expansion.',
    triples=triples, tests=109, timed_launches=1728, accepted_segments=108,
    audit_sha256=sha(p / 'performance_audit.json'), manifest=read(p / 'manifest.json')), indent=2) + '\n')
s = read(p / 'continuation_state.json')
s.update(recorded_at=now, active_runners=[], candidate_adopted=False,
    next_action='Stop joint probability publication mechanism. All handles terminal; do not rerun validation pipeline or short screen. Continue independently waiting JQ72075/1398 and investigate a new mechanism without repeating rejected tile/unroll/fetch/cadence/synchronization scans.',
    results='Rejected.109newtests,34/68machinewordsexact,actualonekernel,synccheck0/racecheckfirstmatchinglaunch0hazards.9cells6basepairs3threewaypoints,1728launch108segments,bothreferencescomplete.100GBvsJQ-0.182/-1.115%,50GB+0.048%.')
s['completed_handles']['59627'] = 0
(p / 'continuation_state.json').write_text(json.dumps(s, indent=2) + '\n')
(p / 'README.md').write_text('''# Joint probability publication, 2026-11-01

Rejected:100GB0/20us slower0.182/1.115%than exactJQcontrol;50GB+0.048%. No full/native/crossshape expansion or synchronization/tile/unroll/fetch/cadence scan. See [report](../../../docs/mla_host_a100_joint_pv_20261101.md), decisions.json, performance_audit.json and composition_comparisons.json.

Original2stage/32KV/108CTA/fetch/LRU unchanged; existing joint-MDO barrier protects KPE->P overwrite, publish both P with one barrier, native descending PV/refill. Four PV barriers become one, odd tile native. Canonical math untouched.109normaltests,34EMU/68normal instructions/resources/machinewords exact,REG252STACK0LOCAL0,actualone108CTA/block32x4x2kernel; separate synccheck0errors,racecheckfirstmatchinglaunch0hazards are diagnostic,notperformance/exhaustiveproof. ReusedJQ109testevidence isnotnewtests.

9cells6basepairs3threewaypoints,1728timedlaunch108segments,bothreferences/raw/source snapshots complete. First sanitizerattachtimeout and missingsourceplans.cpp failure preserved; onlyfailedshortscreen retried, passedgates not repeated. Allrunnersterminal. Originalvalidation_state.json records firstscreenfailure; successful recovery59627 and appendedscreen state preserved. Sourcegenerator/restore refuse different frozen inputs. Root untouched by this candidate; JQretention1398independent,overallgoalactive.
''')
print('REJECTION_REPORT_WRITTEN', len(a['rows']), len(triples))
