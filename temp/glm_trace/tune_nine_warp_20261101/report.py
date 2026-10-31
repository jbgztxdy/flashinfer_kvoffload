"""Report both frozen nine-warp implementations and every slower cell."""
import datetime
import hashlib
import json
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
other = p.parent / 'tune_nine_warp_static_20261101'
read = lambda f: json.loads(f.read_text())
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
audits = [(p, 'nine_warp', read(p / 'performance_audit.json')),
          (other, 'nine_warp_static', read(other / 'performance_audit.json'))]
table = ['| 版本 | 链路 | JQ µs | 九warp µs | 较JQ降低 | Memcpy µs | 纯device µs | 逐步max µs | 高于max |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for folder, variant, audit in audits:
    assert not audit['pending_references'] and not audit['incomplete']
    assert read(folder / 'validation_state.json')['complete']
    assert len(audit['rows']) == (10 if folder == p else 6)
    for r in audit['pairs']:
        assert r['reduction'] < 0
        table.append(f"| {variant} | {r['setting']} | {r['old_us']:.3f} | {r['new_us']:.3f} | {100*r['reduction']:+.3f}% | {r['memcpy_us']:.3f} | {r['local_us']:.3f} | {r['reference_us']:.3f} | {100*r['over_reference']:.2f}% |")
    decision = dict(timestamp=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        goal_status='active', candidate_adopted=False,
        decision='Rejectboth9warpversions;allcells slowerthan exactJQcontrol. Staticstage/scalarpayload didnoteliminateSTACK248. No warp/buffer/unroll/layout sweep or full/crossshape expansion.',
        tests=109, rows=len(audit['rows']), pairs=audit['pairs'],
        audit_sha256=sha(folder / 'performance_audit.json'),
        timed_launches=audit['timed_launches'], accepted_segments=audit['accepted_segments'],
        resources='REG168 STACK248 SHARED1664 LOCAL0; LDL/STL observed, notstackzero.')
    (folder / 'decisions.json').write_text(json.dumps(decision, indent=2) + '\n')
text = '''# A100：完整8个attention warp与独立服务warp（2026-11-01）

两版拒绝。首版100GB0/20µs较冻结JQ控制慢55.426%/56.289%、50GB慢9.247%；正常独立HBM慢69.533%、PCIe慢18.722%。一次固定stage/标量缓冲实现修正后，100GB仍慢54.168%/55.261%、50GB慢8.632%，且248B栈未消失。停止这条方案，不扩大full/跨形状，不继续扫描warp数、缓冲大小、unroll或stage布局。所有负差完整保留。

前期CTA预算诊断表明专用fetch占用SM；旧空间warp方案将256线程分成服务4warp和计算4warp，数学只覆盖32 heads并增加head item。本轮保持完整64 heads的原8个计算warp，另增1个服务warp：一个288线程CTA/block[32,9,1]、108个合作CTA，warp0服务，warp1–8使用虚拟y/z坐标和bar.sync1,256执行原paired-QK/joint-MDO/PV。准备与退出仍由全部288线程参与；每个CTA的计算组先完成原LRU item再算attention，服务组独立推进取数。仍为单cooperative kernel。

为容纳辅助缓冲，QPE不放shared，改按相同MMA片段布局从全局地址读取；完整PE→CKV累加次序保持。shared QPE区由8192B缩为128B对齐padding，计算存储139904B，两个8行cp.async辅助缓冲18432B，共dynamic158336B。第三个8行缓冲在源码中以每线程18个uint4声明，希望留在寄存器。实际static1664B，总shared160000B；正常Graph证实单108CTA、288线程、active1。两版实际REG168/STACK248/LOCAL0，SASS有LDL/STL，不能声称第三缓冲完全在寄存器或没有spill，也不能仅凭源码空间预算认定资源优化成功。

新方案有108个服务warp、每CTA24行在途，共2592行；旧JQ为24个fetch CTA、每CTA128行，共3072行。API输入仍f24，但生产者分工与在途预算明确变化，不称为相同取数策略。QK feature loop固定unroll1是新增288线程寄存器预算的实现选择，不是扫描得到的收益。性能结果属于整体新方案，未证明各部分的独立贡献。

服务组按8行subjob读取：前两个stage用cp.async到shared，第三个stage用普通全局读取到payload。沿用原FAR模型和真实读取之后的tile hook，每个subjob按实际有效行数×1152B登记；未改模拟器、虚拟链路时钟或参考公式。沿用原64行ready flag，完成同一chunk的全部subjob后才release。使用同一副本写staging和已选LRU victim；未就绪的chunk记录到原有界deferred列表，最后从staging补cache。一次deferred chunk可能重复填充已完成subjob，仍写相同page/victim/payload。超出staging的dedup项照常清理。canonical数学头和原5304格矩阵未改。

首版通过109测试与单kernel实查后，EMU和正常HBM均明显变慢，因此不能把回退只归于小请求的模拟器开销。随后只做一次实现控制：把payload数组变成18个命名uint4，三个stage的issue/consume/refill展开为固定表达式，去掉payload地址捕获。任务次序、三缓冲/8行subjob/64行flag、FAR hook、QPE读取、数学、288线程和unroll不变。控制版仍REG168/STACK248，未支持“只要消除数组动态索引即可移除栈”的假设；没有继续改寄存器上限或缓冲数字。栈和global QPE开销可能参与回退，但本轮没有完成因果分解。

两版各通过原105项正常BF16/FP16测试和4项长序列输出/LSE/完整cache载荷/Graph测试，共218项新增测试。长测试含K2048、128池、容量512/4096、淘汰、零miss、空/单元素/padding行及多次重放。实际正常BF16 Graph均只有一个对应新kernel，最大绝对误差0.004128；各自synccheck零错误、racecheck第一个匹配新kernel的launch零hazard/错误/warning（432行、K97），这些诊断不计trace收益或穷尽race证明。每版保留原JQ36EMU/72normal实例的完整指令/资源以及机器字/控制字身份；纯device参考也在其中。JQ控制复用原109测试/binary证据，不算本轮新测试。

全部性能点为GPU3 A100-PCIE/NUMA1、真实GLM-5.3 trace、MTP8/R128、LRU4096/请求、256位置×3冷启动pass、API f24/off，固定JQ→新版本顺序，全部步/pass算术均值。base.cuh是冻结JQ a2f37…c4b7c，**不是生产b6**。normal native_hbm是同卡独立HBM，不是真实NVLink；pcie为正常物理Host链路。首版正常对照用于区分基本原型代价与只在模拟器中出现的开销，未扩展为正常全矩阵。控制版无采用信号，未追加native/full。

{table}

两目录[首版审计](../temp/glm_trace/tune_nine_warp_20261101/performance_audit.json)与[固定stage版审计](../temp/glm_trace/tune_nine_warp_static_20261101/performance_audit.json)共16格/8配对、3072计时launch/192段有效队列；每步/每pass、命中率/unique miss、guard、源码SHA及快照完整，两参考无缺失，没有未完成raw。所有runner已terminal0，未改或安装生产代码。原JQ72075/1398独立等待最后正向三格和保留门槛；主目标仍active，当前较优JQ的大100GB完整trace仍距逐步max约25%–29%。
'''.replace('{table}', '\n'.join(table))
(root / 'docs/mla_host_a100_nine_warp_20261101.md').write_text(text)
print('COMBINED_REJECTION_REPORT_WRITTEN', sum(len(a['rows']) for _, _, a in audits))
