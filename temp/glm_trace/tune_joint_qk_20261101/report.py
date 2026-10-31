"""Render three-way evidence with separate production and joint-MDO comparisons."""
import json
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text())
a=read(p/'performance_audit.json');index={(r['cohort'],r['setting'],r['variant']):r for r in a['rows']}
triples=[]
for cohort in sorted({r['cohort'] for r in a['rows']}):
 for setting in ['g100_l0','g100_l20000','g50_l20000','native_hbm','pcie']:
  if not all((cohort,setting,v) in index for v in ['base','joint_mdo_typed','joint_qk']):continue
  b,j,q=[index[cohort,setting,v] for v in ['base','joint_mdo_typed','joint_qk']]
  triples.append(dict(cohort=cohort,setting=setting,positions=q['positions'],passes=q['passes'],base_us=b['ours_us'],joint_mdo_us=j['ours_us'],joint_qk_us=q['ours_us'],over_base_reduction=1-q['ours_us']/b['ours_us'],over_joint_reduction=1-q['ours_us']/j['ours_us'],memcpy_us=q['memcpy_us'],reference_us=q['reference_us']))
table=['| cohort | 位置/pass | 链路 | 生产µs | 联合MDOµs | 组合QKµs | 较生产降低 | 较联合MDO降低 | 候选Memcpyµs | 候选逐步maxµs |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in triples:
 table.append(f"| {r['cohort']} | {r['positions']}/{r['passes']} | {r['setting']} | {r['base_us']:.3f} | {r['joint_mdo_us']:.3f} | {r['joint_qk_us']:.3f} | {r['over_base_reduction']*100:+.3f}% | {r['over_joint_reduction']*100:+.3f}% | {r['memcpy_us']:.3f} | {r['reference_us']:.3f} |")
text='''# A100：联合MDO与Q fragment复用的组合（2026-11-01）

候选尚未采用，生产保持0f82e9de / b6eea095…e72a。短窗口组合版较生产快2.868%–3.376%，较同轮联合MDO控制再快0.398%–0.916%；需要完整窗口反向确认，不能把较生产的全部收益归于Q复用。

以已独立通过本机98格/49对的joint_mdo_typed（1190…aa59）为算法控制。两个32KV score tile已经同时保留在寄存器中，组合版复用旧P冻结的paired-QK helper，让每个Q fragment加载一次供两个tile使用，随后执行相同joint max/d/O重缩放和逐tile native PV/refill。每score的PE→CKV MMA累加次序保持，与JM相同的在线归约分组；canonical math头34af、原fetch64行双buffer/f24、64head/256thread/32KV/2stage/shared147968B、精确跨步LRU/实际cache载荷、单cooperative kernel均保持。它要求两个stage先ready，失去JM第一QK与第二tile加载的重叠，收益必须实测。复用旧helper的unroll4，不扫unroll/配对长度/cadence/fetch数值。

旧paired-QK-only使用两次独立MDO、无稳定收益；本组合只测试在已经受益的jointMDO中追加Q复用，未修改正在排队的JM源码/计划，不覆盖旧失败证据。派发仍仅SM80原cached whole-row、至少4×SM行、完整staging，其他kernel原34EMU/68normal实例逐一指令/资源相同。

本版新增105项正常BF16/FP16测试和4项长cache-payload/Graph测试通过，共109项。JM控制复用原独立binary及原109测试证据，reused_joint_control.json列精确原件SHA与同字节host/math，不冒充本轮新增测试。实际正常BF16 Graph只有一个新BatchMLAJointQKKernel，108CTA、block[32,4,2]、dynamic147968B/static2304B、active1，probe最大绝对误差0.004128。新EMU/normal均REG251/STACK0/LOCAL0，JM控制REG255/STACK16。新SASS6952条/HMMA299/LDSM198是unroll4静态计数，不能当作动态work、带宽或性能提升的证明。

GPU3为A100-PCIE/NUMA1，far模拟源是同卡独立HBM，不是真实NVLink。MTP8/R128、LRU4096/请求、f24/off。短256位置×3冷pass顺序base→JM→JQ；confirmation_plan.json规定完整1024位置×5pass反向JQ→JM→base。所有步/pass算术均值，冷启动不剔除；纯device同硬件参考、逐步max和完整Memcpy均按原口径，两参考与所有正负差保留。

{table}

完整审计当前{rows}格/{pairs}个对生产配对、{launches}计时launch/{segments}segment，{incomplete}份未完成raw；表只列已完成三方配对。三方结果另见composition_comparisons.json，不能把其中的对JM比较混入对生产配对数量。short_performance_audit.json冻结首次9格/6对、1728launch/108segment；后续追加不覆盖原件。当前活动runner以continuation_state.json和实际handle为准。完整/正常链路/跨形状/真实SXM门槛未完成前不安装，也不声称主目标完成。

JM仍有独立SXM验证器等待GPU0/1空闲；本实验没有远端Slurm提交、没有覆盖排队源或终止其他用户进程。原5304矩阵、生产头、root math和模拟器均未改。
'''.replace('{table}','\n'.join(table))
for k,v in [('rows',len(a['rows'])),('pairs',len(a['pairs'])),('launches',a['timed_launches']),('segments',a['accepted_segments']),('incomplete',len(a['incomplete']))]:text=text.replace('{'+k+'}',str(v))
(root/'docs/mla_host_a100_joint_qk_20261101.md').write_text(text)
(p/'composition_comparisons.json').write_text(json.dumps(dict(candidate_adopted=False,goal_status='active',triples=triples),indent=2)+'\n')
(p/'README.md').write_text('''# JointMDO plus paired-QK fragment reuse, 2026-11-01

Not adopted; productionb6/0f82e9de, goal active. See [report](../../../docs/mla_host_a100_joint_qk_20261101.md), confirmation_plan.json and composition_comparisons.json. Short3way signals need fullreverse/native/crossshape confirmation.

SameJMjointmax/rescale/PV/refill, add frozenPpairedQKhelper/unroll4. Same per-scorePE->CKV order, sameJMFP32online grouping, canonicalmath unchanged. Pairrequiresbothstagesready, trading firstQK/load overlap for Qreuse. Originalfetch/LRU/cachepayload/singlecooperative contract preserved. NewREG251STACK0 versusJM255STACK16,actualone108CTA/block32x4x2kernel,active1,dynamic147968/static2304. Original34EMU/68normal exact.109newtests pass;JMcontrol reuses frozen binary/gates with manifest SHA,zero newcontroltests claimed.

Short9cells/6productionpairs,1728launches/108segments and bothreferences complete;newvsJM100GBfast0.398/0.916%. Fullreverse3way1024x5preplanned. performance_audit.json pairs all againstproduction; composition_comparisons.json separately computesnewvsJMcontrol. Staticunroll4HMMA/LDSM counts do not represent dynamicwork or causality. Allpositives/negatives/source/raw preserved. BaselineW andcontrolJM libraries reused,sourcebytes verified. restore_includes.py refuses overwrites. Original5304matrix untouched; fixedJM/SXMsource/plan unchanged. Read continuation_state.json and originalhandles before any restart.
''')
print('REPORT_WRITTEN',len(a['rows']),len(triples))
