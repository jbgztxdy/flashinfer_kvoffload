"""Render three-way evidence with separate production and joint-MDO comparisons."""
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text())
a=read(p/'performance_audit.json');index={(r['cohort'],r['setting'],r['variant']):r for r in a['rows']}
coverage=read(p/'coverage.json')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
retention=read(p/'joint_qk_retention.json') if (p/'joint_qk_retention.json').exists() else {}
completion=read(p/'completion_state.json') if (p/'completion_state.json').exists() else {}
adopted=retention.get('candidate_adopted',False) and completion.get('complete',False)
if adopted:
 validation=read(p/'tests_production_gpu3.json');proof=read(p/'production_codegen_normal_summary.json');probe=read(p/'dispatch_probe_production_joint_qk.json')
 assert validation['complete'] and validation['exit_code']==0 and validation['expected_tests']==109
 assert validation['production_header_sha256']==retention['candidate_header_sha256']
 assert proof['complete'] and proof['unchanged_kernel_instances']==72 and proof['all_original_and_new_targets_identical']
 assert probe['kernel_nodes']==1
 for key,name in [('sha256','tests_production_gpu3.json'),('codegen_sha256','production_codegen_normal_summary.json'),('dispatch_sha256','dispatch_probe_production_joint_qk.json')]:assert retention['production_validation'][key]==sha(p/name)
 production_status='JQ已通过全部门槛并保留，等价清理源SHA为7193255e…d469。生产root独立新JIT的正常BF16/FP16编译、109项测试、72实例指令/资源和实际单kernel Graph验证均已完成；原72075/1398已terminal0。'
 retention_status='retention脚本已实际安装等价清理源，并完成root独立新JIT的109项测试、72实例代码核验与实际单kernel Graph。全部156矩阵及两顺序完整确认均齐，保留记录joint_qk_retention.json绑定生产验证SHA；这不等于主目标完成。'
else:
 production_status='候选尚未完成生产保留，实际状态以completion_state.json和joint_qk_retention.json为准。'
 retention_status='保留门槛及生产root验证以completion_state.json和joint_qk_retention.json为准，尚不能声称主目标完成。'
forward_missing=[(setting,v) for setting in ['g100_l0','g100_l20000','g50_l20000'] for v in ['base','joint_mdo_typed','joint_qk'] if ('compositionsxmforward1024_f24_',setting,v) not in index]
forward_status=('完整SXM正向确认已齐，后续保留与生产root验证以实际completion_state.json为准。' if not forward_missing else f'完整SXM正向仍缺{len(forward_missing)}格，原72075/1398等待既有进程，不重复启动。')
triples=[]
for cohort in sorted({r['cohort'] for r in a['rows']}):
 for setting in ['g100_l0','g100_l20000','g50_l20000','native_hbm','nvlink','pcie']:
  if not all((cohort,setting,v) in index for v in ['base','joint_mdo_typed','joint_qk']):continue
  b,j,q=[index[cohort,setting,v] for v in ['base','joint_mdo_typed','joint_qk']]
  triples.append(dict(cohort=cohort,setting=setting,positions=q['positions'],passes=q['passes'],hardware=q['hardware'],local_us=q['local_us'],over_reference=q['over_reference'],base_us=b['ours_us'],joint_mdo_us=j['ours_us'],joint_qk_us=q['ours_us'],over_base_reduction=1-q['ours_us']/b['ours_us'],over_joint_reduction=1-q['ours_us']/j['ours_us'],memcpy_us=q['memcpy_us'],reference_us=q['reference_us']))
table=['| 硬件/cohort | 位置/pass | 链路 | 原b6µs | 联合MDOµs | 组合QKµs | 较原b6降低 | 较联合MDO降低 | Memcpyµs | 纯deviceµs | 逐步maxµs | 高于max |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in triples:
 table.append(f"| {r['hardware']}/{r['cohort']} | {r['positions']}/{r['passes']} | {r['setting']} | {r['base_us']:.3f} | {r['joint_mdo_us']:.3f} | {r['joint_qk_us']:.3f} | {r['over_base_reduction']*100:+.3f}% | {r['over_joint_reduction']*100:+.3f}% | {r['memcpy_us']:.3f} | {r['local_us']:.3f} | {r['reference_us']:.3f} | {100*r['over_reference']:.2f}% |")
text='''# A100：联合MDO与Q fragment复用的组合（2026-11-01）

{production_status}GPU3完整1024×5pass反向三方确认，组合版100GB0/20µs较原b6快3.766%/3.856%，较同轮联合MDO控制再快1.026%/0.837%。正常独立HBM较原b6快3.311%、较JM快0.142%；物理PCIe较原b6慢0.031%、较JM慢0.043%，均保留。GPU3/SXM共156矩阵格已完整，60个改变路径配对降低范围−0.188%至+5.253%，均值1.602%，所有负差保留。SXM完整反向100GB0/20µs较原b6快5.228%/4.886%；正常NVLink base/JM/JQ为3305.725/3136.016/3183.982µs，JQ较原b6快3.683%但较JM慢1.530%，PCIe较原b6快0.051%。不能把较原b6的全部收益归于Q复用，也不能声称组合总比JM快。{forward_status}

以已独立通过GPU3/SXM共178格/89对的joint_mdo_typed（1190…aa59）为算法控制。两个32KV score tile已经同时保留在寄存器中，组合版复用旧P冻结的paired-QK helper，让每个Q fragment加载一次供两个tile使用，随后执行相同joint max/d/O重缩放和逐tile native PV/refill。每score的PE→CKV MMA累加次序保持，与JM相同的在线归约分组；canonical math头34af、原fetch64行双buffer/f24、64head/256thread/32KV/2stage/shared147968B、精确跨步LRU/实际cache载荷、单cooperative kernel均保持。它要求两个stage先ready，失去JM第一QK与第二tile加载的重叠，收益必须实测。复用旧helper的unroll4，不扫unroll/配对长度/cadence/fetch数值。

旧paired-QK-only使用两次独立MDO、无稳定收益；本组合只测试在已经受益的jointMDO中追加Q复用，未修改正在排队的JM源码/计划，不覆盖旧失败证据。派发仍仅SM80原cached whole-row、至少4×SM行、完整staging，其他kernel原34EMU/68normal实例逐一指令/资源相同。

本版新增105项正常BF16/FP16测试和4项长cache-payload/Graph测试通过，共109项。JM控制复用原独立binary及原109测试证据，reused_joint_control.json列精确原件SHA与同字节host/math，不冒充本轮新增测试。实际正常BF16 Graph只有一个新BatchMLAJointQKKernel，108CTA、block[32,4,2]、dynamic147968B/static2304B、active1，probe最大绝对误差0.004128。新EMU/normal均REG251/STACK0/LOCAL0，JM控制REG255/STACK16。新SASS6952条/HMMA299/LDSM198是unroll4静态计数，不能当作动态work、带宽或性能提升的证明。

GPU3为A100-PCIE/NUMA1，far模拟源是同卡独立HBM，不是真实NVLink。MTP8/R128、LRU4096/请求、f24/off。短256位置×3冷pass顺序base→JM→JQ；confirmation_plan.json规定完整1024位置×5pass反向JQ→JM→base。所有步/pass算术均值，冷启动不剔除；纯device同硬件参考、逐步max和完整Memcpy均按原口径，两参考与所有正负差保留。

{table}

完整审计当前{rows}格/{pairs}个对原b6配对、{launches}计时launch/{segments}segment，{incomplete}份未完成raw；表只列已完成三方配对。三方结果另见composition_comparisons.json，不能把其中的对JM比较混入对原b6配对数量。short_performance_audit.json冻结首次9格/6对、1728launch/108segment；后续追加不覆盖原件。当前活动runner以continuation_state.json和实际handle为准。GPU3两矩阵76格及SXM两矩阵80格均完成，原66919/18187已terminal0；JM原30328/98811也terminal0。两个未改路径原负差1.113%/1.223%在5pass前后夹测中未复现：MTP1/R256 HBM约慢0.110%，无LRU零远端HBM约快0.064%，controls_audit.json保留全部前/候选/后样本与原负差。原34/68实例机器字含控制字也完全相同，不以此掩盖原独立分配波动。保留源码只改host局部变量名称，36/72实例机器字全相同，并额外通过105+4测试；{retention_status}当前已测SXM100GB仍高于逐步max约25%–27%，GPU3约27%–28%，主目标未解决。

独立硬件计数器诊断使用GPU3固定1024×2048、MTP全集共享、25%新远端行的合成输入，不是GLM trace性能点。Nsight Compute采用application replay，每次重新构建LRU/epoch；两版输出和全部327680有效cache载荷通过。实际Tensor指令数均71827456；short-scoreboard等待26.98%→19.31%，long-scoreboard5.39%→11.13%，barrier8.48%→9.05%。这些warp stall比例不能相加当时间分解；它与shared读取减少/等待两个stage的取舍一致，不能单因解释trace差距。诊断时间、默认cache控制、原始CSV/日志/配置/源码SHA和限制见counters_audit.json，不纳入性能审计或参考。初次诊断导入pytest缺失，在GPU工作前退出；原失败源/日志保存，改用原make_wrapper后成功，没有安装依赖或改测试环境。

JM和JQ的独立SXM矩阵已在GPU0/1自然空闲后完成；本实验没有远端Slurm提交、没有覆盖排队源或终止其他用户进程。原5304矩阵、root math和模拟器均未改；生产头的安装与验证以保留记录为准。
'''.replace('{table}','\n'.join(table)).replace('{forward_status}',forward_status).replace('{production_status}',production_status).replace('{retention_status}',retention_status)
for k,v in [('rows',len(a['rows'])),('pairs',len(a['pairs'])),('launches',a['timed_launches']),('segments',a['accepted_segments']),('incomplete',len(a['incomplete']))]:text=text.replace('{'+k+'}',str(v))
(root/'docs/mla_host_a100_joint_qk_20261101.md').write_text(text)
(p/'composition_comparisons.json').write_text(json.dumps(dict(candidate_adopted=adopted,goal_status='active',triples=triples),indent=2)+'\n')
(p/'README.md').write_text('''# JointMDO plus paired-QK fragment reuse, 2026-11-01

Retention is recorded in joint_qk_retention.json; current production validation is in completion_state.json. Goal remains active. See [report](../../../docs/mla_host_a100_joint_qk_20261101.md), confirmation_plan.json and composition_comparisons.json. All156plannedmatrixcellscomplete;readperformance_audit.jsonforcurrentrow/paircounts. Fullreverse/forwardandnormalnativecomplete;original72075/1398terminal0afterfreshroot109tests/72codegen/singleGraphvalidation. NormalNVLinkJQ+3.683%vsbasebut-1.530%vsJM,preserved;no claimofuniformcompositiongain.

SameJMjointmax/rescale/PV/refill, add frozenPpairedQKhelper/unroll4. Same per-scorePE->CKV order, sameJMFP32online grouping, canonicalmath unchanged. Pairrequiresbothstagesready, trading firstQK/load overlap for Qreuse. Originalfetch/LRU/cachepayload/singlecooperative contract preserved. NewREG251STACK0 versusJM255STACK16,actualone108CTA/block32x4x2kernel,active1,dynamic147968/static2304. Original34EMU/68normal exact.109newtests pass;JMcontrol reuses frozen binary/gates with manifest SHA,zero newcontroltests claimed.

Short9cells/6productionpairs,1728launches/108segments and bothreferences complete;newvsJM100GBfast0.398/0.916%. Fullreverse3way1024x5preplanned. performance_audit.json pairs all againstproduction; composition_comparisons.json separately computesnewvsJMcontrol. Staticunroll4HMMA/LDSM counts do not represent dynamicwork or causality. Allpositives/negatives/source/raw preserved. BaselineW andcontrolJM libraries reused,sourcebytes verified. restore_includes.py refuses overwrites. Original5304matrix untouched; fixedJM/SXMsource/plan unchanged. Read continuation_state.json and originalhandles before any restart.
''')
print('REPORT_WRITTEN',len(a['rows']),len(triples))
