"""Keep route choice and remaining gap explicit, preserving fixed-route source data."""
import hashlib,json,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];load=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
a=load(p/'audit.json');state=load(p/'state.json');check=load(p/'selected_correctness.json');m=load(p/'manifest.json')
assert state['complete'] and all(x['exit_code']==0 for x in state['actions'])
assert len(a['rows'])==30 and len(a['pairs'])==15 and not a['incomplete'] and check['complete']
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==m['header_sha256']
assert all(sha(root/x['path'])==x['sha256'] for x in m['reused_libraries'])
lines=['# A100：按场景择优 staged／direct（2026-11-02）','',
'已用当前生产754706f8重新做真实SXM/NVLink与100GB/s两条路径的同参数对照。结果支持按batch、链路、LRU和远端比例选路径；不能把128请求LRU的约25%差距推广到所有场景，也不能沿用旧版“NVLink一律原地读更快”。','',
'本轮GPU0/1 A100-SXM4-40GB、实际NV12、NUMA0；BF16、64heads、KV512+64、K2048、MTP8。LRU为16/128请求、4096槽/请求、完整1024位置×3冷pass；无LRU为16请求、四个原快照、远端0/10%/30%、3pass。正常构建只测物理NVLink；模拟构建在同一物理NVLink源上测100GB/s＋0/20µs。两条路径共享同一冻结源码和原二进制，只有host_stage切换。16请求f16/split on，128请求f24/split off；不是参数扫描，也不是CUDA源码优化。','',
'| LRU | 请求 | 远端比例 | 链路 | staged µs | direct µs | 选择 | 择优 µs | 纯device µs | 逐步max µs | 高出参考 |','|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|']
for x in a['pairs']:
 lines.append(f"| {'4096/请求' if x['kind']=='lru' else '关闭'} | {x['requests']} | {'trace/LRU实际miss' if x['fraction'] is None else str(round(x['fraction']*100))+'%'} | {x['setting']} | {x['staged_us']:.3f} | {x['inplace_us']:.3f} | {x['selected_route']} | {x['selected_us']:.3f} | {x['local_us']:.3f} | {x['reference_us']:.3f} | {x['over_reference']*100:+.2f}% |")
lines += ['',
'物理NVLink：16请求LRU选择direct，557.725对staged622.968µs，快10.47%，高于参考10.88%；128请求LRU选择staged，3016.416对direct3335.189µs，快9.56%，高于参考22.26%。之前旧路由比较中128请求原地读更快，是在staged还约3.745ms时；后续staged已明显改善，旧赢家不能直接沿用。','',
'100GB/s：两种延迟、两种LRU形状都是staged更快。128请求择优后3289.824/3306.973µs，高于参考22.24%/22.89%；16请求则679.387/708.773µs，高于参考34.55%/40.37%。仅切direct不能消除这些差距。','',
'你记得接近下限的场景确实存在：当前16请求、无LRU、远端10%/30%，NVLink选择staged为511.317/539.904µs，高于参考1.87%/7.24%；100GB/s两档延迟分别高13.38%–18.76%。NVLink10%的两路径差异仅0.22%，视为接近持平，不声称稳定路由收益。0%远端控制也列出；某些staged比纯device基准更快，说明后者是指定本地实现的测量参考，不是严格数学下界，负差没有截断。','',
'历史核对：10月29日MTP8/16请求、LRU4096的NVLink原地读531.456µs对494.592µs，约7.45%；其摘要为后半段逐步median均值、参考用正常构建最快样本。本轮使用所有步骤及冷启动、每pass全部样本算术均值，同硬件参考为原正常计算的逐步均值；原输入绑定/配置与本轮也不同，不能当严格新旧回退对照。history_review.json保留原件SHA和原始样本重新计算。100GB/s曾接近参考的记录也主要是低远端比例/无LRU、中等batch；它不等于128请求LRU已经接近下限。','',
'每条路径分别记录实际unique miss和引用命中率；精确launch级LRU的同stamp并列选择可能使两路径miss略异。参考仍逐步max(本地计算, 本路径实测unique未命中字节/带宽)再求均值，没有max(mean)或改模拟器。direct可能重复读取引用；本表仍用共同的去重搬运乐观参考，不用重复字节抬高参考来缩小差距。','',
'已提供[启动前路由选择器](../temp/glm_trace/route_select_20261102/select_route.py)，读取selected_routes.json中的15个同参数实测配置，用configure_wrapper设置host_stage/fetch/split。它只接受测过的硬件、batch/MTP、K/head/dtype、LRU容量、链路和远端比例，在warmup/Graph capture之前选一次；未覆盖配置明确报缺失，不猜测普适规则。它不在计时内运行，不逐launch在线择优，不切换已经活跃的cache/Graph。本轮没有改变公共wrapper默认路由或CUDA生产源码。','',
'两个选中路径的短K97合成检查完成：16请求direct、128请求staged，各3轮并每轮Graph重放两次，采样output/LSE与全部有效cache载荷/映射均通过。正确性fixture使用容量128，不冒称4096长trace全槽验证；真实性能回放仍实际容量4096并验证miss/命中和输出。首版验证脚本因pytest导入缺失在GPU工作前失败，修正后只重跑此项，原件/错误说明保留。','',
f"合计30格/15对、{a['timed_launches']}计时launch/{a['accepted_segments']}有效段，所有原始路径/正负差/逐步逐pass/guard/源码SHA均保留。复用当前版本正常/EMU二进制且SHA未变，不重跑整套测试/sanitizer或大矩阵。整体优化未完成；本轮收益是路径选择与口径核对，不作为新kernel机制加速。",'',
'[原始记录与复现入口](../temp/glm_trace/route_select_20261102/README.md)。']
(root/'docs/mla_host_route_selection_20261102.md').write_text('\n'.join(lines)+'\n')
(p/'README.md').write_text('''# Current-source matched route selection

All planned30cells/15pairs complete. CurrentCUDA source754706f8 and originalnormal/EMU libraries are unchanged. SXM0/1 NV12/NUMA0, MTP8, BF16/K2048/h64. Sharedgpu01lock and resourcepre/post checks respected. GPU2/3 not used. `plan.json` defines twoLRUshapes16/128 and one16request noLRU shape at0/.1/.3, normalNVLink and100GB0/20us, bothroutes. AllLRUfull1024positions3coldpasses; noLRUfouroriginalsnapshots3passes.

`audit.py` verifies source/bindings/samplemeans/guards/coldmisses and originalsame-hardware stepwise compute/max references; `selected_routes.json` chooses the mean-minimizing route per matched key without deleting slower data. `select_route.configure_wrapper` applies that measured profile before warmup/capture; unmeasured configurations are explicitly unsupported. Not an automatic inference of physical transport, not a CUDA change or universal optimum.

`selected_correctness.json` covers bothchosenLRUpaths on shortK97fixtures, repeatedGraph/output/LSE/allvalidcachepayloads. Initialpytestimportfailure preserved, repaired by extracting twoexistinghelperdefinitions withast without dependency installation. No fullsuite/sanitizer/matrix repetition.

`history_review.json` identifies the oldNVLink7percent16request result and lowfar100GBresults, including oldersteady/fastest-sample reference differences. Do not conflate those with128requestLRU or currentallstepmeans.

Reproduce with `run.py` (skips complete raw per benchmark); read-only audits via `audit.py` and `report.py`. Inspect actualresource/state first; allcurrentrunners are terminal. `continuation_state.json` records finalstatus. Production remains754706f8. Full report: docs/mla_host_route_selection_20261102.md.
''')
checkpoint=dict(timestamp=time.strftime('%FT%T%z'),complete=True,goal_complete=False,production_sha256=m['header_sha256'],kernel_changed=False,route_pairs=15,cells=30,timed_launches=a['timed_launches'],accepted_segments=a['accepted_segments'],active_runners=[],selected_correctness_complete=True,evidence=[dict(file=n,sha256=sha(p/n)) for n in ['manifest.json','state.json','audit.json','selected_routes.json','selected_correctness.json','history_review.json','select_route.py']])
(p/'continuation_state.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
print(json.dumps({k:v for k,v in checkpoint.items() if k!='evidence'},indent=2))
