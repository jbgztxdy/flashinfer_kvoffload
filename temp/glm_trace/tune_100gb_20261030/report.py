"""Audit every candidate/final sample and report the final 100 GB/s matrix."""
import csv,hashlib,json,math,re,statistics
from pathlib import Path
p=Path(__file__).resolve().parent
root=p.parents[2]
source=root/'temp/glm_trace/kernel_report_20261029'
doc=root/'docs/mla_host_a100_100gb_optimization_20261030.md'
load=lambda f:json.loads(f.read_text())
hashfile=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
meta=load(p/'experiment.json')
original=load(source/'matrix.json')
original_lookup={(r['kind'],r['setting'],r['mtp'],r['requests'],r['host_fraction_target']):r for r in original}
local_cache={}
rows=[];files=[];guard_count=0;launches=0;max_error=0
for path in sorted((p/'raw').glob('*.json')):
 d=load(path)
 if not d.get('complete'):continue
 cfg=d['configuration'];n=cfg['steps'];passes=cfg['passes']
 assert cfg['kind'] in ('lru','nocache') and passes in (3,5)
 assert n==(1024//cfg['mtp'])
 assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
 for name,sha in d['source_sha256'].items():
  snap=p/'source_snapshots'/(sha+Path(name).suffix)
  assert snap.is_file() and hashfile(snap)==sha
 assert d['source_sha256']['include/flashinfer/attention/far_link_emu.cuh']=='5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
 for c in d['cases']:
  fraction=c['host_fraction_requested'];kind=cfg['kind'];m=cfg['mtp'];b=cfg['requests'];setting=cfg['setting']['name']
  old=original_lookup[kind,setting,m,b,fraction]
  old_raw=load(root/old['raw']);old_case=next(z for z in old_raw['cases'] if z['host_fraction_requested']==fraction)
  assert d['bindings']==old_raw['bindings'] and c['far_refs']==old_case['far_refs'] and c['far_unique']==old_case['far_unique']
  ticks=list(range(n)) if kind=='lru' else [i//m for i in meta['no_lru_snapshots']]
  assert c['step_indices']==ticks
  if kind=='lru':
   assert cfg['capacity']==4096
   assert all(len(z)==passes for z in c['ours_unique_miss_samples']+c['ours_reference_miss_samples'])
   assert all(z==c['far_unique'][0] for z in c['ours_unique_miss_samples'][0])
   assert all(z==c['far_refs'][0] for z in c['ours_reference_miss_samples'][0])
   assert all(abs(statistics.mean(z)-mean)<1e-8 for z,mean in zip(c['ours_unique_miss_samples'],c['ours_unique_misses']))
   hit=1-sum(map(sum,c['ours_reference_miss_samples']))/passes/sum(c['far_refs'])
   assert abs(hit-c['ours_reference_cache_hit_rate'])<1e-12
  else:
   assert cfg['capacity']==0
   hit=0
  assert max(c['numerical_checks']['max_abs_error'].values())<.01
  max_error=max(max_error,max(c['numerical_checks']['max_abs_error'].values()))
  for mode,t in c['timings'].items():
   samples=t['per_step_samples_us'];assert len(samples)==len(ticks) and all(len(z)==passes for z in samples)
   assert all(math.isfinite(x) and x>0 for z in samples for x in z)
   assert abs(statistics.mean(x for z in samples for x in z)-t['mean_us'])<1e-7
   for k in range(passes):
    assert abs(statistics.mean(z[k] for z in samples)-t['pass_mean_us'][k])<1e-7
    end=0
    for g in (g for g in c['guards'] if g['mode']==mode and g['pass_index']==k):
     assert g['valid'] and g['first']==end
     if g['timing_queue']!='cuda_graph':assert g['guard_ms']>=g['enqueue_ms']+max(1,g['enqueue_ms']*.05)
     end=g['last'];guard_count+=1
    assert end==len(ticks)
   launches+=len(ticks)*passes
  key=(kind,m,b,fraction)
  if key not in local_cache:
   mode='floor' if kind=='lru' else 'snapshots'
   lc=next(z for z in load(source/'raw'/f'{mode}_floor_mtp{m}_b{b}.json')['cases'] if z['host_fraction_requested']==fraction)
   local_cache[key]=[statistics.mean(z) for z in lc['timings']['local']['per_step_samples_us']]
  compute=local_cache[key]
  assert len(compute)==len(ticks)
  bw=cfg['setting'].get('bandwidth_GBps')
  if not bw:
   physical=load(root/'temp/glm_trace/briefing_report_20261030/physical_bandwidth.json')
   bw=next(z['reference_GBps'] for z in physical['settings'] if z['name']==setting)
  ref=statistics.mean(max(t,miss*1152/(1000*bw)) for t,miss in zip(compute,c['ours_unique_misses']))
  t=c['timings']['ours'];copy=c['timings']['memcpy']
  rows.append(dict(variant=cfg['variant'],confirmation=path.name.startswith('confirm_'),kind=kind,setting=setting,mtp=m,requests=b,host_fraction=fraction,fetch_ctas=cfg['fetch_ctas'],host_split=cfg['host_split'],passes=passes,ours_us=t['mean_us'],memcpy_us=copy['mean_us'],local_us=statistics.mean(compute),reference_us=ref,ours_over_reference=t['mean_us']/ref-1,cache_reference_hit_rate=hit,unique_miss_MB=statistics.mean(c['ours_unique_misses'])*1152/1e6,ours_pass_means_us=t['pass_mean_us'],original_ours_us=old['ours_mean_us'],change_vs_original=1-t['mean_us']/old['ours_mean_us'],raw=str(path.relative_to(root))))
 files.append(dict(path=str(path.relative_to(root)),sha256=hashfile(path)))
neighbor_rows=[r for r in rows if Path(r['raw']).name.startswith('neighbor_')]
final=[r for r in rows if r['variant']=='final' and not r['confirmation'] and
       not Path(r['raw']).name.startswith('neighbor_')]
trace=[r for r in final if r['kind']=='lru' and r['setting'].startswith('g100_')]
expected={(int(m),b,f'g100_l{lat}') for m,bs in meta['batches'].items() for b in bs for lat in (0,2000,5000,10000,20000)}
assert {(r['mtp'],r['requests'],r['setting']) for r in trace}<=expected
trace_complete=len(trace)==195
no_lru=[r for r in final if r['kind']=='nocache']
confirmed=[r for r in rows if r['variant']=='final' and r['confirmation']]
regression_shapes={(1,1),(1,4),(1,16),(1,128),(1,512),(8,1),(8,4),(8,16),(8,32),(8,64),(8,128)}
regression_expected={(m,b,s) for m,b in regression_shapes for s in ('g50_l20000','pcie','nvlink')}
regression_actual={(r['mtp'],r['requests'],r['setting']) for r in final if r['kind']=='lru' and not r['setting'].startswith('g100_')}
no_lru_expected={(m,b,s,f) for m,b in ((1,1),(1,512),(8,16),(8,128))
    for s in ('g100_l0','g100_l20000','g50_l20000') for f in meta['host_fractions']}
no_lru_actual={(r['mtp'],r['requests'],r['setting'],r['host_fraction']) for r in no_lru}
confirmation_expected={(v,fetch,m,b,s) for v,fetch in (('base',16),('base',24),('final',24))
    for m,b in ((8,128),(8,64),(1,512)) for s in ('g100_l0','g100_l20000','g50_l20000','nvlink')}
confirmation_actual={(r['variant'],r['fetch_ctas'],r['mtp'],r['requests'],r['setting']) for r in rows if r['confirmation']}
assert regression_actual<=regression_expected and no_lru_actual<=no_lru_expected
assert confirmation_actual<=confirmation_expected
tests=load(p/'tests.json') if (p/'tests.json').exists() else None
complete=(trace_complete and no_lru_actual==no_lru_expected and regression_actual==regression_expected
    and confirmation_actual==confirmation_expected and tests is not None and tests['exit_code']==0)
with (p/'results.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
summary=dict(complete=complete,final_100gb_trace_cells=len(trace),expected_100gb_trace_cells=195,final_no_lru_cells=len(no_lru),final_confirmation_cells=len(confirmed),all_audited_cells=len(rows),timed_launches=launches,accepted_segments=guard_count,sampled_max_abs_error=max_error,rows=rows,original_matrix_sha256=hashfile(source/'matrix.json'),production_header_sha256=hashfile(root/'include/flashinfer/attention/mla_host_batch.cuh'),files=files)
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
label=lambda s: {'pcie':'物理PCIe','nvlink':'物理NVLink','g50_l20000':'50 GB/s＋20 µs'}.get(s,'100 GB/s＋'+s.split('_l')[-1].replace('20000','20').replace('10000','10').replace('5000','5').replace('2000','2')+' µs')
lines=['# A100：100 GB/s 融合 MLA kernel 优化','',f'状态：{"全部验证完成" if complete else "验证进行中"}。100 GB/s完整trace矩阵 {len(trace)}/195 格，无LRU最终比较 {len(no_lru)}/84 格，5pass最终确认 {len(confirmed)}/12 格。','',
 '本轮保持单cooperative kernel、真实GLM-5.3访问trace、4096槽/请求LRU、数学计算、缓存政策与链路模拟规则。生产头文件增加一个大batch的整行staged路径；不采用探索中的动态队列、扩大attention tile、改变fetch块大小或寄存器三级流水。原5304格数据保留，本文将优化结果单独列出。','',
 '## 改动与启用条件','',
 '原路径对尾部行生成partial states并在全grid barrier后merge。新路径在足够多独立行时，每行直接完成attention并输出，省去尾部分片及合并；编译期固定staged、整行及不拆Host工作项，去掉不可能执行的原地读写回分支。容量足以容纳全部选中条目时，attention只读HBM，远端模拟计数/等待仅保留在真正取数的fetch路径；等待staging就绪的协议仍保留。','',
 '自动派发条件为stage=True、host_split=False、64 heads、query行数≥4×SM数、staging容量≥query行数×最大KV长度。未满足条件使用原通用kernel。最终构建中六个原通用kernel的反汇编指令文本与基线一致，证据见compiler_summary.json。','',
 'A100-SXM4的100 GB/s推荐：达到432个query行时设置host_fetch_ctas=24、host_split=False；较小形状沿用fetch16/split on。两者都开启host_stage=True，trace回放开启4096槽LRU；无LRU扫描关闭缓存。新路径不需要新的API或编译开关，尚无本轮H800实测。','',
 '```python','wrapper.host_stage = True','wrapper.host_fetch_ctas = 24','wrapper.host_split = False','wrapper.host_staging_rows = num_query_rows * max_kv_len','```','',
 '## 测量和参考','',
 'A100-SXM4 GPU0计算、GPU1为真实NVLink远端；50/100 GB/s模拟在同一物理路径注入。完整trace每请求1024连续assistant位置，MTP1/8分别1024/128次launch，3次冷启动pass全步算术平均；确认轮为5pass。无LRU使用四个快照和全部七个远端比例。数据载荷固定seed BF16，真实的是访问序列；不包括模型其他层或生成接受率。','',
 'GPU event与guard/16步CUDA Graph避免CPU入队间隙进入计时，miss诊断在事件外。Memcpy保留实际GPU搬运、必要cache scatter和完整attention，CPU LRU规划/Host整理在外。纯device使用原矩阵同输入正常构建参考，完美重叠参考逐步计算mean(max(C_t, unique miss bytes_t/B))；它假设理想重叠，不计附加延迟、LRU维护及SM/HBM争用。','',
 '## 完整trace：100 GB/s全部形状和延迟','',
 '| MTP | 请求 | 延迟 µs | 优化前固定配置 µs | 本轮 µs | Memcpy µs | 纯device µs | 重叠参考 µs | 相对原矩阵降低 | 高于参考 | LRU引用命中率 | fetch/split |',
 '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for r in sorted(trace,key=lambda r:(r['mtp'],r['requests'],int(r['setting'].split('_l')[-1]))):
 lines.append(f"| {r['mtp']} | {r['requests']} | {int(r['setting'].split('_l')[-1])/1000:g} | {r['original_ours_us']:.2f} | {r['ours_us']:.2f} | {r['memcpy_us']:.2f} | {r['local_us']:.2f} | {r['reference_us']:.2f} | {r['change_vs_original']*100:.1f}% | {r['ours_over_reference']*100:.1f}% | {r['cache_reference_hit_rate']*100:.2f}% | {r['fetch_ctas']}/{'on' if r['host_split'] else 'off'} |")
lines+=['','小形状沿用同一通用kernel和原参数，表中小幅变化视为重新测量波动，不作为实现收益。原矩阵固定fetch16/split on；大形状本轮还改变为推荐参数，因此上表是配置整体对照，代码与参数贡献再用下面的同轮控制区分。','',
 '## 同轮5pass确认：默认、同参数旧实现与新实现','',
 '| 链路 | MTP | 请求 | 实现 | fetch/split | 本文 µs | Memcpy µs | 纯device µs | 重叠参考 µs | 高于参考 | pass均值 µs |',
 '|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|']
for r in sorted([r for r in rows if r['confirmation']],key=lambda r:(r['mtp'],r['requests'],r['setting'],r['variant'],r['fetch_ctas'])):
 lines.append(f"| {label(r['setting'])} | {r['mtp']} | {r['requests']} | {r['variant']} | {r['fetch_ctas']}/{'on' if r['host_split'] else 'off'} | {r['ours_us']:.2f} | {r['memcpy_us']:.2f} | {r['local_us']:.2f} | {r['reference_us']:.2f} | {r['ours_over_reference']*100:.1f}% | {', '.join(f'{t:.2f}' for t in r['ours_pass_means_us'])} |")
lines+=['','## 无LRU：四快照、全部远端比例','',
 '| 链路 | MTP | 请求 | 远端目标 | 优化前同轮默认 µs | 本轮 µs | Memcpy µs | 纯device µs | 重叠参考 µs | 高于参考 |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in sorted(no_lru,key=lambda r:(r['mtp'],r['requests'],r['setting'],r['host_fraction'])):
 base=next((b for b in rows if b['variant']=='base' and not b['confirmation'] and b['kind']=='nocache' and all(b[k]==r[k] for k in ('mtp','requests','setting','host_fraction'))),None)
 lines.append(f"| {label(r['setting'])} | {r['mtp']} | {r['requests']} | {r['host_fraction']:.0%} | {base['ours_us']:.2f} | {r['ours_us']:.2f} | {r['memcpy_us']:.2f} | {r['local_us']:.2f} | {r['reference_us']:.2f} | {r['ours_over_reference']*100:.1f}% |" if base else f"| {label(r['setting'])} | {r['mtp']} | {r['requests']} | {r['host_fraction']:.0%} | 待完成 | {r['ours_us']:.2f} | {r['memcpy_us']:.2f} | {r['local_us']:.2f} | {r['reference_us']:.2f} | {r['ours_over_reference']*100:.1f}% |")
lines+=['','## 其他链路回归：完整trace＋LRU4096','',
 '| 链路 | MTP | 请求 | 原矩阵固定配置 µs | 本轮 µs | Memcpy µs | 重叠参考 µs | 高于参考 |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in sorted([r for r in final if r['kind']=='lru' and not r['setting'].startswith('g100_')],key=lambda r:(r['setting'],r['mtp'],r['requests'])):
 lines.append(f"| {label(r['setting'])} | {r['mtp']} | {r['requests']} | {r['original_ours_us']:.2f} | {r['ours_us']:.2f} | {r['memcpy_us']:.2f} | {r['reference_us']:.2f} | {r['ours_over_reference']*100:.1f}% |")
lines+=['','## 邻近fetch CTA复核','',
 '| fetch CTA | split | 延迟 µs | 本文 µs | Memcpy µs | 重叠参考 µs | 高于参考 |',
 '|---:|---|---:|---:|---:|---:|---:|']
for r in sorted([r for r in neighbor_rows if r['setting']=='g100_l20000'],key=lambda r:r['fetch_ctas']):
 lines.append(f"| {r['fetch_ctas']} | {'on' if r['host_split'] else 'off'} | 20 | {r['ours_us']:.2f} | {r['memcpy_us']:.2f} | {r['reference_us']:.2f} | {r['ours_over_reference']*100:.1f}% |")
lines+=['','20与28个CTA的候选使用与final完全相同的header/benchmark路径；其raw文件以neighbor_开头，单独列出，不计入最终195格矩阵。尝试用不同variant标签重复执行时，因隔离include目录未创建而在CUDA分配前失败；原始邻近参数实测保留。','',
 '## 诊断与未采用候选','',
'原staged/fetch16配置的内部时间戳显示：平均约0.202 ms后开始fetch，最后一个fetch CTA约4.010 ms完成，attention约4.234 ms完成，merge最晚约4.406 ms完成；准备/解析约0.20 ms，LRU项通常在0.254 ms前结束。诊断写时间戳会扰动代码，不替代正式计时，具体数组和逐步结果保留在phase_summary.json。LRU不是额外差距的唯一来源，取数吞吐、计算资源分配及尾部合并同样重要。','',
'上述阶段数据来自优化前通用路径；最终整行路径未取得同条件的无竞争阶段剖析，不能将旧阶段耗时直接当作最终实现的归因。最终路径的未插桩总时间、命中和miss字节来自完整测试。','',
 '| 候选 | fetch | split | 100 GB/s＋20 µs、MTP8/128请求 µs |',
 '|---|---:|---|---:|']
for r in sorted([r for r in rows if not r['confirmation'] and r['variant'] not in ('final','profile') and r['kind']=='lru' and r['setting']=='g100_l20000' and r['mtp']==8 and r['requests']==128],key=lambda r:r['ours_us']):
 lines.append(f"| {r['variant']} | {r['fetch_ctas']} | {'on' if r['host_split'] else 'off'} | {r['ours_us']:.2f} |")
lines+=['','## 正确性与结论边界','',f"本目录已审计{len(rows)}个比较格、{launches:,}次计时launch、{guard_count:,}段有效队列，采样最大绝对误差{max_error:.8f}。源码、绑定、每步miss、数学输出、guard和均值逐项核对，原矩阵SHA未变。完整测试状态：{tests['exit_code'] if tests else '尚在排队'}，输出见pytest_final.log；新增整行路径测试覆盖完整输出、Graph连续回放、padding、全本地切换和缓存内容。",'',
 '本轮取得的是指定100 GB/s条件下的单kernel改进，不把max参考之上的差值全部称为不可避免开销。参考未计入附加延迟、准备/维护、SM和HBM争用及流水尾部；需要继续按实测距离评价，没有达到10%内的格子保留实际差距。小batch沿用原实现，不能将大batch收益推广到所有形状。','',
 '所有原始样本、逐pass/逐步数据、失败和未采用候选、源代码快照均保留。平面数据：[results.csv](../temp/glm_trace/tune_100gb_20261030/results.csv)，审计：[audit.json](../temp/glm_trace/tune_100gb_20261030/audit.json)。']
lines+=['','寄存器三级流水保留一块数据在寄存器中，资源检查无local spill，但12/16/20个fetch CTA仍分别需要约5.121/4.128/3.631 ms（100 GB/s＋20 µs、MTP8/128请求），均未超过最终约3.56 ms的确认结果，因此未合入。较少fetch CTA的带宽不足与较多fetch CTA占用attention资源之间仍存在取舍，不能仅靠增加在途块数消除当前差距。早期一次尝试在前一基准进程退出时触发独占检查，未分配实验数据；失败日志保留，正式regpipe3由监督脚本串行完成。']
if complete:
 headline=['## 主要结果','',
     '以下均为完整GLM-5.3 trace、staged＋LRU4096槽/请求、100 GB/s＋20 µs，5次冷启动完整窗口确认。旧默认fetch16/split on，旧同参数和新实现均为fetch24/split off。','',
     '| MTP | 请求 | 旧默认 µs | 旧同参数 µs | 新实现 µs | 比旧默认降低 | 比旧同参数降低 | Memcpy加速 | 高于重叠参考 |',
     '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for m,b in ((1,512),(8,64),(8,128)):
  selected=[r for r in rows if r['confirmation'] and r['setting']=='g100_l20000' and r['mtp']==m and r['requests']==b]
  old=next(r for r in selected if r['variant']=='base' and r['fetch_ctas']==16)
  control=next(r for r in selected if r['variant']=='base' and r['fetch_ctas']==24)
  new=next(r for r in selected if r['variant']=='final')
  headline.append(f"| {m} | {b} | {old['ours_us']:.2f} | {control['ours_us']:.2f} | {new['ours_us']:.2f} | {(1-new['ours_us']/old['ours_us'])*100:.1f}% | {(1-new['ours_us']/control['ours_us'])*100:.1f}% | {new['memcpy_us']/new['ours_us']:.3f}× | {new['ours_over_reference']*100:.1f}% |")
 headline+=['','相对旧默认的变化包含参数与代码收益；同参数列单独核对代码收益。参考以上的剩余差距仍需保留，不能把本轮改进写成所有100 GB/s参数都已在理论参考10%以内。','']
 image=root/'docs/assets/glm53_100gb_optimization_20261030/trace_optimization.png'
 if image.exists():
  headline+=['![完整trace、staged、LRU4096下的优化前后及两种baseline](assets/glm53_100gb_optimization_20261030/trace_optimization.png)','',
      '图：完整trace、staged＋LRU4096、3pass矩阵均值。旧列为原矩阵固定配置，新列为最终实现与推荐参数；黑虚线为逐步max参考。5pass同轮确认见上表。','']
 lines[4:4]=headline
doc.write_text('\n'.join(lines)+'\n')
audit=dict(complete=complete,source_matrix_sha256=hashfile(source/'matrix.json'),production_header_sha256=hashfile(root/'include/flashinfer/attention/mla_host_batch.cuh'),summary_sha256=hashfile(p/'summary.json'),csv_sha256=hashfile(p/'results.csv'),document_sha256=hashfile(doc),files=files,neighbor_candidate_rows=len(neighbor_rows),source_sha256={z.name:hashfile(z) for z in p.glob('*.py')},tests=tests)
audit['test_source_sha256']=hashfile(root/'tests/attention/test_mla_host_batch.py')
if tests:
 assert '103 passed' in (p/'pytest_final.log').read_text()
 audit['test_log_sha256']=hashfile(p/'pytest_final.log')
image=root/'docs/assets/glm53_100gb_optimization_20261030/trace_optimization.png'
if image.exists():
 figure=load(p/'figure_data.json')
 assert len(figure)==6 and all(r in rows for r in figure)
 audit['figure_sha256']=hashfile(image)
 audit['figure_data_sha256']=hashfile(p/'figure_data.json')
(p/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','files')}))
