"""Render retained JQ evidence without rewriting the original matrix or figures."""
import hashlib
import json
from pathlib import Path

p=Path(__file__).resolve().parent
root=p.parents[2]
read=lambda name:json.loads((p/name).read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
a=read('performance_audit.json');r=read('joint_qk_retention.json');s=read('completion_state.json')
t=read('tests_production_gpu3.json');code=read('production_codegen_normal_summary.json');probe=read('dispatch_probe_production_joint_qk.json')
assert s['complete'] and r['candidate_adopted'] and t['complete'] and t['expected_tests']==109 and t['exit_code']==0
assert code['complete'] and code['unchanged_kernel_instances']==72 and code['all_original_and_new_targets_identical']
assert probe['kernel_nodes']==1
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==r['candidate_header_sha256']=='7193255e2fdf83d3baa9347dea1a02b892c29ed746da719818e30fdf4066d469'
for key,name in [('sha256','tests_production_gpu3.json'),('codegen_sha256','production_codegen_normal_summary.json'),('dispatch_sha256','dispatch_probe_production_joint_qk.json')]:
    assert r['production_validation'][key]==sha(p/name)
rows=[]
for cohort,mtp,requests,setting in [
    ('joint_qk_shapes_sxm_limited_',1,512,'g100_l20000'),
    ('joint_qk_shapes_sxm_limited_',8,64,'g100_l20000'),
    ('compositionsxmreverse1024_f24_',8,128,'g100_l0'),
    ('compositionsxmreverse1024_f24_',8,128,'g100_l20000'),
    ('compositionsxmreverse1024_f24_',8,128,'g50_l20000'),
    ('compositionsxmforward1024_f24_',8,128,'g100_l0'),
    ('compositionsxmforward1024_f24_',8,128,'g100_l20000'),
    ('compositionsxmforward1024_f24_',8,128,'g50_l20000'),
    ('compositionsxmnative1024_f24_',8,128,'nvlink'),
    ('compositionsxmnative1024_f24_',8,128,'pcie')]:
    rows.append(next(x for x in a['pairs'] if x['cohort']==cohort and x['mtp']==mtp and x['requests']==requests and x['setting']==setting and x['variant']=='joint_qk'))
table=['| MTP/请求 | cohort/pass | 链路 | 原b6 µs | JQ生产 µs | 降低 | Memcpy µs | 纯device µs | 逐步max µs | 高于max |',
       '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for x in rows:
    cohort='reverse' if 'reverse' in x['cohort'] else ('forward' if 'forward' in x['cohort'] else ('native' if 'native' in x['cohort'] else 'matrix'))
    table.append(f"| {x['mtp']}/{x['requests']} | {cohort}/{x['passes']} | {x['setting']} | {x['old_us']:.3f} | {x['new_us']:.3f} | {x['reduction']:+.3%} | {x['memcpy_us']:.3f} | {x['local_us']:.3f} | {x['reference_us']:.3f} | {x['over_reference']:.2%} |")
section='''### 7.9 已保留：联合softmax重缩放与Q片段复用

11月1日07:17生产验证完成：**JQ已逐字节安装，生产header为7193255e…d469。** 两个32KV tile共用一次在线softmax的max和d/O缩放，并复用Q片段读取；保持完整576特征、各score的PE→CKV累加顺序、精确LRU/真实cache载荷和单cooperative kernel。在线归约分组变化，数学方程不变；canonical数学头、模拟器、两参考公式和原5304格矩阵均保留。

GPU3/SXM新增156格矩阵及三组三方完整确认均齐，共210格/110个对原b6配对。60个改变路径配对的未加权耗时降低均值1.602%、范围−0.188%至+5.253%；全部正负差保留。两个超过1%的未改路径原负差经完整5pass前/候选/后夹测未复现，原样本仍在。下面均为A100-SXM4/NUMA0、真实trace完整1024位置、LRU4096/请求、fetch24/split off。同轮b6为原生产基线；3pass矩阵与5pass正向/反向/正常链路分别标注。

{table}

100GB的完整两顺序确认较原b6快约3.84%–5.23%，但仍高于逐步max约25%–27%，未达到10%以内的目标。收益包含联合重缩放和Q复用，不能全部归于Q复用。正常NVLink三方为b6/JM/JQ=3305.725/3136.016/3183.982µs；JQ较JM慢1.530%，该负差保留在[组合报告](mla_host_a100_joint_qk_20261101.md)，不能声称组合总比JM快。

原测源a2f37f…保留；生产清理源只改host局部别名，两者全部36EMU/72normal实例的指令、资源、机器字和控制字完全一致。测量源与清理源各自通过105+4正常测试后，原自动1398完成全部审计/保留门槛并安装；随后从生产root新建独立JIT，再次通过BF16/FP16编译、109项测试、72实例代码核验和实际单108CTA/256线程Graph。原72075/1398均terminal0，[保留记录](../temp/glm_trace/tune_joint_qk_20261101/joint_qk_retention.json)绑定生产验证原件SHA。

后续独立候选中的联合P发布、lane私有分母、九warp及固定stage控制、小batch私有直方图均未保留；各实现、负差和失败记录仍在对应报告。整体优化目标继续active，本节没有改写原矩阵或图1–6。后继[QPE片段全item复用](mla_host_a100_cached_pe_20261101.md)的实现及跨场景验证单列，该实验不属于本次JQ生产保留。

'''.replace('{table}','\n'.join(table))
f=root/'docs/mla_host_glm53_report.md';text=f.read_text()
start=text.index('### 7.9 ');end=text.index('## 8. 全参数汇总、计时口径与结论边界')
text=text[:start]+section+text[end:]
text=text.replace('最新未安装候选及约25%–27%的剩余差距单列于§7.9','最新已保留JQ及约25%–27%的剩余差距单列于§7.9')
f.write_text(text)
print('RETAINED_MAIN_REPORT_WRITTEN')
