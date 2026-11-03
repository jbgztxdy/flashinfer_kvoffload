from pathlib import Path
import json,hashlib,statistics,subprocess,ast
p=Path(__file__).resolve().parent;root=p.parents[2]
r=json.loads((p/'bounded_anchor_native_unlimited_f24.json').read_text());assert r['complete'];assert r['header_sha256']==hashlib.sha256((p/'bounded_anchor.cuh').read_bytes()).hexdigest()
for lib in r['libraries']:assert hashlib.sha256((root/lib['path']).read_bytes()).hexdigest()==lib['sha256']
means=[statistics.mean(x['us']) for x in r['passes']]
a=dict(pass_means_us=means,mean_us=statistics.mean(means),focused_tests=5,fp64_model_cases=72,source_library_sha_verified=True,retained=False,scope='normal BF16 staged onGPU3 same-device farHBM; no100GB/physical-link/fullmatrix conclusion')
(p/'bounded_anchor_audit.json').write_text(json.dumps(a,indent=2)+'\n')
(p/'BOUNDED_ANCHOR_20261104.md').write_text(f'''# 有界归一化基点实验\n\n从rescale_skip（仅factor==1跳过O/d乘法）出发，改变online softmax归一化基点的更新规则：当前分数的log2最大值超过旧基点8单位才更新，否则沿用旧基点。每个tile仍计算全部QK、分数exp、rowsum、PV，必要的O/d重缩放仍执行。归一化方程无需基点等于最大值；旧状态用exp2(old-new)换基，分子分母同步缩放。P操作数在精确算术下不超过256（实际FP32舍入须另计），使用BF16原有指数范围，未减少尾数位数。8是明确的P范围选择，不是参数扫描。与旧仅最大值未变跳过不同，本方案即使最大值小幅上升也可保持基点。\n\n72项FP64方程模型包括n1/31/32/65/128/2048、尺度.05/1/4/32及正负1000log2跳变，归一化输出/LSE误差<1e-11且P<=256；它不是GPU BF16数值证明。实际GPU定向K65及K2048，Q缩放.05/1/4/32共5case，包含原全mask/padding/stride80/Graph及全有效cache载荷检查，通过，未修改原数学/LSE容差。未新增原输入之外的数值稳定性保证。\n\nGPU3/A100-PCIE/NUMA1，同卡独立farHBM、BF16/h64/MTP8/R128/K2048/LRU4096，256位置32步×3coldpass为{means}µs，均{a['mean_us']:.6f}。近期新clean3099.381330µs（跨轮非夹测），未见净收益，不保留、不安装、不扩展100GB。数学采样/最终cache载荷和实际加载库/header SHA均核验。主函数同统计器9349指令、REG255/STACK16；静态codegen不证明动态重缩放次数或原因，未测动态anchor更新率/分支效率，不认定浮点乘法成本已经被因果排除。\n\n跨query unique所有权、远端字节/真实双写、服务时序、release/acquire及精确LRU代码保持。未改变生产、参考、模拟器，未新增sanitizer/完整矩阵/物理链路结论。\n''')
h=root/'docs/HANDOFF.md';s=h.read_text();marker='## 2. 当前运行状态与新一轮进度\n';assert marker in s
s=s.replace(marker,marker+'''\n**11/04全线程payload与有界softmax基点实验：** 固定20页包将1440uint4展平给256线程，5完整轮+160线程尾轮；原stage/cache真实写入/unique协议保持，normal3206.048µs。由codegen膨胀补rolled-loop控制3160.512µs，仍未胜近期clean3099.381，各2定向检查通过。另一有界归一化基点原型：仅分数超旧anchor8log2单位才重缩放，完整QK/PV保持；72FP64方程case、5GPU不同Q尺度case通过且容差不变，但normal3172.971µs无收益。三版均拒绝不测100GB；无动态anchor/分支因果计数。报告FLAT_PAYLOAD_20261104.md、BOUNDED_ANCHOR_20261104.md及raw/audit/source/实际库SHA齐。生产81093保持，runner均结束，不重复此映射或anchor阈值扫描。\n''')
h.write_text(s)
production={x:hashlib.sha256((root/'include/flashinfer/attention'/x).read_bytes()).hexdigest() for x in ['mla_host_batch.cuh','mla.cuh','far_link_emu.cuh']}
assert production['mla_host_batch.cuh']=='81093ba5ab33cc5d84b2df04c702975697387b4054ebe8f1991f558f1bd6cdd2'
files=[root/x for x in subprocess.check_output(['git','ls-files','--others','--exclude-standard',str(p.relative_to(root))],cwd=root,text=True).splitlines()]
for f in files:
 if f.suffix=='.py':ast.parse(f.read_text())
 if f.suffix=='.json':json.loads(f.read_text())
m=p/'payload_anchor_manifest.json';m.write_text(json.dumps(dict(files=[dict(path=str(f.relative_to(root)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in files],production=production),indent=2)+'\n')
subprocess.run(['git','add','--',*[str(f.relative_to(root)) for f in files],str(m.relative_to(root)),'docs/HANDOFF.md'],cwd=root,check=True)
print('new files',len(files),'anchor mean',a['mean_us'])
