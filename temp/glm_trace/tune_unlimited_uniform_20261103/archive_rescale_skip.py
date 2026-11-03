from pathlib import Path
import json,hashlib,statistics,subprocess,ast
p=Path(__file__).resolve().parent;root=p.parents[2]
raw={v:json.loads((p/f'{v}_native_unlimited_f24.json').read_text()) for v in ['rescale_skip','clean_rescale_control_retry']}
means={v:[statistics.mean(x['us']) for x in r['passes']] for v,r in raw.items()}
for r in raw.values():
 assert r['complete']
 for lib in r['libraries']:assert hashlib.sha256((root/lib['path']).read_bytes()).hexdigest()==lib['sha256']
assert raw['rescale_skip']['header_sha256']==hashlib.sha256((p/'rescale_skip.cuh').read_bytes()).hexdigest()
audit=dict(pass_means_us=means,means_us={v:statistics.mean(x) for v,x in means.items()},slowdown_percent=(statistics.mean(means['rescale_skip'])/statistics.mean(means['clean_rescale_control_retry'])-1)*100,focused_tests=2,header_actual_library_sha_verified=True,retained=False,scope='GPU3 normal same-device farHBM; no100GB or physical link conclusion',failures=['initial constructor match-count assertion prevented header creation; downstream compile missing include, preserved failed_setup log','first fresh clean control failed diagnostics whitelist after timing; incomplete raw retained; retry changed only harness whitelist/output paths'])
(p/'rescale_skip_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
(p/'RESCALE_SKIP_20261104.md').write_text(f'''# 恒等softmax缩放跳过实验\n\n从clean全CTA unique staged冻结头出发，仅joint MDO在原exp2算出的factor精确等于1时跳过O/d乘1。分数exp、QK/PV、归一化、搬运/发布、精确LRU及真实stage/cache双写均保持。不同于已有包调度/同步/最终目的地实验。没有改变容差或远端字节。\n\n正常GPU3/A100-PCIE、NUMA1、同卡独立HBM源、BF16/h64/MTP8/R128/K2048/LRU4096，256位置32步×3coldpass。候选逐pass {means['rescale_skip']}µs，均{audit['means_us']['rescale_skip']:.6f}；随后同参数未修改clean {means['clean_rescale_control_retry']}，均{audit['means_us']['clean_rescale_control_retry']:.6f}。候选慢{audit['slowdown_percent']:.3f}%，每pass均慢，无采用信号；不安装、不扩展100GB模拟。顺序单向，不当严格因果性能分解。\n\n两项K65/K2048定向数学/LSE/全mask/padding/QPE stride/Graph/cache载荷检查通过，性能数学采样与所有最终有效cache载荷通过。沿用旧容差。source和完整raw实际加载库SHA已核验；机器码有额外4个BRA/BSSY/BSYNC，REG255/STACK16与clean相同。这证明分支已生成，不能从静态代码单独归因退化。未统计动态因子分布/分支效率，未新增sanitizer/全矩阵/物理链路结论。\n\n初构造器未限定重复数学函数导致assert，随后构建缺include失败；日志保留。新clean控制首次未放行禁用的service诊断白名单，计时后assert，incomplete原件保留；仅修runner白名单并使用新输出文件重试。未覆盖失败raw，未使用失败计时作成绩。\n''')
h=root/'docs/HANDOFF.md';s=h.read_text();marker='## 2. 当前运行状态与新一轮进度\n';assert marker in s
s=s.replace(marker,marker+f'''\n**11/04恒等rescale跳过实验：** clean全CTA unique staged仅factor精确等于1跳过O/d乘法，完整QK/softmax/PV及LRU/搬运保持；2定向检查通过，normal3pass3147.851µs，随后新clean3099.381µs，慢1.564%，拒绝不测100GB。REG255/STACK16相同，额外分支实际生成；无动态因子统计，不单归因分支。初构造/控制白名单失败和incomplete raw保留，retry仅补控制。见RESCALE_SKIP_20261104.md、audit和source/实际库SHA。生产81093保持，runner结束。此前第三归档提交1007e898。\n''')
h.write_text(s)
files=[root/x for x in subprocess.check_output(['git','ls-files','--others','--exclude-standard',str(p.relative_to(root))],cwd=root,text=True).splitlines()]
for f in files:
 if f.suffix=='.py':ast.parse(f.read_text())
 if f.suffix=='.json':json.loads(f.read_text())
m=p/'rescale_skip_manifest.json';m.write_text(json.dumps(dict(files=[dict(path=str(f.relative_to(root)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in files]),indent=2)+'\n')
subprocess.run(['git','add','--',*[str(f.relative_to(root)) for f in files],str(m.relative_to(root)),'docs/HANDOFF.md'],cwd=root,check=True)
print(json.dumps(audit,indent=2))
