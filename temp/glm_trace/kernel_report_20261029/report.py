"""Generate a self-contained Chinese report and machine-readable full matrix, only from measured data."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
DEST=Path(__file__).resolve().parent
DOC=ROOT/"docs/mla_host_glm53_kernel_matrix_20261029.md"
META=json.loads((DEST/"experiment.json").read_text())


def setting_label(setting):
    if setting["name"]=="nvlink":return "物理 NVLink"
    if setting["name"]=="pcie":return "物理 Host PCIe"
    bw=setting["bandwidth_GBps"]
    return (f"{bw} GB/s" if bw else "不限速")+f"＋{setting['latency_ns']/1000:g} µs"


def read(path):return json.loads(path.read_text())


def atomic_write(path,text):
    temp=path.with_suffix(path.suffix+".new")
    temp.write_text(text);temp.replace(path)


def floor(mtp,batch,ticks,fraction=None):
    kind="floor" if fraction is None else "snapshots"
    path=DEST/"raw"/f"{kind}_floor_mtp{mtp}_b{batch}.json"
    if not path.exists():return None
    data=read(path)
    if not data["complete"]:return None
    case=next(c for c in data["cases"] if c["host_fraction_requested"]==fraction)
    times=case["timings"]["local"]["per_step_samples_us"]
    if fraction is not None:ticks=list(range(len(times)))
    mean=sum(sum(times[t]) for t in ticks)/sum(len(times[t]) for t in ticks)
    best=min(v for t in ticks for v in times[t])
    return mean,best


def gather():
    rows=[];files=[]
    for path in sorted((DEST/"raw").glob("*.json")):
        if path.name.startswith("smoke_"):continue
        try:data=read(path)
        except FileNotFoundError:continue
        if not data["complete"]:continue
        cfg=data["configuration"]
        if cfg["kind"]=="nocache" and cfg.get("memcpy_path_version",1)<2:continue
        files.append(dict(file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          source_sha256=data["source_sha256"]))
        if cfg["kind"] in ("floor","snapshots"):continue
        for case in data["cases"]:
            bounds=floor(cfg["mtp"],cfg["requests"],case["step_indices"],case["host_fraction_requested"])
            if bounds is None:continue
            local,best=bounds
            ours=case["timings"]["ours"]["mean_us"];copy=case["timings"]["memcpy"]["mean_us"]
            n=len(case["step_indices"]);entries=cfg["query_rows"]*2048
            refs=sum(case["far_refs"]);unique=sum(case["far_unique"]);miss=sum(case["ours_unique_misses"])
            c=case["memcpy_plan_counts"]
            m_misses=sum(x[2] for x in c) if c else unique
            m_hits=sum(x[3] for x in c) if c else 0
            bw=cfg["setting"].get("bandwidth_GBps",0)
            transfer_us=miss/n*1152/(bw*1000) if bw else None
            rows.append(dict(kind=cfg["kind"],setting=cfg["setting"]["name"],mtp=cfg["mtp"],
                             bandwidth_GBps=cfg["setting"].get("bandwidth_GBps"),
                             added_latency_us=cfg["setting"].get("latency_ns",0)/1000,
                             emulated=cfg["setting"].get("emulated",False),
                             requests=cfg["requests"],query_rows=cfg["query_rows"],
                             host_fraction_target=case["host_fraction_requested"],
                             host_fraction_actual=case["host_fraction_actual"],
                             ours_mean_us=ours,memcpy_mean_us=copy,local_mean_us=local,local_best_us=best,
                             ours_to_memcpy=ours/copy,ours_to_local_mean=ours/local,ours_to_local_best=ours/best,
                             unique_far_pages_per_step=unique/n,ours_misses_per_step=miss/n,
                             ours_unique_cache_hit_rate=(1-miss/unique if unique else 0),
                             ours_reference_cache_hit_rate=case.get("ours_reference_cache_hit_rate",0),
                             ours_remote_reference_fraction=(1-case.get("ours_reference_cache_hit_rate",0))*refs/(n*entries),
                             final_remote_unique_fraction=miss/(n*entries),
                             memcpy_misses_per_step=m_misses/n,
                             memcpy_unique_cache_hit_rate=(1-m_misses/unique if unique else 0),
                             memcpy_reference_cache_hit_rate=m_hits/max(1,refs),
                             remote_unique_bytes_per_step=miss/n*1152,
                             ideal_unique_transfer_us=transfer_us,
                             ideal_compute_transfer_bound_us=max(local,transfer_us) if transfer_us is not None else None,
                             ours_pass_means_us=case["timings"]["ours"]["pass_mean_us"],
                             memcpy_pass_means_us=case["timings"]["memcpy"]["pass_mean_us"],
                             ours_pass_range_over_mean=(max(case["timings"]["ours"]["pass_mean_us"])-
                                 min(case["timings"]["ours"]["pass_mean_us"]))/ours,
                             memcpy_pass_range_over_mean=(max(case["timings"]["memcpy"]["pass_mean_us"])-
                                 min(case["timings"]["memcpy"]["pass_mean_us"]))/copy,
                             raw=str(path.relative_to(ROOT))))
    return rows,files


def document(rows,files):
    expected0=len(META["settings"])*sum(len(v) for v in META["batches"].values())*7
    expected1=expected0//7
    done0=sum(x["kind"]=="nocache" for x in rows);done1=len(rows)-done0
    audit=read(DEST/"audit.json") if (DEST/"audit.json").exists() else {}
    yield "# GLM-5.3 真实 trace：远端 KV sparse MLA kernel 完整性能对比\n"
    yield "报告更新时间："+time.strftime("%Y-%m-%d %H:%M:%S %z")+"。\n"
    yield f"实验范围：不开 LRU {done0}/{expected0} 格；连续 LRU {done1}/{expected1} 格。"
    yield ("**状态：全组合实验与审计完成。**\n" if (done0,done1)==(expected0,expected1) and audit.get("complete") else
           "**状态：实验进行中。本报告只列已完成的实测，不以估算填补空格。**\n")
    stability=read(DEST/"stability_repeat.json") if (DEST/"stability_repeat.json").exists() else {}
    if not stability.get("complete"):
        done=len(stability.get("repeated",[]));total=len(stability.get("files",[]))
        yield f"稳定性复测：{done}/{total} 组完成。无LRU虽然采样已覆盖全部参数，存在波动较大、等待重测的旧样本；在复测完成前，本报告不能作为最终性能结论。\n"
    resume=read(DEST/"resume_state.json") if (DEST/"resume_state.json").exists() else {}
    if resume.get("state")=="waiting_for_gpu":
        yield f"续跑状态（{resume['timestamp']}）：等待原A100 GPU0/1空闲；现有样本已保留。\n"
    elif resume.get("state")=="interrupted_sigterm":
        yield f"续跑状态（{resume['timestamp']}）：测试进程收到SIGTERM，退出码143；当前未运行GPU实验，等待确认GPU使用安排。剩余组合仍保留，不能将本报告作为全组合最终版。\n"
    if rows:
        yield "## 实测摘要\n"
        for kind,title in (("nocache","不开LRU"),("lru","真实trace＋LRU")):
            values=[r for r in rows if r["kind"]==kind]
            if values:
                wins=sum(r["ours_to_memcpy"]<1 for r in values)
                faster=sum(r["ours_to_memcpy"]<.97 for r in values)
                slower=sum(r["ours_to_memcpy"]>1.03 for r in values)
                yield (f"- {title}：已完成{len(values)}格，本文快于Memcpy {wins}格；"
                       f"按±3%阅读阈值：快于3% {faster}格、接近{len(values)-faster-slower}格、慢于3% {slower}格。"
                       f"本文÷Memcpy范围{min(r['ours_to_memcpy'] for r in values):.3f}–{max(r['ours_to_memcpy'] for r in values):.3f}，"
                       f"本文÷本地平均范围{min(r['ours_to_local_mean'] for r in values):.3f}–{max(r['ours_to_local_mean'] for r in values):.3f}。")
        yield ""
        yield "### 典型参数摘录\n"
        yield "这里固定摘录1/4/16/128请求及四类链路；无LRU取目标Host 30%。所有其他比例和batch仍见后面的完整逐格表，不按赢家挑选配置。时间单位µs。\n"
        yield "| 场景 | 链路 | MTP | 请求 | 本文 | Memcpy | 纯本地 | 本文÷Memcpy | 本文÷本地 |"
        yield "|---|---|---:|---:|---:|---:|---:|---:|---:|"
        selected=[r for r in rows if r["requests"] in (1,4,16,128) and r["setting"] in ("g100_l20000","g50_l20000","nvlink","pcie")
                  and (r["kind"]=="lru" or r["host_fraction_target"]==.3)]
        for r in sorted(selected,key=lambda x:(x["kind"],x["setting"],x["mtp"],x["requests"])):
            setting=next(s for s in META["settings"] if s["name"]==r["setting"])
            yield (f"| {'trace＋LRU' if r['kind']=='lru' else '无LRU，Host30%'} | {setting_label(setting)} | {r['mtp']} | {r['requests']} | "
                   f"{r['ours_mean_us']:.2f} | {r['memcpy_mean_us']:.2f} | {r['local_mean_us']:.2f} | {r['ours_to_memcpy']:.3f} | {r['ours_to_local_mean']:.3f} |")
        yield ""
        yield "±3%是便于阅读的差异分组，不是统计显著性检验。小于pass间波动的差距不宜宣称稳定收益；CSV/JSON保留每次pass均值及极差。\n"
        yield "### 参数分布图\n"
        yield "图中无LRU展示目标Host 30%；其他比例仍见完整逐格表。灰色为未完成，图标题注明已测格数；数值越低越好。\n"
        yield "![无LRU性能比值](assets/glm53_kernel_matrix_20261029/nocache.png)\n"
        yield "![连续LRU性能比值](assets/glm53_kernel_matrix_20261029/lru.png)\n"
        yield "### 按链路汇总\n"
        yield "各格等权；几何平均用于概览，详细结论以对应batch/MTP/Host比例的逐格实测为准。部分完成时不作全矩阵结论。\n"
        yield "| 类别 | 链路 | 已测格 | 快于Memcpy格 | 几何平均加速比 Memcpy/本文 | 几何平均 本文/本地 |"
        yield "|---|---|---:|---:|---:|---:|"
        for kind in ("nocache","lru"):
            for setting in META["settings"]:
                sub=[r for r in rows if r["kind"]==kind and r["setting"]==setting["name"]]
                if not sub:continue
                speed=math.exp(sum(math.log(1/r["ours_to_memcpy"]) for r in sub)/len(sub))
                gap=math.exp(sum(math.log(r["ours_to_local_mean"]) for r in sub)/len(sub))
                yield f"| {kind} | {setting['name']} | {len(sub)} | {sum(r['ours_to_memcpy']<1 for r in sub)} | {speed:.3f} | {gap:.3f} |"
        yield ""
    yield "## 1. 研究对象与报告口径\n"
    yield ("只测 sparse MLA kernel 及必需 KV 搬运，不模拟模型其他部分。三种比较对象是："
           "本文单 cooperative kernel、先 Memcpy/GPU gather 后本地 attention、纯本地 attention。"
           "全部查询形状为64头、BF16、512+64维KV、top-k2048。Q/KV为相同随机BF16值，"
           "访问索引来自GLM-5.3权重对固定语料的因果计算；不是完整模型生成或真实MTP接受率测试。\n")
    yield ("Host比例在本报告中指**原始KV需要从远端读取的条目比例**，不是LRU命中率。"
           "无LRU部分按固定(request,position)映射指定0/5/10/15/20/25/30%，同时报告实际比例；"
           "源trace局部性保持，按四个间隔位置采样单步。LRU部分除最新token外KV原始驻留远端，"
           "不人为指定缓存命中率，连续跑完1024位置后记录缓存及最终远端比例。\n")
    yield ("LRU命中率是对真实top-k索引，在本报告指定的驻留规则、4096槽容量和逐launch LRU政策上"
           "回放得到的结果，并非从生产部署直接采集的缓存计数；换容量、驻留策略或请求调度后可能改变。\n")
    yield "## 2. 数据、设备和完整参数\n"
    yield ("设备：A100-SXM4-40GB双卡，GPU0运行，GPU1为真实NV12远端；物理PCIe为NUMA0 Host pinned内存。"
           "模拟链路是在同一真实远端路径上使用原有虚拟链路时钟，带宽按字节收费、访问附加延迟。"
           "模拟构建与纯本地下限分别使用独立JIT；不启用可选SMALL_PREPARE优化。全部形状固定使用"
           "batched staged路径、fetch CTA=16、Host item=64；单行也显式走该路径，避免旧单query实现"
           "缺少完整模拟钩子造成低估。这里评估固定配置，不逐格挑选最优配置。\n")
    yield "- MTP1 请求数："+", ".join(map(str,META["batches"]["1"]))+"。"
    yield "- MTP8 请求数："+", ".join(map(str,META["batches"]["8"]))+"；query行数=请求数×MTP。"
    yield "- 模拟组合：不限速/100/50 GB/s × 附加0/2/5/10/20 µs；另加物理NVLink和PCIe。"
    yield "- 每格3次测量pass；LRU每pass从冷开始，包含第一步；主表为全部步骤/全部pass的算术平均。"
    yield "- 无LRU取真实窗口偏移0/256/512/768的四个MTP分组，索引随位置改变；每格均检查输出。"
    yield "- 连续LRU用完整1024位置：MTP1=1024次launch，MTP8=128次；每请求4096槽。"
    yield "- 请求绑定在所有batch间使用相同前缀，不随batch重新随机选层/窗口。超过84组可用层×窗口后重复绑定，地址空间仍独立，报告不将其称为512个独立真实会话。\n"
    yield ("trace来自4个完整assistant窗口，每个1024位置；doc1起点35184/41887/48296，doc3起点50621，"
           "每窗口21个indexer层，共84组层×窗口。原始top-k有效性和assistant检查通过；原语料全角色"
           "block loss gate没有通过，不能据此声称完整模型质量已经验证。MTP8是连续8个实测位置组合，"
           "不包含推测生成接受率。batch>128时本地near物理KV按128请求复用，远端和LRU槽按请求独立；"
           "这影响本地cache局部性，应将此作为实验范围限制。输入来源和每请求层/窗口绑定保存于raw JSON。\n")
    yield "## 3. 计时与正确性\n"
    calibration=DEST/"calibration.json"
    if calibration.exists():
        cal=read(calibration)
        yield f"链路校准：15种模拟组合全部通过={cal['complete']}；原始记录为同目录calibration.json。\n"
        yield "| 设置 | 设置GB/s | 实测GB/s | 附加延迟µs | 实测依赖读µs | 校准 |"
        yield "|---|---:|---:|---:|---:|---|"
        for c in cal["settings"]:
            yield f"| {c['name']} | {c['bandwidth_GBps']} | {c['measured_GBps']:.2f} | {c['latency_ns']/1000:.1f} | {c['latency_ns_measured']/1000:.3f} | {'通过' if c['ok'] else '失败'} |"
        yield ""
    yield ("CUDA event括住实际GPU工作。每16步一段预先入队，段前GPU等待保护区，记录CPU入队时间和"
           "GPU保护区耗时，正式样本必须全部在GPU保护区结束前完成入队，否则整任务失败且不标完成。"
           "段间同步保留LRU状态；模型加载、索引上传、CPU规划、参考计算和warm-up不计入主耗时。"
           "首次检查pass不参与平均；正式3pass记录全部逐步原始样本及pass均值。\n")
    yield ("连续LRU后续使用每16步CUDA Graph录制并仅回放一次，每步仍用external CUDA event记录；"
           "录制本身不执行kernel，不改变LRU，整段GPU工作一次提交，因此没有CPU逐步入队间隙。"
           "不含capture/graph launch主机时间，仍保留逐步事件、miss及graph输出正确性锚点。"
           "与原guard计时的完整窗口交叉检查：100GB/s＋20µs、MTP8单请求，本文差−0.16%，"
           "Memcpy差−1.75%；两份数据均留档。早期5个完整LRU校验格中4格仍用已通过guard的计时，"
           "原始JSON逐段记录timing_queue，报告不把两种计时方法隐去。\n")
    yield ("可复用的Graph在不同pass重新回放；每次pass开始都清空本文LRU，Memcpy按同一离线冷启动规划"
           "逐步重新装入数据。Graph只复用操作描述，不复用上一pass的LRU热缓存。\n")
    yield ("Memcpy baseline维护真实缓存KV和有效索引：离线CPU计算精确逐launch LRU、发出未命中搬运，"
           "持久槽保存命中页，放不进缓存的本步页写临时槽；attention在正确命中/未命中数据上执行，"
           "采样query行与数学参考比较。GPU gather/H2D、写缓存槽及attention均计时；CPU LRU规划和"
           "PCIe CPU gather不计时，后者另保存耗时，所以它是GPU部分对照，不是完整CPU+GPU延迟。"
           "没有沿用此前只搬未命中却缺完整缓存结果的计时替身。\n")
    yield ("大batch的Host源数据按段打包，避免为完整窗口一次保留上百GB的pinned内存。"
           "每步拥有独立源缓冲，并在该段GPU完成后才释放；CPU打包和索引上传仍在事件外，"
           "实际H2D、缓存scatter与attention仍全部计时。每份新数据记录其预打包内存上限。\n")
    yield ("无LRU Memcpy使用连续的本地staging，直接H2D/gather写入该区域，不额外scatter。"
           "修正前已通过的本文样本原样保留，只重测对应Memcpy，前后源码SHA与旧Memcpy样本都记录；"
           "主表只接受memcpy_path_version=2的对照。LRU因持久槽不连续，保留必要的scatter。\n")
    yield ("纯本地baseline正常构建、每个选中KV均预先放在本地且输出正确，数据准备在计时外，"
           "只计attention。主表用算术平均；另外保留同形状最快实测样本作乐观计算下限。"
           "纯本地是零远端传输参考，不是数学证明的绝对最快kernel；所有比值原样保留，不强行截断到≥1。\n")
    yield ("本地参考将每一步所需的远端页去重、紧凑放置在本地staging，再用相同query和完整selected KV计算。"
           "因此它是可容纳于40GB设备的全本地工作集参考，不要求把所有请求的整个64K历史同时放入本地。"
           "对应no-LRU比较逐个Host比例使用完全相同映射的四个位置本地样本，另有273个本地对照格；"
           "LRU比较取完整1024位置的本地均值。\n")
    yield ("如何解释距离本地极限：若每步仍需搬运B字节，带宽W GB/s，则仅传输理想时间为"
           "B/(1000W) µs；以max(本地计算实测,理想传输)作为理想重叠参考。该参考中的本地时间来自"
           "指定attention实现，不能视为所有可能kernel的数学下界。"
           "因此远端流量较高且链路只有50GB/s时，本文/本地显著大于1并不全是调度浪费。"
           "matrix.csv/json额外保存这一乐观参考，不计附加延迟且不用于替代实测。\n")
    yield ("baseline的离线LRU规划另通过独立Python oracle，覆盖逐步淘汰、同stamp并列、MTP内重复、"
           "容量溢出与实际payload映射，结果为planner_check.json。计时保护区和诊断读取不计入kernel事件；"
           "CPU标量写GPU造成的隐式同步已在正式采样前排除。失败和预实验保留在failures/，不进入汇报表。\n")
    yield "## 4. 不开 LRU：全组合结果\n"
    yield "时间单位µs。本文÷本地越接近1越好，本文÷Memcpy小于1代表本文更快。"
    for setting in META["settings"]:
        name=setting["name"]
        yield f"### {setting_label(setting)}（{name}）\n"
        for mtp in (1,8):
            sub=[r for r in rows if r["kind"]=="nocache" and r["setting"]==name and r["mtp"]==mtp]
            yield f"#### MTP{mtp}\n"
            yield "| 请求 | query行 | 目标Host% | 实际Host% | 本文µs | Memcpyµs | 纯本地µs | 本文÷Memcpy | 本文÷本地 |"
            yield "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            for r in sorted(sub,key=lambda x:(x["requests"],x["host_fraction_target"])):
                yield (f"| {r['requests']} | {r['query_rows']} | {r['host_fraction_target']*100:.0f} | "
                       f"{r['host_fraction_actual']*100:.2f} | {r['ours_mean_us']:.2f} | {r['memcpy_mean_us']:.2f} | "
                       f"{r['local_mean_us']:.2f} | {r['ours_to_memcpy']:.3f} | {r['ours_to_local_mean']:.3f} |")
            yield ""
    yield "## 5. 真实 trace＋LRU：完整窗口平均\n"
    yield ("缓存命中率按去重远端工作集：1−未命中unique页/远端unique页；按3次pass未命中数的均值统计。最终远端%为"
           "未命中unique页/全部top-k条目。引用命中率=1−miss引用/原始Host引用；miss引用%为"
           "miss引用/全部top-k条目，搬运unique%再扣除同launch重复。各分母不同，不能混用。Memcpy的LRU相同但同stamp淘汰"
           "并列顺序可不同，分别列出其缓存命中率。\n")
    for setting in META["settings"]:
        name=setting["name"];yield f"### {setting_label(setting)}（{name}）\n"
        yield "| MTP | 请求 | query行 | 本文µs | Memcpyµs | 纯本地µs | 本文÷Memcpy | 本文÷本地 | 原始Host% | 本文unique命中% | 本文引用命中% | Memcpy unique命中% | miss引用% | 搬运unique% | 未命中页/步 |"
        yield "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        sub=[r for r in rows if r["kind"]=="lru" and r["setting"]==name]
        for r in sorted(sub,key=lambda x:(x["mtp"],x["requests"])):
            yield (f"| {r['mtp']} | {r['requests']} | {r['query_rows']} | {r['ours_mean_us']:.2f} | "
                   f"{r['memcpy_mean_us']:.2f} | {r['local_mean_us']:.2f} | {r['ours_to_memcpy']:.3f} | "
                   f"{r['ours_to_local_mean']:.3f} | {r['host_fraction_actual']*100:.2f} | {r['ours_unique_cache_hit_rate']*100:.2f} | "
                   f"{r['ours_reference_cache_hit_rate']*100:.2f} | "
                   f"{r['memcpy_unique_cache_hit_rate']*100:.2f} | {r['ours_remote_reference_fraction']*100:.2f} | "
                   f"{r['final_remote_unique_fraction']*100:.2f} | "
                   f"{r['ours_misses_per_step']:.1f} |")
        yield ""
    yield "## 6. 原始数据与复现\n"
    yield ("同名HTML为自包含离线版本，含完整正文、内嵌图和可筛选的实测表，支持导出筛选后的CSV。"
           "Markdown和HTML均标明完成度，不能将部分完成的LRU统计解读为全矩阵结论。\n")
    yield ("参数在 `temp/glm_trace/kernel_report_20261029/experiment.json`，完整平面表 `matrix.csv`/`matrix.json`，"
           "每个参数格及所有逐步样本在 `raw/*.json`；`progress.json`为任务状态，`run.log`为流水日志。"
           "`bench.py`运行单batch，`run.py`遍历全组合、跳过已完成文件，不覆写完成数据；`plans.cpp`"
           "生成Memcpy的真实缓存规划。源码SHA/输入绑定/远端物理驻留均写入原始文件。\n")
    yield "完整性要求：39个batch×17种链路=663个LRU格，另×7个Host比例=4641个无LRU格；本地另有39份完整回放＋273个按Host比例匹配的快照对照格。"


rows,files=gather()
atomic_write(DEST/"matrix.json",json.dumps(rows,separators=(",",":"))+"\n")
if rows:
    temp=DEST/"matrix.csv.new"
    with temp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(DEST/"matrix.csv")
atomic_write(DOC,"\n".join(document(rows,files))+"\n")
atomic_write(DEST/"files.json",json.dumps(files,indent=2)+"\n")
print(json.dumps(dict(completed_cells=len(rows),document=str(DOC),measured_files=len(files))))
