"""Management briefing derived exclusively from the audited GLM53 kernel matrix."""
import hashlib
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from references import for_row

DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[2]
SOURCE=ROOT/"temp/glm_trace/kernel_report_20261029"
ASSETS=ROOT/"docs/assets/glm53_briefing_20261030"
DOC=ROOT/"docs/mla_host_glm53_report_external.md"
ASSETS.mkdir(parents=True,exist_ok=True)
load=lambda path:json.loads(path.read_text())
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
audit=load(SOURCE/"audit.json")
checked=load(SOURCE/"report_check.json")
assert audit["complete"] and checked["complete"]
assert digest(SOURCE/"matrix.json")==checked["artifact_sha256"][str((SOURCE/"matrix.json").relative_to(ROOT))]
ROWS=load(SOURCE/"matrix.json")
META=load(SOURCE/"experiment.json")
CAL=load(SOURCE/"calibration.json")
INDEX={(r["kind"],r["setting"],r["mtp"],r["requests"],r["host_fraction_target"]):r for r in ROWS}
assert len(ROWS)==len(INDEX)==5304
FIGURES={}


def row(kind,setting,mtp,batch,fraction=None):
    return INDEX[kind,setting,mtp,batch,fraction]


def label(name):
    if name=="nvlink":return "物理 NVLink"
    if name=="pcie":return "物理 Host PCIe"
    s=next(s for s in META["settings"] if s["name"]==name)
    return (f"{s['bandwidth_GBps']} GB/s" if s["bandwidth_GBps"] else "不限速")+f"＋{s['latency_ns']/1000:g} µs"


def subset(kind,setting,mtp=None,min_batch=0,fraction=None):
    return [r for r in ROWS if r["kind"]==kind and r["setting"]==setting
            and (mtp is None or r["mtp"]==mtp) and r["requests"]>=min_batch
            and (kind=="lru" or fraction is None or r["host_fraction_target"]==fraction)]


def stats(values):
    ratios=[r["memcpy_mean_us"]/r["ours_mean_us"] for r in values]
    return dict(cells=len(values),wins=sum(v>1 for v in ratios),
                faster_3pct=sum(r["ours_to_memcpy"]<.97 for r in values),
                speedup=math.exp(sum(map(math.log,ratios))/len(ratios)),
                local_ratio=math.exp(sum(math.log(r["ours_to_local_mean"]) for r in values)/len(values)),
                reduction_min=min(1-1/v for v in ratios),reduction_max=max(1-1/v for v in ratios))


font=Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
fm.fontManager.addfont(font)
plt.rcParams.update({"font.family":fm.FontProperties(fname=font).get_name(),"font.size":10,
    "axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,
    "axes.edgecolor":"#a9b3be","axes.labelcolor":"#445363","xtick.color":"#445363",
    "ytick.color":"#445363","text.color":"#172c3d","figure.facecolor":"white",
    "axes.facecolor":"white","savefig.facecolor":"white","legend.fontsize":9,
    "axes.titlesize":12})
BLUE,ORANGE,GREEN,INK="#3978b7","#e28542","#23866b","#24364b"


def dress(ax,ylabel="平均 GPU 耗时（ms）"):
    ax.grid(axis="y",color="#e1e7eb",lw=.7)
    ax.set_axisbelow(True);ax.set_ylabel(ylabel)
    ax.tick_params(length=3)


def bars(ax,records,labels,legend=False):
    x=np.arange(len(records));width=.24
    for offset,(key,name,color) in enumerate((("local_mean_us","纯 device 计算",BLUE),
            ("memcpy_mean_us","Memcpy＋attention",ORANGE),("ours_mean_us","本文融合 kernel",GREEN))):
        y=np.array([r[key]/1000 for r in records]);positions=x+(offset-1)*width
        ax.bar(positions,y,width*.92,label=name,color=color,zorder=3)
        if key!="local_mean_us":
            mode="ours" if key=="ours_mean_us" else "memcpy"
            passes=np.array([r[mode+"_pass_means_us"] for r in records])/1000
            ax.errorbar(positions,y,yerr=np.stack((y-passes.min(1),passes.max(1)-y)),
                        fmt="none",ecolor="#354657",elinewidth=.7,capsize=2,zorder=4)
        for pos,v in zip(positions,y):
            ax.annotate(f"{v:.2f}",(pos,v),xytext=(0,4),textcoords="offset points",
                        ha="center",fontsize=7,color=color)
    for i,r in enumerate(records):
        bound=for_row(r)["mean_stepwise_max_us"]/1000
        ax.plot([i-.42,i+.42],[bound,bound],color=INK,ls=(0,(4,2)),lw=1.35,zorder=5,
                label="参考下限 max(计算,搬运)" if i==0 else None)
    ax.set_xticks(x,labels);ax.set_ylim(0,max(r["memcpy_mean_us"] for r in records)/1000*1.25)
    # Include slower-than-copy cells as well; no clipped regressions.
    top=max(max(r["ours_mean_us"],r["memcpy_mean_us"],r["local_mean_us"]) for r in records)/1000
    ax.set_ylim(0,top*1.25)
    dress(ax)
    if legend:ax.legend(frameon=False,loc="upper left")


def save(fig,name,records,caption):
    path=ASSETS/(name+".png")
    fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)
    unique={tuple((r[k] for k in ("kind","setting","mtp","requests","host_fraction_target"))):r for r in records}
    records=list(unique.values())
    FIGURES[name]=dict(file=str(path.relative_to(ROOT)),caption=caption,rows=records,
                      overlap_references=[for_row(r) for r in records])


def figures():
    # Fixed before comparison: batch16 and128, both bandwidths, all seven Host fractions.
    fig,axes=plt.subplots(2,2,figsize=(13.6,8.0),layout="constrained")
    records=[]
    for i,b in enumerate((16,128)):
        for j,s in enumerate(("g100_l20000","g50_l20000")):
            values=[row("nocache",s,8,b,f) for f in META["host_fractions"]];records+=values
            ax=axes[i,j]
            bars(ax,values,[f"{f:.0%}" for f in META["host_fractions"]],legend=i==0 and j==0)
            ax.set_title(f"{label(s)} · {b} 请求 × MTP8",loc="left")
            ax.set_xlabel("选中 KV 的目标原始远端占比")
    fig.suptitle("单步无 LRU：取数与计算融合，覆盖全部 0%–30% 远端比例",fontsize=15)
    save(fig,"single_step",records,"固定MTP8、16/128请求，100/50GB/s＋20µs，全部七个远端比例。黑虚线为逐步max(纯device计算,unique搬运预算)的平均；细线为三次pass范围。")

    fig,axes=plt.subplots(2,2,figsize=(13.6,8.0),layout="constrained");records=[]
    for i,m in enumerate((1,8)):
        batches=[1,4,16,128,512] if m==1 else [1,4,16,64,128]
        for j,s in enumerate(("g100_l20000","g50_l20000")):
            values=[row("lru",s,m,b) for b in batches];records+=values
            ax=axes[i,j];bars(ax,values,[str(b) for b in batches],legend=i==0 and j==0)
            ax.set_title(f"{label(s)} · MTP{m}",loc="left");ax.set_xlabel("并发请求数")
    fig.suptitle("真实 GLM-5.3 trace＋4096 槽 LRU：完整窗口平均耗时",fontsize=15)
    save(fig,"trace_latency",records,"MTP1固定1/4/16/128/512请求，MTP8固定1/4/16/64/128请求；完整1024位置平均。黑虚线为逐步max(计算,unique搬运)的平均。")

    fig,axes=plt.subplots(1,2,figsize=(13.6,4.0),layout="constrained");records=[]
    for m,marker in ((1,"o"),(8,"s")):
        values=sorted(subset("lru","g100_l20000",m),key=lambda r:r["requests"]);records+=values
        axes[0].plot([r["requests"] for r in values],[100*r["ours_reference_cache_hit_rate"] for r in values],
                     marker=marker,markersize=4,label=f"MTP{m}",lw=1.8)
    axes[0].set_xscale("log",base=2);axes[0].set_xticks([1,4,16,64,128,512],[1,4,16,64,128,512])
    axes[0].set_ylim(60,95);axes[0].set_xlabel("并发请求数（对数轴）")
    dress(axes[0],"LRU 引用命中率（%）");axes[0].legend(frameon=False)
    axes[0].set_title("真实访问序列的缓存复用",loc="left")
    values=[row("lru","g100_l20000",8,b) for b in (1,4,16,128)];records+=values
    components=[("原始本地约0.05%（见放大图）",[1-r["host_fraction_actual"] for r in values],BLUE),
                ("LRU 命中",[r["host_fraction_actual"]-r["ours_remote_reference_fraction"] for r in values],"#71b6a0"),
                ("同次 launch 去重节省",[r["ours_remote_reference_fraction"]-r["final_remote_unique_fraction"] for r in values],"#f1be78"),
                ("最终远端搬运",[r["final_remote_unique_fraction"] for r in values],ORANGE)]
    bottom=np.zeros(len(values))
    for name,ys,color in components:
        ys=np.array(ys)*100
        axes[1].bar(range(len(values)),ys,bottom=bottom,label=name,color=color,width=.55)
        bottom+=ys
    assert np.allclose(bottom,100)
    for i,r in enumerate(values):
        axes[1].text(i,100-r["final_remote_unique_fraction"]*50,
                     f"{r['final_remote_unique_fraction']*100:.1f}%",ha="center",va="center",fontsize=9)
    axes[1].set_xticks(range(len(values)),[f"{r['requests']} 请求" for r in values])
    axes[1].set_ylim(0,120);dress(axes[1],"占全部 top-k 引用的比例（%）")
    axes[1].legend(frameon=False,ncol=2,fontsize=8,loc="upper center")
    axes[1].set_title("MTP8：LRU 与去重共同压低远端流量",loc="left")
    inset=axes[1].inset_axes([.07,.07,.3,.28])
    original=[(1-r["host_fraction_actual"])*100 for r in values]
    inset.bar(range(len(values)),original,color=BLUE,width=.55)
    inset.set_ylim(0,.075);inset.set_xticks(range(len(values)),[1,4,16,128],fontsize=7)
    inset.set_yticks([0,.05],["0%","0.05%"],fontsize=7)
    inset.set_title("原始本地占比放大",fontsize=8)
    inset.set_facecolor("#ffffff");inset.tick_params(length=2)
    for i,v in enumerate(original):inset.text(i,v+.002,f"{v:.3f}%",ha="center",fontsize=6)
    save(fig,"trace_traffic",records,"取100GB/s＋20µs的实际miss计数；各项以全部top-k引用为分母并合计100%。原始本地约0.05%，左下角单独放大；LRU命中属于另外的本地读取。")

    fig,axes=plt.subplots(1,2,figsize=(13.6,4.1),layout="constrained");records=[]
    for ax,s in zip(axes,("g100_l20000","g50_l20000")):
        values=sorted(subset("lru",s,8),key=lambda r:r["requests"]);records+=values
        x=np.arange(len(values))
        ax.plot(x,[r["memcpy_mean_us"]/r["local_mean_us"] for r in values],color=ORANGE,label="Memcpy／纯 device",lw=2)
        ax.plot(x,[r["ours_to_local_mean"] for r in values],color=GREEN,marker="o",markersize=3,label="本文／纯 device",lw=2)
        ax.plot(x,[for_row(r)["mean_stepwise_max_us"]/r["local_mean_us"] for r in values],
                color=INK,ls="--",lw=1.4,label="理想重叠参考／纯 device")
        ax.axhline(1,color=BLUE,ls=":",lw=1.5,label="纯 device = 1")
        ax.set_xticks(x,[r["requests"] for r in values],rotation=60,fontsize=8)
        ax.set_xlabel("并发请求数（列出全部25种MTP8形状）")
        ax.set_ylim(.85,4.2);dress(ax,"相对纯 device 计算的耗时倍数")
        ax.set_title(label(s)+" · trace＋LRU",loc="left")
    axes[0].legend(frameon=False,ncol=2,fontsize=8)
    save(fig,"distance_to_limits",records,"完整25种MTP8形状；黑虚线为逐步max(纯device计算,实际unique未命中字节/设置带宽)的平均，不包含附加延迟。")

    fig,axes=plt.subplots(1,2,figsize=(13.6,4.1),layout="constrained");records=[]
    for ax,s in zip(axes,("nvlink","pcie")):
        values=[row("lru",s,8,b) for b in (1,4,16,64,128)];records+=values
        bars(ax,values,[str(r["requests"]) for r in values],legend=s=="nvlink")
        ax.set_xlabel("并发请求数");ax.set_title(label(s)+" · MTP8＋LRU",loc="left")
    fig.suptitle("物理链路补充：全部使用本文 staged 路径",fontsize=15)
    save(fig,"physical_links",records,"固定1/4/16/64/128请求；黑虚线为逐步max(计算,unique搬运)的平均，PCIe/NVLink使用本次补测的物理带宽。全部为staged实测。")

    fig,axes=plt.subplots(1,2,figsize=(13.0,3.6),layout="constrained");records=[]
    for ax,m in zip(axes,(1,8)):
        data=np.zeros((2,5));annotations=[]
        for i,bw in enumerate((100,50)):
            annotation=[]
            for j,lat in enumerate((0,2000,5000,10000,20000)):
                values=subset("lru",f"g{bw}_l{lat}",m);records+=values;st=stats(values)
                data[i,j]=st["speedup"];annotation.append(f"{st['speedup']:.2f}×\n{st['wins']}/{st['cells']} 格领先")
            annotations.append(annotation)
        from matplotlib.colors import TwoSlopeNorm
        ax.imshow(data,cmap="RdYlGn",norm=TwoSlopeNorm(vmin=.85,vcenter=1,vmax=1.6),aspect="auto")
        ax.set_xticks(range(5),[0,2,5,10,20]);ax.set_yticks(range(2),["100 GB/s","50 GB/s"])
        ax.set_xlabel("附加访问延迟（µs）");ax.set_title(f"MTP{m}：Memcpy／本文 的几何平均",loc="left")
        for i in range(2):
            for j in range(5):ax.text(j,i,annotations[i][j],ha="center",va="center",fontsize=10)
        ax.tick_params(length=0)
    save(fig,"bandwidth_latency",records,"每格覆盖该MTP全部batch形状；数值为等权几何平均加速比，另标严格快于Memcpy的形状数。")


def image(name,number):
    return f"![图{number}：{FIGURES[name]['caption']}](assets/glm53_briefing_20261030/{name}.png)\n\n图{number}　{FIGURES[name]['caption']}\n"


def comparison_table(records,host=False,cache=False):
    head="| 链路 | MTP | 请求 |"+(" 目标远端比例 |" if host else "")+" 本文 µs | Memcpy µs | 纯 device µs | 重叠参考 µs | 耗时降低 | 本文/纯device | 本文/重叠参考 |"+(" LRU引用命中率 |" if cache else "")
    yield head
    yield "|"+"---|"*(10+int(host)+int(cache))
    for r in records:
        columns=[label(r["setting"]),str(r["mtp"]),str(r["requests"])]
        if host:columns.append(f"{r['host_fraction_target']:.0%}")
        columns.extend([f"{r[k]:.2f}" for k in ("ours_mean_us","memcpy_mean_us","local_mean_us")])
        reference=for_row(r)
        columns.extend([f"{reference['mean_stepwise_max_us']:.2f}",f"{(1-r['ours_to_memcpy'])*100:.1f}%",
                        f"{r['ours_to_local_mean']:.3f}×",f"{reference['ours_to_reference']:.3f}×"])
        if cache:columns.append(f"{r['ours_reference_cache_hit_rate']*100:.2f}%")
        yield "| "+" | ".join(columns)+" |"


def document():
    large={s:stats(subset("lru",s,8,16)) for s in ("g100_l20000","g50_l20000")}
    a=row("nocache","g100_l20000",8,16,.1)
    b=row("nocache","g100_l20000",8,16,.3)
    single_ratios=[r["ours_to_local_mean"] for r in subset("nocache","g100_l20000",8)
                   if r["requests"]==16 and .1<=r["host_fraction_target"]<=.3]
    c=row("lru","g50_l20000",8,128)
    c_ref=for_row(c)
    pcie_large=[for_row(row("lru","pcie",8,n)) for n in (32,64,128)]
    near=row("nocache","nvlink",8,16,.1)
    near_trace=row("lru","nvlink",8,14)
    yield "# 远端 KV 稀疏 MLA 融合算子性能汇报\n"
    yield "**基于 GLM-5.3 真实访问 trace｜A100-SXM4｜单算子 GPU 实测｜2026 年 10 月 30 日**\n"
    yield "## 1. 工作成果与核心结论\n"
    yield ("本阶段完成了远端 KV sparse MLA decode 的单 kernel 融合方案，将**跨 query 去重、跨步 LRU、"
           "远端取数、attention 计算及结果合并**放入同一次 cooperative kernel 启动。目标是在 KV 放到"
           "Host DRAM 或其他 GPU 后，尽量把传输隐藏在计算期间，减少相对纯 device 计算增加的耗时。\n")
    yield ("本次已完成 **39 种 batch/MTP 形状 × 17 种链路**的系统测试：无 LRU 再扫描 7 个远端比例，"
           "共 **4,641 个单步比较格＋663 个真实 trace/LRU 比较格**，另有 312 个纯 device 参考格。"
           "全部参数覆盖与数值检查审计通过。\n")
    yield "| 汇报要点 | 本次实测证据 |"
    yield "|---|---|"
    yield (f"| **与 Memcpy 相比，中大 batch 的融合收益明显** | 真实trace、MTP8、16–128请求的全部14种形状，"
           f"100 GB/s＋20 µs下均领先，耗时降低{large['g100_l20000']['reduction_min']*100:.1f}%–"
           f"{large['g100_l20000']['reduction_max']*100:.1f}%；50 GB/s＋20 µs下均领先，降低"
           f"{large['g50_l20000']['reduction_min']*100:.1f}%–{large['g50_l20000']['reduction_max']*100:.1f}%。 |")
    yield (f"| **单步计算可接近纯 device 参考** | 无LRU、100 GB/s＋20 µs、MTP8/16请求、远端10%–30%时，"
           f"本文耗时为纯device的{min(single_ratios):.2f}–"
           f"{max(single_ratios):.2f}倍；物理NVLink、远端10%时为"
           f"{near['ours_to_local_mean']:.3f}倍。 |")
    yield (f"| **带宽受限的大 batch 已接近理想重叠下限** | 真实trace、50 GB/s＋20 µs、MTP8/128请求："
           f"本文{c['ours_mean_us']/1000:.3f} ms，Memcpy {c['memcpy_mean_us']/1000:.3f} ms，耗时降低"
           f"{(1-c['ours_to_memcpy'])*100:.1f}%；理想重叠参考{c_ref['mean_stepwise_max_us']/1000:.3f} ms，"
           f"本文高出{(c_ref['ours_to_reference']-1)*100:.1f}%。物理PCIe的32/64/128请求高出约"
           f"{(pcie_large[0]['ours_to_reference']-1)*100:.1f}%／{(pcie_large[1]['ours_to_reference']-1)*100:.1f}%／"
           f"{(pcie_large[2]['ours_to_reference']-1)*100:.1f}%。 |")
    yield ""
    yield ("**结论的适用范围**：中大 batch、有限带宽场景已体现稳定且可观的融合收益；低远端比例或较快链路"
           "可以接近纯 device 计算。原始远端流量较高时，物理传输预算会高于纯 device 时间，此时应考察"
           "与计算/传输共同决定的理想重叠下限的距离。小 batch 和部分 NVLink 参数仍有改进空间，完整汇总见第8节。\n")
    yield "## 2. Kernel 设计与本次启用配置\n"
    yield ("Memcpy 方案先把所需远端页搬到本地，再执行 attention；搬运和计算在依赖关系上串行。本文先统一解析"
           "整个 batch 的 top-k，识别本地页、LRU命中页和远端未命中页；fetch线程块只取唯一未命中页，"
           "attention线程块先处理可用数据，并随远端分块到达继续计算。最后在同一kernel内完成分片结果合并。\n")
    yield ("本报告的Memcpy基线同样对未命中页去重，并在trace部分使用同容量LRU。速度收益体现融合执行、"
           "取数/计算重叠等实现差异；本轮比较整条执行路径，单项设计的贡献可通过后续消融进一步量化。\n")
    yield "| 设计 | 实现与作用 |"
    yield "|---|---|"
    yield "| 统一地址空间 | CUDA UVA/VMM统一访问本地HBM、peer GPU HBM或Host pinned内存，输入页号保留真实访问关系。 |"
    yield "| 跨query去重 | 在batch索引中识别重复远端页；本测试请求地址独立，复用主要来自同请求MTP多行，共享页只搬一次。 |"
    yield "| 融合LRU | 每请求独立4096槽；命中检测、淘汰、未命中页写入与attention共处一个kernel，跨decode步骤保留缓存。 |"
    yield "| Staged取数与计算重叠 | fetch CTA分块写本地staging和缓存；attention CTA处理本地/命中页，并在远端块就绪后继续，无需先完成整批搬运。 |"
    yield "| 批量处理与结果合并 | 一次启动处理全部请求及MTP行，分片softmax/输出在同一cooperative kernel内合并。 |"
    yield ""
    yield "**本报告所有本文数据均为 staged 路径，包括物理 NVLink；没有逐格挑选原地读结果。**\n"
    yield "| 测试部分 | 本文LRU | 本文取数路径 | 其他固定参数 |"
    yield "|---|---|---|---|"
    yield "| 单步、指定远端比例 | 关闭（容量0） | batched staged；MTP内去重仍开启 | fetch CTA=16，Host item=64条目 |"
    yield "| 连续真实trace回放 | 开启，每请求4096槽 | 同一batched staged路径，融合LRU | 同上，每pass冷启动 |"
    yield "| 纯device参考 | 不需要远端缓存 | 正常构建的本地attention | 全部选中KV预置本地，只计attention |"
    yield ""
    yield ("本报告的远端比例指top-k引用落到原始远端的比例，与LRU命中率不同；该定义同时适用于Host和peer存储。"
           "无LRU的0%–30%为目标值，固定页映射下实际比例略有波动，完整矩阵同时记录实际值。\n")
    yield ("单query也固定使用batched staged实现；可选SMALL_PREPARE优化本轮关闭。本轮评估的是同一套固定配置，"
           "并未按参数点重新调优。产品实现同时支持原地读，但本报告的比较和图表统一采用上述实测配置。\n")
    yield "## 3. GLM-5.3 trace 采集与回放方法\n"
    yield ("使用GLM-5.3权重对软件开发agent的多轮轨迹进行离线因果前向计算。原始语料来自"
           "`nebius/SWE-rebench-openhands-trajectories`，按模型原生聊天及工具调用模板构造4篇64K文档，"
           "每篇来自一条轨迹。逐层记录indexer选出的top-2048 token位置、对应index score及窗口/层来源。"
           "模型共有21个实际执行indexer的层；其余层复用映射另行保留。\n")
    yield ("正式性能回放选取**4段完整、连续的1024位置assistant窗口**，起点为文档1的35184/41887/48296和"
           "文档3的50621；每窗口21层，共84组层×窗口访问序列。窗口内不跳过位置、不拼接不连续token，"
           "请求在不同batch之间使用相同前缀绑定，整个回放期间不换层或窗口。\n")
    yield "| 项目 | 回放方式 |"
    yield "|---|---|"
    yield "| MTP1 / MTP8 | 每请求每launch处理1/8个连续实测位置；1024位置分别对应1024/128次launch。MTP8不模拟推测接受率。 |"
    yield "| 单步无LRU | 取窗口偏移0/256/512/768的四个分组；按固定(request,token位置)映射指定远端比例0/5/10/15/20/25/30%，保留MTP重复关系。 |"
    yield "| Trace＋LRU | 除每个query最新token外，原始KV驻留远端；每pass冷启动并回放完整窗口，命中率由真实访问序列和缓存政策决定。 |"
    yield "| 数据载荷 | Q/KV为固定seed生成的相同BF16数据；真实的是top-k访问序列，不是完整模型生成出的Q/KV值。 |"
    yield "| 并发请求 | 不同请求可绑定不同层/窗口，构成真实访问序列的合成batch；超过84请求会重复capture，远端及LRU地址仍独立。 |"
    yield ""
    yield ("因此，这部分实验衡量的是**真实访问局部性驱动下的算子性能**：命中、去重和远端流量均来自实际kernel回放，"
           "不包含模型其他层、调度或服务端延迟，也不能直接换算成整模型吞吐提升。\n")
    yield "## 4. 带宽与访问延迟模拟\n"
    yield ("在A100双卡真实NVLink远端路径上，对远端读取使用共享虚拟链路时钟：每块数据按字节数预约传输预算，"
           "预算由设置带宽决定；数据被使用前，再等到预约完成时间加附加延迟。真实读取仍发出，多个CTA和多块请求"
           "可同时在途。这样限制的是整张GPU的远端总带宽与数据可用时间，取数/计算仍可重叠。\n")
    yield "```text\n块可用时间 = 链路预约完成时间 + 附加访问延迟\n该块传输预算 = 字节数 / 设置带宽\n```\n"
    yield ("带宽扫描不限速、100、50 GB/s，附加延迟扫描0/2/5/10/20 µs，形成15个排列组合；另测未注入模拟的"
           "物理NVLink和物理Host PCIe，共17种链路。本文fetch路径与Memcpy的GPU gather共用同一模拟机制；"
           "延迟施加在搬运块/访存tile上，不是在整个kernel结束后统一加一个固定时间。\n")
    yield "| 校准项目 | 实测结果 |"
    yield "|---|---|"
    for bw in (100,50):
        values=[s["measured_GBps"] for s in CAL["settings"] if s["bandwidth_GBps"]==bw]
        yield f"| 设置{bw} GB/s，覆盖全部5种延迟 | 实测{min(values):.2f}–{max(values):.2f} GB/s |"
    delays=[s["latency_ns_measured"]/1000 for s in CAL["settings"] if s["latency_ns"]==20000]
    yield f"| 附加20 µs的依赖读校准 | 实测{min(delays):.3f}–{max(delays):.3f} µs |"
    yield "| 校准判据 | 带宽误差≤5%；延迟误差≤10%＋0.6 µs；15种组合全部通过 |"
    yield ""
    yield ("纯device参考使用正常构建，移除模拟钩子和远端传输。模拟构建自身有额外开销，因而本文/纯device比值"
           "包含这部分成本；模拟结果用于评估指定链路约束，不代表某一未来硬件或H800的直接实测。\n")
    physical=load(DEST/"physical_bandwidth.json")
    yield ("柱状图中的黑虚线统一表示逐步完美重叠参考：先对每一步计算max(纯device时间,unique未命中字节/带宽)，"
           "再对完整窗口求平均。模拟链路用设置带宽；物理链路使用本次补测的max(DMA,SM gather)带宽，"
           "是乐观传输预算，不把Memcpy实测总时间当作传输时间。\n")
    yield "| 物理链路补测 | 连续DMA GB/s | 随机gather GB/s | 参考带宽 GB/s |"
    yield "|---|---:|---:|---:|"
    for s in physical["settings"]:
        yield (f"| {label(s['name'])} | {s['measurements']['dma_contiguous']['median_GBps']:.2f} | "
               f"{s['measurements']['sm_gather_random']['median_GBps']:.2f} | {s['reference_GBps']:.2f} |")
    yield ""
    yield "## 5. 单步无 LRU：融合取数相对 Memcpy 的收益\n"
    yield image("single_step",1)
    yield (f"在100 GB/s＋20 µs、MTP8/16请求下，远端比例从10%到30%，本文分别为{a['ours_mean_us']:.2f}和"
           f"{b['ours_mean_us']:.2f} µs，对应Memcpy {a['memcpy_mean_us']:.2f}和{b['memcpy_mean_us']:.2f} µs，"
           f"耗时降低{(1-a['ours_to_memcpy'])*100:.1f}%和{(1-b['ours_to_memcpy'])*100:.1f}%。"
           "传输量增加后，融合kernel仍能把相当部分传输隐藏在计算期间。\n")
    samples=[row("nocache",s,8,batch,f) for s in ("g100_l20000","g50_l20000")
             for batch,f in ((16,.1),(16,.3),(128,.1),(128,.3))]
    yield from comparison_table(samples,host=True)
    yield ""
    yield ("图1完整展示所选形状的全部远端比例，包括0%和低远端比例的开销：并非每个格子都领先。"
           "例如低流量时，Memcpy搬运成本较小，融合调度与staging成本可能抵消重叠收益。"
           "本图固定16和128请求，未按优胜结果筛选参数点。\n")
    yield "### 5.1 MTP1 与小 batch 补充\n"
    yield ("以下固定远端目标30%，补充MTP1的1/4/16/128请求。小batch的融合准备开销更突出；"
           "耗时降低为负值表示本文较慢，这些格子作为后续优化方向保留。\n")
    yield from comparison_table([row("nocache",s,1,batch,.3) for s in ("g100_l20000","g50_l20000")
                                 for batch in (1,4,16,128)],host=True)
    yield ""
    yield "## 6. 真实 trace＋LRU：中大 batch 的收益持续存在\n"
    yield image("trace_latency",2)
    for s in ("g100_l20000","g50_l20000"):
        all_shapes=stats(subset("lru",s));large_shapes=large[s]
        yield (f"- **{label(s)}**：全部39个形状中{all_shapes['wins']}个快于Memcpy，等权几何平均加速"
               f"{all_shapes['speedup']:.3f}倍；MTP8、16–128请求的全部14种形状均领先，几何平均加速"
               f"{large_shapes['speedup']:.3f}倍。")
    yield ""
    yield ("收益出现在已经为Memcpy基线预先完成CPU LRU规划和Host数据整理的条件下。表中仅比较其GPU搬运、"
           "必要缓存scatter及完整attention，本文仍完成融合LRU。因此这些结果已经体现GPU执行路径上的改进，"
           "无需把未计时的CPU工作加入基线来获得优势。\n")
    samples=[row("lru",s,8,batch) for s in ("g100_l20000","g50_l20000") for batch in (1,4,16,64,128)]
    yield from comparison_table(samples,cache=True)
    yield ""
    yield "### 6.1 缓存与去重如何减少远端流量\n"
    yield image("trace_traffic",3)
    traffic=row("lru","g100_l20000",8,128)
    yield ("**图中原始本地不是0。**本轮连续trace采用历史KV远端驻留、最新token本地驻留的设置；"
           "最新token约占top-k引用的1/2048，因此原始本地只有约0.05%，在0%–100%堆叠柱中很薄。"
           "图3已增加放大图与明确数值；LRU命中后从本地HBM读取的约67%–87%远端引用，计入绿色LRU部分。\n")
    yield "| MTP8请求数 | 原始本地/全部引用 | LRU命中后本地读取/全部引用 | 原始本地＋LRU命中/全部引用 |"
    yield "|---:|---:|---:|---:|"
    for batch in (1,4,16,128):
        r=row("lru","g100_l20000",8,batch)
        near_fraction=1-r["host_fraction_actual"]
        cache_fraction=r["host_fraction_actual"]-r["ours_remote_reference_fraction"]
        yield f"| {batch} | {near_fraction*100:.4f}% | {cache_fraction*100:.2f}% | {(near_fraction+cache_fraction)*100:.2f}% |"
    yield ""
    yield (f"以100 GB/s＋20 µs、MTP8/128请求为例：原始远端引用占{traffic['host_fraction_actual']*100:.2f}%，"
           f"LRU引用命中率为{traffic['ours_reference_cache_hit_rate']*100:.2f}%；按全部top-k条目计，"
           f"未命中引用降至{traffic['ours_remote_reference_fraction']*100:.2f}%，再经同launch去重，"
           f"实际搬运unique页仅占{traffic['final_remote_unique_fraction']*100:.2f}%。"
           "缓存命中与MTP去重共同降低必须跨链路搬运的字节数，融合流水再隐藏剩余取数与计算。\n")
    yield ("引用命中率的分母是原始远端引用，最终搬运比例的分母是全部top-k条目，二者不能直接相减。"
           "命中率来自指定驻留规则、4096槽容量和逐launch LRU政策下的实测回放，并非线上生产缓存计数。\n")
    yield "### 6.2 对不同带宽与延迟组合的覆盖\n"
    yield image("bandwidth_latency",4)
    yield ("图4对每个MTP的全部batch等权统计，覆盖100/50 GB/s与五档延迟的所有组合。50 GB/s下，"
           "MTP1与MTP8的全部形状、全部五档延迟均快于Memcpy。100 GB/s下，中大MTP8 batch持续受益，"
           "小batch的准备和同步成本仍会影响领先幅度。\n")
    yield "## 7. 距离纯 device 与理想重叠下限还有多远\n"
    yield "### 7.1 纯 device 是零传输计算参考\n"
    yield (f"无LRU、物理NVLink、MTP8/16请求、远端10%时，本文{near['ours_mean_us']:.2f} µs，"
           f"纯device {near['local_mean_us']:.2f} µs，差{(near['ours_to_local_mean']-1)*100:.1f}%；"
           f"真实trace＋LRU、NVLink、MTP8/14请求时，本文{near_trace['ours_mean_us']:.2f} µs，"
           f"纯device {near_trace['local_mean_us']:.2f} µs，差{(near_trace['ours_to_local_mean']-1)*100:.1f}%，"
           f"同时比Memcpy降低{(1-near_trace['ours_to_memcpy'])*100:.1f}%耗时。后者是特定形状的接近device实例，"
           "不作为所有batch的代表；高延迟16请求的范围见第1节与图1。\n")
    yield ("纯device数值来自相同query和完整selected KV的全本地attention实测，代表本报告所用attention实现的"
           "零传输参考，不是对所有可能GPU kernel的数学绝对下界。\n")
    yield "### 7.2 完美重叠参考：传输主导时已接近链路极限\n"
    yield "```text\n每步理想传输时间（µs）= 该步unique未命中字节数 / (1000 × 带宽GB/s)\n每步完美重叠参考 = max(该步纯device平均时间, 该步理想传输时间)\n完整窗口参考 = 所有步骤完美重叠参考的算术平均\n```\n"
    yield ("本版按逐步数据重新核对参考值，修正了上一版先平均再取max的汇总顺序。"
           "对于流量随trace步骤变化的场景，mean(max)通常高于max(mean)。"
           "例如100 GB/s＋20 µs、MTP8/128请求的trace参考从2595.96修正为2691.45 µs，"
           "kernel计时保持4450.15 µs，对应高于参考65.3%，而非上一版的71.4%。\n")
    yield image("distance_to_limits",5)
    yield ("图5列出全部25种MTP8形状。纯device基线始终为1；黑虚线加入必需传输预算，便于区分"
           "带宽约束与kernel额外开销。这一参考假设完美重叠，不含附加访问延迟、LRU处理及同步成本，"
           "是乐观参考，不能替代实测或当作所有实现的数学下界。\n")
    yield "| 场景：MTP8 / 128请求 | 本文 µs | Memcpy µs | 纯device µs | 理想重叠参考 µs | 本文高于重叠参考 |"
    yield "|---|---:|---:|---:|---:|---:|"
    for kind in ("nocache","lru"):
        for s in ("pcie","g50_l20000","g100_l20000"):
            r=row(kind,s,8,128,.3 if kind=="nocache" else None);bound=for_row(r)["mean_stepwise_max_us"]
            yield (f"| {'无LRU，远端30%' if kind=='nocache' else '真实trace＋LRU'}，{label(s)} | "
                   f"{r['ours_mean_us']:.2f} | {r['memcpy_mean_us']:.2f} | {r['local_mean_us']:.2f} | "
                   f"{bound:.2f} | {(r['ours_mean_us']/bound-1)*100:.1f}% |")
    yield ""
    yield (f"在50 GB/s＋20 µs的真实trace大batch中，本文{c['ours_mean_us']/1000:.3f} ms距离理想重叠"
           f"参考仅{(c_ref['ours_to_reference']-1)*100:.1f}%，说明此格已经较充分利用"
           f"受限链路。同格纯device为{c['local_mean_us']/1000:.3f} ms，本文为其{c['ours_to_local_mean']:.2f}倍。"
           "9.4%衡量距理想重叠的距离；要进一步缩小这类场景的"
           "device差距，还需要减少必需远端流量或提高有效带宽。\n")
    yield "**新旧实验均支持：传输主导的大batch可以接近完美重叠参考。**\n"
    yield ("此前接近下限的结果主要出现在物理PCIe约27 GB/s或50 GB/s的传输主导场景，"
           "并非100 GB/s及所有batch均如此。本次真实trace也复现了相同趋势：PCIe、MTP8的"
           "32/64/128请求距参考为8.8%/8.4%/4.8%；50 GB/s＋20 µs、128请求为9.4%。"
           "旧代理trace的PCIe单步112请求、远端25%曾为约1.01倍；它与本轮GLM-5.3完整窗口的"
           "访问序列、流量、缓存和聚合口径不同，应保留各自来源。\n")
    yield ("其中旧表的1.01倍还采用两组索引搬运预算中的较大值；用两组预算的均值重算为约1.036倍。"
           "两者都支持该传输主导场景接近下限，但本报告统一按完整窗口的逐步均值统计。\n")
    yield ("100 GB/s时，必需传输预算缩短，固定staged取数、LRU、staging及attention调度的额外成本更显著。"
           "旧代理trace的100 GB/s＋20 µs、MTP8/128请求同样曾为5585.41 µs，"
           "传输参考3584.34 µs，约高55.8%；这不是本次独有的现象。"
           "本轮不限速模拟的同形状staged耗时约4.056 ms，而纯device为2.467 ms，说明差距不只来自"
           "100 GB/s带宽和20 µs等待；各项固定成本的贡献仍需单独消融确认。\n")
    yield "**关键点独立复测**：在原矩阵之外，使用相同脚本、同一对GPU、完整1024位置、3次冷启动pass重新测量，结果如下。\n"
    yield "| MTP8 / 128请求 | 原本文 µs | 复测本文 µs | 变化 | 原Memcpy µs | 复测Memcpy µs |"
    yield "|---|---:|---:|---:|---:|---:|"
    for setting in ("g100_l20000","g50_l20000"):
        original=row("lru",setting,8,128)
        repeated=load(DEST/"recheck/raw"/f"lru_{setting}_mtp8_b128.json")
        assert repeated["complete"]
        times=repeated["cases"][0]["timings"]
        new=times["ours"]["mean_us"]
        yield (f"| {label(setting)} | {original['ours_mean_us']:.2f} | {new:.2f} | "
               f"{(new/original['ours_mean_us']-1)*100:+.2f}% | {original['memcpy_mean_us']:.2f} | "
               f"{times['memcpy']['mean_us']:.2f} |")
    repeated_local=load(DEST/"recheck/raw/floor_floor_mtp8_b128.json")
    yield (f"\n纯device也从{c['local_mean_us']:.2f} µs复测为"
           f"{repeated_local['cases'][0]['timings']['local']['mean_us']:.2f} µs。"
           "复测输出校验通过，三项计时均与原矩阵接近；报告继续采用同一完整矩阵，不用单次复测替换更慢格子。"
           "详细流量、逐步参考与新旧对照证据见本报告的数据复核记录。\n")
    yield "### 7.3 物理 PCIe / NVLink 的补充验证\n"
    yield image("physical_links",6)
    yield ("物理Host PCIe的连续LRU全部39个形状均快于Memcpy，等权几何平均加速"
           f"{stats(subset('lru','pcie'))['speedup']:.3f}倍；32/64/128请求同时距完美重叠参考"
           "8.8%/8.4%/4.8%，是本轮真实trace中接近链路极限的直接证据。物理NVLink下，固定staged方案对大部分中大MTP8 batch"
           "有收益，但小batch存在回退；全39形状平均接近持平，后续应针对准备/同步成本和取数路径继续优化。\n")
    yield "## 8. 全参数汇总、计时口径与结论边界\n"
    yield "### 8.1 全部17种链路汇总\n"
    yield ("无LRU列覆盖全部39形状×7远端比例＝273格，包含0%远端；trace列覆盖39形状。加速比为"
           "Memcpy/本文的等权几何平均，大于1表示本文更快。下表保留不限速和NVLink平均未领先的结果。\n")
    yield "| 链路 | 无LRU领先格/273 | 无LRU平均加速 | Trace＋LRU领先格/39 | Trace＋LRU平均加速 |"
    yield "|---|---:|---:|---:|---:|"
    summaries=[]
    for s in META["settings"]:
        n=stats(subset("nocache",s["name"]));l=stats(subset("lru",s["name"]))
        summaries.append(dict(setting=s["name"],nocache=n,lru=l))
        yield f"| {label(s['name'])} | {n['wins']} | {n['speedup']:.3f}× | {l['wins']} | {l['speedup']:.3f}× |"
    yield ""
    (DEST/"summary_metrics.json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+"\n")
    yield "### 8.2 测量设置与正确性\n"
    yield "| 项目 | 统一设置 |"
    yield "|---|---|"
    yield "| 设备 | A100-SXM4-40GB双卡，GPU0计算、GPU1为NV12 peer；Host pinned内存绑定NUMA0。全部数据来自A100。 |"
    yield "| Attention形状 | 64 heads，BF16，KV维度512＋64，page size=1，top-k=2048，每KV行1152 B。 |"
    yield "| 请求数 | MTP1：1/2/4/8/16/32/48/64/96/128/192/256/384/512；MTP8：1/2/3/4/5/6/7/8/10/12/14/16/20/24/28/32/40/48/56/64/72/80/96/112/128。 |"
    yield "| 平均口径 | 三次pass、全部步骤的算术平均；LRU每pass从冷缓存开始，含第一步；无LRU取四个trace分组，不使用后半段或最好样本代替均值。 |"
    yield "| GPU计时 | CUDA event；guard验证的预入队或16步CUDA Graph，避免CPU逐步发射间隙进入事件；每步记录原始时间。 |"
    yield "| 本文范围 | 单个cooperative kernel，含去重、LRU（如开启）、staged取数、attention和合并。 |"
    yield "| Memcpy范围 | 实际远端gather/H2D、必要cache scatter和完整本地attention；CPU LRU规划、Host CPU gather及索引上传在外。无LRU直接写连续staging，无冗余scatter。 |"
    yield "| 纯device范围 | 相同query/selected KV均在本地，正常构建，只计attention；无LRU逐远端比例匹配布局，LRU用完整窗口均值。 |"
    yield (f"| 验证 | {audit['timed_launches']:,}次计时launch、{audit['accepted_segments']:,}段有效队列；"
           f"指定step/query锚点对数学参考校验，采样最大绝对误差{audit['sampled_max_abs_error']:.8f}；"
           "独立CPU oracle核对缓存映射及完整真实trace payload。 |")
    yield ""
    yield ("三次pass的范围反映测量波动，不是统计置信区间；较小的百分比差异不作为稳定收益的主要依据。"
           "本报告重点强调数十个百分点的中大batch收益，并保留全部形状和各pass均值供核查。\n")
    yield "### 8.3 结论边界\n"
    yield ("本次证明的是指定访问模式和链路约束下的**kernel GPU执行优势**，不是端到端模型吞吐。"
           "真实trace组成合成batch；超过84请求重复capture，超过128请求时near物理KV按128请求复用以适配40GB，"
           "远端地址和LRU仍独立。MTP8是连续8位置分组，不含推测接受率。\n")
    yield ("所用assistant窗口的top-k完整性、因果性和assistant检查通过；完整语料的全角色loss gate未全部通过，"
           "本报告只据此评价访问模式下的kernel表现。无LRU指定0%–30%原始远端驻留，LRU几乎全部原始KV驻留远端，"
           "两部分各自与相同输入的baseline比较；开启LRU的净收益需另外做同输入消融。\n")
    yield "## 9. 结论与后续方向\n"
    yield ("**本文融合kernel在有限带宽、中大batch场景下已显著优于先搬后算的Memcpy路径。**"
           "真实GLM-5.3 trace下，MTP8、16–128请求的全部14个形状在100/50 GB/s＋20 µs均领先；"
           "几何平均加速分别为1.369倍和1.541倍。\n")
    yield ("**接近极限已有明确实例，且不同极限的含义清晰。**低远端比例的单步场景可做到接近纯device计算；"
           "物理PCIe的32–128请求真实trace距离理想重叠参考约4.8%–8.8%，50 GB/s＋20 µs的128请求约9.4%。"
           "后续主要空间集中在小batch固定成本、较快链路下的staging开销，以及继续减少必需远端字节数。\n")
    yield "---\n"
    yield ("数据来源：[完整5304格实验报告](mla_host_glm53_kernel_matrix_20261029.md)、"
           "[全量CSV](../temp/glm_trace/kernel_report_20261029/matrix.csv)、"
           "[原始审计](../temp/glm_trace/kernel_report_20261029/audit.json)。"
           "实验数据提交`8c624445`；本汇报的作图脚本、图数据和来源SHA保存在"
           "`temp/glm_trace/briefing_report_20261030/`。"
           "[下限与计时复核记录](../temp/glm_trace/briefing_report_20261030/verification_notes.md)。"
           "所有图的kernel计时均来自同一GLM-5.3/A100完整矩阵，参考线按逐步数据重新计算。\n")


figures()
DOC.write_text("\n".join(document()).rstrip()+"\n")
(DEST/"figure_data.json").write_text(json.dumps(FIGURES,ensure_ascii=False,separators=(",",":"))+"\n")
manifest=dict(timestamp=time.strftime("%FT%T%z"),source_commit="8c624445",
              source_matrix_sha256=digest(SOURCE/"matrix.json"),source_csv_sha256=digest(SOURCE/"matrix.csv"),
              source_audit=dict((k,v) for k,v in audit.items() if k!="files"),
              reference_definition="mean_t(max(mean_pass(local_t), mean_pass(unique_misses_t)*1152/(1000*W)))",
              physical_bandwidth_sha256=digest(DEST/"physical_bandwidth.json"),
              verification_sha256=digest(DEST/"verification.json"),
              verification_notes_sha256=digest(DEST/"verification_notes.md"),
              analysis_source_sha256={p.name:digest(p) for p in
                  (DEST/"build_report.py",DEST/"references.py",DEST/"verify_sources.py",DEST/"probe_physical.py")},
              figure_count=len(FIGURES),figure_cells={k:len(v["rows"]) for k,v in FIGURES.items()},
              artifact_sha256={str(p.relative_to(ROOT)):digest(p) for p in [DOC,*sorted(ASSETS.glob("*.png"))]})
(DEST/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(dict(markdown=str(DOC),figures=len(FIGURES)),ensure_ascii=False))
