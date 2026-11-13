# Purpose: Figures for docs/mla_host_lru_report.md from temp/a100_full (A100) and the H800
#          validation job: cross-decode LRU vs floor and baselines, single step across far share,
#          LRU gain, LRU capacity, HiSparse.
# Usage: MPLCONFIGDIR=temp/matplotlib python3 temp/report_lru/plot.py
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

CJK = None
for path in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if Path(path).exists():
        fm.fontManager.addfont(path)
        CJK = fm.FontProperties(fname=path).get_name()
        break

A100 = Path("temp/a100_full")
H800 = Path("temp/h800/results/fi-validate-697235")
OUT = Path("temp/report_lru")
BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SURFACE = "#fcfcfb"
INK, INK2, GRID, SPINE = "#0b0b0b", "#52514e", "#E6E9EC", "#ADB4BA"
ROW = 1152  # bytes per token per layer



def compute_floor(result):
    """Fastest all-local attention measured for this batch: the native kernel with every row
    remapped to HBM, or either kernel at 0% far (native's fixed KV split is not optimal at every
    shape, so taking only it would put the floor above what was measured)."""
    zero = next(c for c in result["cases"] if c["far_fraction"] == 0)
    return min(min(c["attn"]["median_us"] for c in result["cases"]), zero["batch"]["median_us"], zero["direct"]["median_us"])

def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": CJK or "DejaVu Sans", "font.size": 8.5, "axes.unicode_minus": False,
        "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.titlesize": 9.5, "legend.fontsize": 8,
    })


def dress(axis, ylabel):
    axis.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(SPINE)
        axis.spines[spine].set_linewidth(0.8)
    axis.set_ylabel(ylabel, color=INK2)
    axis.tick_params(length=2.5, width=0.8)


def grouped(axis, labels, series, fmt, fontsize=6.2, clip=None):
    """series: [(name, color, values)]; one thin bar per series per label."""
    n = len(series)
    width = 0.8 / n
    for s, (name, color, ys) in enumerate(series):
        xs = [i + (s - (n - 1) / 2) * width for i in range(len(labels))]
        shown = [min(y, clip) if clip else y for y in ys]
        axis.bar(xs, shown, width * 0.92, label=name, color=color, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        for x, y, v in zip(xs, shown, ys):
            axis.annotate(fmt(v), (x, y), textcoords="offset points", xytext=(0, 2.5), ha="center",
                          fontsize=fontsize, color=INK2, zorder=4)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels)


def load(name):
    return json.loads((A100 / name).read_text())


def steady(r, mode):
    tail = range(r["steps"] // 2, r["steps"])
    return sum(r["per_step"][mode][t]["median_us"] for t in tail) / len(tail)


def rounds_rows(name, ours, min_requests=0):
    d = load(name)
    bw = max(d["bandwidth"]["sm_gather_random_GBps"], d["bandwidth"]["dma_contiguous_GBps"])
    rows = []
    for r in d["results"]:
        if r["requests"] < min_requests:
            continue
        tail = range(r["steps"] // 2, r["steps"])
        miss = sum(r["cached_misses"][ours][t] for t in tail) / len(tail)
        floor = max(steady(r, "floor_fa2"), miss * ROW / bw / 1e3)
        rows.append(dict(req=r["requests"], rows=r["query_rows"], floor=floor, ours=steady(r, ours),
                         memcpy=steady(r, "memcpy_lru"), direct=steady(r, "direct_lru"),
                         nocache=steady(r, "ours_nocache") if "ours_nocache" in r["per_step"] else None))
    return rows


def fig_rounds(capacity=4096):
    suffix = "" if capacity == 4096 else f"_c{capacity}"
    panels = [
        ("host", "ours_stage", "PCIe，MTP8"),
        ("peer", "ours_inplace", "NVLink，MTP8"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for axis, (tier, ours, title) in zip(axes, panels):
        d = load(f"rounds_{tier}_mtp8{suffix}.json")
        base = {r["requests"]: r for r in load(f"rounds_{tier}_mtp8.json")["results"]}
        bw = max(d["bandwidth"]["sm_gather_random_GBps"], d["bandwidth"]["dma_contiguous_GBps"])
        rows = []
        for r in d["results"]:
            if r["requests"] not in (1, 4, 16, 64, 128):
                continue
            tail = range(r["steps"] // 2, r["steps"])
            miss = sum(r["cached_misses"][ours][t] for t in tail) / len(tail)
            floor = max(steady(base[r["requests"]], "floor_fa2"), miss * ROW / bw / 1e3)
            rows.append(dict(req=r["requests"], rows=r["query_rows"], floor=floor,
                             ours=steady(r, ours), memcpy=steady(r, "memcpy_lru"),
                             direct=steady(r, "direct_lru")))
        grouped(axis, [f"{r['req']} 请求\n{r['rows']} 行" for r in rows], [
            ("Direct + LRU", BLUE, [r["direct"] / 1e3 for r in rows]),
            ("Memcpy + LRU", ORANGE, [r["memcpy"] / 1e3 for r in rows]),
            ("本文 kernel + LRU", GREEN, [r["ours"] / 1e3 for r in rows]),
        ], lambda v: f"{v:.1f}" if v >= 1 else f"{v:.2f}", fontsize=5.8)
        for i, r in enumerate(rows):
            axis.plot([i - 0.42, i + 0.42], [r["floor"] / 1e3] * 2,
                      color=INK, linestyle=(0, (4, 3)), linewidth=1.1, zorder=5)
        axis.set_title(title, loc="left")
        dress(axis, "每步 GPU 耗时 (ms)")
        top = max(max(r["direct"], r["memcpy"], r["ours"]) for r in rows) / 1e3
        axis.set_ylim(0, top * 1.20)
    axes[0].plot([], [], color=INK, linestyle=(0, (4, 3)), label="参考下限")
    axes[0].legend(loc="upper left", frameon=False, ncol=2, fontsize=7)
    fig.suptitle(f"A100，MTP8，跨 decode 稳态：每请求 {capacity} 槽 LRU，虚线为参考下限",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    filename = "rounds_vs_floor.png" if capacity == 4096 else f"rounds_vs_floor_c{capacity}.png"
    fig.savefig(OUT / filename, dpi=170)
    plt.close(fig)


def fig_single_step():
    d = load("sweep_host_mtp8.json")
    bw = max(d["bandwidth"]["sm_gather_random_GBps"], d["bandwidth"]["dma_contiguous_GBps"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for axis, req in zip(axes, (16, 128)):
        r = next(x for x in d["results"] if x["requests"] == req)
        cases = [c for c in r["cases"] if c["far_fraction"] <= 0.15]
        labels = [f"{c['far_fraction']:.0%}" for c in cases]
        grouped(axis, labels, [
            ("原生直访 Direct", BLUE, [c["direct"]["median_us"] / 1e3 for c in cases]),
            ("提前拷贝 Memcpy", ORANGE, [c["copy"]["median_us"] / 1e3 for c in cases]),
            ("本文 kernel", GREEN, [c["batch"]["median_us"] / 1e3 for c in cases]),
        ], lambda v: f"{v:.1f}" if v >= 1 else f"{v:.2f}", fontsize=6.0)
        floors = [max(compute_floor(r), sorted(c["far_unique_bytes"])[len(c["far_unique_bytes"]) // 2] / bw / 1e3) / 1e3
                  for c in cases]
        axis.plot(range(len(cases)), floors, color=INK2, marker="o", markersize=4, markerfacecolor=SURFACE,
                  linewidth=1.2, zorder=5, label="参考下限")
        axis.set_xlabel("远端（Host）命中率")
        dress(axis, "GPU 耗时 (ms)")
        axis.set_title(f"{req} 请求 × MTP8 = {r['query_rows']} 行", loc="left")
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    rm, rd = [], []
    for req in (16, 128):
        for c in next(x for x in d["results"] if x["requests"] == req)["cases"]:
            if 0.1 <= c["far_fraction"] <= 0.15:
                rm.append(c["batch"]["median_us"] / c["copy"]["median_us"])
                rd.append(c["batch"]["median_us"] / c["direct"]["median_us"])
    fig.suptitle(f"单步、不开缓存，PCIe，A100：远端占比 10%～15% 时本文 kernel 耗时是 Memcpy 的 {min(rm):.2f}～{max(rm):.2f} 倍、"
                 f"Direct 的 {min(rd):.2f}～{max(rd):.2f} 倍", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "single_step_pcie_mtp8.png", dpi=170)
    plt.close(fig)


def fig_single_step_nvlink():
    d = load("sweep_peer_mtp8.json")
    bw = max(d["bandwidth"]["sm_gather_random_GBps"], d["bandwidth"]["dma_contiguous_GBps"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for axis, req in zip(axes, (64, 128)):
        r = next(x for x in d["results"] if x["requests"] == req)
        cases = r["cases"]
        grouped(axis, [f"{c['far_fraction']:.0%}" for c in cases], [
            ("提前拷贝 Memcpy", ORANGE, [c["copy"]["median_us"] / 1e3 for c in cases]),
            ("优化后", GREEN, [c["direct"]["median_us"] / 1e3 for c in cases]),
        ], lambda v: f"{v:.2f}", fontsize=6.0)
        floors = [max(compute_floor(r), sorted(c["far_unique_bytes"])[len(c["far_unique_bytes"]) // 2] / bw / 1e3) / 1e3
                  for c in cases]
        axis.plot(range(len(cases)), floors, color=INK2, marker="o", markersize=4,
                  markerfacecolor=SURFACE, linewidth=1.2, zorder=5, label="参考下限")
        axis.set_xlabel("远端（peer HBM）选中 KV 占比")
        dress(axis, "GPU 耗时 (ms)")
        axis.set_title(f"{req} 请求 × MTP8 = {r['query_rows']} 行", loc="left")
        top = max(c[k]["median_us"] / 1e3 for c in cases for k in ("direct", "copy"))
        axis.set_ylim(0, top * 1.30)
    axes[0].legend(loc="upper left", frameon=False, ncol=2, fontsize=7)
    fig.suptitle("单步、不开缓存，NVLink，A100：本文原地读不开缓存时即原生 kernel，与 Direct 相同，远端占比 ≥ 10% 时快于 Memcpy",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "single_step_nvlink_mtp8_inplace.png", dpi=170)
    plt.close(fig)


def fig_lru_gain():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0))
    for axis, (name, ours, title) in zip(axes, [
            ("rounds_host_mtp8.json", "ours_stage", "PCIe，MTP8"), ("rounds_host_mtp1.json", "ours_stage", "PCIe，MTP1")]):
        rows = [r for r in rounds_rows(name, ours) if r["req"] >= 4]
        grouped(axis, [str(r["req"]) for r in rows], [
            ("本文 kernel，不开缓存", GRAY, [r["nocache"] / 1e3 for r in rows]),
            ("本文 kernel + LRU", GREEN, [r["ours"] / 1e3 for r in rows]),
        ], lambda v: f"{v:.1f}" if v >= 1 else f"{v:.2f}", fontsize=6.0)
        for i, r in enumerate(rows):
            axis.annotate(f"{r['nocache'] / r['ours']:.1f}×", (i, r["nocache"] / 1e3), textcoords="offset points",
                          xytext=(0, 12), ha="center", fontsize=7.2, color=GREEN, weight="bold")
        axis.set_xlabel("请求数")
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        dress(axis, "每步 GPU 耗时 (ms，对数)")
        axis.set_title(title, loc="left")
        axis.set_ylim(top=max(r["nocache"] for r in rows) / 1e3 * 4)
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("跨 decode LRU 的收益（A100，全部 KV 在 Host）：每步快 2.0～5.6 倍", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "lru_gain.png", dpi=170)
    plt.close(fig)


def fig_hisparse():
    cases = [("rounds_host_tiny.json", 1, "MTP1，1 请求"), ("rounds_host_tiny.json", 4, "MTP1，4 请求"),
             ("rounds_host_mtp8_tiny.json", 2, "MTP8，2 请求（16 行）")]
    vals = {k: [] for k in ("hisparse", "ours")}
    for name, req, _ in cases:
        d = json.loads((H800 / name).read_text())
        r = next(x for x in d["results"] if x["requests"] == req)
        vals["hisparse"].append(steady(r, "hisparse"))
        vals["ours"].append(steady(r, "ours_stage"))  # the PCIe configuration
    fig, axis = plt.subplots(figsize=(8.5, 4.0))
    grouped(axis, [c[2] for c in cases], [
        ("SGLang HiSparse（swap-in + FlashMLA）", BLUE, vals["hisparse"]),
        ("本文 kernel + LRU（staged）", GREEN, vals["ours"]),
    ], lambda v: f"{v:.0f} µs", fontsize=7.2)
    for i, (h, o) in enumerate(zip(vals["hisparse"], vals["ours"])):
        axis.annotate(f"快 {h / o:.1f}×", (i, h), textcoords="offset points", xytext=(0, 14), ha="center",
                      fontsize=8, color=GREEN, weight="bold")
    axis.set_ylim(0, max(vals["hisparse"]) * 1.25)
    axis.legend(loc="upper left", frameon=False)
    dress(axis, "每步 GPU 耗时 (µs)")
    axis.set_title("H800，PCIe，跨 decode 稳态，每请求 4096 槽（验证作业，小规模）", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "hisparse_h800.png", dpi=170)
    plt.close(fig)
    return vals


def fig_capacity():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.0))
    for row, (tier, ours, title) in enumerate([
            ("host", "ours_stage", "PCIe"), ("peer", "ours_inplace", "NVLink")]):
        for col, cap in enumerate((2048, 4096, 8192)):
            suffix = "" if cap == 4096 else f"_c{cap}"
            data = load(f"rounds_{tier}_mtp8{suffix}.json")
            rows = [dict(req=r["requests"], ours=steady(r, ours),
                         memcpy=steady(r, "memcpy_lru"), direct=steady(r, "direct_lru"))
                    for r in data["results"] if r["requests"] in (1, 4, 16, 64, 128)]
            axis = axes[row, col]
            grouped(axis, [str(r["req"]) for r in rows], [
                ("优化后 + LRU", GREEN, [r["ours"] / 1e3 for r in rows]),
                ("同容量较快基线", ORANGE,
                 [min(r["memcpy"], r["direct"]) / 1e3 for r in rows]),
            ], lambda v: f"{v:.2f}", fontsize=6.0)
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            # Match scales across capacities within each link.
            axis.set_ylim((0.07, 40) if tier == "host" else (0.04, 10))
            dress(axis, "每步 GPU 耗时 (ms，对数)")
            axis.set_xlabel("请求数")
            axis.set_title(f"{title}，每请求 {cap} 槽", loc="left")
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=7)
    fig.suptitle("A100，MTP8：LRU 容量扫描（跨步稳态，各基线使用相同缓存容量）",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "lru_capacity_comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    style()
    fig_rounds()
    fig_rounds(8192)
    fig_single_step()
    fig_single_step_nvlink()
    fig_lru_gain()
    fig_capacity()
    print(fig_hisparse())
