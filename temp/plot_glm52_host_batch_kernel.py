# Purpose: Figures for the batched Host MLA kernel: hit-rate sweep at the target batch, per-row
#          scaling across batch sizes, and the cross-decode-round cache effect.
# Usage: MPLCONFIGDIR=temp/matplotlib python temp/plot_glm52_host_batch_kernel.py
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# The labels are Chinese; DejaVu Sans has no CJK glyphs, so register a CJK face if present.
CJK = None
for path in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if Path(path).exists():
        fm.fontManager.addfont(path)
        CJK = fm.FontProperties(fname=path).get_name()
        break

# Validated categorical slots 1-3 (see the dataviz palette reference); ours last so it reads as
# the subject against the two baselines.
SERIES = [
    ("direct", "原生直访 Direct", "#2a78d6"),
    ("copy", "提前拷贝 Memcpy", "#eb6834"),
    ("batch", "本文 kernel", "#1baf7a"),
]
SURFACE = "#fcfcfb"
INK, INK2, GRID, SPINE = "#0b0b0b", "#52514e", "#E6E9EC", "#ADB4BA"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": CJK or "DejaVu Sans", "font.size": 8.5,
        "axes.unicode_minus": False,
        "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.titlesize": 9.5, "legend.fontsize": 8,
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


def grouped(axis, labels, data, scale, fmt, annotate=True):
    """One group per label, one thin bar per series, a 2px surface gap between fills."""
    width = 0.26
    for s, (key, name, color) in enumerate(SERIES):
        xs = [i + (s - 1) * width for i in range(len(labels))]
        ys = [data[key][i] / scale for i in range(len(labels))]
        axis.bar(xs, ys, width * 0.92, label=name, color=color,
                 edgecolor=SURFACE, linewidth=1.0, zorder=3)
        if annotate:
            for x, y in zip(xs, ys):
                axis.annotate(fmt(y), (x, y), textcoords="offset points", xytext=(0, 2.5),
                              ha="center", fontsize=6.4, color=INK2, zorder=4)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels)


def load_csv(path):
    return {(int(r["query_rows"]), float(r["host_fraction"])): r for r in csv.DictReader(open(path))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtp", type=Path, default=Path("temp/host_attention_batch_kernel/mtp_v3.csv"))
    parser.add_argument("--attn", type=Path, default=Path("temp/host_attention_batch_kernel/attn.json"))
    parser.add_argument("--rounds", type=Path, default=Path("temp/host_attention_rounds/rounds128.json"))
    parser.add_argument("--fetch", type=Path,
                        default=Path("temp/host_attention_batch_kernel/fetch_only.json"))
    parser.add_argument("--out", type=Path, default=Path("temp/host_attention_batch_kernel"))
    args = parser.parse_args()
    style()
    rows = load_csv(args.mtp)
    fractions = sorted({k[1] for k in rows})
    batches = sorted({k[0] for k in rows})

    # Pure-attention reference (all KV in HBM): the floor transfer can be hidden against.
    floor = None
    if args.attn.exists():
        a = json.loads(args.attn.read_text())
        b = [x for x in a["results"] if x["query_rows"] == 1024]
        if b:
            floor = min(c["attn"]["median_us"] for c in b[0]["cases"]) / 1000

    # --- 1. hit-rate sweep at the target batch -------------------------------------------
    figure, axis = plt.subplots(figsize=(7.0, 3.5))
    data = {k: [rows[(1024, f)][f"{k}_us"] and float(rows[(1024, f)][f"{k}_us"]) for f in fractions]
            for k, _, _ in SERIES}
    grouped(axis, [f"{f:.0%}" for f in fractions], data, 1000, lambda v: f"{v:.1f}")
    # Bars occupy every group, so both reference marks are labelled in the axes' free space.
    note = []
    if floor:
        axis.axhline(floor, color=INK2, linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
        note.append(f"— — 纯计算下限 {floor:.1f} ms（KV 全在 HBM、零传输）")
    if args.fetch.exists():
        # Measured transfer-only time: the same kernel with the attention phase compiled out.
        fo = json.loads(args.fetch.read_text())["results"][0]["cases"]
        got = {c["host_fraction"]: c["batch"]["median_us"] / 1000 for c in fo}
        axis.plot(range(len(fractions)), [got.get(f) for f in fractions], marker="o",
                  markersize=4.0, linewidth=1.4, color=INK2, zorder=5,
                  markeredgecolor=SURFACE, markeredgewidth=1.0)
        note.append("—●— 纯搬运耗时（同 kernel 去掉 attention，仅把命中页搬进 HBM）")
    for i, text in enumerate(note):
        axis.annotate(text, (0.015, 0.885 - i * 0.068), xycoords="axes fraction",
                      fontsize=7.2, color=INK2)
    dress(axis, "GPU 耗时 (ms)")
    axis.set_ylim(0, max(max(v) for v in data.values()) / 1000 * 1.18)
    axis.set_xlabel("Host 命中率", color=INK2)
    axis.set_title("1024 query 行（128 请求 × 8 MTP），top-k 2048，A100-PCIe\n"
                   "top-k 回放自实测 MTP 重叠：跨行去重后唯一页占 41%",
                   color=INK, loc="left", fontsize=9)
    axis.legend(frameon=False, loc="upper left", ncol=3, handlelength=1.1, columnspacing=1.3)
    figure.tight_layout()
    figure.savefig(args.out / "batch_kernel_1024rows.png", dpi=200)
    plt.close(figure)

    # --- 2. per-row scaling across batch sizes -------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.5))
    for axis, frac in zip(axes, (0.10, 0.30)):
        data = {k: [float(rows[(n, frac)][f"{k}_us"]) / n for n in batches] for k, _, _ in SERIES}
        grouped(axis, [str(n) for n in batches], data, 1, lambda v: f"{v:.0f}", annotate=False)
        dress(axis, "摊薄到每行的耗时 (µs)")
        axis.set_xlabel("query 行数", color=INK2)
        axis.set_title(f"Host 命中率 {frac:.0%}", color=INK, loc="left")
    axes[0].legend(frameon=False, loc="upper right", handlelength=1.1)
    figure.suptitle("摊薄到每 query 行的耗时（总时间 ÷ 行数；行之间是并行的，非每行独立耗时）\n"
                    "小 batch 处三者都偏高是因 GPU 未被填满；top-k 同样回放自实测 MTP 重叠",
                    color=INK, fontsize=9, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(args.out / "batch_kernel_scaling.png", dpi=200)
    plt.close(figure)

    # --- 3. cross-round persistent cache --------------------------------------------------
    if args.rounds.exists():
        d = json.loads(args.rounds.read_text())
        nc = d["variants"][0]["rounds"]
        ca = d["variants"][1]["rounds"]
        labels = ["无缓存"] + [f"轮 {r['round']}" for r in ca]
        data = {k: [sum(r[k]["median_us"] for r in nc) / len(nc)] + [r[k]["median_us"] for r in ca]
                for k, _, _ in SERIES}
        figure, axis = plt.subplots(figsize=(7.4, 3.5))
        grouped(axis, labels, data, 1000, lambda v: f"{v:.1f}")
        if floor:
            axis.axhline(floor, color=INK2, linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
            # Left of the first group: the only region the bars never reach.
            axis.annotate(f"纯计算下限 {floor:.1f} ms", (-0.48, floor),
                          textcoords="offset points", xytext=(0, 5), ha="left",
                          fontsize=7, color=INK2)
        miss = [r["host_miss_fraction"] for r in ca]
        axis.annotate("实际过 PCIe 的比例：" + " → ".join(f"{m:.1%}" for m in miss),
                      (0.5, 0.80), xycoords="axes fraction", fontsize=7.2, color=INK2)
        dress(axis, "GPU 耗时 (ms)")
        axis.set_ylim(0, max(max(v) for v in data.values()) / 1000 * 1.18)
        axis.set_xlabel("连续 verify 轮（持久缓存逐轮预热）", color=INK2)
        axis.set_title("跨 decode 轮持久缓存：1024 行，30% Host 命中（按页放置模拟，输出逐位不变）",
                       color=INK, loc="left")
        axis.legend(frameon=False, loc="upper right", ncol=3, handlelength=1.1, columnspacing=1.3)
        figure.tight_layout()
        figure.savefig(args.out / "batch_kernel_rounds.png", dpi=200)
        plt.close(figure)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
