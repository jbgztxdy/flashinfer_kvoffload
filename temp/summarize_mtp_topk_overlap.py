# Purpose: Summarize MTP top-k overlap captured by measure_mtp_topk_overlap.py.
# Usage: python temp/summarize_mtp_topk_overlap.py temp/mtp_overlap/overlap_ctx*.json
import argparse
import json
from pathlib import Path
import statistics

import torch


def quantiles(values):
    ordered = sorted(values)
    pick = lambda q: ordered[int((len(ordered) - 1) * q)]  # noqa: E731
    return dict(p10=pick(0.1), median=statistics.median(ordered), p90=pick(0.9))


def random_union(context, topk, tokens):
    """Expected |union| / (tokens * topk) for independent uniform top-k sets."""
    return context * (1 - (1 - topk / context) ** tokens) / (tokens * topk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--recent", type=int, default=4096, help="keys this close stay in HBM")
    parser.add_argument("--output", type=Path, default=Path("temp/mtp_overlap/summary"))
    args = parser.parse_args()
    summary = []
    per_layer = []
    for path in args.inputs:
        meta = json.loads(path.read_text())
        topk = meta["configuration"]["topk"]
        mtp = meta["configuration"]["mtp"]
        context = meta["configuration"]["context"]
        for document in meta["documents"]:
            data = torch.load(document["selections"])
            selections, starts = data["selections"].long(), data["group_starts"]
            layers, groups = selections.shape[:2]
            metrics = {key: [] for key in ("u2", "u4", "u8", "jaccard", "far_share", "far_u8")}
            layer_u8 = [[] for _ in range(layers)]
            layer_far = [[] for _ in range(layers)]
            for layer in range(layers):
                for group in range(groups):
                    sets = selections[layer, group]
                    for m in (2, 4, 8):
                        if m <= mtp:
                            metrics[f"u{m}"].append(len(sets[:m].unique()) / (m * topk))
                    for i in range(mtp - 1):
                        a, b = set(sets[i].tolist()), set(sets[i + 1].tolist())
                        metrics["jaccard"].append(len(a & b) / len(a | b))
                    far = sets[sets < int(starts[group]) - args.recent]
                    metrics["far_share"].append(far.numel() / sets.numel())
                    if far.numel():
                        metrics["far_u8"].append(len(far.unique()) / far.numel())
                        layer_far[layer].append(metrics["far_u8"][-1])
                    layer_u8[layer].append(metrics["u8"][-1])
            mean_start = float(starts.float().mean())
            row = dict(
                context=context,
                document=document["name"],
                tail_loss=document.get("tail_loss"),
                random_u8=random_union(mean_start, topk, 8),
                random_far_u8=random_union(mean_start - args.recent, topk, 8),
            )
            row |= {key: quantiles(values) for key, values in metrics.items()}
            summary.append(row)
            for layer in range(layers):
                per_layer.append(
                    dict(
                        context=context,
                        document=document["name"],
                        layer=layer,
                        u8=statistics.median(layer_u8[layer]),
                        far_u8=statistics.median(layer_far[layer]) if layer_far[layer] else None,
                    )
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(dict(recent=args.recent, documents=summary, per_layer=per_layer), indent=2) + "\n"
    )
    lines = [
        "| 上下文 | 文档 | tail loss | 8 token 并集比例 | 随机基线 | 远端占比 | 远端并集比例 | 随机远端基线 | 相邻 Jaccard |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['context'] // 1024}K | {row['document']} | {row['tail_loss']:.2f} | "
            f"{row['u8']['median']:.3f} ({row['u8']['p10']:.3f}–{row['u8']['p90']:.3f}) | {row['random_u8']:.3f} | "
            f"{row['far_share']['median']:.2f} | "
            f"{row['far_u8']['median']:.3f} ({row['far_u8']['p10']:.3f}–{row['far_u8']['p90']:.3f}) | "
            f"{row['random_far_u8']:.3f} | {row['jaccard']['median']:.3f} |"
        )
    contexts = sorted({row["context"] for row in per_layer})
    lines += ["", "| 层 | " + " | ".join(f"{c // 1024}K 远端并集比例" for c in contexts) + " |",
              "|---:|" + "---:|" * len(contexts)]
    for layer in sorted({row["layer"] for row in per_layer}):
        cells = []
        for c in contexts:
            values = [r["far_u8"] for r in per_layer if r["layer"] == layer and r["context"] == c and r["far_u8"] is not None]
            cells.append(f"{statistics.median(values):.3f}" if values else "-")
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
