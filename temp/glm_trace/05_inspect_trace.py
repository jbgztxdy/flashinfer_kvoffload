"""Inspect saved GLM trace data and loss, on CPU, before starting formal capture.

loss[r] predicts token[r+1], so loss roles belong to the target position.
Usage: python 05_inspect_trace.py --trace traces/check/doc0.pt \
    --loss traces/check/loss.pt --out results/check.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def describe(values):
    if not values.numel():
        return dict(count=0, mean=None)
    q = values.float().quantile(torch.tensor([0.5, 0.9, 0.99]))
    return dict(count=values.numel(), mean=values.double().mean().item(),
                p50=q[0].item(), p90=q[1].item(), p99=q[2].item(), max=values.max().item())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(args):
    trace = torch.load(args.trace, map_location="cpu")
    loss = torch.load(args.loss, map_location="cpu")
    sel, scores, stats = (trace[k] for k in ("selections", "scores", "row_stats"))
    positions = trace["positions"]
    n = trace["tokens"]
    checks = {}
    checks["shape"] = (sel.ndim == 4 and sel.shape == scores.shape
                       and sel.shape[-1] == 2048 and stats.shape == sel.shape[:-1] + (5,)
                       and positions.shape == sel.shape[1:3])
    expected_layers = [0, 1, 2] + list(range(6, 78, 4))
    checks["all_21_indexer_layers"] = trace["layers"] == expected_layers and sel.shape[0] == 21
    checks["all_78_layers"] = trace["layers_run"] == 78 and trace["source_layer"] == [
        max(v for v in expected_layers if v <= layer) for layer in range(78)]
    checks["positions"] = bool(((positions >= 0) & (positions < n)).all())
    checks["consecutive_windows"] = torch.equal(
        positions, trace["group_starts"][:, None] + torch.arange(sel.shape[2]))
    checks["dtypes"] = sel.dtype == torch.int32 and scores.dtype == torch.float32
    checks["score_rows"] = True
    # Check all recorded rows in small slices to limit scratch memory on 64K traces.
    for layer in range(sel.shape[0]):
        ids, sc, st = (x[layer].reshape(-1, x.shape[-1]) for x in (sel, scores, stats))
        pos = positions.flatten()
        for start in range(0, len(pos), 64):
            s, v, r = ids[start:start + 64], sc[start:start + 64], st[start:start + 64]
            p = pos[start:start + 64]
            valid = s >= 0
            sorted_ids = s.sort(-1).values
            duplicate = (sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (sorted_ids[:, 1:] >= 0)
            checks["score_rows"] &= bool(
                (s >= -1).all() and (s <= p[:, None]).all()
                and ((s == -1) == torch.isneginf(v)).all()
                and torch.isfinite(v[valid]).all() and not duplicate.any()
                and (v[:, 1:] <= v[:, :-1]).all()
                and (valid.sum(-1) == (p + 1).clamp(max=2048)).all()
                and (r[:, 0] == p + 1).all()
                and torch.allclose(r[:, 3], v[:, 0], rtol=1e-5, atol=1e-3)
                and torch.allclose(r[:, 4], v.where(valid, 0).sum(-1), rtol=1e-5, atol=1e-3))

    doc = trace["doc"]
    begin, end = map(int, loss["doc_bounds"][doc:doc + 2])
    assert end - begin == n
    per_token = loss["loss"][begin:end]
    roles = trace["role"]
    assert roles is not None and len(roles) == n
    checks["loss_finite"] = bool(torch.isfinite(per_token[:-1]).all() and torch.isnan(per_token[-1]))
    target_roles = roles[1:]
    role_names = trace["role_names"]
    by_role = {name: describe(per_token[:-1][(target_roles == i) & torch.isfinite(per_token[:-1])])
               for i, name in enumerate(role_names)}
    blocks = []
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        targets = roles[start + 1:min(stop + 1, n)]
        blocks.append(dict(start=start, end=stop, mean=per_token[start:stop].nanmean().item(),
                           target_role_counts={name: int((targets == i).sum())
                                               for i, name in enumerate(role_names)}))
    high_blocks = [b for b in blocks[1:] if b["mean"] >= args.block_limit]
    assistant = by_role["assistant"]
    gates = dict(data=all(checks.values()),
                 assistant_loss=assistant["count"] > 0 and assistant["mean"] <= args.assistant_limit,
                 blocks_after_first=len(high_blocks) == 0)
    result = dict(trace=str(args.trace), loss=str(args.loss), trace_sha256=digest(args.trace),
                  loss_sha256=digest(args.loss), tokens=n, selections_shape=list(sel.shape),
                  layers=trace["layers"], members=trace["members"], meta=trace.get("meta"),
                  recorded_window_loss=trace["loss"], all_position_loss=describe(per_token[torch.isfinite(per_token)]),
                  loss_by_target_role=by_role, blocks_256=blocks, high_blocks_after_first=high_blocks,
                  thresholds=dict(assistant_mean=args.assistant_limit, block_mean=args.block_limit),
                  checks=checks, gates=gates, ready_for_formal_capture=all(gates.values()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(loss_by_target_role=by_role, high_blocks_after_first=high_blocks,
                          checks=checks, gates=gates, ready_for_formal_capture=all(gates.values())), indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--loss", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--assistant-limit", type=float, default=1.5)
    ap.add_argument("--block-limit", type=float, default=6.0)
    args = ap.parse_args()
    torch.set_num_threads(4)
    raise SystemExit(inspect(args))
