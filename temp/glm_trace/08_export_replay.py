"""Export unchanged top-k tensors for replay; keep scores in the original trace.

The optional role filter retains whole consecutive windows, never joins tokens.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(trace_path, out_path, assistant_only=False):
    if out_path.exists():
        raise FileExistsError(out_path)
    trace = torch.load(trace_path, map_location="cpu")
    selections = trace["selections"]
    groups = list(range(selections.shape[1]))
    if assistant_only:
        assistant = trace["role_names"].index("assistant")
        groups = [g for g in groups if bool((trace["role"][trace["positions"][g]] == assistant).all())]
    if not groups:
        raise ValueError("no complete windows match the requested role")
    keep = ("group_starts", "positions", "layers", "source_layer", "topk", "doc", "tokens",
            "members", "corpus", "config", "model", "source_sha256", "window_plan_sha256",
            "meta", "role", "role_names", "layers_run")
    replay = {k: trace[k] for k in keep}
    replay["selections"] = selections
    if len(groups) != selections.shape[1]:
        index = torch.tensor(groups)
        for key, dim in (("selections", 1), ("positions", 0), ("group_starts", 0)):
            replay[key] = replay[key].index_select(dim, index)
    assert torch.equal(replay["positions"], replay["group_starts"][:, None]
                       + torch.arange(replay["selections"].shape[2]))
    provenance = dict(source_trace=str(trace_path), source_trace_sha256=digest(trace_path),
                      source_groups=groups, assistant_only=assistant_only,
                      scores_preserved_in_source=True, topk_values_unchanged=True,
                      selections_shape=list(replay["selections"].shape),
                      group_starts=replay["group_starts"].tolist(),
                      export_script_sha256=digest(Path(__file__)))
    replay["replay_provenance"] = provenance
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pending = out_path.with_suffix(out_path.suffix + ".new")
    torch.save(replay, pending)
    pending.replace(out_path)
    provenance.update(replay=str(out_path), replay_sha256=digest(out_path),
                      bytes=out_path.stat().st_size)
    out_path.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--assistant-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    export(args.trace, args.out, args.assistant_only)
