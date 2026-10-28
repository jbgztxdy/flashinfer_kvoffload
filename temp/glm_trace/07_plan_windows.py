"""Keep uniform windows, moving nearby windows to cover full assistant runs when available.

The positions remain consecutive within each window; no roles or intervening tokens are removed.
Usage: python 07_plan_windows.py --corpus corpus_swe.pt --out results/swe_windows_64k.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from trace_windows import recording_starts


def make_plan(corpus, corpus_path, groups, span):
    lengths = [len(t) for t in corpus["tokens"]]
    uniform = recording_starts(corpus, lengths, groups, span)
    documents = []
    for i, (n, role, original) in enumerate(zip(lengths, corpus["role"], uniform)):
        active = (role == corpus["role_names"].index("assistant")).int()
        edges = torch.diff(torch.cat([torch.tensor([0]), active, torch.tensor([0])]))
        begin, end = ((edges == v).nonzero().flatten().tolist() for v in (1, -1))
        runs = [(max(a, n // 2), b) for a, b in zip(begin, end) if b - max(a, n // 2) >= span]
        starts = original.tolist()
        moved = []
        used = set()
        for a, b in runs:
            candidates = sorted((j for j in range(groups) if j not in used), key=lambda j: abs(starts[j] - a))
            for j in candidates:
                trial = sorted(starts[:j] + [a] + starts[j + 1:])
                if all(y - x >= span for x, y in zip(trial, trial[1:])):
                    moved.append(dict(original_start=starts[j], start=a, assistant_run_end=b))
                    starts[j] = a
                    used.add(j)
                    break
            else:
                raise ValueError(f"cannot fit assistant run {a}:{b} in document {i}")
        starts.sort()
        windows = []
        for s in starts:
            counts = {name: int((role[s:s + span] == k).sum()) for k, name in enumerate(corpus["role_names"])}
            windows.append(dict(start=s, end=s + span, query_role_counts=counts,
                                all_assistant=counts["assistant"] == span))
        documents.append(dict(doc=i, tokens=n, members=corpus["members"][i], starts=starts,
                              uniform_starts=original.tolist(), moved_windows=moved, windows=windows))
    return dict(corpus=str(corpus_path), corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                groups=groups, span=span, documents=documents)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--span", type=int, default=1024)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    corpus = torch.load(args.corpus, map_location="cpu")
    plan = make_plan(corpus, args.corpus, args.groups, args.span)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    recording_starts(corpus, [len(t) for t in corpus["tokens"]], args.groups, args.span, args.out, args.corpus)
    print(json.dumps([dict(doc=d["doc"], all_assistant_windows=sum(w["all_assistant"] for w in d["windows"]),
                           moved_windows=d["moved_windows"]) for d in plan["documents"]], indent=2))
