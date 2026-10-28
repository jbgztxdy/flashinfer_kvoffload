"""CPU-only recording-window selection shared by the collector and corpus inspection."""
import hashlib
import json
from pathlib import Path

import torch


def recording_starts(corpus, lengths, groups, span, plan_path=None, corpus_path=None):
    if groups < 1 or span < 1 or any(n < 2 * span for n in lengths):
        raise ValueError("need positive groups/span and enough tokens in each document's second half")
    if plan_path is None:
        return [torch.linspace(n // 2, n - span, groups).long() for n in lengths]
    plan = json.loads(Path(plan_path).read_text())
    if plan["groups"] != groups or plan["span"] != span:
        raise ValueError("window plan groups/span do not match the capture")
    if hashlib.sha256(Path(corpus_path).read_bytes()).hexdigest() != plan["corpus_sha256"]:
        raise ValueError("window plan belongs to a different corpus")
    starts = []
    for i, n in enumerate(lengths):
        doc = plan["documents"][i]
        st = torch.tensor(doc["starts"], dtype=torch.long)
        if doc["tokens"] != n or doc["members"] != corpus["members"][i]:
            raise ValueError(f"window plan document {i} does not match the capture")
        if (len(st) != groups or not bool(((st >= n // 2) & (st + span <= n)).all())
                or not bool((st[1:] - st[:-1] >= span).all())):
            raise ValueError(f"invalid, overlapping or out-of-range windows in document {i}")
        starts.append(st)
    return starts
