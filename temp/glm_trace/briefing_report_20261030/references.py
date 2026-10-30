"""Ideal-overlap reference: average of per-step max(compute, unique-byte budget)."""
from functools import lru_cache
import json
from pathlib import Path
import statistics

DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[2]
SOURCE=ROOT/"temp/glm_trace/kernel_report_20261029"


@lru_cache(None)
def read(name):return json.loads((ROOT/name).read_text())


@lru_cache(None)
def reference(kind,setting,mtp,batch,fraction):
    mode="floor" if kind=="lru" else "snapshots"
    local_path=SOURCE/"raw"/f"{mode}_floor_mtp{mtp}_b{batch}.json"
    local=read(str(local_path.relative_to(ROOT)))
    local=next(c for c in local["cases"] if c["host_fraction_requested"]==fraction)
    raw_path=SOURCE/"raw"/f"{kind}_{setting}_mtp{mtp}_b{batch}.json"
    raw=read(str(raw_path.relative_to(ROOT)))
    case=next(c for c in raw["cases"] if c["host_fraction_requested"]==fraction)
    cfg=raw["configuration"];bw=cfg["setting"].get("bandwidth_GBps",0)
    if bw:
        bandwidth_source="configured_emulated_bandwidth"
    else:
        probe=read(str((DEST/"physical_bandwidth.json").relative_to(ROOT)))
        physical=setting if setting in ("pcie","nvlink") else "nvlink"
        bw=next(s["reference_GBps"] for s in probe["settings"] if s["name"]==physical)
        bandwidth_source="postcheck_max_physical_DMA_or_gather"
    compute=[statistics.mean(t) for t in local["timings"]["local"]["per_step_samples_us"]]
    transfer=[miss*1152/(bw*1000) for miss in case["ours_unique_misses"]]
    assert len(compute)==len(transfer)
    per_step=[max(c,t) for c,t in zip(compute,transfer)]
    bound=statistics.mean(per_step)
    max_of_means=max(statistics.mean(compute),statistics.mean(transfer))
    assert bound>=max_of_means-1e-8
    return dict(kind=kind,setting=setting,mtp=mtp,requests=batch,host_fraction_target=fraction,
        bandwidth_GBps=bw,bandwidth_source=bandwidth_source,mean_compute_us=statistics.mean(compute),
        mean_unique_transfer_us=statistics.mean(transfer),mean_stepwise_max_us=bound,
        legacy_max_of_means_us=max_of_means,
        ours_to_reference=case["timings"]["ours"]["mean_us"]/bound,
        per_step_compute_us=compute,per_step_transfer_us=transfer,per_step_reference_us=per_step,
        raw=str(raw_path.relative_to(ROOT)),local_raw=str(local_path.relative_to(ROOT)))


def for_row(r):
    return reference(r["kind"],r["setting"],r["mtp"],r["requests"],r["host_fraction_target"])
