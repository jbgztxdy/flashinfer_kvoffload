"""Audit exact requested coverage, arithmetic means, data checks, and every accepted guard."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
DEST=Path(__file__).resolve().parent
meta=json.loads((DEST/"experiment.json").read_text())
expected={(kind,s["name"],mtp,b,f) for kind in ("nocache","lru") for s in meta["settings"]
          for mtp in (1,8) for b in meta["batches"][str(mtp)]
          for f in (meta["host_fractions"] if kind=="nocache" else [None])}
seen=set();local_seen=set();guards=launches=0;errors=0.;files=[];missing_files=[]
for path in sorted((DEST/"raw").glob("*.json")):
    if path.name.startswith("smoke_"):continue
    try:data=json.loads(path.read_text())
    except FileNotFoundError:continue
    if not data["complete"]:missing_files.append(path.name);continue
    cfg=data["configuration"];assert cfg["passes"]==meta["passes"]
    if cfg["kind"]=="nocache" and cfg.get("memcpy_path_version",1)<2:continue
    assert len(data["bindings"])==cfg["requests"]
    assert cfg["mtp"] in (1,8) and cfg["requests"] in meta["batches"][str(cfg["mtp"])]
    assert cfg["query_rows"]==cfg["requests"]*cfg["mtp"] and cfg["topk"]==2048 and cfg["heads"]==64
    assert cfg["steps"]==meta["trace_positions"]//cfg["mtp"]
    assert cfg["capacity"]==(meta["capacity"] if cfg["kind"]=="lru" else 0)
    assert data["device"]=="NVIDIA A100-SXM4-40GB"
    if cfg["kind"] not in ("floor","snapshots"):
        assert cfg["setting"]==next(s for s in meta["settings"] if s["name"]==cfg["setting"]["name"])
        assert bool("-DFLASHINFER_FAR_EMU" in data["extra_cuda_flags"])==cfg["setting"].get("emulated",False)
    for case in data["cases"]:
        if cfg["kind"] in ("floor","snapshots"):
            key=(cfg["kind"],cfg["mtp"],cfg["requests"],case["host_fraction_requested"])
            assert key not in local_seen;local_seen.add(key)
        required_steps=meta["trace_positions"]//cfg["mtp"]
        if cfg["kind"] in ("lru","floor"):
            assert case["step_indices"]==list(range(required_steps))
        else:
            assert case["step_indices"]==[x//cfg["mtp"] for x in meta["no_lru_snapshots"]]
        assert set(case["numerical_checks"]["max_abs_error"])==set(case["timings"])
        assert all(math.isfinite(v) and v>=0 for v in case["numerical_checks"]["max_abs_error"].values())
        n=len(case["step_indices"]);entries=cfg["query_rows"]*cfg["topk"]
        assert len(case["far_refs"])==len(case["far_unique"])==len(case["ours_unique_misses"])==n
        assert abs(case["host_fraction_actual"]-sum(case["far_refs"])/(n*entries))<1e-12
        assert all(0<=u<=r<=entries for u,r in zip(case["far_unique"],case["far_refs"]))
        if cfg["kind"]=="lru":
            miss=case["ours_unique_miss_samples"]
            assert len(miss)==required_steps and all(len(v)==cfg["passes"] for v in miss)
            assert all(abs(sum(v)/len(v)-m)<1e-9 for v,m in zip(miss,case["ours_unique_misses"]))
            assert all(0<=m<=u for m,u in zip(case["ours_unique_misses"],case["far_unique"]))
            assert case["ours_unique_misses"][0]==case["far_unique"][0]
            ref_miss=case["ours_reference_miss_samples"]
            assert len(ref_miss)==required_steps and all(len(v)==cfg["passes"] for v in ref_miss)
            assert all(all(0<=m<=r for m in v) for v,r in zip(ref_miss,case["far_refs"]))
            assert all(m==case["far_refs"][0] for m in ref_miss[0])
            assert abs(case["ours_reference_cache_hit_rate"]-(1-sum(map(sum,ref_miss))/cfg["passes"]/sum(case["far_refs"])))<1e-12
            assert all(v[0]==r and v[1]==u and 0<=v[2]<=u and 0<=v[3]<=r
                       for v,r,u in zip(case["memcpy_plan_counts"],case["far_refs"],case["far_unique"]))
        assert abs(case["final_remote_unique_fraction"]-sum(case["ours_unique_misses"])/(n*entries))<1e-12
        if cfg["kind"] not in ("floor","snapshots"):
            key=(cfg["kind"],cfg["setting"]["name"],cfg["mtp"],cfg["requests"],case["host_fraction_requested"])
            assert key in expected and key not in seen,key
            assert cfg["ours_dispatch"]=="batched_staged_all_shapes"
            seen.add(key)
        n=len(case["step_indices"]);errors=max(errors,*case["numerical_checks"]["max_abs_error"].values())
        for mode,timing in case["timings"].items():
            raw=timing["per_step_samples_us"];assert len(raw)==n
            assert all(len(x)==cfg["passes"] for x in raw)
            average=sum(map(sum,raw))/(n*cfg["passes"])
            assert abs(average-timing["mean_us"])<1e-9
            assert all(v>0 for step in raw for v in step)
            assert all(abs(sum(step[p] for step in raw)/n-v)<1e-9
                       for p,v in enumerate(timing["pass_mean_us"]))
            launches+=n*cfg["passes"]
            for p in range(cfg["passes"]):
                accepted=[g for g in case["guards"] if g["mode"]==mode and g["pass_index"]==p]
                end=0
                for g in accepted:
                    assert g["valid"] and g["first"]==end
                    if g.get("timing_queue")=="cuda_graph":
                        assert cfg["kind"]=="lru"
                    else:
                        assert g["guard_ms"]>=g["enqueue_ms"]+max(1,g["enqueue_ms"]*.05)
                    end=g["last"];guards+=1
                assert end==n
    files.append(dict(file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      source_sha256=data["source_sha256"]))
    for source_set in (data["source_sha256"],data.get("preserved_ours_source_sha256",{})):
        for name,sha in source_set.items():
            snap=DEST/"source_snapshots"/(sha+Path(name).suffix)
            assert snap.exists() and hashlib.sha256(snap.read_bytes()).hexdigest()==sha,(path,name)
calibration=json.loads((DEST/"calibration.json").read_text()) if (DEST/"calibration.json").exists() else None
local_expected={(kind,m,b,f) for kind in ("floor","snapshots") for m in (1,8)
                for b in meta["batches"][str(m)]
                for f in ([None] if kind=="floor" else meta["host_fractions"])}
complete=seen==expected and local_seen==local_expected and not missing_files and calibration is not None and calibration["complete"]
stability=json.loads((DEST/"stability_repeat.json").read_text()) if (DEST/"stability_repeat.json").exists() else {}
complete=complete and stability.get("complete",False)
if stability.get("complete"):
    assert {r["file"] for r in stability["repeated"]}==set(stability["files"])
    for repeated in stability["repeated"]:
        assert hashlib.sha256((ROOT/repeated["file"]).read_bytes()).hexdigest()==repeated["sha256"]
        assert all(v["new_pass_range_over_mean"]<=.20 for c in repeated["changes"] for v in c["modes"].values())
summary=dict(expected_cells=len(expected),completed_cells=len(seen),missing_cells=len(expected-seen),
             incomplete_files=missing_files,accepted_segments=guards,timed_launches=launches,
             local_reference_cases=len(local_seen),expected_local_reference_cases=len(local_expected),
             stability_repeats_complete=stability.get("complete",False),
             sampled_max_abs_error=errors,calibration_passed=bool(calibration and calibration["complete"]),
             complete=complete,files=files)
temp=DEST/"audit.json.new";temp.write_text(json.dumps(summary,indent=2)+"\n");temp.replace(DEST/"audit.json")
print(json.dumps({k:v for k,v in summary.items() if k!="files"},indent=2))
if not complete:raise SystemExit(2)
