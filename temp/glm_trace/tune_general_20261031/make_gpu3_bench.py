"""Derive a separately labelled benchmark without modifying queued GPU01 inputs."""
from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'bench.py').read_text()
changes=[
 ('from resources import require_exclusive','from resources_gpu3 import require_exclusive'),
 ('LOCK=(ROOT/"temp/glm_trace/kernel_report_20261029/gpu01.lock").open("w")','LOCK=(DEST/"gpu3_correctness.lock").open("w")'),
 ('"nvidia-smi","--id=0,1"','"nvidia-smi","--id=3"'),
 ('str((DEST / "resources.py").relative_to(ROOT))','str((DEST / "resources_gpu3.py").relative_to(ROOT))'),
 ('directory=DEST/"source_snapshots";directory.mkdir(exist_ok=True)','directory=DEST/"gpu3/source_snapshots";directory.mkdir(parents=True,exist_ok=True)'),
 ('outpath = DEST / "raw" / (label + ".json")','outpath = DEST / "gpu3/raw" / (label + ".json")'),
 ('failed=DEST/"failures";failed.mkdir(exist_ok=True)','failed=DEST/"gpu3/failures";failed.mkdir(parents=True,exist_ok=True)'),
 ('tiers.TieredKV(near,far,staging,0,setting["tier"],1)','tiers.TieredKV(near,far,staging,0,setting["tier"],0)'),
 ('steps=STEPS,positions=STEPS*M,route=A.route,passes=PASSES,capacity=', 'steps=STEPS,positions=STEPS*M,route=A.route,hardware_group="gpu3_a100_pcie",far_peer_physical_device=3,far_peer_is_same_device=True,passes=PASSES,capacity=')]
for old,new in changes:
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
s=s.replace('assert SETTINGS\n', 'assert SETTINGS\nassert all(s["name"] in ("floor","pcie") or s["name"].startswith(("g50_","g100_")) for s in SETTINGS), "GPU3 same-device far is not physical NVLink"\n')
(p/'bench_gpu3.py').write_text(s)
