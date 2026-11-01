"""Check cold/warm reference misses before the launch and expose legacy remap undercount."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
variant = os.environ['TUNE_VARIANT']
assert variant in ['base', 'final_destination']
assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
from resources_gpu3 import snapshot, require_exclusive
lock = (p.parent / 'tune_general_20261031/gpu3_correctness.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX)
while not snapshot()['available']:
    print('WAITING_GPU3', snapshot(), flush=True)
    time.sleep(10)
before = require_exclusive()
import torch
sys.path.insert(0, str(root / 'temp'))
from bench_glm52_host_batch import make_wrapper, reference
from flashinfer.jit import env
env.FLASHINFER_INCLUDE_DIR = p / ('include_' + variant)
torch.manual_seed(20261103)
rows, length, pages, host_start = 432, 97, 8192, 4096
kv = torch.randn(pages, 1, 576, device='cuda', dtype=torch.bfloat16)
q = torch.randn(rows, 64, 512, device='cuda', dtype=torch.bfloat16)
qp = torch.randn(rows, 64, 64, device='cuda', dtype=torch.bfloat16)
idx = torch.empty(rows * length, device='cuda', dtype=torch.int32)
w = make_wrapper(rows, idx, torch.empty(256 << 20, device='cuda', dtype=torch.uint8), length)
w.host_batch = w.host_stage = True
w.host_split = False
w.host_fetch_ctas = 24
w.host_staging_rows = idx.numel()
w._host_prefetch_shape = False
w.set_host_cache(torch.tensor([0, rows], dtype=torch.int32), 128)
# The wrapper allocates cache backing lazily on the first eager run.
idx.copy_((host_start + torch.arange(length, device='cuda', dtype=torch.int32)).repeat(rows))
w.run(q, qp, kv[..., :512], kv[..., 512:], host_page_start=host_start)
torch.cuda.synchronize()
w.release_host_cache(torch.tensor([0], dtype=torch.int32))
steps = []
for step, delta in enumerate([0, 0, 256, 256]):
    selected = (host_start + delta + torch.arange(length, device='cuda', dtype=torch.int32)).repeat(rows).contiguous()
    idx.copy_(selected)
    # Fully external diagnostic, before the measured operator. No kernel source
    # changes, no fabricated reconstruction from the candidate's final remap.
    resident, slot_page, _, store = w._host_cache_views()
    misses = resident[(selected - host_start).long()] < 0
    refs = int(misses.sum())
    unique = int(torch.unique(selected[misses]).numel())
    assert refs == (rows * length if step % 2 == 0 else 0)
    assert unique == (length if step % 2 == 0 else 0)
    out = w.run(q, qp, kv[..., :512], kv[..., 512:], host_page_start=host_start)
    torch.cuda.synchronize()
    got_unique = int(w._host_batch_buffers[5][1])
    legacy = int((w._host_batch_buffers[3] <= -(128 + 1)).sum())
    assert got_unique == unique
    assert legacy == (0 if variant == 'final_destination' else refs)
    expected = reference(kv, q, qp, selected.view(rows, length), [0, rows - 1])
    torch.testing.assert_close(out[[0, rows - 1]].float(), expected, rtol=.01, atol=.01)
    slots = (slot_page[:-1] >= 0).nonzero().flatten()
    cached = slot_page[slots].long()
    assert torch.equal(resident[cached - host_start], slots.int())
    assert torch.equal(store['kv'][slots], kv[cached, 0])
    steps.append(dict(step=step, before_launch_reference_misses=refs, before_launch_unique_misses=unique,
                      kernel_unique_counter=got_unique, legacy_postlaunch_threshold_count=legacy,
                      output_and_all_cache_payload_checked=True))
record = dict(complete=True, timestamp=time.strftime('%FT%T%z'), variant=variant, steps=steps,
              resources_start=before, resources_end=require_exclusive(), performance_claim=False,
              header_sha256=hashlib.sha256((p / (variant + '.cuh')).read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              limitation='Four deterministic launches, proves diagnostic incompatibility; does not recover historical trace reference hit rates.')
(p / ('remap_diagnosis_' + variant + '.json')).write_text(json.dumps(record, indent=2) + '\n')
print('REMAP_DIAGNOSIS_OK', variant)
