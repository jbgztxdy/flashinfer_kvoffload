"""Observe actual CPE local remap order on real trace; no performance measurements."""
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

p = Path(__file__).resolve().parent
root = p.parents[2]
previous = p.parent / 'tune_cached_pe_20261101'
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
sys.path.insert(0, str(previous))
from resources_gpu3 import snapshot, require_exclusive
lock = (p.parent / 'tune_general_20261031/gpu3_correctness.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX)
while not snapshot()['available']:
    print('WAITING_GPU3', snapshot(), flush=True)
    time.sleep(15)
before = require_exclusive()
import torch
torch.set_num_threads(4)
sys.path.insert(0, str(root / 'temp'))
from bench_glm52_host_batch import make_wrapper, reference
from flashinfer.jit import env
env.FLASHINFER_INCLUDE_DIR = previous / 'include_cached_pe'
assert sha(env.FLASHINFER_INCLUDE_DIR / 'flashinfer/attention/mla_host_batch.cuh') == 'a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
caps = []
for f in sorted((root / 'temp/glm_trace/assistant64k').glob('doc*.pt')):
    d = torch.load(f, map_location='cpu', weights_only=False)
    assert d['replay_provenance']['assistant_only']
    for layer in range(len(d['layers'])):
        for group in range(len(d['group_starts'])):
            caps.append((d['selections'][layer, group], dict(file=str(f.relative_to(root)), layer=d['layers'][layer],
                window_start=int(d['group_starts'][group]), source_trace_sha256=d['replay_provenance']['source_trace_sha256'])))
random.Random(42).shuffle(caps)
requests, mtp, length, context = 128, 8, 2048, 65536
rows = requests * mtp
host_start = requests * context
torch.manual_seed(42)
kv = torch.randn(2 * host_start, 1, 576, device='cuda', dtype=torch.bfloat16)
q = torch.randn(rows, 64, 512, device='cuda', dtype=torch.bfloat16)
qp = torch.randn(rows, 64, 64, device='cuda', dtype=torch.bfloat16)
idx = torch.empty(rows * length, device='cuda', dtype=torch.int32)
w = make_wrapper(rows, idx, torch.empty(256 << 20, device='cuda', dtype=torch.uint8), length)
w.host_batch = w.host_stage = True
w.host_split = False
w.host_fetch_ctas = 24
w._host_prefetch_shape = False
w.host_staging_rows = idx.numel()
w.set_host_cache(torch.arange(requests + 1, dtype=torch.int32) * mtp, 4096)
captures = []
sample_requests = list(range(0, requests, 8))
for step in range(16):
    page_rows = []
    for request in range(requests):
        values, binding = caps[request % len(caps)]
        selected = values[step * mtp:(step + 1) * mtp].long()
        newest = binding['window_start'] + torch.arange(step * mtp, (step + 1) * mtp).view(mtp, 1)
        page_rows.append(torch.where(selected < newest, host_start + request * context + selected,
                                     request * context + selected).int())
    selected = torch.cat(page_rows).cuda()
    idx.copy_(selected.flatten())
    out = w.run(q, qp, kv[..., :512], kv[..., 512:], host_page_start=host_start)
    torch.cuda.synchronize()
    if step in [0, 7, 15]:
        expected = reference(kv, q, qp, selected, [0, rows - 1])
        torch.testing.assert_close(out[[0, rows - 1]].float(), expected, rtol=.01, atol=.01)
        remap = w._host_batch_buffers[3].view(rows, length)
        for request in sample_requests:
            encoded = remap[request * mtp:(request + 1) * mtp].cpu().tolist()
            tails = []
            for row in encoded:
                host = sum(x <= -(requests * 4096 + 1) for x in row)
                assert all(x <= -(requests * 4096 + 1) for x in row[:host])
                assert all(x > -(requests * 4096 + 1) for x in row[host:])
                tails.append(row[host:])
            captures.append(dict(step=step, request=request, tails=tails))
resident, slot_page, _, store = w._host_cache_views()
slots = (slot_page[:-1] >= 0).nonzero().flatten()
cached = slot_page[slots].long()
assert torch.equal(resident[cached - host_start], slots.int())
assert torch.equal(store['kv'][slots], kv[cached, 0])
raw = p / 'local_order_snapshots.json.gz'
with gzip.GzipFile(filename=str(raw), mode='wb', mtime=0) as f:
    f.write(json.dumps(captures, separators=(',', ':')).encode())
def footprint(tails):
    refs = unique = windows = 0
    for tile in range((max(map(len, tails)) + 31) // 32):
        values = [v for row in tails for v in row[max(0, len(row) - (tile + 1) * 32):max(0, len(row) - tile * 32)]]
        refs += len(values)
        unique += len(set(values))
        windows += 1
    return dict(references=refs, window_unique_pages=unique, windows=windows)
summary = {}
for name, sorted_order in [('original', False), ('sorted', True)]:
    values = [footprint([sorted(row) if sorted_order else row for row in x['tails']]) for x in captures]
    summary[name] = {key: sum(x[key] for x in values) for key in ['references', 'window_unique_pages', 'windows']}
record = dict(complete=True, timestamp=time.strftime('%FT%T%z'), performance_claim=False,
    header_sha256=sha(previous / 'cached_pe.cuh'), source_sha256=sha(Path(__file__)), raw_sha256=sha(raw),
    physical_gpu=3, hardware='A100-PCIE NUMA1, independent same-card farHBM; not NVLink',
    resources_start=before, resources_end=require_exclusive(), steps=16, snapshots=[0, 7, 15],
    sample_requests=sample_requests, summary=summary,
    relative_window_footprint_reduction=1 - summary['sorted']['window_unique_pages'] / summary['original']['window_unique_pages'],
    limitation='Actual CPE remap, real trace and full cache payload checked. Aligned32row windows across8queries are a locality proxy, not measured cache transactions, misses, stall, concurrency or latency. Sorting cost and numerical grouping require a new candidate.')
(p / 'local_order_review.json').write_text(json.dumps(record, indent=2) + '\n')
print('LOCAL_ORDER_REVIEW_COMPLETE', record['relative_window_footprint_reduction'], flush=True)
