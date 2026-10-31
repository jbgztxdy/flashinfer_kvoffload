"""Exercise long local queues, >SM LRU pools, eviction, zero-miss and Graph reuse."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tests/attention'))
from test_mla_host_batch import _make_indices, _plan, _skip_if_unsupported


@pytest.mark.parametrize('capacity', [512, 4096])
def test_long_fetch_queues_and_cache_payload(capacity):
    _skip_if_unsupported()
    requests, mtp, length = 128, 8, 2048
    rows = requests * mtp
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    assert sm == 108 and requests > sm and rows >= 4 * sm
    pages = requests * length * 8
    host_start = pages // 2
    torch.manual_seed(20261101 + capacity)
    kv = torch.randn(pages, 1, 576, device='cuda', dtype=torch.bfloat16)
    q = torch.randn(rows, 64, 512, device='cuda', dtype=kv.dtype)
    qp = torch.randn(rows, 64, 64, device='cuda', dtype=kv.dtype)
    indices = torch.empty(rows * length, device='cuda', dtype=torch.int32)
    w = _plan(rows, length, indices)
    w.host_stage, w.host_split = True, False
    w.host_fetch_ctas = 24
    w.host_staging_rows = indices.numel()
    w.set_host_cache(torch.arange(requests + 1, dtype=torch.int32) * mtp, capacity)
    for launch, fraction in enumerate([1.0, 0.5, 1.0]):
        selected = _make_indices(requests, mtp, length, pages, host_start,
                                 fraction, 0.75, 9101 + launch).cuda()
        selected[0] = -1
        selected[1, 1:] = -1
        selected[2::17, 3:6] = -1
        indices.copy_(selected.flatten())
        w.run(q, qp, kv[..., :512], kv[..., 512:], host_page_start=host_start)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = w.run(q, qp, kv[..., :512], kv[..., 512:], host_page_start=host_start)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        # Include empty/single-entry rows, CTA boundaries and the last row.
        for row in sorted({0, 1, sm - 1, sm, 4 * sm - 1, rows // 2, rows - 1}):
            chosen = selected[row][selected[row] >= 0].long()
            if chosen.numel() == 0:
                assert torch.all(out[row] == 0)
                continue
            values = kv[chosen, 0].float()
            scores = (q[row].float() @ values[:, :512].T +
                      qp[row].float() @ values[:, 512:].T) / 16
            torch.testing.assert_close(out[row].float(), scores.softmax(-1) @ values[:, :512],
                                       rtol=1e-2, atol=1e-2)
        resident, slot_page, _, store = w._host_cache_views()
        filled = (slot_page[:-1] >= 0).nonzero().flatten()
        cached_pages = slot_page[filled].long()
        assert torch.equal(resident[cached_pages - host_start], filled.int())
        assert torch.equal(store['kv'][filled], kv[cached_pages, 0])
        assert len(set(cached_pages.cpu().tolist())) == len(filled)
