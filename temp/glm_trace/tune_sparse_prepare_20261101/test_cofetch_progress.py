"""Exercise long local queues, >SM LRU pools, eviction, zero-miss and Graph reuse."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tests/attention'))
from test_mla_host_batch import _make_indices, _plan, _skip_if_unsupported


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_sparse_pool_unequal_rows_and_empty_prefix(dtype):
    """New path: aligned K256, unequal/empty-miss pools, odd tails and Graph eviction."""
    _skip_if_unsupported()
    rows, length, pages, host_start = 512, 256, 8192, 4096
    torch.manual_seed(20261104)
    kv = torch.randn(pages, 1, 576, device='cuda', dtype=dtype)
    q = torch.randn(rows, 64, 512, device='cuda', dtype=dtype)
    qp = torch.randn(rows, 64, 80, device='cuda', dtype=dtype)[..., :64]
    indices = torch.empty(rows * length, device='cuda', dtype=torch.int32)
    w = _plan(rows, length, indices, dtype=dtype)
    w.host_stage, w.host_split, w.host_fetch_ctas = True, False, 24
    w.host_staging_rows = indices.numel()
    w.set_host_cache(torch.tensor([0, 1, 7, 16, rows], dtype=torch.int32), 128)
    for delta in [0, 2048, 0]:
        selected = (host_start + delta + (torch.arange(rows * length, device='cuda', dtype=torch.int32).view(rows, length) * 7) % 1024)
        selected[0] = -1  # Leading pool has no claims: repeated dense-job prefixes.
        selected[1, 1:] = -1
        selected[3, -33:] = -1  # Seven KV tiles, including an odd tail.
        selected[17::19, 3:6] = -1
        indices.copy_(selected.flatten())
        w.run(q, qp, kv[..., :512], kv[..., 512:], host_page_start=host_start)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, lse = w.run(q, qp, kv[..., :512], kv[..., 512:], return_lse=True, host_page_start=host_start)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        for row in [0, 1, 3, 7, 16, 107, 108, rows - 1]:
            chosen = selected[row][selected[row] >= 0].long()
            if not chosen.numel():
                assert torch.all(out[row] == 0) and torch.all(lse[row] <= -1e4)
                continue
            values = kv[chosen, 0].float()
            scores = (q[row].float() @ values[:, :512].T + qp[row].float() @ values[:, 512:].T) / 16
            torch.testing.assert_close(out[row].float(), scores.softmax(-1) @ values[:, :512], rtol=.01, atol=.01)
            torch.testing.assert_close(lse[row], torch.logsumexp(scores, -1) / torch.log(torch.tensor(2.0)), rtol=.001, atol=.001)
        resident, slot_page, _, store = w._host_cache_views()
        slots = (slot_page[:-1] >= 0).nonzero().flatten()
        cached = slot_page[slots].long()
        assert torch.equal(resident[cached - host_start], slots.int())
        assert torch.equal(store['kv'][slots], kv[cached, 0])
        assert len(set(cached.cpu().tolist())) == slots.numel()


@pytest.mark.parametrize('capacity', [512, 4096])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_long_fetch_queues_and_cache_payload(capacity, dtype):
    _skip_if_unsupported()
    requests, mtp, length = 128, 8, 2048
    rows = requests * mtp
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    assert sm == 108 and requests > sm and rows >= 4 * sm
    pages = requests * length * 8
    host_start = pages // 2
    torch.manual_seed(20261101 + capacity)
    kv = torch.randn(pages, 1, 576, device='cuda', dtype=dtype)
    q = torch.randn(rows, 64, 512, device='cuda', dtype=kv.dtype)
    qp = torch.randn(rows, 64, 64, device='cuda', dtype=kv.dtype)
    indices = torch.empty(rows * length, device='cuda', dtype=torch.int32)
    w = _plan(rows, length, indices, dtype=dtype)
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
        selected[3, length - 33:] = -1  # 63 tiles: paired loop plus a nontrivial odd tail
        indices.copy_(selected.flatten())
        w.run(q, qp, kv[..., :512], kv[..., 512:], host_page_start=host_start)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, lse = w.run(q, qp, kv[..., :512], kv[..., 512:], return_lse=True, host_page_start=host_start)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        # Include empty/single-entry rows, CTA boundaries and the last row.
        for row in sorted({0, 1, 3, sm - 1, sm, 4 * sm - 1, rows // 2, rows - 1}):
            chosen = selected[row][selected[row] >= 0].long()
            if chosen.numel() == 0:
                assert torch.all(out[row] == 0)
                assert torch.all(lse[row] <= -1e4)
                continue
            values = kv[chosen, 0].float()
            scores = (q[row].float() @ values[:, :512].T +
                      qp[row].float() @ values[:, 512:].T) / 16
            torch.testing.assert_close(out[row].float(), scores.softmax(-1) @ values[:, :512],
                                       rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(lse[row].float(), torch.logsumexp(scores, -1) / torch.log(torch.tensor(2.0)),
                                       rtol=1e-3, atol=1e-3)
        resident, slot_page, _, store = w._host_cache_views()
        filled = (slot_page[:-1] >= 0).nonzero().flatten()
        cached_pages = slot_page[filled].long()
        assert torch.equal(resident[cached_pages - host_start], filled.int())
        assert torch.equal(store['kv'][filled], kv[cached_pages, 0])
        assert len(set(cached_pages.cpu().tolist())) == len(filled)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_cached_query_pe_head_stride(dtype):
    """Exercise the new query-fragment addressing with a padded head stride."""
    _skip_if_unsupported()
    rows, length, pages, host_start = 432, 65, 8192, 4096
    torch.manual_seed(20261102)
    kv = torch.randn(pages, 1, 576, device='cuda', dtype=dtype)
    q = torch.randn(rows, 64, 512, device='cuda', dtype=dtype)
    qp = torch.randn(rows, 64, 80, device='cuda', dtype=dtype)[..., :64]
    assert qp.stride(-2) == 80 and not qp.is_contiguous()
    selected = (torch.arange(rows * length, dtype=torch.int32) % pages).view(rows, length)
    indices = selected.flatten().cuda()
    w = _plan(rows, length, indices, dtype=dtype)
    w.host_stage, w.host_split, w.host_fetch_ctas = True, False, 24
    w.host_staging_rows = indices.numel()
    w.set_host_cache(torch.tensor([0, rows], dtype=torch.int32), 512)
    w.run(q, qp, kv[..., :512], kv[..., 512:], host_page_start=host_start)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = w.run(q, qp, kv[..., :512], kv[..., 512:], return_lse=True,
                         host_page_start=host_start)
    graph.replay()
    torch.cuda.synchronize()
    for row in [0, 1, 107, 108, rows - 1]:
        values = kv[selected[row].long().cuda(), 0].float()
        scores = (q[row].float() @ values[:, :512].T + qp[row].float() @ values[:, 512:].T) / 16
        torch.testing.assert_close(out[row].float(), scores.softmax(-1) @ values[:, :512], rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse[row], torch.logsumexp(scores, -1) / torch.log(torch.tensor(2.0)), rtol=1e-3, atol=1e-3)
    resident, slot_page, _, store = w._host_cache_views()
    slots = (slot_page[:-1] >= 0).nonzero().flatten()
    cached_pages = slot_page[slots].long()
    assert torch.equal(resident[cached_pages - host_start], slots.int())
    assert torch.equal(store['kv'][slots], kv[cached_pages, 0])


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_shared_claims_reset_and_replay(dtype):
    """Many CTA rows claim the same pages, then switch to disjoint pages across launches."""
    _skip_if_unsupported()
    rows, length, pages, host_start = 432, 97, 8192, 4096
    torch.manual_seed(20261103)
    kv = torch.randn(pages, 1, 576, device='cuda', dtype=dtype)
    q = torch.randn(rows, 64, 512, device='cuda', dtype=dtype)
    qp = torch.randn(rows, 64, 64, device='cuda', dtype=dtype)
    indices = torch.empty(rows * length, device='cuda', dtype=torch.int32)
    w = _plan(rows, length, indices, dtype=dtype)
    w.host_stage, w.host_split, w.host_fetch_ctas = True, False, 24
    w.host_staging_rows = indices.numel()
    w.set_host_cache(torch.tensor([0, rows], dtype=torch.int32), 512)
    for round_id in range(4):
        if round_id % 2 == 0:
            selected = (torch.arange(length, dtype=torch.int32)[None, :].expand(rows, -1).clone() + host_start + 1024 * round_id)
        else:
            selected = (torch.arange(rows * length, dtype=torch.int32).view(rows, length) % 2048 + host_start)
        selected[0] = -1
        selected[1, 1:] = -1
        indices.copy_(selected.flatten().cuda())
        w.run(q, qp, kv[..., :512], kv[..., 512:], host_page_start=host_start)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, lse = w.run(q, qp, kv[..., :512], kv[..., 512:], return_lse=True, host_page_start=host_start)
        graph.replay(); graph.replay(); torch.cuda.synchronize()
        for row in [0, 1, 107, 108, rows-1]:
            chosen = selected[row][selected[row] >= 0].long().cuda()
            if not chosen.numel():
                assert torch.all(out[row] == 0) and torch.all(lse[row] <= -1e4)
                continue
            values = kv[chosen, 0].float()
            scores = (q[row].float() @ values[:, :512].T + qp[row].float() @ values[:, 512:].T) / 16
            torch.testing.assert_close(out[row].float(), scores.softmax(-1) @ values[:, :512], rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(lse[row], torch.logsumexp(scores, -1) / torch.log(torch.tensor(2.0)), rtol=1e-3, atol=1e-3)
        resident, slot_page, _, store = w._host_cache_views()
        slots = (slot_page[:-1] >= 0).nonzero().flatten()
        cached = slot_page[slots].long()
        assert torch.equal(resident[cached - host_start], slots.int())
        assert torch.equal(store['kv'][slots], kv[cached, 0])
        assert cached.unique().numel() == cached.numel()
