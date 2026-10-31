"""Check exact radix victims, protected hits, padding and all cached payloads."""
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tests/attention'))
from test_mla_host_batch import _plan, _skip_if_unsupported


@pytest.mark.parametrize('capacity', [512, 4096])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_multibyte_radix_victims(capacity, dtype):
    _skip_if_unsupported()
    rows, length, host_start = 8, 1024, 64
    torch.manual_seed(20261101 + capacity)
    kv = torch.randn(host_start + 3 * capacity, 1, 576, device='cuda', dtype=dtype)
    q = torch.randn(rows, 64, 512, device='cuda', dtype=dtype)
    qp = torch.randn(rows, 64, 64, device='cuda', dtype=dtype)
    indices = torch.full((rows * length,), host_start, dtype=torch.int32, device='cuda')
    w = _plan(rows, length, indices, dtype=dtype)
    w.host_stage, w.host_split, w.host_fetch_ctas = True, True, 16
    w.host_staging_rows = rows * length
    w.set_host_cache(torch.tensor([0, rows], dtype=torch.int32), capacity)
    w.run(q, qp, kv[..., :512], kv[..., 512:], host_page_start=host_start)
    resident, slot_page, stamps, store = w._host_cache_views()
    for layout_index, layout in enumerate(['all_equal', 'one_byte', 'four_bytes']):
        resident.fill_(-1)
        slot_page.fill_(-1)
        slot_page[:capacity] = torch.arange(host_start, host_start + capacity, device='cuda', dtype=torch.int32)
        resident[:capacity] = torch.arange(capacity, device='cuda', dtype=torch.int32)
        store['kv'][:capacity] = kv[host_start:host_start + capacity, 0]
        if layout == 'all_equal':
            keys = [271] * capacity
        elif layout == 'one_byte':
            keys = [0x01020300 + (i * 37) % 251 for i in range(capacity)]
        else:
            keys = [(i * 0x01020307) % 0x03000000 for i in range(capacity)]
        stamps[:capacity] = torch.tensor(keys, device='cuda', dtype=torch.int32)
        epoch = 0x03040506 + 16 * layout_index
        w._host_batch_buffers[5][3] = epoch
        hits = list(range(host_start, host_start + 32))
        misses = list(range(host_start + capacity, host_start + capacity + 97))
        chosen = torch.tensor(hits + misses, dtype=torch.int32)
        selected = chosen[torch.arange(rows * length) % chosen.numel()].view(rows, length)
        selected[0] = -1
        selected[1, 1:] = -1
        selected[2:, 3:6] = -1
        indices.copy_(selected.flatten().cuda())
        out, lse = w.run(q, qp, kv[..., :512], kv[..., 512:], return_lse=True,
                         host_page_start=host_start)
        for row in range(rows):
            valid = selected[row][selected[row] >= 0].long().cuda()
            if valid.numel() == 0:
                assert torch.all(out[row] == 0)
                assert torch.all(lse[row] <= -1e4)
                continue
            values = kv[valid, 0].float()
            scores = (q[row].float() @ values[:, :512].T + qp[row].float() @ values[:, 512:].T) / 16
            torch.testing.assert_close(out[row].float(), scores.softmax(-1) @ values[:, :512],
                                       rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(lse[row], torch.logsumexp(scores, -1) / torch.log(torch.tensor(2.0)),
                                       rtol=1e-3, atol=1e-3)
        # The original tie traversal is thread-major, then stride256 within a thread.
        expected_victims = set(sorted(range(32, capacity),
            key=lambda i: (keys[i], i % 256, i // 256))[:len(misses)])
        now = slot_page[:capacity].cpu().tolist()
        actual_victims = {i for i, page in enumerate(now) if page != host_start + i}
        assert actual_victims == expected_victims, layout
        assert set(now[i] for i in actual_victims) == set(misses)
        assert all(now[i] == host_start + i for i in range(32))
        pages = slot_page[:capacity].long()
        assert torch.equal(resident[pages - host_start], torch.arange(capacity, device='cuda', dtype=torch.int32))
        assert torch.equal(store['kv'][:capacity], kv[pages, 0])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            w.run(q, qp, kv[..., :512], kv[..., 512:], host_page_start=host_start)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        assert slot_page[:capacity].cpu().tolist() == now
        assert torch.equal(store['kv'][:capacity], kv[pages, 0])
