"""Exercise the newly routed 2-SM-wave boundary, including exact cache payloads."""
import importlib.util
from pathlib import Path
import pytest
import torch
root=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('mla_host_batch_test_helpers',root/'tests/attention/test_mla_host_batch.py')
helpers=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
_skip_if_unsupported=helpers._skip_if_unsupported
_plan=helpers._plan
_make_indices=helpers._make_indices

@pytest.mark.parametrize("cached", [False, True])
def test_medium_whole_rows_graph_and_padding(cached):
    """Medium staged rows survive graph replay, changing misses and an all-local launch."""
    _skip_if_unsupported()
    device = torch.device("cuda")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    mtp, kv_len = 8, 97
    requests = (2 * sm_count + mtp - 1) // mtp
    rows = requests * mtp
    num_pages = requests * kv_len * 8
    host_start = num_pages // 2
    torch.manual_seed(731)
    cache = torch.randn(num_pages, 1, 576, dtype=torch.bfloat16, device=device)
    q = torch.randn(rows, 64, 512, dtype=cache.dtype, device=device)
    qp = torch.randn(rows, 64, 64, dtype=cache.dtype, device=device)
    indices = torch.empty(rows * kv_len, dtype=torch.int32, device=device)
    w = _plan(rows, kv_len, indices)
    w.host_stage, w.host_split = True, False
    w.host_fetch_ctas = 24
    w.host_staging_rows = indices.numel()
    if cached:
        w.set_host_cache(torch.arange(requests + 1, dtype=torch.int32) * mtp, 256)
    for launch, fraction in enumerate((1.0, 0.3, 0.0, 0.3)):
        selected = _make_indices(requests, mtp, kv_len, num_pages, host_start,
                                 fraction, 0.6, 731 + launch).to(device)
        selected[0, :] = -1
        selected[1, 1:] = -1
        selected[2::3, 3:6] = -1
        indices.copy_(selected.flatten())
        # Warm before capture, then exercise the same persistent state across repeated replay.
        w.run(q, qp, cache[..., :512], cache[..., 512:], host_page_start=host_start)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, lse = w.run(q, qp, cache[..., :512], cache[..., 512:],
                             return_lse=True, host_page_start=host_start)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        for row in range(rows):
            pages = selected[row][selected[row] >= 0].long()
            if not pages.numel():
                assert torch.all(out[row] == 0)
                assert torch.all(lse[row] <= -1e4)
                continue
            kv = cache[pages, 0].float()
            scores = (q[row].float() @ kv[:, :512].T + qp[row].float() @ kv[:, 512:].T) / 16
            torch.testing.assert_close(out[row].float(), scores.softmax(-1) @ kv[:, :512],
                                       rtol=1e-2, atol=1e-2)
        if cached:
            resident, slot_page, _, store = w._host_cache_views()
            filled = (slot_page[:-1] >= 0).nonzero().flatten()
            pages = slot_page[filled].long()
            assert torch.equal(resident[pages - host_start], filled.int())
            assert torch.equal(store["kv"][filled], cache[pages, 0])
