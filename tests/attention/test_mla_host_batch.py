"""Batched Host MLA decode: dedup staging, overflow, chunk merging and launch-to-launch state.

Pages at or above ``host_page_start`` are treated as Host pages. The kernel logic does not depend
on where they physically live, so these tests keep the whole cache in device memory.
"""

import pytest
import torch

from flashinfer.mla import BatchMLAPagedAttentionWrapper
from flashinfer.utils import get_compute_capability


def _skip_if_unsupported():
    if not torch.cuda.is_available() or get_compute_capability(torch.device("cuda"))[0] < 8:
        pytest.skip("Requires SM80+")


def _make_indices(requests, mtp, kv_len, num_pages, host_page_start, host_fraction, shared, seed):
    """Each request owns a page range; its MTP tokens select overlapping top-k sets."""
    generator = torch.Generator().manual_seed(seed)
    local_pages = host_page_start // requests
    host_pages = (num_pages - host_page_start) // requests
    rows = []
    for r in range(requests):
        host_count = round(host_fraction * kv_len)
        base_local = torch.randperm(local_pages, generator=generator)[: kv_len] + r * local_pages
        base_host = (
            torch.randperm(host_pages, generator=generator)[: kv_len] + host_page_start + r * host_pages
        )
        for t in range(mtp):
            # Keep `shared` of each part from the request's common pool, draw the rest per token.
            keep = round(shared * kv_len)
            local = torch.cat(
                [base_local[:keep], base_local[keep:][torch.randperm(kv_len - keep, generator=generator)]]
            )[: kv_len - host_count]
            host = torch.cat(
                [base_host[:keep], base_host[keep:][torch.randperm(kv_len - keep, generator=generator)]]
            )[:host_count]
            row = torch.cat([local, host])
            rows.append(row[torch.randperm(kv_len, generator=generator)])
    return torch.stack(rows).int()


def _reference(q_nope, q_pe, ckv, kpe, indices, sm_scale):
    outs, lses = [], []
    for row in range(q_nope.shape[0]):
        pages = indices[row].long()
        k = ckv[pages, 0].float()
        scores = (q_nope[row].float() @ k.T + q_pe[row].float() @ kpe[pages, 0].float().T) * sm_scale
        outs.append(scores.softmax(-1) @ k)
        lses.append(torch.logsumexp(scores, -1))
    return torch.stack(outs), torch.stack(lses)


def _run(
    requests,
    mtp,
    num_heads,
    kv_len,
    host_fraction,
    shared,
    dtype=torch.bfloat16,
    staging_rows=None,
    fetch_ctas=4,
    seed=0,
    repeats=2,
):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    rows = requests * mtp
    num_pages = max(8 * kv_len * requests, 4096)
    host_page_start = num_pages // 2
    cache = torch.randn(num_pages, 1, 576, dtype=dtype, device=device)
    ckv, kpe = cache[..., :512], cache[..., 512:]
    q_nope = torch.randn(rows, num_heads, 512, dtype=dtype, device=device)
    q_pe = torch.randn(rows, num_heads, 64, dtype=dtype, device=device)
    indices = _make_indices(
        requests, mtp, kv_len, num_pages, host_page_start, host_fraction, shared, seed
    ).to(device)
    sm_scale = 1 / 16
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 2**20, dtype=torch.uint8, device=device), backend="fa2"
    )
    wrapper.plan(
        torch.arange(rows + 1, dtype=torch.int32, device=device),
        torch.arange(rows + 1, dtype=torch.int32, device=device) * kv_len,
        indices.view(-1),
        torch.full((rows,), kv_len, dtype=torch.int32, device=device),
        num_heads,
        512,
        64,
        1,
        False,
        sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    wrapper.host_staging_rows = staging_rows
    wrapper.host_fetch_ctas = fetch_ctas
    expected_o, expected_lse = _reference(q_nope, q_pe, ckv, kpe, indices, sm_scale)
    for _ in range(repeats):
        out, lse = wrapper.run(
            q_nope, q_pe, ckv, kpe, return_lse=True, host_page_start=host_page_start
        )
        torch.testing.assert_close(out.float(), expected_o, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse, expected_lse / torch.log(torch.tensor(2.0)), rtol=1e-3, atol=1e-3)
    return wrapper, (q_nope, q_pe, ckv, kpe, host_page_start, indices, expected_o)


@pytest.mark.parametrize("requests,mtp", [(1, 2), (1, 8), (4, 8), (16, 8), (64, 4)])
@pytest.mark.parametrize("host_fraction", [0.0, 0.2, 1.0])
def test_host_batch_matches_reference(requests, mtp, host_fraction):
    _skip_if_unsupported()
    _run(requests, mtp, 64, 2048, host_fraction, shared=0.5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_heads", [64, 128])
def test_host_batch_heads_and_dtypes(dtype, num_heads):
    _skip_if_unsupported()
    _run(4, 8, num_heads, 2048, 0.3, shared=0.5, dtype=dtype)


@pytest.mark.parametrize("kv_len", [1, 31, 100, 513])
def test_host_batch_short_rows(kv_len):
    _skip_if_unsupported()
    _run(2, 4, 64, kv_len, 0.5, shared=0.5)


@pytest.mark.parametrize("staging_rows", [64, 1000])
def test_host_batch_staging_overflow(staging_rows):
    """Pages beyond the staging capacity are read directly and stay correct across launches."""
    _skip_if_unsupported()
    _run(8, 8, 64, 2048, 0.3, shared=0.3, staging_rows=staging_rows, repeats=3)


@pytest.mark.parametrize("fetch_ctas", [1, 8])
def test_host_batch_fetch_ctas(fetch_ctas):
    _skip_if_unsupported()
    _run(8, 8, 64, 2048, 0.3, shared=0.5, fetch_ctas=fetch_ctas)


def test_host_batch_base_e_and_index_update():
    _skip_if_unsupported()
    wrapper, (q_nope, q_pe, ckv, kpe, host_page_start, indices, _) = _run(
        4, 8, 64, 2048, 0.3, shared=0.5
    )
    # New indices in the planned buffer must not reuse stale dedup or ready state.
    # `indices` aliases that buffer, so keep a copy of the new order.
    indices = indices.flip(0).clone()
    wrapper._kv_indices_buf.copy_(indices.reshape(-1))
    out, lse = wrapper.run(
        q_nope, q_pe, ckv, kpe, return_lse=True, return_lse_base_on_e=True,
        host_page_start=host_page_start,
    )
    expected_o, expected_lse = _reference(q_nope, q_pe, ckv, kpe, indices, 1 / 16)
    torch.testing.assert_close(out.float(), expected_o, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("order", [(0.3, 0.0), (0.0, 0.3), (0.0, 0.0, 0.3, 0.0)])
def test_host_batch_alternating_zero_host(order):
    """Launches with no Host page take a fast path that skips the resolve pass, the second grid
    barrier and the fetch. Alternating with Host-bearing launches must not leave stale state."""
    _skip_if_unsupported()
    device = torch.device("cuda")
    torch.manual_seed(0)
    requests, mtp, num_heads, kv_len = 4, 8, 64, 1024
    rows = requests * mtp
    num_pages = max(8 * kv_len * requests, 4096)
    host_page_start = num_pages // 2
    cache = torch.randn(num_pages, 1, 576, dtype=torch.bfloat16, device=device)
    ckv, kpe = cache[..., :512], cache[..., 512:]
    q_nope = torch.randn(rows, num_heads, 512, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(rows, num_heads, 64, dtype=torch.bfloat16, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 2**20, dtype=torch.uint8, device=device), backend="fa2"
    )
    indices_buf = torch.empty(rows * kv_len, dtype=torch.int32, device=device)
    wrapper.plan(
        torch.arange(rows + 1, dtype=torch.int32, device=device),
        torch.arange(rows + 1, dtype=torch.int32, device=device) * kv_len,
        indices_buf,
        torch.full((rows,), kv_len, dtype=torch.int32, device=device),
        num_heads, 512, 64, 1, False, 1 / 16,
        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
    )
    for step, fraction in enumerate(order):
        indices = _make_indices(
            requests, mtp, kv_len, num_pages, host_page_start, fraction, 0.5, seed=step
        ).to(device)
        indices_buf.copy_(indices.view(-1))
        out = wrapper.run(q_nope, q_pe, ckv, kpe, host_page_start=host_page_start)
        expected, _ = _reference(q_nope, q_pe, ckv, kpe, indices, 1 / 16)
        torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)


def test_host_batch_cuda_graph():
    _skip_if_unsupported()
    wrapper, (q_nope, q_pe, ckv, kpe, host_page_start, indices, expected_o) = _run(
        8, 8, 64, 2048, 0.25, shared=0.5
    )
    out = torch.empty_like(q_nope)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q_nope, q_pe, ckv, kpe, out=out, host_page_start=host_page_start)
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected_o, rtol=1e-2, atol=1e-2)


def _lru_rounds(requests, mtp, kv_len, host_fraction, keep, rounds, seed, share=0.0):
    """Top-k of consecutive decode rounds: each round keeps `keep` of a request's previous Host
    pages and draws the rest fresh, so pages are reused, go stale and get evicted. With `share`,
    request r also reuses that fraction of request r - 1's Host pages (prefix sharing)."""
    generator = torch.Generator().manual_seed(seed)
    host_count = round(host_fraction * kv_len)
    span = 4 * rounds * kv_len
    num_pages = 2 * requests * span
    host_page_start = num_pages // 2
    previous = [None] * requests
    out = []
    for _ in range(rounds):
        rows = []
        current = []
        for r in range(requests):
            host_base = host_page_start + r * span
            fresh = torch.randperm(span, generator=generator)[:host_count] + host_base
            if previous[r] is not None:
                k = round(keep * host_count)
                fresh[:k] = previous[r][torch.randperm(host_count, generator=generator)[:k]]
            if share and r > 0:
                k = round(share * host_count)
                fresh[-k:] = current[r - 1][:k]
            fresh = torch.unique(fresh)
            fresh = torch.cat([fresh, torch.randperm(span, generator=generator)[: host_count - len(fresh)] + host_base])
            current.append(fresh)
            for _ in range(mtp):
                host = fresh[torch.randperm(host_count, generator=generator)[: host_count * 3 // 4]]
                local = torch.randperm(span, generator=generator)[: kv_len - len(host)] + r * span
                row = torch.cat([local, host])
                rows.append(row[torch.randperm(kv_len, generator=generator)])
        previous = current
        out.append(torch.stack(rows).int())
    return out, num_pages, host_page_start


@pytest.mark.parametrize(
    "requests,mtp,capacity,share",
    [(1, 8, 512, 0.0), (4, 8, 640, 0.0), (16, 2, 448, 0.0), (6, 4, 512, 0.3), (3, 8, 192, 0.0)],
)
def test_host_batch_lru_cache(requests, mtp, capacity, share):
    _skip_if_unsupported()
    device = torch.device("cuda")
    kv_len, rounds, num_heads, dtype = 1024, 6, 64, torch.bfloat16
    rows = requests * mtp
    rounds_indices, num_pages, host_page_start = _lru_rounds(
        requests, mtp, kv_len, 0.4, 0.6, rounds, seed=requests, share=share
    )
    cache = torch.randn(num_pages, 1, 576, dtype=dtype, device=device)
    ckv, kpe = cache[..., :512], cache[..., 512:]
    q_nope = torch.randn(rows, num_heads, 512, dtype=dtype, device=device)
    q_pe = torch.randn(rows, num_heads, 64, dtype=dtype, device=device)
    indices = torch.empty(rows * kv_len, dtype=torch.int32, device=device)
    sm_scale = 1 / 16
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 2**20, dtype=torch.uint8, device=device), backend="fa2"
    )
    wrapper.plan(
        torch.arange(rows + 1, dtype=torch.int32, device=device),
        torch.arange(rows + 1, dtype=torch.int32, device=device) * kv_len,
        indices,
        torch.full((rows,), kv_len, dtype=torch.int32, device=device),
        num_heads, 512, 64, 1, False, sm_scale, q_data_type=dtype, kv_data_type=dtype,
    )
    wrapper.host_fetch_ctas = 4
    wrapper.set_host_cache(torch.arange(requests + 1, dtype=torch.int32) * mtp, capacity)
    last_used = {}
    resident_before = set()
    pool_before = {}
    hits = 0
    for t, round_indices in enumerate(rounds_indices):
        wrapper._kv_indices_buf.copy_(round_indices.view(-1).to(device))
        out, _ = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True, host_page_start=host_page_start)
        expected, _ = _reference(q_nope, q_pe, ckv, kpe, round_indices.to(device), sm_scale)
        torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)

        host_pages = wrapper._host_batch_buffers[0][1]
        persist = wrapper._host_cache["persist"]
        slots = requests * capacity
        resident = persist[:host_pages].cpu()
        slot_page = persist[host_pages : host_pages + slots].cpu()
        staging = wrapper._host_batch_buffers[1]
        staged_rows = staging.shape[0] - slots
        cached = (resident >= 0).nonzero().view(-1) + host_page_start
        resident_now = set(cached.tolist())
        # Maps agree, and every cached copy holds its page's bytes.
        assert torch.equal(slot_page[resident[cached - host_page_start].long()], cached.int())
        filled = slot_page >= 0
        assert int(filled.sum()) == len(resident_now)
        cache_rows = staging[staged_rows:][filled.nonzero().view(-1).to(device)]
        assert torch.equal(cache_rows, cache.view(num_pages, 576)[slot_page[filled].long().to(device)])

        selected = set(int(p) for p in round_indices.view(-1).tolist() if p >= host_page_start)
        hits += len(selected & resident_before)
        for page in selected:
            last_used[page] = t
        pool_now = {p: int(resident[p - host_page_start]) // capacity for p in resident_now}
        for r in range(requests):
            assert sum(1 for p in pool_now.values() if p == r) <= capacity
        # When a pool's pages of this launch fit, all of them must be resident now.
        if round(kv_len * 0.4) <= capacity and share == 0.0:
            assert selected <= resident_now
        # LRU within each pool: an evicted page was never used after a page that stayed there
        # without being used now.
        for r in range(requests):
            evicted = {p for p in resident_before - resident_now if pool_before[p] == r}
            kept_idle = {
                p for p in (resident_now & resident_before) - selected if pool_now[p] == r
            }
            if evicted and kept_idle:
                assert max(last_used[p] for p in evicted) <= min(last_used[p] for p in kept_idle)
        resident_before, pool_before = resident_now, pool_now
    # Rounds reuse 60% of the previous Host pages, so later rounds must hit.
    assert hits > 0


def test_host_batch_lru_reset_and_disable():
    _skip_if_unsupported()
    wrapper, (q_nope, q_pe, ckv, kpe, host_page_start, indices, expected) = _run(
        2, 8, 64, 512, 0.3, 0.5, repeats=1
    )
    wrapper.set_host_cache(torch.tensor([0, 8, 16], dtype=torch.int32), 1024)
    for _ in range(3):
        out = wrapper.run(q_nope, q_pe, ckv, kpe, host_page_start=host_page_start)
        torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)
    wrapper.reset_host_cache()
    out = wrapper.run(q_nope, q_pe, ckv, kpe, host_page_start=host_page_start)
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)
    wrapper.set_host_cache(torch.tensor([0, 16], dtype=torch.int32), 0)
    out = wrapper.run(q_nope, q_pe, ckv, kpe, host_page_start=host_page_start)
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)
