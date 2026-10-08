# Purpose: Check mixed-tier MLA outputs, LSE, graph reuse, and Host mutations.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/test_glm52_host_overlap.py
import argparse
import gc
import json
import math

import torch

from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost, make_indices
from flashinfer.mla import BatchMLAPagedAttentionWrapper


def check_case(wrapper, q, qpe, cache, host, selected, out, lse, graph, base_e):
    device = q.device
    boundary = cache.shape[0] // 2
    wrapper._kv_indices_buf.copy_(selected)
    graph.replay()
    torch.cuda.synchronize(device)
    local = selected < boundary
    rows = torch.empty((2048, 576), dtype=cache.dtype)
    if local.any():
        rows[local] = cache[selected[local].long(), 0].cpu()
    if (~local).any():
        rows[~local] = host[selected[~local].long() - boundary, 0]
    rows = rows.to(device).float()
    scores = (q[0].float() @ rows[:, :512].T + qpe[0].float() @ rows[:, 512:].T) / 16
    expected = scores.softmax(-1) @ rows[:, :512]
    expected_lse = scores.logsumexp(-1)
    if not base_e:
        expected_lse /= math.log(2)
    torch.testing.assert_close(out[0].float(), expected, rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(lse[0], expected_lse, rtol=2e-3, atol=2e-3)
    return (out[0].float() - expected).abs().max().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(1)
    torch.set_num_threads(8)
    torch.manual_seed(123)
    cases = 0
    maximum_error = 0.0
    for dtype in [torch.bfloat16] if args.quick else [torch.bfloat16, torch.float16]:
        allocation = MixedHBMHost(131072, 1)
        cache = allocation.tensor(1).view(dtype)
        host = allocation.host_tensor().view(dtype)
        cache[:65536].normal_()
        host.normal_()
        q = torch.randn((1, 64, 512), device="cuda:1", dtype=dtype)
        qpe = torch.randn((1, 64, 64), device="cuda:1", dtype=dtype)
        out = torch.empty_like(q)
        lse = torch.empty((1, 64), device="cuda:1", dtype=torch.float32)
        indices = torch.empty(2048, device="cuda:1", dtype=torch.int32)
        workspace = torch.empty(32 * 2**20, dtype=torch.uint8, device="cuda:1")
        wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
        wrapper.plan(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 2048], dtype=torch.int32),
            indices,
            torch.tensor([2048], dtype=torch.int32),
            64,
            512,
            64,
            1,
            False,
            1 / 16,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        indices.copy_(torch.arange(2048, dtype=torch.int32))
        wrapper.run(
            q,
            qpe,
            cache[..., :512],
            cache[..., 512:],
            out=out,
            lse=lse,
            return_lse=True,
            host_page_start=65536,
        )
        graphs = []
        for base_e in [False] if args.quick else [False, True]:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                wrapper.run(
                    q,
                    qpe,
                    cache[..., :512],
                    cache[..., 512:],
                    out=out,
                    lse=lse,
                    return_lse=True,
                    return_lse_base_on_e=base_e,
                    host_page_start=65536,
                )
            graphs.append(graph)
            for layout in ["random"] if args.quick else ["random", "head", "tail"]:
                for seed in [5] if args.quick else [5, 43]:
                    selected_sets = make_indices(
                        131072,
                        2048,
                        (
                            [0, 0.2, 0.25, 0.3, 1]
                            if args.quick
                            else [
                                0,
                                0.1,
                                0.15,
                                0.2,
                                448 / 2048,
                                449 / 2048,
                                0.25,
                                513 / 2048,
                                0.3,
                                0.4,
                                896 / 2048,
                                897 / 2048,
                                0.6,
                                0.8,
                                1,
                            ]
                        ),
                        seed,
                        layout,
                    )
                    for selected in selected_sets.values():
                        maximum_error = max(
                            maximum_error,
                            check_case(
                                wrapper,
                                q,
                                qpe,
                                cache,
                                host,
                                selected,
                                out,
                                lse,
                                graph,
                                base_e,
                            ),
                        )
                        cases += 1
            # Changing Host contents must invalidate data from the previous replay.
            selected = make_indices(131072, 2048, [0.2], 71, "random")[0.2]
            for change in [0.25, -0.5, 0.75]:
                torch.cuda.synchronize()
                host.add_(change)
                maximum_error = max(
                    maximum_error,
                    check_case(
                        wrapper, q, qpe, cache, host, selected, out, lse, graph, base_e
                    ),
                )
                cases += 1
        del graph, graphs, wrapper, cache, host, q, qpe, out, lse, workspace, indices
        gc.collect()
        torch.cuda.synchronize()
        allocation.close()
    print(
        json.dumps(
            dict(
                passed=True,
                cases=cases,
                maximum_output_absolute_error=maximum_error,
                dtypes=["bfloat16"] if args.quick else ["bfloat16", "float16"],
                lse_bases=["2"] if args.quick else ["2", "e"],
                quick_mode=args.quick,
                host_mutations=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
