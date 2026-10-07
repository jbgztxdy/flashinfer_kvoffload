# Purpose: Verify native MLA regression and mixed-tier input validation.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/test_glm52_host_overlap_api.py
import json
import math

import torch

from bench_glm52_decode import BatchMLAPagedAttentionWrapper


def make_wrapper(qptr, kvptr, indices, lens, heads, page_size, causal, dtype):
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(32 * 2**20, dtype=torch.uint8, device="cuda:1"), backend="fa2"
    )
    wrapper.plan(
        torch.tensor(qptr, dtype=torch.int32),
        torch.tensor(kvptr, dtype=torch.int32),
        indices,
        torch.tensor(lens, dtype=torch.int32),
        heads,
        512,
        64,
        page_size,
        causal,
        1 / 16,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    return wrapper


def main():
    torch.cuda.set_device(1)
    torch.manual_seed(76)
    native_cases = 0
    for dtype in (torch.bfloat16, torch.float16):
        for heads, page_size, batch, query_length, causal in (
            (64, 16, 2, 2, False),
            (64, 16, 2, 2, True),
            (128, 1, 1, 1, False),
        ):
            length = 70
            pages = (length + page_size - 1) // page_size
            cache = torch.randn(
                (pages * batch, page_size, 576), dtype=dtype, device="cuda:1"
            )
            q = torch.randn(
                (query_length * batch, heads, 512), dtype=dtype, device="cuda:1"
            )
            qpe = torch.randn(
                (query_length * batch, heads, 64), dtype=dtype, device="cuda:1"
            )
            indices = torch.arange(pages * batch, dtype=torch.int32, device="cuda:1")
            wrapper = make_wrapper(
                [i * query_length for i in range(batch + 1)],
                [i * pages for i in range(batch + 1)],
                indices,
                [length] * batch,
                heads,
                page_size,
                causal,
                dtype,
            )
            output, lse = wrapper.run(
                q, qpe, cache[..., :512], cache[..., 512:], return_lse=True
            )
            for request in range(batch):
                rows = (
                    cache[request * pages : (request + 1) * pages]
                    .reshape(-1, 576)[:length]
                    .float()
                )
                for offset in range(query_length):
                    token = request * query_length + offset
                    scores = (
                        q[token].float() @ rows[:, :512].T
                        + qpe[token].float() @ rows[:, 512:].T
                    ) / 16
                    if causal:
                        scores[:, length - query_length + offset + 1 :] = -torch.inf
                    expected = scores.softmax(-1) @ rows[:, :512]
                    torch.testing.assert_close(
                        output[token].float(), expected, rtol=1e-2, atol=5e-3
                    )
                    torch.testing.assert_close(
                        lse[token], scores.logsumexp(-1) / math.log(2), rtol=2e-3, atol=2e-3
                    )
            native_cases += 1

    dtype = torch.bfloat16
    indices = torch.arange(2048, dtype=torch.int32, device="cuda:1")
    cache = torch.randn((2048, 1, 576), dtype=dtype, device="cuda:1")
    q = torch.randn((1, 64, 512), dtype=dtype, device="cuda:1")
    qpe = torch.randn((1, 64, 64), dtype=dtype, device="cuda:1")
    wrapper = make_wrapper([0, 1], [0, 2048], indices, [2048], 64, 1, False, dtype)
    invalid_cases = 0

    def expect_error(call, error=ValueError):
        nonlocal invalid_cases
        try:
            call()
        except error:
            invalid_cases += 1
        else:
            raise AssertionError("Invalid Host overlap input was accepted")

    for boundary in (-1, 2049, 1024.5):
        expect_error(
            lambda: wrapper.run(
                q, qpe, cache[..., :512], cache[..., 512:], host_page_start=boundary
            )
        )
    expect_error(
        lambda: wrapper.run(
            q, qpe.float(), cache[..., :512], cache[..., 512:], host_page_start=1024
        )
    )
    expect_error(
        lambda: wrapper.run(
            q, qpe, cache[..., :256], cache[..., 512:], host_page_start=1024
        )
    )
    strided = torch.empty((2048, 1, 1024), dtype=dtype, device="cuda:1")[..., ::2]
    expect_error(
        lambda: wrapper.run(q, qpe, strided, cache[..., 512:], host_page_start=1024)
    )
    cutlass = BatchMLAPagedAttentionWrapper(
        wrapper._float_workspace_buffer, backend="cutlass"
    )
    expect_error(
        lambda: cutlass.run(
            q, qpe, cache[..., :512], cache[..., 512:], host_page_start=1024
        )
    )
    short_wrapper = make_wrapper(
        [0, 1], [0, 2047], indices[:2047], [2047], 64, 1, False, dtype
    )
    expect_error(
        lambda: short_wrapper.run(
            q, qpe, cache[..., :512], cache[..., 512:], host_page_start=1024
        )
    )
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "passed": True,
                "native_cases": native_cases,
                "invalid_input_cases": invalid_cases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
