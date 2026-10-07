# Purpose: Compare local and remote CUDA VMM KV gather latency without MLA compute.
# Usage: PYTHONPATH=. python temp/bench_glm52_sparse_decode_p2p_gather.py

import statistics
import importlib.util
import pathlib

import torch

_path = pathlib.Path(__file__).with_name("bench_glm52_sparse_decode_p2p.py")
_spec = importlib.util.spec_from_file_location("p2p_bench", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
MixedHBM = _module.MixedHBM
make_indices = _module.make_indices


def timed_index_select(source, indices, repeats=100):
    output = torch.empty((indices.numel(), source.shape[-1]), dtype=source.dtype, device=source.device)
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        start.record()
        for _ in range(repeats):
            torch.index_select(source[:, 0], 0, indices, out=output)
        end.record()
    graph.replay()
    end.synchronize()
    samples = []
    for _ in range(20):
        graph.replay()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    return statistics.median(samples), output


def main():
    torch.cuda.set_device(0)
    tokens, topk = 131072, 2048
    allocation = MixedHBM(tokens, 0, 1)
    cache = allocation.tensor(0)
    local_cache = allocation.tensor(0, 0, tokens // 2)
    remote_cache = allocation.tensor(1, tokens // 2, tokens // 2)
    with torch.cuda.device(0):
        torch.randn(local_cache.shape, out=local_cache)
    with torch.cuda.device(1):
        torch.randn(remote_cache.shape, out=remote_cache)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    ckv = cache[..., :512]
    sets = make_indices(tokens, topk, [0.0, 1.0], 42, "random")
    local = sets[0.0].to(device="cuda:0", dtype=torch.int64)
    remote = sets[1.0].to(device="cuda:0", dtype=torch.int64)
    local_us, local_out = timed_index_select(ckv, local)
    remote_us, remote_out = timed_index_select(ckv, remote)
    torch.cuda.synchronize(0)
    print(f"local_index_select_us={local_us:.3f}")
    print(f"remote_index_select_us={remote_us:.3f}")
    print(f"remote_to_local_ratio={remote_us / local_us:.3f}")
    print(f"residency={allocation.residency}")
    assert local_out.shape == remote_out.shape == (topk, 512)
    del local_out, remote_out, local, remote, ckv, cache, local_cache, remote_cache
    allocation.close()


if __name__ == "__main__":
    main()
