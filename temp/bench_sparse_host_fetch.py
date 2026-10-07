# Purpose: Bound sparse Host transfer latency without attention math.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/bench_sparse_host_fetch.py > temp/sparse_host_fetch_bandwidth.json
import ctypes
import json
import statistics
import torch
from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost, make_indices

torch.cuda.set_device(1)
driver = ctypes.CDLL('libcuda.so.1')
module, function = ctypes.c_void_p(), ctypes.c_void_p()
def check(status):
    if status != 0:
        raise RuntimeError(f'CUDA Driver error {status}')
check(driver.cuModuleLoad(ctypes.byref(module), b'temp/bench_sparse_host_fetch.cubin'))
check(driver.cuModuleGetFunction(ctypes.byref(function), module, b'sparse_fetch'))
results = []
for context in [131072, 1048576]:
    allocation = MixedHBMHost(context, 1)
    cache = allocation.tensor(1)
    allocation.host_tensor().normal_()
    indices = torch.empty(2048, dtype=torch.int32, device='cuda:1')
    output = torch.empty((2048, 576), dtype=torch.bfloat16, device='cuda:1')
    values = [ctypes.c_uint64(cache.data_ptr() + context//2*1152), ctypes.c_uint64(indices.data_ptr()), ctypes.c_uint64(output.data_ptr()), ctypes.c_int(context//2)]
    params = (ctypes.c_void_p * 4)(*[ctypes.addressof(v) for v in values])
    for blocks in [64, 108, 216, 432]:
        stream = torch.cuda.Stream(device=1)
        def launch():
            check(driver.cuLaunchKernel(function, blocks, 1, 1, 256, 1, 1, 0, ctypes.c_void_p(stream.cuda_stream), params, None))
        indices.copy_(make_indices(context, 2048, [0.2], 42, 'random')[0.2])
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            launch()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(100): launch()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for fraction in [.1, .15, .2, 1.]:
                selected = make_indices(context, 2048, [fraction], 42, 'random')[fraction]
                indices.copy_(selected)
                times = []
                for _ in range(20):
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end)*10)
                mask = selected >= context//2
                reference = allocation.host_tensor()[selected[mask].long()-context//2, 0]
                torch.testing.assert_close(output[selected.ge(context//2).nonzero().flatten().to('cuda:1')].cpu(), reference, rtol=0, atol=0)
                latency = statistics.median(times)
                results.append(dict(context=context, blocks=blocks, fraction=fraction, median_us=latency, GB_s=int(mask.sum())*1152/latency/1000))
        del graph
    del cache
    allocation.close()
print(json.dumps(results, indent=2))
