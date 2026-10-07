# Purpose: Compare cacheable and write-combined pinned DRAM sparse read latency.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/bench_sparse_host_allocators.py > temp/sparse_host_allocators.json
import ctypes
import gc
import json
import os
import statistics

import torch
from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost, make_indices

os.sched_setaffinity(0, set(range(32)) | set(range(64, 96)))
torch.set_num_threads(8)
torch.cuda.set_device(1)
torch.empty(1, device='cuda:1')
driver = ctypes.CDLL('libcuda.so.1')
def check(status):
    if status != 0:
        raise RuntimeError(f'CUDA Driver error {status}')
module, function = ctypes.c_void_p(), ctypes.c_void_p()
check(driver.cuModuleLoad(ctypes.byref(module), b'temp/bench_sparse_host_fetch.cubin'))
check(driver.cuModuleGetFunction(ctypes.byref(function), module, b'sparse_fetch'))
context = 1048576
indices = torch.empty(2048, dtype=torch.int32, device='cuda:1')
output = torch.empty((2048, 576), dtype=torch.bfloat16, device='cuda:1')
stream = torch.cuda.Stream(device=1)
results = []
for kind, flags in [('vmm_host_numa', None), ('mapped_cacheable', 3), ('mapped_write_combined', 7)]:
    if flags is None:
        allocation = MixedHBMHost(context, 1)
        cache = allocation.tensor(1)
        mapped = cache.data_ptr() + context//2*1152
        host = allocation.host_tensor()
    else:
        cpu_pointer, device_pointer = ctypes.c_void_p(), ctypes.c_uint64()
        check(driver.cuMemHostAlloc(ctypes.byref(cpu_pointer), ctypes.c_size_t(context//2*1152), ctypes.c_uint(flags)))
        check(driver.cuMemHostGetDevicePointer_v2(ctypes.byref(device_pointer), cpu_pointer, ctypes.c_uint(0)))
        mapped = device_pointer.value
        buffer = (ctypes.c_uint16 * (context//2*576)).from_address(cpu_pointer.value)
        host = torch.frombuffer(buffer, dtype=torch.bfloat16).view(context//2, 1, 576)
    host.copy_(torch.randn_like(host))
    values = [ctypes.c_uint64(mapped), ctypes.c_uint64(indices.data_ptr()), ctypes.c_uint64(output.data_ptr()), ctypes.c_int(context//2)]
    params = (ctypes.c_void_p * 4)(*[ctypes.addressof(v) for v in values])
    def launch():
        check(driver.cuLaunchKernel(function, 108, 1, 1, 256, 1, 1, 0, ctypes.c_void_p(stream.cuda_stream), params, None))
    with torch.cuda.stream(stream):
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
            slots = mask.nonzero().flatten().to('cuda:1')
            expected = host[selected[mask].long()-context//2, 0].clone()
            torch.testing.assert_close(output[slots].cpu(), expected, rtol=0, atol=0)
            if fraction == .2:
                # Verify a fresh CPU write is visible after repeated graph executions.
                host.copy_(torch.zeros_like(host))
                graph.replay()
                stream.synchronize()
                torch.testing.assert_close(output[slots].cpu(), torch.zeros_like(expected), rtol=0, atol=0)
            latency = statistics.median(times)
            results.append(dict(allocator=kind, flags=flags, fraction=fraction, median_us=latency, GB_s=int(mask.sum())*1152/latency/1000, cpu_pointer=hex(host.data_ptr()), mapped_pointer=hex(mapped), verified=True))
    del graph, host
    gc.collect()
    torch.cuda.synchronize()
    if flags is None:
        del cache
        allocation.close()
    else:
        check(driver.cuMemFreeHost(cpu_pointer))
print(json.dumps(results, indent=2))
