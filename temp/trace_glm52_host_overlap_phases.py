# Purpose: Instrument local computation, Host row readiness, and Host partial writes.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/trace_glm52_host_overlap_phases.py
import functools
import sys

import bench_glm52_sparse_decode_hbm_host_direct as benchmark
from flashinfer.jit import gen_batch_mla_module
from flashinfer.mla import _core


@functools.cache
def instrumented_module(backend, *args, host_prefetch=False):
    specification = gen_batch_mla_module(backend, *args, host_prefetch=host_prefetch)
    if host_prefetch:
        specification.name += "_phase_trace"
        specification.extra_cuda_cflags.append("-DFLASHINFER_MLA_HOST_TRACE")
    return specification.build_and_load()


_core.get_batch_mla_module = instrumented_module
sys.argv = [
    sys.argv[0],
    "--device",
    "1",
    "--contexts",
    "1048576",
    "--fractions",
    "0",
    ".2",
    "--rounds",
    "1",
    "--index-sets",
    "1",
    "--hot-repeats",
    "10",
    "--host-prefetch",
    "--record-host-trace",
    "--output",
    "temp/glm52_overlap_host_final_phase_trace.json",
]
benchmark.main()
