# Purpose: Validate the direct-to-shared Host attention prototype.
# Usage: PYTHONPATH=. /mnt/data/mhy-vllm-venv/bin/python temp/test_glm52_pcie20.py --quick
import functools
import os

from flashinfer.jit import gen_batch_mla_module
from flashinfer.mla import _core
import test_glm52_host_overlap as validation


@functools.cache
def private_module(backend, *args, host_prefetch=False):
    specification = gen_batch_mla_module(backend, *args, host_prefetch=host_prefetch)
    if host_prefetch:
        chunks = int(os.environ.get("MLA_PRIVATE_CHUNKS", "28"))
        pipeline = os.environ.get("MLA_PRIVATE_PIPELINE", "1") == "1"
        specification.name += f"_private_shared_{chunks}_{int(pipeline)}"
        specification.extra_cuda_cflags.extend(
            [
                "-DFLASHINFER_MLA_HOST_PRIVATE_SHARED",
                f"-DFLASHINFER_MLA_HOST_PRIVATE_CHUNKS={chunks}",
            ]
        )
        if pipeline:
            specification.extra_cuda_cflags.append(
                "-DFLASHINFER_MLA_HOST_PRIVATE_PIPELINE"
            )
    return specification.build_and_load()


_core.get_batch_mla_module = private_module
if __name__ == "__main__":
    validation.main()
