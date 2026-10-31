# Experiment-only helpers for the emulated far link: the emulated row gather (Memcpy baselines)
# and the environment for the kernels' emulation build.
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[2]
_mod = None


def kernel_env(gbps, latency_ns, far_begin, far_end):
    """Settings read by every kernel launch (see far_link_emu.cuh); [far_begin, far_end) is the
    far tier's address range (the native kernel counts reads inside it as far)."""
    os.environ["FLASHINFER_FAR_EMU"] = f"{gbps}:{int(latency_ns)}:{int(far_begin)}:{int(far_end)}"


def _load():
    global _mod
    if _mod is None:
        _mod = load("far_emu_gather", [str(Path(__file__).with_name("emu_gather.cu"))],
                    extra_include_paths=[str(ROOT / "include")], extra_cuda_cflags=["-O3", "-arch=sm_80"],
                    build_directory=str(ROOT / "temp/sim_link/build"), verbose=False)
    return _mod


def latency_ns(src, ids, steps=200):
    """Mean latency of one dependent far row read over the emulated link."""
    return _load().latency(src, ids.to(torch.int64), steps)


def gather(src, ids, out):
    """out[i] = src[ids[i]] over the emulated link (rows of 1152 bytes)."""
    _load().gather(src, ids.to(torch.int64), out)
    return out
