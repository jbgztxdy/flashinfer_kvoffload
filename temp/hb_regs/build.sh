#!/bin/bash
# Purpose: Compile one batched Host kernel instantiation and print its register/stack use.
# Usage: temp/hb_regs/build.sh [extra nvcc flags]
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
/usr/local/cuda-13.1/bin/nvcc -cubin -o "$ROOT/temp/hb_regs/hb.cubin" "$ROOT/temp/hb_regs/hb.cu" \
  -I"$ROOT/include" -isystem /usr/local/cuda-13.1/include/cccl \
  -I"$ROOT/3rdparty/cutlass/include" -I"$ROOT/3rdparty/cutlass/tools/util/include" \
  --expt-relaxed-constexpr -gencode=arch=compute_80,code=sm_80 -std=c++17 -use_fast_math \
  -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DNDEBUG -O3 "$@"
/usr/local/cuda-13.1/bin/cuobjdump -res-usage "$ROOT/temp/hb_regs/hb.cubin" | grep -E "REG" | sed -E 's/.*REG:([0-9]+) STACK:([0-9]+).*/regs=\1 stack=\2/'
