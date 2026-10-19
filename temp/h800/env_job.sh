# Source inside a Slurm job on kaituo1000 (g40, H800). Everything stays under ~/WORK.
RTX=/WORK/PUBLIC/luyouy_work/rtx
FI=$RTX/yjw/flashinfer
source "$RTX/env.sh"
# Host /usr/local/cuda is 12.4; torch is cu130, so JIT with the self-consistent CUDA 13.0 shim.
export CUDA_HOME=$RTX/cache/cuda130_home CUDA_PATH=$RTX/cache/cuda130_home
export PATH=$CUDA_HOME/bin:$PATH
export FLASHINFER_WORKSPACE_BASE=$FI/temp/h800/jit
export FLASHINFER_CUDA_ARCH_LIST=9.0a
export TMPDIR=$FI/temp/h800/tmp
export SGLANG_CACHE_DIR=$RTX/cache/sglang SGLANG_JIT_CACHE_DIR=$RTX/cache/sglang/jit
export TVM_FFI_CACHE_DIR=$RTX/cache/tvm_ffi
export PYTHONPATH=$FI:$FI/temp/h800/pydeps
mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$TMPDIR"
cd "$FI"
