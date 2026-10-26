# Emulated far link at 100 and 50 GB/s (plus the unlimited emulation build as the hook-overhead
# reference): cross-decode rounds with the LRU for ours (in place, staged with 8 and 16 fetch
# CTAs) and the LRU baselines. The all-HBM floor comes from the normal build (no hooks).
set -e
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. FLASHINFER_CUDA_ARCH_LIST=8.0 TMPDIR=$PWD/temp CUDA_VISIBLE_DEVICES=0,1 CUDA_HOME=/usr/local/cuda
O=temp/sim_link/bw; mkdir -p $O
R="numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python temp/h800/bench_rounds.py --tier peer --passes 3"
Q8="4 16 64 128"; Q1="16 128 256"
[ -f $O/floor_mtp8.json ] || FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_a100_final $R --mtp 8 --requests $Q8 --modes floor_fa2 --output $O/floor_mtp8.json
[ -f $O/floor_mtp1.json ] || FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_a100_final $R --mtp 1 --requests $Q1 --modes floor_fa2 --output $O/floor_mtp1.json
export FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_FAR_EMU FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_emu
for G in 0 100 50; do
  /mnt/data/mhy-vllm-venv/bin/python temp/sim_link/validate.py $G 0 $O/validate_${G}.json
  for M in 8 1; do
    Q=$([ $M = 8 ] && echo $Q8 || echo $Q1)
    [ -f $O/g${G}_mtp${M}.json ] || $R --mtp $M --requests $Q --modes ours_inplace ours_stage memcpy_lru direct_lru --emu $G:0 --output $O/g${G}_mtp${M}.json
    [ -f $O/g${G}_mtp${M}_f16.json ] || $R --mtp $M --requests $Q --modes ours_stage --fetch-ctas 16 --emu $G:0 --output $O/g${G}_mtp${M}_f16.json
  done
done
echo BW_DONE
