# Emulated far-link latency (extra 2/5/10/20 us per access) x bandwidth (unlimited/100/50 GB/s):
# cross-decode rounds with the LRU, ours (in place; staged with 16 fetch CTAs) vs LRU baselines.
set -e
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. FLASHINFER_CUDA_ARCH_LIST=8.0 TMPDIR=$PWD/temp CUDA_VISIBLE_DEVICES=0,1 CUDA_HOME=/usr/local/cuda
export FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_FAR_EMU FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_emu
O=temp/sim_link/lat; mkdir -p $O
R="numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python temp/h800/bench_rounds.py --tier peer --passes 3"
for L in 2000 5000 10000 20000; do
  for G in 0 100 50; do
    /mnt/data/mhy-vllm-venv/bin/python temp/sim_link/validate.py $G $L $O/validate_g${G}_l${L}.json
    [ -f $O/g${G}_l${L}_mtp8.json ] || $R --mtp 8 --requests 4 16 64 128 --modes ours_inplace memcpy_lru direct_lru --emu $G:$L --output $O/g${G}_l${L}_mtp8.json
    [ -f $O/g${G}_l${L}_mtp8_f16.json ] || $R --mtp 8 --requests 4 16 64 128 --modes ours_stage --fetch-ctas 16 --emu $G:$L --output $O/g${G}_l${L}_mtp8_f16.json
    [ -f $O/g${G}_l${L}_mtp1.json ] || $R --mtp 1 --requests 16 128 256 --modes ours_inplace ours_stage memcpy_lru direct_lru --fetch-ctas 16 --emu $G:$L --output $O/g${G}_l${L}_mtp1.json
  done
done
echo LAT_DONE
