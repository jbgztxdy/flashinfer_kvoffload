# Final A100 measurement set at the current kernel (GPU0 near; far = NUMA0 pinned DRAM over PCIe,
# or GPU1 HBM over NV12 NVLink). HiSparse needs FlashMLA (SM90+): H800 only.
set -x
cd /home/rtx/projects/flashinfer
export CUDA_DEVICE_ORDER=PCI_BUS_ID FLASHINFER_CUDA_ARCH_LIST=8.0 TMPDIR=$PWD/temp CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PWD FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_a100_final
P="numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python"
O=temp/a100_full
run() { local out=$1; shift; [ -f $O/$out ] && return; "$@" --output $O/$out 2>&1 | grep -v Warning; }
R8="1 2 3 4 5 6 7 8 10 12 14 16 20 24 28 32 40 48 56 64 72 80 96 112 128"
R1="1 2 4 8 16 32 48 64 96 128 192 256 384 512"
R1P="1 2 4 8 16 32 48 64 96 128 192 256"
F="0 0.05 0.1 0.15 0.2 0.25 0.3"
# Single decode step, no cache: compute floor, Memcpy, Direct, ours (staged).
run sweep_host_mtp8.json $P temp/h800/bench_sweep.py --tier host --mtp 8 --requests $R8 --fractions $F --rounds 8 --modes attn copy direct batch
run sweep_peer_mtp8.json $P temp/h800/bench_sweep.py --tier peer --mtp 8 --requests $R8 --fractions $F --rounds 8 --modes attn copy direct batch
run sweep_host_mtp1.json $P temp/h800/bench_sweep.py --tier host --mtp 1 --requests $R1 --fractions $F --rounds 8 --modes attn copy direct batch
run sweep_peer_mtp1.json $P temp/h800/bench_sweep.py --tier peer --mtp 1 --requests $R1P --fractions $F --rounds 8 --modes attn copy direct batch
# Consecutive decode steps with the cross-step LRU (all KV far but the newest token), against the
# same LRU around Memcpy and Direct.
Q8="1 2 3 4 6 8 12 16 24 32 48 64 96 128"
Q1="1 2 4 8 16 32 64 128 256 512"
Q1P="1 2 4 8 16 32 64 128 256"
run rounds_host_mtp8.json $P temp/h800/bench_rounds.py --tier host --mtp 8 --requests $Q8 --passes 5 --modes ours_stage ours_nocache memcpy_lru direct_lru floor_fa2
run rounds_peer_mtp8.json $P temp/h800/bench_rounds.py --tier peer --mtp 8 --requests $Q8 --passes 5 --modes ours_inplace ours_nocache memcpy_lru direct_lru floor_fa2
run rounds_host_mtp1.json $P temp/h800/bench_rounds.py --tier host --mtp 1 --requests $Q1 --passes 5 --modes ours_stage ours_nocache memcpy_lru direct_lru floor_fa2
run rounds_peer_mtp1.json $P temp/h800/bench_rounds.py --tier peer --mtp 1 --requests $Q1P --passes 5 --modes ours_inplace ours_nocache memcpy_lru direct_lru floor_fa2
# LRU capacity per request (MTP8).
for C in 2048 8192; do
  run rounds_host_mtp8_c$C.json $P temp/h800/bench_rounds.py --tier host --mtp 8 --requests 1 4 16 64 128 --capacity $C --passes 5 --modes ours_stage memcpy_lru direct_lru
  run rounds_peer_mtp8_c$C.json $P temp/h800/bench_rounds.py --tier peer --mtp 8 --requests 1 4 16 64 128 --capacity $C --passes 5 --modes ours_inplace memcpy_lru direct_lru
done
echo ALL_DONE
