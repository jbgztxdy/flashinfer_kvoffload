# Full local A100 measurement set (GPU0 near, GPU0's PCIe Host link or GPU1 over NV12 as the far
# tier). HiSparse needs FlashMLA (SM90+), so it runs on H800 only.
set -x
cd /home/rtx/projects/flashinfer
export CUDA_DEVICE_ORDER=PCI_BUS_ID FLASHINFER_CUDA_ARCH_LIST=8.0 TMPDIR=$PWD/temp CUDA_VISIBLE_DEVICES=0,1
P="numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python"
O=temp/a100_final
BASE=/home/rtx/projects/flashinfer_base
R8="1 2 3 4 5 6 8 10 12 14 16 20 24 28 32 40 48 56 64 80 96 112 128"
R1="1 2 4 8 16 32 48 64 96 128 192 256 384 512"
R1P="1 2 4 8 16 32 48 64 96 128 192 256"
F="0 0.05 0.1 0.15 0.2 0.25 0.3"
run() { local out=$1; shift; [ -f $O/$out ] && return; "$@" --output $O/$out 2>&1 | grep -v Warning; }
export PYTHONPATH=$PWD FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_a100_final
# 1. A/B against the kernel before the LRU work (cache off), two interleaved rounds.
for i in 1 2; do
  (cd $BASE && PYTHONPATH=$BASE FLASHINFER_WORKSPACE_BASE=$BASE/temp/jit_ab $P temp/h800/bench_sweep.py --tier host --mtp 8 \
     --requests 1 4 16 64 128 --fractions 0 0.1 0.3 --modes attn batch --rounds 6 --no-check \
     --output $PWD/../flashinfer/$O/ab_base_r$i.json 2>&1 | grep "far=")
  run ab_new_r$i.json $P temp/h800/bench_sweep.py --tier host --mtp 8 --requests 1 4 16 64 128 --fractions 0 0.1 0.3 \
     --modes attn batch --rounds 6 --no-check
done
# 2. Single-step batch sweeps.
run sweep_host_mtp8.json $P temp/h800/bench_sweep.py --tier host --mtp 8 --requests $R8 --fractions $F --rounds 8 --modes attn copy direct batch
run sweep_peer_mtp8.json $P temp/h800/bench_sweep.py --tier peer --mtp 8 --requests $R8 --fractions $F --rounds 8 --modes attn copy direct batch batch:f24
run sweep_host_mtp1.json $P temp/h800/bench_sweep.py --tier host --mtp 1 --requests $R1 --fractions $F --rounds 8 --modes attn copy direct batch
run sweep_peer_mtp1.json $P temp/h800/bench_sweep.py --tier peer --mtp 1 --requests $R1P --fractions $F --rounds 8 --modes attn copy direct batch batch:f24
# 3. Consecutive decode steps with the cross-step LRU (all KV far except the newest token).
M="ours_stage ours_inplace ours_nocache floor_fa2"
run rounds_host_mtp8.json $P temp/h800/bench_rounds.py --tier host --mtp 8 --requests 1 2 4 8 16 32 64 128 --passes 5 --modes $M
run rounds_peer_mtp8.json $P temp/h800/bench_rounds.py --tier peer --mtp 8 --requests 1 2 4 8 16 32 64 128 --passes 5 --modes $M
run rounds_host_mtp1.json $P temp/h800/bench_rounds.py --tier host --mtp 1 --requests 1 2 4 8 16 32 64 128 256 512 --passes 5 --modes $M
run rounds_peer_mtp1.json $P temp/h800/bench_rounds.py --tier peer --mtp 1 --requests 1 2 4 8 16 32 64 128 256 --passes 5 --modes $M
# 4. LRU capacity per request, MTP8.
for C in 2048 8192; do
  run rounds_host_mtp8_c$C.json $P temp/h800/bench_rounds.py --tier host --mtp 8 --requests 1 4 16 64 --capacity $C --passes 5 --modes ours_stage ours_inplace
  run rounds_peer_mtp8_c$C.json $P temp/h800/bench_rounds.py --tier peer --mtp 8 --requests 1 4 16 64 --capacity $C --passes 5 --modes ours_stage ours_inplace
done
echo ALL_DONE
