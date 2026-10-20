# NVLink fetch-CTA sweep: single-step (cached steady-state proxy: far = misses) and real LRU rounds.
set -x
cd /home/rtx/projects/flashinfer
export CUDA_DEVICE_ORDER=PCI_BUS_ID FLASHINFER_CUDA_ARCH_LIST=8.0 TMPDIR=$PWD/temp CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PWD FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_a100_final
P="numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python"
O=temp/a100_final
FM="batch:f8 batch:f16 batch:f24 batch:f32 batch:f48 batch:f64 batch:f80 batch:f96"
$P temp/h800/bench_sweep.py --tier peer --mtp 8 --requests 1 2 4 8 16 32 64 128 --fractions 0 0.1 0.2 0.3 --rounds 6 \
  --modes attn copy direct $FM --output $O/fetch_peer_mtp8.json 2>&1 | grep "far="
$P temp/h800/bench_sweep.py --tier host --mtp 8 --requests 1 4 16 64 128 --fractions 0 0.1 0.3 --rounds 6 \
  --modes attn copy direct batch:f4 batch:f8 batch:f16 batch:f32 --output $O/fetch_host_mtp8.json 2>&1 | grep "far="
for f in 16 32 64 96; do
  $P temp/h800/bench_rounds.py --tier peer --mtp 8 --requests 1 4 16 64 128 --passes 5 --fetch-ctas $f \
    --modes ours_stage ours_inplace --output $O/rounds_peer_mtp8_f$f.json 2>&1 | grep steady
done
for f in 16 32 64; do
  $P temp/h800/bench_rounds.py --tier peer --mtp 1 --requests 1 8 64 256 --passes 5 --fetch-ctas $f \
    --modes ours_stage ours_inplace --output $O/rounds_peer_mtp1_f$f.json 2>&1 | grep steady
done
echo FETCH_DONE
