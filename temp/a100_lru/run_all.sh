# Supplementary A100 runs (local GPU0, NVLink peer GPU1) while the H800 queue waits.
set -x
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_host_lru FLASHINFER_CUDA_ARCH_LIST=8.0 TMPDIR=$PWD/temp CUDA_VISIBLE_DEVICES=0,1
P=/mnt/data/mhy-vllm-venv/bin/python
R="1 2 3 4 6 8 12 16 24 32 48 64 80 96 112 128"
$P temp/h800/bench_sweep.py --tier host --mtp 8 --requests $R --rounds 6 --output temp/a100_lru/sweep_host_mtp8.json
$P temp/h800/bench_sweep.py --tier peer --mtp 8 --requests $R --rounds 6 --modes attn copy direct batch batch:f24 batch_inplace --output temp/a100_lru/sweep_peer_mtp8.json
$P temp/h800/bench_rounds.py --tier host --mtp 8 --requests 1 2 4 8 16 32 64 --passes 5 --modes ours_stage ours_inplace ours_nocache floor_fa2 --output temp/a100_lru/rounds_host_mtp8.json
$P temp/h800/bench_rounds.py --tier peer --mtp 8 --requests 1 2 4 8 16 32 64 --passes 5 --modes ours_stage ours_inplace ours_nocache floor_fa2 --output temp/a100_lru/rounds_peer_mtp8.json
