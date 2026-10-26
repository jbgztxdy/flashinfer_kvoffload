# Gate: emulation build with an unlimited emulated link vs the normal build, on the rounds
# workload the tests use (NVLink MTP8/MTP1). Every method must stay within a few percent.
set -e
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. FLASHINFER_CUDA_ARCH_LIST=8.0 TMPDIR=$PWD/temp CUDA_VISIBLE_DEVICES=0,1 CUDA_HOME=/usr/local/cuda
O=temp/sim_link/gate; mkdir -p $O
P="numactl -N0 -m0 /mnt/data/mhy-vllm-venv/bin/python temp/h800/bench_rounds.py --tier peer --passes 3"
M="ours_inplace ours_stage memcpy_lru direct_lru floor_fa2"
FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_a100_final $P --mtp 8 --requests 4 16 64 128 --modes $M --output $O/normal_mtp8.json 2>&1 | grep steady
FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_FAR_EMU FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_emu $P --mtp 8 --requests 4 16 64 128 --modes $M --emu 0:0 --output $O/emu_mtp8.json 2>&1 | grep steady
FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_a100_final $P --mtp 1 --requests 16 128 256 --modes $M --output $O/normal_mtp1.json 2>&1 | grep steady
FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_FAR_EMU FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_emu $P --mtp 1 --requests 16 128 256 --modes $M --emu 0:0 --output $O/emu_mtp1.json 2>&1 | grep steady
