#!/usr/bin/env bash
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --output=/WORK/PUBLIC/luyouy_work/rtx/yjw/test/glm_trace/logs/%x-%j.out
# GLM-5.3 indexer top-k traces on A800 (1 GPU by default; up to 8 approved for this task: NPROC and --gres).
# Usage: sbatch --job-name=glm-trace --time=02:00:00 03_trace.sbatch <02_prefill_trace.py args>
set -euo pipefail
cd /WORK/PUBLIC/luyouy_work/rtx
source env.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 TMPDIR=/WORK/PUBLIC/luyouy_work/rtx/tmp NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
cd yjw/test/glm_trace
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
torchrun --standalone --nproc-per-node ${NPROC:-1} 02_prefill_trace.py "$@"
echo "end=$(date -Is) status=$?"
