#!/usr/bin/env bash
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=/WORK/PUBLIC/luyouy_work/rtx/yjw/test/glm_trace/logs/%x-%j.out
# CPU-only corpus build (tokenizing) on an A800 node. Usage: sbatch --job-name=glm-corpus --time=01:00:00 04_corpus.sbatch <01b args>
set -euo pipefail
cd /WORK/PUBLIC/luyouy_work/rtx
source env.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1
cd yjw/test/glm_trace
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
python 01b_build_corpus_swe.py "$@"
echo "end=$(date -Is)"
