#!/usr/bin/env bash
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/WORK/PUBLIC/luyouy_work/rtx/yjw/test/glm_trace/logs/%x-%j.out
# Inspect every document, preserving the strict gate exit status in each JSON.
set -euo pipefail
cd /WORK/PUBLIC/luyouy_work/rtx
source env.sh >/dev/null 2>&1
cd yjw/test/glm_trace
trace_dir=$1
result_prefix=$2
status=0
for doc in 0 1 2 3; do
  if ! python 05_inspect_trace.py --trace "$trace_dir/doc$doc.pt" \
      --loss "$trace_dir/loss.pt" --out "${result_prefix}_doc$doc.json"; then
    status=1
  fi
done
exit "$status"
