#!/usr/bin/env bash
#SBATCH --partition=h01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --chdir=/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer_tuned_20261029
#SBATCH --output=/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer/temp/h800/logs/%x-%j.out
# Pass host (one GPU, includes full tests) or peer (two GPUs) with an explicit Slurm allocation.
set -euo pipefail
root=/WORK/PUBLIC/luyouy_work/rtx
tree=$root/yjw/flashinfer_tuned_20261029
source "$root/yjw/flashinfer/temp/h800/env_job.sh" >/dev/null 2>&1
cd "$tree"
export PYTHONPATH=$tree:$tree/temp/h800:$root/yjw/flashinfer/temp/h800/pydeps
export FLASHINFER_WORKSPACE_BASE=$tree/temp/h800/jit TMPDIR=$tree/temp/h800/tmp
mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$TMPDIR"
tier=$1
out=$root/yjw/flashinfer/temp/h800/results/$SLURM_JOB_NAME-$SLURM_JOB_ID
mkdir -p "$out"
echo "job=$SLURM_JOB_ID tree=$tree tier=$tier start=$(date -Is)"
cat SOURCE_MANIFEST.json
sha256sum include/flashinfer/attention/mla_host_batch.cuh temp/h800/bench_rounds.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
if [[ $tier == host ]]; then
  python -m pytest -q -p no:cacheprovider -p tuned_jitpaths tests/attention/test_mla_host_batch.py \
    > "$out/pytest.log" 2>&1
  tail -n 2 "$out/pytest.log"
fi
capture="$root/yjw/test/glm_trace/replay/swe64k_assistant_doc*_20261029.pt"
for mtp in 1 8; do
  python temp/h800/bench_rounds.py --tier "$tier" --mtp "$mtp" --requests 1 4 16 \
    --captures "$capture" --context 65536 --steps "$((1024 / mtp))" --passes 7 \
    --fetch-ctas 16 --modes ours_stage ours_inplace memcpy_lru direct_lru hisparse floor_fa2 floor_flashmla \
    --output "$out/mtp$mtp.json" > "$out/mtp$mtp.log" 2>&1
  tail -n 4 "$out/mtp$mtp.log"
done
echo "end=$(date -Is)"
