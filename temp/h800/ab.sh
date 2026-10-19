# A/B on the same GPU: the batched Host kernel before the LRU work (52f289a1) and now, cache off.
set -x
BASE=/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer_base
NEW=/WORK/PUBLIC/luyouy_work/rtx/yjw/flashinfer
for round in 1 2; do
  for tree in $BASE $NEW; do
    tag=$(basename $tree)
    (cd $tree && FLASHINFER_WORKSPACE_BASE=$tree/temp/h800/jit_ab PYTHONPATH=$tree:$NEW/temp/h800/pydeps \
      python temp/h800/bench_sweep.py --tier host --mtp 8 --requests 1 4 16 64 128 --fractions 0 0.1 0.3 \
      --modes attn batch --rounds 6 --no-check --output $OUT/ab_${tag}_r$round.json 2>&1 | grep -v Warn | grep "far=")
  done
done
