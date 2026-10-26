# H800 NVLink tuning: H800 NVLink (~177 GB/s) is slow relative to its compute, so the in-place
# duplicate reads may cost more than on A100; compare with staged (dedup) at several fetch-CTA counts.
set -x
for F in 16 32 64; do
  python temp/h800/bench_rounds.py --tier peer --mtp 8 --requests 16 64 128 --passes 3 --fetch-ctas $F --modes ours_stage memcpy_lru --output $OUT/rounds_mtp8_stage_f$F.json 2>&1 | grep steady
done
python temp/h800/bench_rounds.py --tier peer --mtp 8 --requests 16 64 128 --passes 3 --modes ours_inplace hisparse floor_fa2 --output $OUT/rounds_mtp8_inplace.json 2>&1 | grep steady
