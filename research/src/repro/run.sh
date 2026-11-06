#!/usr/bin/env bash
# Reproduction: E2/SN2 on the five public Heid and Green folds (explicit H) and RDB7 on our three
# random 80/10/10 splits with explicit H maps. E2/SN2 at 100 epochs (Heid and Green), RDB7 at 50
# epochs (the grid setting, as reported); fixed config.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/repro
mkdir -p $OUT
jobs=()
for k in 0 1 2 3 4; do
  jobs+=("--data e2sn2 --split-seed $k --splits data/heidfold_e2sn2_$k.json --train-cond explicit --test-conds explicit --epochs 100 --out $OUT/e2sn2_explicit_heidfold$k")
done
for s in 0 1 2; do
  jobs+=("--data rdb7 --split-seed $s --train-cond explicit --test-conds explicit --epochs 50 --out $OUT/rdb7_explicit_s$s")
done
printf '%s\n' "${jobs[@]}" | xargs -P 2 -I{} sh -c 'AMN_DROP_CKPT=1 /usr/bin/time -v src/lib/memlimit.sh 12 python3 -I src/lib/train.py {} > /dev/null 2> "$(echo {} | sed "s/.*--out //").log"'
python3 -I src/repro/summarise.py
