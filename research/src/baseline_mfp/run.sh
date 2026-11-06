#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results/baseline_mfp
for ds in e2sn2 rdb7 cyclo; do for kind in diff concat; do for s in 0 1 2; do
  cap=1600; [ $ds = rdb7 ] && cap=5000
  [ -s results/baseline_mfp/${ds}_${kind}_s$s.json ] || OMP_NUM_THREADS=1 nice -n 10 /usr/bin/time -v src/lib/memlimit.sh 8 python3 -I src/baseline_mfp/mfp.py $ds $s results/baseline_mfp $kind $cap 2>> results/baseline_mfp/time.log
done; done; done
