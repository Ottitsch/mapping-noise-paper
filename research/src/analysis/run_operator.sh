#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for ds in e2sn2 rdb7 cyclo; do
  n=10; [ $ds = cyclo ] && n=3
  [ -s results/operator/${ds}_fractions.csv ] || OMP_NUM_THREADS=1 nice -n 10 /usr/bin/time -v src/lib/memlimit.sh 8 python3 -I src/analysis/operator_table.py $ds $n results/operator 2> results/operator/$ds.log
done
OMP_NUM_THREADS=1 nice -n 10 /usr/bin/time -v src/lib/memlimit.sh 8 python3 -I src/analysis/operator_table.py cyclo 7 results/operator 3 2> results/operator/cyclo_extra.log
