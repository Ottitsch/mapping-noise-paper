#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for ds in rdb7 cyclo; do
  /usr/bin/time -v src/lib/memlimit.sh 8 python3 -I src/rxnmapper_rate/rate.py $ds results/rxnmapper_rate 2> results/rxnmapper_rate/$ds.log
done
