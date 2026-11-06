#!/usr/bin/env bash
# Run every job of the queue that has no metrics.json yet, two at a time, 2 threads each.
# JOB_FILTER (an extended regex on the job line) restricts the run to matching jobs.
# Each run writes <out>.log with /usr/bin/time -v output (peak memory) and any error.
set -uo pipefail
cd "$(dirname "$0")/../.."
python3 -I src/grid/make_jobs.py > results/grid/jobs.txt
run_one () {
  out=$(echo "$1" | sed 's/.*--out //')
  [ -s "$out/metrics.json" ] && return 0
  # another queue is running this job: its epoch log or run log changed in the last 10 minutes
  for f in "$out/log/version_0/metrics.csv" "$out.log"; do
    [ -f "$f" ] && [ -n "$(find "$f" -mmin -10)" ] && return 0
  done
  mkdir -p "$(dirname "$out")"
  AMN_DROP_CKPT=1 /usr/bin/time -v src/lib/memlimit.sh 12 python3 -I src/lib/train.py $1 > /dev/null 2> "$out.log" \
    && echo "$(date -u +%H:%M:%S) done $out" || echo "$(date -u +%H:%M:%S) FAILED $out"
}
export -f run_one
grep -v '^$' results/grid/jobs.txt | grep -E "${JOB_FILTER:-.}" | xargs -P 2 -I{} bash -c 'run_one "{}"' >> results/grid/progress.txt 2>&1
