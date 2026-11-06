#!/usr/bin/env bash
# Download raw inputs into research/data/raw (not committed). Pinned to commit SHAs recorded in data/raw/SOURCES.txt.
set -euo pipefail
cd "$(dirname "$0")/../../data/raw"
get () { # repo path out
  local sha; sha=$(gh api "repos/$1/commits/HEAD" -q .sha 2>/dev/null || true)
  [ -s "$3" ] || curl -sSfL "https://raw.githubusercontent.com/$1/${sha:-HEAD}/$2" -o "$3"
  echo "$1 ${sha:-HEAD} $2 -> $3 sha256=$(sha256sum "$3" | cut -c1-16)" >> SOURCES.txt
}
: > SOURCES.txt
get lcmd-epfl/benchmark-barrier-learning data/gdb7-22-ts/ccsdtf12_dz.csv gdb_bench.csv
get lcmd-epfl/benchmark-barrier-learning data/cyclo/full_dataset.csv cyclo_bench.csv
get lcmd-epfl/reply-physics-reactions data/gdb7-22-ts/submit_true_rxns.csv gdb_reply_true.csv
get lcmd-epfl/reply-physics-reactions data/gdb7-22-ts/submit_rxnmapper_rxns.csv gdb_reply_rxnmapper.csv
get hesther/reactiondatabase data/e2sn2.csv e2sn2.csv
get hesther/reactiondatabase data_splits/e2sn2.tar.gz e2sn2_splits.tar.gz
[ -s rdb7_zenodo.csv ] || curl -sSfL "https://zenodo.org/api/records/6618262/files/ccsdtf12_dz.csv/content" -o rdb7_zenodo.csv
echo "zenodo 6618262 ccsdtf12_dz.csv -> rdb7_zenodo.csv sha256=$(sha256sum rdb7_zenodo.csv | cut -c1-16)" >> SOURCES.txt
mkdir -p e2sn2_splits && tar -xzf e2sn2_splits.tar.gz -C e2sn2_splits
