#!/usr/bin/env bash
# Run a command with a hard address space limit, so a memory mistake kills the command, not the runner.
# Usage: src/lib/memlimit.sh <GB> <command...>
# Example: src/lib/memlimit.sh 12 python3 -I src/lib/train.py --data rdb7 ...
set -euo pipefail
gb="${1:?usage: memlimit.sh <GB> <command...>}"; shift
ulimit -v $(( gb * 1024 * 1024 ))
exec "$@"
