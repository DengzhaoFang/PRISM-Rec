#!/bin/bash
#
# PRISM Stage 1 — Tokenizer Batch Runner
#
# Usage:
#   bash scripts/prism/run_stage1.sh [DATASET]
#
#   GPUS="0,1,2,3" bash scripts/prism/run_stage1.sh beauty

set -euo pipefail
DATASET="${1:-beauty}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPUS="${GPUS:-}"

cd "$PROJECT_ROOT"
source .venv/bin/activate

CMD=(python scripts/prism/batch/stage1.py "$DATASET")

if [[ -n "$GPUS" ]]; then
    CMD+=(--gpus "$GPUS")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
