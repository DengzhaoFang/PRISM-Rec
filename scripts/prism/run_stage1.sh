#!/bin/bash
#
# PRISM Stage 1 — Tokenizer Batch Runner
#
# Usage:
#   bash scripts/prism/run_stage1.sh [DATASET]
#
#   STAGE1_OUTPUT="hparam_stage1_v2" GPUS="0,1,2,3" bash scripts/prism/run_stage1.sh beauty

set -euo pipefail
DATASET="${1:-cds}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Editable config ──────────────────────────────────────────────────────
STAGE1_OUTPUT="${STAGE1_OUTPUT:-hparam_stage1_PASCL}"
GPUS="${GPUS:-}"
# ──────────────────────────────────────────────────────────────────────────

cd "$PROJECT_ROOT"
source .venv/bin/activate

CMD=(python scripts/prism/batch/stage1.py "$DATASET"
     --output-base "scripts/output/prism_tokenizer/${DATASET}/${STAGE1_OUTPUT}")

if [[ -n "$GPUS" ]]; then
    CMD+=(--gpus "$GPUS")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
