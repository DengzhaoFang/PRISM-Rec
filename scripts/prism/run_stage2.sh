#!/bin/bash
#
# PRISM Stage 2 — Recommender Batch Runner
#
# Usage:
#   bash scripts/prism/run_stage2.sh [DATASET]
#
#   STAGE1_DIR="scripts/output/prism_tokenizer/beauty/hparam_stage1" \
#   STAGE1_EXPERIMENTS="cma_mcd,cma_mcd_saco_c00625" \
#   STAGE2_OUTPUT="hparam_stage2" \
#   bash scripts/prism/run_stage2.sh beauty

set -euo pipefail
DATASET="${1:-beauty}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Editable config ──────────────────────────────────────────────────────
STAGE1_DIR="${STAGE1_DIR:-scripts/output/prism_tokenizer/beauty/hparam_stage1}"
STAGE1_EXPERIMENTS="${STAGE1_EXPERIMENTS:-}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-hparam_stage2}"
FAST_DEV_CONFIG="${FAST_DEV_CONFIG:-}"
GPUS="${GPUS:-}"
# ──────────────────────────────────────────────────────────────────────────

cd "$PROJECT_ROOT"
source .venv/bin/activate

CMD=(python scripts/prism/batch/stage2.py "$DATASET")

if [[ -n "$STAGE1_DIR" ]]; then
    CMD+=(--stage1-base "$STAGE1_DIR")
fi

if [[ -n "$STAGE1_EXPERIMENTS" ]]; then
    CMD+=(--experiments "$STAGE1_EXPERIMENTS")
fi

if [[ -n "$STAGE2_OUTPUT" ]]; then
    CMD+=(--output-base "scripts/output/recommender/prism/${DATASET}/${STAGE2_OUTPUT}")
fi

if [[ -n "$FAST_DEV_CONFIG" ]]; then
    CMD+=(--fast-dev-config "$FAST_DEV_CONFIG")
fi

if [[ -n "$GPUS" ]]; then
    CMD+=(--gpus "$GPUS")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
