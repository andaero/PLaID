#!/usr/bin/env bash

# Run novelty evaluation for a set of space-group (sg) numbers.
# -----------------------------------------------------------------------------
# Usage: ./run_novelty.sh <model_folder> [e_hull_max] [model_name]
#
#   <model_folder>  The folder name inside ../results/lora/ that contains the
#                   per-space-group CSV files, e.g. qwen_7b_wyckoff_sg_temp_0.7
#   [e_hull_max]    Optional e_above_hull_maximum value (default: 0.1)
#   [model_name]    Optional suffix used in result filenames (default: eqv2)
#
# Example:
#   ./run_novelty.sh qwen_7b_wyckoff_sg_temp_0.7 0.1 eqv3
# -----------------------------------------------------------------------------

set -euo pipefail

# ------------------------------ Config ---------------------------------------
SG_NUMBERS=(1 15 38 119 143 194 216)   # Space-group numbers to iterate over
RESULTS_ROOT="../results/lora"        # Base directory containing model results
SUN_DIR="csv_results"                 # Where sun_* CSVs will be written
# -----------------------------------------------------------------------------

# ---------------------------- Arguments --------------------------------------
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <model_folder> [e_hull_max] [model_name]" >&2
  exit 1
fi

FOLDER="$1"                        # e.g. qwen_7b_wyckoff_sg_temp_0.7
shift                                # shift off folder arg

# Default values
E_HULL_MAX="0.1"
MODEL_NAME="eqv2"

# Consume optional args if provided
if [[ $# -gt 0 ]]; then
  E_HULL_MAX="$1"; shift
fi
if [[ $# -gt 0 ]]; then
  MODEL_NAME="$1"; shift
fi
# -----------------------------------------------------------------------------

# Ensure output directory exists
mkdir -p "${SUN_DIR}"

# ----------------------------- Main loop -------------------------------------
RESULTS_DIR="${RESULTS_ROOT}/${FOLDER}"

for SG in "${SG_NUMBERS[@]}"; do
  INPUT_CSV="${RESULTS_DIR}/${SG}_${MODEL_NAME}_ehull_results.csv"
  OUT_JSON="test.json"
  SUN_OUT="${SUN_DIR}/${SG}_sun_${MODEL_NAME}.csv"

  echo "[+] Running novelty.py for SG ${SG} (e_hull_max=${E_HULL_MAX})"
  python evals/novelty.py "${INPUT_CSV}" "${OUT_JSON}" "${FOLDER}" \
    --e_above_hull_maximum "${E_HULL_MAX}" \
    --sg "${SG}" \
    --sun_out "${SUN_OUT}"

done

echo "All novelty evaluations completed." 