#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash src/bbo/scripts/run_week_pipeline.sh <week-number>"
  echo "Example: bash src/bbo/scripts/run_week_pipeline.sh 13"
  exit 1
fi

week="$1"
bbo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
week_dir="$bbo_root/week-$week"
scripts_dir="$week_dir/scripts"
python_cmd="${PYTHON_CMD:-python}"

if ! command -v "$python_cmd" >/dev/null 2>&1; then
  echo "Python command not found: $python_cmd"
  echo "Set PYTHON_CMD=<python-interpreter> and retry."
  exit 1
fi

if [[ ! -d "$scripts_dir" ]]; then
  echo "No scripts directory found for week-$week at: $scripts_dir"
  exit 1
fi

echo "Running weekly pipeline for week-$week"

echo "[1/3] Looking for data preparation script"
if [[ -f "$scripts_dir/prepare_week$((week-1))_data.py" ]]; then
  "$python_cmd" "$scripts_dir/prepare_week$((week-1))_data.py"
else
  echo "No prepare script found for week-$week (skipping)"
fi

echo "[2/3] Looking for query generation script"
if [[ -f "$scripts_dir/nn_round${week}_queries.py" ]]; then
  "$python_cmd" "$scripts_dir/nn_round${week}_queries.py"
elif [[ -f "$scripts_dir/run_all_queries.py" ]]; then
  "$python_cmd" "$scripts_dir/run_all_queries.py"
else
  echo "No query generation script found for week-$week"
fi

echo "[3/3] Looking for submission writer script"
if [[ -f "$scripts_dir/write_submission_file.py" ]]; then
  "$python_cmd" "$scripts_dir/write_submission_file.py"
else
  echo "No submission writer found for week-$week (skipping)"
fi

echo "Done for week-$week"
