#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash src/bbo/scripts/setup_and_run_week.sh <week-number> [--skip-install] [--python <python-cmd>] [--venv-dir <path>]"
  echo "Examples:"
  echo "  bash src/bbo/scripts/setup_and_run_week.sh 13"
  echo "  bash src/bbo/scripts/setup_and_run_week.sh 4 --skip-install"
  echo "  bash src/bbo/scripts/setup_and_run_week.sh 7 --python python3"
  echo "  bash src/bbo/scripts/setup_and_run_week.sh 7 --venv-dir .venv-capstone"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

week="$1"
shift

skip_install=false
python_cmd="python"
venv_dir=".venv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-install)
      skip_install=true
      shift
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "Error: --python requires a command (e.g., python or python3)."
        exit 1
      fi
      python_cmd="$2"
      shift 2
      ;;
    --venv-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: --venv-dir requires a path."
        exit 1
      fi
      venv_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if ! [[ "$week" =~ ^[0-9]+$ ]]; then
  echo "Error: <week-number> must be an integer (received: $week)"
  usage
  exit 1
fi

bbo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(cd "$bbo_root/../.." && pwd)"
requirements_file="$project_root/requirements.txt"
run_script="$bbo_root/scripts/run_week_pipeline.sh"
venv_path="$project_root/$venv_dir"
venv_python="$venv_path/Scripts/python.exe"

if ! command -v "$python_cmd" >/dev/null 2>&1; then
  echo "Python command not found: $python_cmd"
  echo "Tip: pass --python <cmd> to choose a different interpreter."
  exit 1
fi

if [[ ! -f "$requirements_file" ]]; then
  echo "requirements.txt not found at: $requirements_file"
  exit 1
fi

if [[ ! -f "$run_script" ]]; then
  echo "run_week_pipeline.sh not found at: $run_script"
  exit 1
fi

# Check base Python compatibility with pinned requirements.
py_version="$($python_cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
if [[ "$py_version" == "3.13"* ]]; then
  echo "Warning: requirements.txt pins package versions that may not support Python $py_version."
  echo "Recommended: use --python with 3.11 or 3.10 (for example: --python python3.11)."
fi

if [[ ! -f "$venv_python" ]]; then
  echo "[setup] Creating virtual environment at: $venv_path"
  "$python_cmd" -m venv "$venv_path"
fi

if [[ ! -f "$venv_python" ]]; then
  echo "Failed to locate venv Python at: $venv_python"
  exit 1
fi

if [[ "$skip_install" == false ]]; then
  echo "[setup] Installing dependencies into venv from requirements.txt"
  "$venv_python" -m pip install -r "$requirements_file"
else
  echo "[setup] Skipping dependency install (--skip-install)"
fi

echo "[run] Executing weekly pipeline for week-$week"
PYTHON_CMD="$venv_python" bash "$run_script" "$week"
