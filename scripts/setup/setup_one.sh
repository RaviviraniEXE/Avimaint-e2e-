#!/usr/bin/env bash
set -euo pipefail

environment_key="${1:-}"
case "$environment_key" in
  core|normalization|ie-classical|ie-neural|spert|retrieval|dashboard|dev) ;;
  *) echo "Usage: setup_one.sh {core|normalization|ie-classical|ie-neural|spert|retrieval|dashboard|dev}"; exit 2 ;;
esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
environment_file="$project_root/envs/$environment_key/environment.yml"
requirements_file="$project_root/envs/$environment_key/requirements.txt"
environment_name="$(awk '/^name:/ {print $2; exit}' "$environment_file")"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is not available. Initialize Conda first."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$environment_name"; then
  conda env update --name "$environment_name" --file "$environment_file" --prune
else
  conda env create --file "$environment_file"
fi

conda run --name "$environment_name" python -m pip install --upgrade pip
conda run --name "$environment_name" python -m pip install --requirement "$requirements_file"
conda run --name "$environment_name" python -m pip install --editable "$project_root" --no-deps
conda run --name "$environment_name" python -m avimaint.cli doctor --environment "$environment_name"
