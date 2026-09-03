#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

data_config="configs/normalization/data.yaml"
split_config="configs/normalization/split.yaml"
model_config="configs/normalization/byt5_gold.yaml"

conda run -n avimaint-normalization python -m avimaint.normalization audit --config "$data_config"
echo "Review data/aviation/interim/normalization_manual_review.csv before continuing."

if [[ "${1:-}" == "--train" ]]; then
  conda run -n avimaint-normalization python -m avimaint.normalization prepare --config "$data_config"
  conda run -n avimaint-normalization python -m avimaint.normalization split --config "$split_config"
  conda run -n avimaint-normalization python -m avimaint.normalization train --config "$model_config"
fi
