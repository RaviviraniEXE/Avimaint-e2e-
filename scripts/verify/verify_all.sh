#!/usr/bin/env bash
set -euo pipefail

environment_names=(
  avimaint-core
  avimaint-normalization
  avimaint-ie-classical
  avimaint-ie-neural
  avimaint-spert
  avimaint-retrieval
  avimaint-dashboard
  avimaint-dev
)

for environment_name in "\${environment_names[@]}"; do
  echo "Verifying $environment_name"
  conda run --name "$environment_name" python -m avimaint.cli doctor --environment "$environment_name"
  conda run --name "$environment_name" python -m pip check
done

