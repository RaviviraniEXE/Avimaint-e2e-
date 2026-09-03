#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
environment_keys=(core normalization ie-classical ie-neural spert retrieval dashboard dev)

for environment_key in "${environment_keys[@]}"; do
  bash "$project_root/scripts/setup/setup_one.sh" "$environment_key"
done

bash "$project_root/scripts/setup/clone_official_spert.sh"

echo "All AviMaint environments are installed."
