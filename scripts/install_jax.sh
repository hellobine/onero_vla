#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
openpi_dir="${1:-$project_dir/../openpi}"
[[ "$(uname -s)" == Linux ]] || {
  echo "JAX GPU inference requires Linux." >&2; exit 2;
}
case "$(uname -m)" in
  aarch64) runtime_dir="$project_dir/jax_runtime" ;;
  x86_64) runtime_dir="$project_dir/jax_runtime_5090" ;;
  *) echo "Supported JAX platforms: Linux aarch64 and x86_64." >&2; exit 2 ;;
esac
cd -- "$project_dir"
uv run --no-sync python - "$openpi_dir" "$runtime_dir/upstream-files.json" <<'PY'
import sys
from pathlib import Path
from lerobot_robot_onero.jax_policy import verify_openpi
verify_openpi(Path(sys.argv[1]).resolve(strict=True), Path(sys.argv[2]))
PY
uv sync --project "$runtime_dir" --frozen --no-dev
echo "JAX installed: $runtime_dir/.venv/bin/python"
echo "Configure model.backend, jax_python and openpi_source before preflight."
