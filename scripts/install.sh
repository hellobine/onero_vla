#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
lerobot_dir="$project_dir/../lerobot"
release="0.3.4"
extras=()
while (($#)); do
  case "$1" in
    --lerobot-path) lerobot_dir="${2:?Missing LeRobot path}"; shift 2 ;;
    --release) release="${2:?Missing release}"; shift 2 ;;
    --extra)
      case "${2:-}" in
        pi|smolvla|diffusion) extras+=(--extra "$2"); shift 2 ;;
        *) echo "Supported extras: pi, smolvla, diffusion" >&2; exit 2 ;;
      esac ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$release" == "0.3.4" ]] || { echo "Unsupported release: $release" >&2; exit 2; }
lerobot_dir="$(cd -- "$lerobot_dir" && pwd -P)"
expected_dir="$(cd -- "$project_dir/../lerobot" && pwd -P)"
[[ "$lerobot_dir" == "$expected_dir" ]] || {
  echo "Keep the official lerobot checkout next to this project, named lerobot." >&2; exit 2;
}
expected_commit="410c095eacc0011d998a73383eb94abed2715af0"
[[ "$(git -C "$lerobot_dir" rev-parse HEAD)" == "$expected_commit" ]] || {
  echo "LeRobot must be the exact v0.7.0 commit: $expected_commit" >&2; exit 2;
}
[[ -z "$(git -C "$lerobot_dir" status --porcelain --untracked-files=no)" ]] || {
  echo "The official LeRobot checkout has tracked modifications." >&2; exit 2;
}
command -v uv >/dev/null
cd -- "$project_dir"
if [[ ! -d .venv ]]; then
  uv venv --python 3.12
fi
if grep -q '^include-system-site-packages = true' .venv/pyvenv.cfg; then
  echo "Use an isolated venv, and source ROS setup.bash instead of exposing system packages." >&2
  exit 2
fi
if [[ -n "${extras[*]:-}" ]]; then
  uv sync --frozen --no-dev "${extras[@]}"
else
  uv sync --frozen --no-dev
fi
uv run --no-sync onero-infer --help
echo "Installed 0.3.4. Configure configs/customer.yaml before connecting hardware."
