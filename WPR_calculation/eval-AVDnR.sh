#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/Cnn14_DecisionLevelMax_mAP=0.385.pth}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 NAME=PATH [NAME=PATH ...]" >&2
  echo "Example: $0 AVCASS=/path/to/avcass mrx=/path/to/mrx bandit=/path/to/bandit" >&2
  exit 2
fi

METHOD_ARGS=()
for method in "$@"; do
  METHOD_ARGS+=(--method "$method")
done

python3 "${PROJECT_DIR}/evaluate_wpr.py" \
  "${METHOD_ARGS[@]}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR:-${PROJECT_DIR}/results}" \
  --threshold "${THRESHOLD:-0.15}" \
  --device "${DEVICE:-auto}"
