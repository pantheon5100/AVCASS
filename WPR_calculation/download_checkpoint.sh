#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${PROJECT_DIR}/Cnn14_DecisionLevelMax_mAP=0.385.pth"
URL="https://zenodo.org/records/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
EXPECTED_SHA256="dd3b4043a87d4ec13df8082c0fcfee3fb5084151808e47e060987a95eabdd142"

if command -v curl >/dev/null 2>&1; then
  curl --fail --location --output "${OUTPUT}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget --output-document="${OUTPUT}" "${URL}"
else
  echo "Install curl or wget to download the checkpoint." >&2
  exit 1
fi

echo "${EXPECTED_SHA256}  ${OUTPUT}" | sha256sum --check -
