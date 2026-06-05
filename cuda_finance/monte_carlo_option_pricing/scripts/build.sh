#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -d "cuda_finance/monte_carlo_option_pricing" ]]; then
    SOURCE_DIR="cuda_finance/monte_carlo_option_pricing"
    BUILD_DIR="cuda_finance/monte_carlo_option_pricing/build"
else
    SOURCE_DIR="${MODULE_DIR}"
    BUILD_DIR="${MODULE_DIR}/build"
fi

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --config Release

EXECUTABLE="${BUILD_DIR}/monte_carlo_option_pricing"
if [[ -f "${BUILD_DIR}/Release/monte_carlo_option_pricing.exe" ]]; then
    EXECUTABLE="${BUILD_DIR}/Release/monte_carlo_option_pricing.exe"
elif [[ -f "${BUILD_DIR}/monte_carlo_option_pricing.exe" ]]; then
    EXECUTABLE="${BUILD_DIR}/monte_carlo_option_pricing.exe"
fi

echo "Executable: ${EXECUTABLE}"
