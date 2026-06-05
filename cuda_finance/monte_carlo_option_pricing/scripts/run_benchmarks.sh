#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MODULE_DIR}/../.." && pwd)"
BUILD_DIR="${MODULE_DIR}/build"
RESULTS_FILE="${MODULE_DIR}/results/monte_carlo_benchmark.csv"
PYTHON_BIN="python3"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

cmake -S "${MODULE_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --config Release

EXECUTABLE="${BUILD_DIR}/monte_carlo_option_pricing"
if [[ -f "${BUILD_DIR}/Release/monte_carlo_option_pricing.exe" ]]; then
    EXECUTABLE="${BUILD_DIR}/Release/monte_carlo_option_pricing.exe"
elif [[ -f "${BUILD_DIR}/monte_carlo_option_pricing.exe" ]]; then
    EXECUTABLE="${BUILD_DIR}/monte_carlo_option_pricing.exe"
fi

rm -f "${RESULTS_FILE}"

for paths in 100000 1000000 5000000 10000000; do
    "${EXECUTABLE}" \
        --paths "${paths}" \
        --S0 100 \
        --K 100 \
        --r 0.05 \
        --sigma 0.2 \
        --T 1.0 \
        --type call \
        --seed 1234 \
        --output "${RESULTS_FILE}"
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_results.py" \
    --input "${RESULTS_FILE}" \
    --reports-dir "${MODULE_DIR}/reports"

echo "Benchmark CSV: ${RESULTS_FILE}"
echo "Plots: ${MODULE_DIR}/reports/runtime_vs_paths.png"
echo "Plots: ${MODULE_DIR}/reports/speedup_vs_paths.png"
