#!/usr/bin/env bash
set -u

missing=0

check_tool() {
    local name="$1"
    local version_cmd="$2"

    if command -v "${name}" >/dev/null 2>&1; then
        echo "PASS ${name}: $(command -v "${name}")"
        bash -lc "${version_cmd}" || true
        echo
    else
        echo "FAIL ${name}: not found"
        echo
        missing=1
    fi
}

check_nvidia_smi() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "PASS nvidia-smi: $(command -v nvidia-smi)"
        nvidia-smi
        echo
    else
        echo "FAIL nvidia-smi: not found"
        echo
        missing=1
    fi
}

check_nvidia_smi
check_tool "cmake" "cmake --version | head -n 1"
check_tool "g++" "g++ --version | head -n 1"
check_tool "nvcc" "nvcc --version"

if [[ "${missing}" -ne 0 ]]; then
    echo "CUDA toolchain check failed: one or more required tools are missing."
    exit 1
fi

echo "CUDA toolchain check passed."
