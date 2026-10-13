#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
experiment_python="${PCBPARSE_EXPERIMENT_PYTHON:-$project_root/../runtime/edarace-py312/bin/python}"
if [[ ! -x "$experiment_python" ]]; then
    printf 'Set PCBPARSE_EXPERIMENT_PYTHON to a Python runtime with torch and onnxruntime.\n' >&2
    exit 2
fi
exec "$experiment_python" -m data_tools.pin_package_infer "$@"
