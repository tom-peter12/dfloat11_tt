#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${TT_METAL_HOME:-}" ]]; then
    if [[ -d "$REPO_ROOT/../tt-metal" ]]; then
        TT_METAL_HOME="$(cd "$REPO_ROOT/../tt-metal" && pwd)"
    else
        echo "Set TT_METAL_HOME to your tt-metal checkout." >&2
        exit 1
    fi
fi
PYTHON_BIN="${PYTHON_BIN:-$TT_METAL_HOME/python_env/bin/python3}"

export TT_METAL_HOME
unset TT_METAL_VISIBLE_DEVICES
unset TT_MESH_GRAPH_DESC_PATH
export DFLOAT11_TT_DEVICE_ID="${DFLOAT11_TT_DEVICE_ID:-3}"
export DFLOAT11_TRACE_LINEAR="${DFLOAT11_TRACE_LINEAR:-1}"
export DFLOAT11_CACHE_WEIGHTS="${DFLOAT11_CACHE_WEIGHTS:-1}"
export PYTHONPATH="$(dirname "$REPO_ROOT"):${PYTHONPATH:-}"

cd "$REPO_ROOT"
echo "[dfloat11] Config: eval/configs/llama-3.2-1b-full.yaml"
if [[ -n "${DFLOAT11_RESULTS_SUFFIX:-}" ]]; then
    echo "[dfloat11] Results suffix: $DFLOAT11_RESULTS_SUFFIX"
fi
echo "[dfloat11] Phase order: CPU compress bundle -> open TT device -> upload/decompress/infer"
echo "[dfloat11] Compression verification is off unless the config sets compression_check: true"
echo "[dfloat11] Decompressed TT weights are cached during generation unless DFLOAT11_CACHE_WEIGHTS=0"
"$PYTHON_BIN" -m dfloat11_tt.eval --config eval/configs/llama-3.2-1b-full.yaml
