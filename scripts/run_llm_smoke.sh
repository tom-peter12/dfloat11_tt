#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-/home/tomasissac/tt-metal}"
PYTHON_BIN="${PYTHON_BIN:-$TT_METAL_HOME/python_env/bin/python3}"
CONFIG="${1:-eval/configs/smollm2-135m-one-layer-smoke.yaml}"

export TT_METAL_HOME
export TT_METAL_VISIBLE_DEVICES="${TT_METAL_VISIBLE_DEVICES:-0}"
export TT_MESH_GRAPH_DESC_PATH="${TT_MESH_GRAPH_DESC_PATH:-$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto}"
export DFLOAT11_TT_DEVICE_ID="${DFLOAT11_TT_DEVICE_ID:-3}"
export DFLOAT11_TRACE_LINEAR="${DFLOAT11_TRACE_LINEAR:-1}"
export DFLOAT11_CACHE_WEIGHTS="${DFLOAT11_CACHE_WEIGHTS:-1}"
export PYTHONPATH="$(dirname "$REPO_ROOT"):${PYTHONPATH:-}"

cd "$REPO_ROOT"
echo "[dfloat11] Config: $CONFIG"
echo "[dfloat11] Phase order: CPU compress bundle -> open TT device -> upload/decompress/infer"
echo "[dfloat11] Compression verification is off unless the config sets compression_check: true"
echo "[dfloat11] Decompressed TT weights are cached during generation unless DFLOAT11_CACHE_WEIGHTS=0"
"$PYTHON_BIN" -m dfloat11_tt.eval --config "$CONFIG"
