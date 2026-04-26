"""Memory tests: compression ratio per-tensor and aggregate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from loguru import logger

from dfloat11_tt.compress.bundle import load_model_bundle


def run_memory_test(
    model_id: str,
    bundle_path: Path,
    results_dir: Path,
    expected_ratio: float = 0.71,
) -> Dict:
    results_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_model_bundle(bundle_path)

    tensor_results: List[Dict] = []
    total_orig   = 0
    total_comp   = 0

    for name, bundle in bundles.items():
        orig_bytes = bundle["n_elements"] * 2  # BF16
        comp_bytes = (
            bundle["encoded_exponent"].nbytes
            + bundle["sign_mantissa"].nbytes
            + bundle["gaps"].nbytes
            + bundle["output_positions"].nbytes
            + bundle["luts"].nbytes
        )
        ratio      = comp_bytes / orig_bytes
        eff_bits   = ratio * 16.0  # effective bits per parameter

        tensor_results.append({
            "name": name,
            "shape": bundle["shape"],
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "ratio": ratio,
            "eff_bits_per_param": eff_bits,
        })

        total_orig += orig_bytes
        total_comp += comp_bytes

    agg_ratio = total_comp / total_orig if total_orig > 0 else 0.0
    within_spec = abs(agg_ratio - expected_ratio) <= 0.005

    summary = {
        "model_id": model_id,
        "test": "memory",
        "total_orig_bytes": total_orig,
        "total_comp_bytes": total_comp,
        "aggregate_ratio": agg_ratio,
        "expected_ratio": expected_ratio,
        "within_spec": within_spec,
        "aggregate_eff_bits": agg_ratio * 16.0,
        "tensors": tensor_results,
    }

    out_path = results_dir / "memory.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    status = "PASS" if within_spec else "WARN"
    logger.info(
        f"[memory] {status}: aggregate ratio = {agg_ratio:.4f} "
        f"(expected {expected_ratio:.3f} ± 0.005). "
        f"{total_orig/1e9:.2f} GB → {total_comp/1e9:.2f} GB."
    )
    return summary
