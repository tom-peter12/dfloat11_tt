"""Performance tests: decompression throughput, token latency, memory usage.

Mirrors Figure 6 and Figure 7 of the DFloat11 paper.
All measurements use actual on-device timing via tt-metal profiler or wall clock.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger


def _measure_decompress_throughput(
    bundle_path: Path,
    model_id: str,
    tensor_sizes: List[int],  # column counts for synthetic tensors
    tt_device: Any,
) -> List[Dict]:
    """Measure decompression throughput (GB/s) for various matrix sizes."""
    from dfloat11_tt.compress.compressor import compress_tensor
    from dfloat11_tt.nn._df11_split import DEFAULT_MAX_CORES, compute_core_ranges

    results = []
    R = 4096
    for C in tensor_sizes:
        w = torch.randn(R, C, dtype=torch.bfloat16)
        bundle = compress_tensor(w)

        import ttnn
        import numpy as np

        def _to_ttnn_uint8(arr):
            return ttnn.from_torch(
                torch.from_numpy(arr.flatten().astype(np.uint8)),
                dtype=ttnn.uint8, device=tt_device, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        def _to_ttnn_uint32(arr):
            return ttnn.from_torch(
                torch.from_numpy(arr.flatten().astype(np.uint32)),
                dtype=ttnn.uint32, device=tt_device, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        enc_tt    = _to_ttnn_uint8(bundle["encoded_exponent"])
        sm_tt     = _to_ttnn_uint8(bundle["sign_mantissa"])
        luts_tt   = _to_ttnn_uint8(bundle["luts"])
        gaps_tt   = _to_ttnn_uint8(bundle["gaps"])
        outpos_tt = _to_ttnn_uint8(bundle["output_positions"].view(np.uint8))
        elem_starts, elem_counts, bit_starts = compute_core_ranges(
            bundle, bundle["R_pad"], bundle["C_pad"], max_cores=DEFAULT_MAX_CORES
        )
        elem_starts_tt = _to_ttnn_uint32(elem_starts)
        elem_counts_tt = _to_ttnn_uint32(elem_counts)
        bit_starts_tt = _to_ttnn_uint32(bit_starts)
        elem_starts_host = elem_starts.tolist()
        elem_counts_host = elem_counts.tolist()
        bit_starts_host = bit_starts.tolist()

        try:
            from dfloat11_tt_cpp import dfloat11_decompress

            # Warmup
            w_tt = dfloat11_decompress(
                enc_tt, sm_tt, luts_tt, gaps_tt, outpos_tt,
                elem_starts_tt,
                elem_counts_tt,
                bit_starts_tt,
                elem_starts_host,
                elem_counts_host,
                bit_starts_host,
                bundle["k"], bundle["n"], bundle["T"], bundle["B"],
                R, C, bundle["R_pad"], bundle["C_pad"],
                bundle["n_elements"], bundle["n_bytes"],
            )
            w_tt.deallocate(force=True)

            # Timed run
            N_ITER = 5
            t0 = time.perf_counter()
            for _ in range(N_ITER):
                w_tt = dfloat11_decompress(
                    enc_tt, sm_tt, luts_tt, gaps_tt, outpos_tt,
                    elem_starts_tt,
                    elem_counts_tt,
                    bit_starts_tt,
                    elem_starts_host,
                    elem_counts_host,
                    bit_starts_host,
                    bundle["k"], bundle["n"], bundle["T"], bundle["B"],
                    R, C, bundle["R_pad"], bundle["C_pad"],
                    bundle["n_elements"], bundle["n_bytes"],
                )
                w_tt.deallocate(force=True)
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) / N_ITER * 1000
            # Input bandwidth: compressed bytes
            comp_bytes = bundle["encoded_exponent"].nbytes + bundle["sign_mantissa"].nbytes
            # Output bandwidth: decompressed BF16 bytes
            decomp_bytes = R * C * 2
            throughput_gbps_in  = comp_bytes / (elapsed_ms * 1e-3) / 1e9
            throughput_gbps_out = decomp_bytes / (elapsed_ms * 1e-3) / 1e9
            n_tiles = (bundle["R_pad"] // 32) * (bundle["C_pad"] // 32)
            tiles_per_sec = n_tiles / (elapsed_ms * 1e-3)

            res = {
                "R": R, "C": C, "shape": f"{R}x{C}",
                "comp_bytes": comp_bytes, "decomp_bytes": decomp_bytes,
                "elapsed_ms": elapsed_ms,
                "throughput_gbps_in": throughput_gbps_in,
                "throughput_gbps_out": throughput_gbps_out,
                "tiles_per_sec": tiles_per_sec,
            }
        except Exception as e:
            logger.warning(f"Throughput test {R}x{C} failed: {e}")
            res = {"R": R, "C": C, "error": str(e)}

        results.append(res)
        logger.info(
            f"  {R}x{C}: {res.get('elapsed_ms', 'N/A'):.1f}ms, "
            f"{res.get('throughput_gbps_out', 'N/A'):.2f} GB/s (output)"
        )

    return results


def _measure_decode_latency(
    model_id: str,
    bundle_path: Path,
    patterns: List[str],
    batch_sizes: List[int],
    tt_device: Any,
    max_new_tokens: int = 32,
) -> List[Dict]:
    """Measure end-to-end token decode latency and throughput."""
    from transformers import AutoTokenizer
    from dfloat11_tt.nn.hf_patch import from_pretrained_df11

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = from_pretrained_df11(model_id, bundle_path, tt_device, patterns=patterns)

    prompt = "The future of artificial intelligence is"
    results = []

    for bs in batch_sizes:
        try:
            inputs = tokenizer(
                [prompt] * bs, return_tensors="pt", padding=True, truncation=True
            )

            # Warmup
            with torch.no_grad():
                model.generate(inputs["input_ids"], max_new_tokens=4, do_sample=False)

            N_ITER = 3
            t0 = time.perf_counter()
            for _ in range(N_ITER):
                with torch.no_grad():
                    out = model.generate(
                        inputs["input_ids"], max_new_tokens=max_new_tokens, do_sample=False
                    )
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) / N_ITER * 1000
            tokens_generated = (out.shape[1] - inputs["input_ids"].shape[1]) * bs
            throughput_tps = tokens_generated / ((t1 - t0) / N_ITER)

            results.append({
                "batch_size": bs,
                "elapsed_ms": elapsed_ms,
                "tokens_per_sec": throughput_tps,
                "latency_per_token_ms": elapsed_ms / max_new_tokens,
            })
            logger.info(
                f"  bs={bs}: {elapsed_ms:.1f}ms total, "
                f"{throughput_tps:.1f} tok/s, "
                f"{elapsed_ms/max_new_tokens:.1f}ms/tok"
            )
        except Exception as e:
            logger.warning(f"  bs={bs}: FAILED — {e}")
            results.append({"batch_size": bs, "error": str(e)})

    return results


def run_perf_test(
    model_id: str,
    bundle_path: Path,
    results_dir: Path,
    patterns: List[str],
    tt_device: Optional[Any] = None,
) -> Dict:
    results_dir.mkdir(parents=True, exist_ok=True)

    results: Dict = {"model_id": model_id, "test": "performance"}

    # Decompression throughput vs matrix size
    if tt_device is not None:
        logger.info("[perf] Measuring decompression throughput ...")
        sizes = [2048, 4096, 8192, 16384, 32768, 65536]
        results["decompress_throughput"] = _measure_decompress_throughput(
            bundle_path, model_id, sizes, tt_device
        )

        # Token decode latency
        logger.info("[perf] Measuring token decode latency ...")
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        results["decode_latency"] = _measure_decode_latency(
            model_id, bundle_path, patterns, batch_sizes, tt_device
        )
    else:
        logger.warning("[perf] No TT device — skipping on-device perf measurements.")
        results["decompress_throughput"] = []
        results["decode_latency"] = []

    out_path = results_dir / "performance.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"[perf] Saved to {out_path}")
    return results
