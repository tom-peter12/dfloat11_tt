"""Evaluation runner: orchestrates all test categories for a model config.

CLI: python -m dfloat11_tt.eval --config eval/configs/<model>.yaml
     python -m dfloat11_tt.eval --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

ROOT = Path(__file__).parent.parent


def _get_bundle_path(model_id: str) -> Path:
    slug = model_id.replace("/", "__")
    return ROOT / "compressed" / f"{slug}.df11tt"


def _get_results_dir(model_id: str) -> Path:
    slug = model_id.replace("/", "__")
    return ROOT / "results" / slug / "raw"


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_config(config_path: Path) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _bundle_path_for_config(cfg: Dict) -> Path:
    model_id = cfg["model_id"]
    return _resolve_repo_path(cfg["bundle_path"]) if "bundle_path" in cfg else _get_bundle_path(model_id)


def ensure_bundle(config_path: Path) -> Path:
    """Run CPU compression before any TT device is opened."""
    cfg = _load_config(config_path)
    model_id = cfg["model_id"]
    patterns = cfg.get("compress_patterns", [])
    bundle_path = _bundle_path_for_config(cfg)
    check_compression = bool(cfg.get("compression_check", False))

    if bundle_path.exists():
        logger.info(f"Compressed bundle already exists: {bundle_path}")
        return bundle_path

    logger.info(f"Bundle not found at {bundle_path}")
    logger.info("Running CPU compression before opening the TT device ...")
    if not check_compression:
        logger.info("Compression bit-identity check is disabled for this pipeline run.")
    from dfloat11_tt.compress.__main__ import compress_model

    compress_model(model_id, bundle_path, patterns=patterns, check_correctness=check_compression)
    return bundle_path


def run_config(config_path: Path, tt_device: Optional[Any] = None) -> Dict:
    cfg = _load_config(config_path)

    model_id = cfg["model_id"]
    patterns = cfg.get("compress_patterns", [])
    suite    = cfg.get("eval_suite", ["bit_identity"])
    bundle_path  = _bundle_path_for_config(cfg)
    results_dir  = _resolve_repo_path(cfg["results_dir"]) if "results_dir" in cfg else _get_results_dir(model_id)
    prompt_file  = ROOT / cfg.get("prompt_file", "eval/prompts/50_prompts.txt")
    max_new_tokens = int(cfg.get("max_new_tokens", 50))
    prompt_limit = cfg.get("prompt_limit")
    compare_logits = bool(cfg.get("compare_logits", True))
    wrap_blocks = bool(cfg.get("wrap_blocks", True))
    expected_ratio = cfg.get("expected_compression_ratio", 0.71)

    # Keep this guard for direct run_config() callers. The CLI calls ensure_bundle()
    # before opening the TT device, so this should normally be a no-op there.
    if not bundle_path.exists():
        ensure_bundle(config_path)

    agg_results: Dict = {"model_id": model_id, "config": str(config_path)}

    if "bit_identity" in suite:
        from dfloat11_tt.eval.test_bit_identity import run_bit_identity_test
        agg_results["bit_identity"] = run_bit_identity_test(
            model_id, bundle_path, results_dir, patterns
        )

    if "output_equivalence" in suite:
        from dfloat11_tt.eval.test_output_equivalence import run_output_equivalence_test
        agg_results["output_equivalence"] = run_output_equivalence_test(
            model_id,
            bundle_path,
            results_dir,
            patterns,
            prompt_file,
            tt_device=tt_device,
            max_new_tokens=max_new_tokens,
            prompt_limit=prompt_limit,
            compare_logits=compare_logits,
            wrap_blocks=wrap_blocks,
        )

    if "benchmarks" in suite:
        from dfloat11_tt.eval.test_benchmarks import run_benchmark_test
        agg_results["benchmarks"] = run_benchmark_test(model_id, results_dir)

    if "performance" in suite:
        from dfloat11_tt.eval.test_perf import run_perf_test
        agg_results["performance"] = run_perf_test(
            model_id, bundle_path, results_dir, patterns, tt_device=tt_device
        )

    if "memory" in suite:
        from dfloat11_tt.eval.test_memory import run_memory_test
        agg_results["memory"] = run_memory_test(
            model_id, bundle_path, results_dir, expected_ratio=expected_ratio
        )

    # Write per-model summary
    summary_dir = results_dir.parent
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.json").write_text(json.dumps(agg_results, indent=2))

    # Generate human-readable report
    _write_report(agg_results, summary_dir / "report.md")

    return agg_results


def _write_report(results: Dict, path: Path) -> None:
    model_id = results.get("model_id", "unknown")
    lines = [f"# Evaluation Report: {model_id}\n"]

    if "bit_identity" in results:
        bi = results["bit_identity"]
        lines.append(f"## Bit-Identity Test")
        lines.append(f"- Result: {'✅ PASS' if bi.get('all_pass') else '❌ FAIL'}")
        lines.append(f"- Tensors tested: {bi.get('n_tensors', 0)}")
        lines.append(f"- Failures: {bi.get('n_fail', 0)}\n")

    if "output_equivalence" in results:
        oe = results["output_equivalence"]
        lines.append("## Output Equivalence")
        lines.append(f"- Result: {'PASS' if oe.get('all_pass') else 'FAIL'}")
        lines.append(f"- Prompts tested: {oe.get('n_prompts', 0)}")
        lines.append(f"- Failures: {oe.get('n_fail', 0)}")
        first = (oe.get("results") or [{}])[0]
        if first:
            lines.append(f"- Prompt: {first.get('prompt', '')}")
            lines.append(f"- BF16 completion: {first.get('ref_completion', '')}")
            lines.append(f"- DF11 completion: {first.get('df11_completion', '')}")
            if first.get("df11_generate_seconds") is not None:
                lines.append(f"- BF16 generation: {first.get('ref_generate_seconds', 0):.2f}s")
                lines.append(f"- DF11 generation: {first.get('df11_generate_seconds', 0):.2f}s")
                lines.append(f"- DF11 tokens/sec: {first.get('df11_tokens_per_sec', 0):.2f}")
        lines.append("")

    if "memory" in results:
        mem = results["memory"]
        lines.append(f"## Memory")
        lines.append(f"- Compression ratio: {mem.get('aggregate_ratio', 0):.4f}")
        lines.append(f"- Effective bits/param: {mem.get('aggregate_eff_bits', 0):.2f}")
        lines.append(f"- Original: {mem.get('total_orig_bytes', 0)/1e9:.3f} GB")
        lines.append(f"- Compressed: {mem.get('total_comp_bytes', 0)/1e9:.3f} GB\n")

    if "performance" in results:
        perf = results["performance"]
        if perf.get("decode_latency"):
            lines.append(f"## Token Decode Latency")
            lines.append("| Batch | ms/total | tok/s |")
            lines.append("|-------|----------|-------|")
            for r in perf["decode_latency"]:
                if "error" not in r:
                    lines.append(
                        f"| {r['batch_size']} | {r.get('elapsed_ms',0):.1f} | {r.get('tokens_per_sec',0):.1f} |"
                    )
            lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="DFloat11-TT evaluation runner")
    parser.add_argument("--config", type=Path, help="Single model config YAML")
    parser.add_argument("--all", action="store_true", help="Run all configs")
    default_device_id = int(os.environ.get("DFLOAT11_TT_DEVICE_ID", "0"))
    parser.add_argument("--tt-device-id", type=int, default=default_device_id, help="TT device ID")
    parser.add_argument("--no-device", action="store_true",
                        help="Skip on-device tests (CPU-only mode)")
    args = parser.parse_args()

    configs_dir = ROOT / "eval" / "configs"

    if args.all:
        configs = list(configs_dir.glob("*.yaml"))
    elif args.config:
        configs = [args.config]
    else:
        parser.error("Specify --config <path> or --all")
        return

    configs = sorted(configs)

    logger.info("Preparing compressed bundles before TT device open ...")
    for cfg in configs:
        ensure_bundle(cfg)

    tt_device = None
    if not args.no_device:
        try:
            import ttnn
            tt_device = ttnn.open_device(device_id=args.tt_device_id)
            logger.info(f"Opened TT device {args.tt_device_id}")
        except Exception as e:
            logger.warning(f"Could not open TT device: {e} -- running CPU-only mode.")

    try:
        all_results: List[Dict] = []
        for cfg in configs:
            logger.info(f"Running config: {cfg}")
            result = run_config(cfg, tt_device=tt_device)
            all_results.append(result)

        # Aggregate JSON for visualizer
        agg_path = ROOT / "results" / "aggregate.json"
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        agg_path.write_text(json.dumps(all_results, indent=2))
        logger.info(f"Aggregate results -> {agg_path}")
    finally:
        if tt_device is not None:
            try:
                import ttnn
                ttnn.close_device(tt_device)
            except Exception:
                pass


if __name__ == "__main__":
    main()
