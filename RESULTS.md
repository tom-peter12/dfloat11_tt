# RESULTS.md — Performance on Tenstorrent Blackhole vs NVIDIA (DFloat11 Paper)

## Compression Ratios

| Model | Paper (NVIDIA) | This Port (Blackhole) | Within Spec (±0.5%) |
|-------|---------------|-----------------------|---------------------|
| Llama-3.2-1B | ~70% | *[fill after eval-all]* | — |
| Llama-3.1-8B | ~70% | *[fill after eval-all]* | — |
| Mistral-Small | ~70% | *[fill after eval-all]* | — |
| Qwen3-8B | ~70% | *[fill after eval-all]* | — |

## Decompression Throughput

*[Fill from `results/<model>/raw/performance.json` after device runs.]*

The NVIDIA DFloat11 paper reports ~250 GB/s decompression throughput (Figure 7) on an A100.
Tenstorrent Blackhole targets: 32 GB/s aggregate DRAM bandwidth × 130 cores = potentially higher
per-tensor throughput when encoding fits in L1.

## Key Architectural Tradeoffs

### Where Blackhole is expected to be faster

1. **LUT reuse**: the 1280-byte LUT lives entirely in L1 SRAM. No cache misses (unlike CUDA's texture cache which can miss under register pressure). Every LUT lookup is a deterministic ~1-cycle L1 access.

2. **No sync overhead**: the Blelloch prefix sum in CUDA requires 9 rounds of `__syncthreads()` for 512 threads. On Blackhole, the prefix sum is a simple sequential scan on one RISC-V core — no barrier stalls at all.

3. **Deterministic memory latency**: Blackhole's NoC has deterministic latency, unlike GPU caches which depend on occupancy and access patterns.

### Where NVIDIA is expected to be faster

1. **Raw thread count**: CUDA launches 512×N_blocks threads simultaneously; our Blackhole port processes them sequentially on 130 cores. CUDA exploits massively more parallelism per clock.

2. **Memory bandwidth**: A100 HBM bandwidth (~2 TB/s) vastly exceeds Blackhole GDDR6 (~32 GB/s total). For bandwidth-bound workloads, NVIDIA wins by ~60×.

3. **Matmul overlap**: CUDA can pipeline the decompression CUDA stream with a separate matmul stream on the tensor cores. Blackhole's single-stream programming model requires decompression to complete before matmul starts (though batched block decompression partially recovers this).

### Where they are comparable

- **L1 locality**: the DFloat11 decode kernel is primarily compute-bound (Huffman walk + LUT) rather than bandwidth-bound once data is in L1. Both architectures have ample L1 for the LUTs.
- **Correctness**: both produce bit-identical BF16 outputs by construction.

## Profiler Data

*[Insert tracy profiler screenshots from `TT_METAL_PROFILER=1 make eval-1b` once device runs are complete.]*

Profiling command:
```bash
TT_METAL_PROFILER=1 python -m dfloat11_tt.eval \
    --config eval/configs/llama-3.2-1b.yaml
```

The profiler output will be in `generated/profiler/output/`.
