# DFloat11-TT

**Lossless LLM compression (DFloat11) ported to Tenstorrent Blackhole.**

DFloat11 (arXiv:2504.11651) reduces BFloat16 LLM weights to ~70% of original size with **bit-for-bit identical outputs**, by Huffman-coding only the 8-bit exponent field. This repository ports the CUDA implementation to Tenstorrent Blackhole using the TT-Metalium SDK.

## Quickstart

For the exact Blackhole reproduction commands used during bring-up, see
[REPRODUCE.md](REPRODUCE.md).

```bash
# 1. Build C++ extension (requires TT_METAL_HOME pointing to tt-metal)
make build

# 2. Compress a model (Llama-3.2-1B example)
python -m dfloat11_tt.compress \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --out compressed/llama-3.2-1b.df11tt

# 3. Run evaluation (requires Blackhole device)
make eval-1b

# 4. Open the visualization notebook
make viz
```

## Results

| Model | Orig (GB) | Comp (GB) | Ratio | MMLU | Δ |
|-------|-----------|-----------|-------|------|---|
| Llama-3.2-1B | 2.47 | 1.75 | 70.9% | ≡ BF16 | 0.00 |
| Llama-3.1-8B | 15.0 | 10.5 | 70.0% | ≡ BF16 | 0.00 |

*Full results in `results/aggregate.json` after `make eval-all`.*

## Architecture

The core mapping from NVIDIA CUDA to Tenstorrent Blackhole:
- **1 CUDA block (512 threads)** → **1 Tensix core** (scalar RISC-V on TRISC1)
- **Blelloch prefix sum** → **sequential forward scan** in L1 SRAM
- **Shared memory write buffer** → **L1 SRAM scratch** at fixed address
- **LUT texture cache** → **L1-resident array** (reader pre-loads once)
- **Row-major output** → **32×32 tiled output** (direct, Path A)
- **CUDA grid** → **130 Blackhole compute cores** via SPMD

See [INVESTIGATION.md](INVESTIGATION.md) and [ARCHITECTURE.md](ARCHITECTURE.md) for full details.

## Structure

```
compress/     Python compressor (Huffman, LUTs, bundle format)
kernels/      TT-Metalium RISC-V kernels (reader, compute, writer)
host/         Metalium Program assembly and launch
op/           TT-NN op registration + Python pybind
nn/           DF11Linear, DF11TransformerBlock, HF patch
eval/         Multi-model testing pipeline
viz/          Presentation notebook
tests/        Unit + integration tests
```
