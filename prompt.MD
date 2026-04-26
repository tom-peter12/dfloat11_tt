# Prompt: Port DFloat11 to Tenstorrent Blackhole — Full Implementation

You are an experienced systems-and-ML engineer with deep familiarity with both NVIDIA CUDA and the Tenstorrent tt-metal stack. Your task is to **port DFloat11 (arXiv:2504.11651) — a lossless LLM compression framework — from NVIDIA GPU to Tenstorrent Blackhole**, with a complete production-grade implementation, a testing pipeline for multiple models, and a visualizer for presentation purposes.

This is a course-project deliverable. **The expectation is a full, working implementation — not a proof of concept.** Treat any "TODO", "for brevity", or stubbed functions as failures. If something is hard, do it correctly the first time.

---

## 0. What DFloat11 actually is (read before designing)

DFloat11 reduces a BFloat16 LLM to ~70% of its original size with **bit-for-bit identical outputs**. It exploits the fact that the 8-bit BFloat16 exponent carries only ~2.6 bits of entropy. The core ideas are:

1. **Format:** keep sign (1b) and mantissa (7b) raw and bit-packed into `PackedSignMantissa`; Huffman-encode only the exponents into a tightly packed bitstream `EncodedExponent`.
2. **Hierarchical LUTs:** the Huffman tree (max code length ~24–32 bits) is decomposed into a chain of compact 256-entry LUTs (each decoding one byte). Unused exponent values 240–255 act as pointers to the next LUT in the hierarchy. Whole LUT structure fits in on-chip SRAM (~9 KB).
3. **Two-phase parallel decode kernel:**
   - Each thread is assigned `n` bytes (n=8) of the encoded exponent stream.
   - Auxiliary arrays: `Gaps` (5b per thread, bit offset where the first complete code begins) and `BlockOutputPos` (32b per thread-block, where the block writes its first decoded element).
   - **Phase 1:** each thread walks its byte-range counting how many symbols it produces; threads in a block do a Blelloch prefix-sum to compute per-thread output positions inside the block.
   - **Phase 2:** each thread re-walks its byte-range, this time writing decoded BFloat16 values into a SRAM write-buffer; one coalesced write to HBM at the end.
4. **Transformer-block-level batched decompression:** all weight matrices in one transformer block are decompressed as a single batch right before the block's forward pass, then immediately freed. Matmul itself is unchanged — it consumes regular BF16.

**Critical invariant:** the decompressed BF16 tensor must be **bit-identical** to the original. This is the central correctness test.

---

## 1. Repositories you have access to

- `~/DFloat11` — original NVIDIA implementation (CUDA kernels, Python integration, HF Transformers patches).
- `~/tt-metal` — Tenstorrent Metalium SDK (low-level kernels) and TT-NN (operator library + PyTorch frontend).

**Step 0 of your work is to read both, in this order:**

1. `cd ~/DFloat11 && find . -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h' -o -name '*.py' \) | head -100` — then read every CUDA kernel, the Python compressor, and the HF integration. Take notes on:
   - the exact bit layout of `EncodedExponent`, `PackedSignMantissa`, `Gaps`, `BlockOutputPos`, `LUT[1..k]`, `CodeLengths`;
   - the exact Huffman tree construction (tie-breaking matters for bit-identity across re-compressions);
   - the kernel launch parameters (threads/block, n, k, max code length L);
   - how the kernel is integrated into `torch.nn.Linear` and HF Transformers blocks.
2. `~/tt-metal/METALIUM_GUIDE.md`, `~/tt-metal/tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md`, `~/tt-metal/tt_metal/programming_examples/`, and the LLK headers under `~/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/`. You must understand: Tensix core layout, the 5 RISC-V "baby cores" (BRISC, NCRISC, TRISC0/1/2), L1 SRAM (1.5 MB per Tensix), circular buffers, NoC, the reader/compute/writer kernel triplet, the 32×32 tile layout, and how TT-NN ops are registered.

Produce a short `INVESTIGATION.md` in the project root summarizing your findings before writing any kernels. This document is part of the deliverable and proves you understood both stacks.

---

## 2. The fundamental architectural difference (and why a naive port fails)

A naive 1:1 port of the CUDA kernel will be wrong and slow. The two architectures have **incompatible execution models**:

| Concept | NVIDIA GPU (DFloat11 original) | Tenstorrent Blackhole |
|---|---|---|
| Parallelism unit | 1000s of SIMT threads in blocks of ~256 | 140 Tensix cores; each has 5 scalar RISC-V baby cores + 1 matrix engine (FPU) + 1 vector engine (SFPU) |
| Memory hierarchy | Registers → shared mem (~100 KB/block) → L2 → HBM | Registers → L1 SRAM (1.5 MB/Tensix, **explicitly managed, no cache**) → GDDR6 over NoC |
| Sync primitive | `__syncthreads()`, warp shuffles | Circular buffers with semaphores, NoC barriers |
| Native data unit | Scalar / warp (32 lanes) | **32×32 tile** (matmul/SFPU operate on tiles, not scalars) |
| Data transfer | Implicit caching, coalescing helpers | **Explicit** reader/writer kernels move data between DRAM ⇄ L1 over NoC |
| Programming model | Single kernel, SIMT | Triplet per Tensix core: reader (BRISC/NCRISC) + compute (TRISC0/1/2) + writer; communicate via circular buffers |
| "Block" | A grid block of ~256 threads | A single Tensix core (or a logical group of cores via the work-distribution scheme) |

### Mapping decisions you must make explicit

The DFloat11 kernel is **bit-manipulation-heavy and not matmul-shaped**. It barely uses the FPU and only modestly uses the SFPU. Most of the work — bit-level Huffman decoding, LUT lookups, prefix sums, byte writes — is naturally **scalar control-flow code on the RISC-V baby cores**, not tile compute.

Therefore, the canonical mapping for this port is:

- **CUDA grid block ↔ one Tensix core.** The total work (bytes of encoded exponent stream) is sharded across the 140 Tensix cores of Blackhole.
- **CUDA threads inside a block ↔ logical thread chunks processed sequentially or pipelined across the 5 baby cores within a Tensix.** Inside a Tensix you do **not** have 256 hardware threads. Instead:
  - **NCRISC (data-movement-1)**: streams encoded exponents and `PackedSignMantissa` from DRAM into L1 circular buffers.
  - **BRISC (data-movement-0)**: streams `Gaps` and `LUTs` and `CodeLengths` into L1; also issues the final coalesced writeback to DRAM.
  - **TRISC1 (math) + TRISC0 (unpack) + TRISC2 (pack)**: not directly useful for Huffman decode. Instead, use the **SFPU via SFPI** to vectorize the per-thread byte-walk where possible — each SFPU lane plays the role of one CUDA thread, walking its own n=8 byte chunk. The SFPU on Blackhole has 32 lanes per face; treat it as a 32-wide SIMD unit. This recovers a meaningful fraction of CUDA's intra-block parallelism.
  - If the SFPU programming for variable-length bit walks is too divergent (it likely will be — Huffman is inherently variable-length), fall back to running the per-thread loops on **TRISC1 as plain RISC-V scalar code**, and use the SFPU only for the prefix-sum step (Blelloch). Decide based on profiling, but **implement and benchmark both**.
- **Shared memory (CUDA) ↔ L1 SRAM**. The hierarchical LUTs (~9 KB), the encoded-exponent slab for this Tensix (~nT bytes), the gaps array slice, and the BF16 write-buffer all live in L1. Plenty of room (1.5 MB).
- **`__syncthreads()` ↔ circular-buffer push/pop semaphores** between the reader, compute, and writer kernels of the same Tensix. For prefix-sum across "threads" inside a Tensix, use plain memory + an `noc_async_write_barrier`-style barrier within the compute kernel's RISC-V flow.
- **Coalesced HBM write ↔ a single `noc_async_write` of the L1 write-buffer to DRAM, followed by a barrier.**
- **Tile layout vs row-major.** BF16 matmul on Tenstorrent expects **tiled (32×32) layout**. The decompressed weight matrix is the **input** to a TT-NN matmul. Therefore the writer must produce tile-major output, not row-major. Two paths — pick one and justify it in `INVESTIGATION.md`:
  - **(A) Decompress directly into tiled layout.** Compute the tile (row, col) and intra-tile (r, c) for each output element from its linear index in the decompressed stream. Slightly more arithmetic in the writer; saves a tilize op afterwards. Preferred.
  - **(B) Decompress row-major into a scratch L1 buffer and run TT-NN's `tilize` immediately after.** Simpler code, costs an extra pass over the output. Acceptable fallback only if (A) hits a pathological case.

- **Transformer-block-level batched decompression ↔ one program (Metalium `Program`) that decompresses several weight tensors in parallel across the 140 Tensix cores**, with the cores partitioned by tensor (e.g., q_proj on cores 0–27, k_proj on 28–55, etc.) so all matrices in a block come up at once before the block's forward pass.

### What is the same

The **algorithm** — Huffman tree, hierarchical LUTs, `Gaps` + `BlockOutputPos`, two-phase decode — is unchanged. **Only the execution mapping changes.** The host-side compressor (Python) is mostly a straight port; the device-side decompressor is a complete rewrite in tt-metal.

---

## 3. Implementation plan (deliver in this order)

### Phase 1 — Host-side compressor (Python)
Path: `dfloat11_tt/compress/`

- Port `~/DFloat11`'s Python compressor verbatim where possible (Huffman tree builder, exponent histogram, code assignment, packing).
- Output a **format-versioned binary blob** per weight tensor: header (magic `DF11TT01`, dtype, original shape, k = number of LUTs, L = max code length, n = bytes per "thread", T = threads per Tensix block, B = total Tensix blocks), then `LUT[1..k]`, `CodeLengths[256]`, `EncodedExponent`, `PackedSignMantissa`, `Gaps`, `BlockOutputPos`. Document the layout in `dfloat11_tt/compress/FORMAT.md`.
- Provide a CLI: `python -m dfloat11_tt.compress --model meta-llama/Llama-3.1-8B-Instruct --out ./compressed/llama-3.1-8b.df11tt`. It must traverse the HF model, compress every targeted matrix (all Linear weights + token embedding + LM head, matching the original DFloat11 selection rules), and write the bundle.
- **Bit-identity test:** load the compressed bundle, run a pure-Python decoder, compare to the original BF16 tensor — must match every bit.

### Phase 2 — Tenstorrent decompression kernels (C++ tt-metal)
Path: `dfloat11_tt/kernels/`

Three kernel files per Tensix core role, plus a host program that builds and launches them:

1. **Reader kernel** (`reader_df11.cpp`, runs on NCRISC + BRISC):
   - Inputs from DRAM: `EncodedExponent` slab, `PackedSignMantissa` slab, `Gaps` slice, `LUTs`, `CodeLengths`, `BlockOutputPos[block_id]`, `BlockOutputPos[block_id+1]`.
   - Pushes them onto L1 circular buffers `cb_encoded`, `cb_signmant`, `cb_gaps`, `cb_luts`.
2. **Compute kernel** (`decompress_df11.cpp`, runs on TRISC1 with optional SFPU use):
   - Phase 1: loop over the T logical threads; for each, walk its n bytes using LUT chain, count produced symbols, write into `num_elements[t]`. Then in-Tensix Blelloch prefix-sum.
   - Phase 2: loop again, this time emitting BF16 values. Each emit:
     - decoded exponent from LUT chain;
     - sign+mantissa byte from `PackedSignMantissa[ThreadOutputPos[t]]`;
     - assemble `bf16 = (sign << 15) | (exponent << 7) | mantissa`;
     - write to `WriteBuffer` at the correct **tiled** offset (or row-major if doing path B).
   - Use the SFPU only where it's a clear win (e.g., the prefix sum's upsweep/downsweep using SFPI vector ops). Implement the inner Huffman walk as straight RISC-V C++ code — it's branchy and pointer-chasing, which is fine on a baby core.
   - Push tiles onto `cb_decoded` for the writer.
3. **Writer kernel** (`writer_df11.cpp`, runs on BRISC):
   - Pops decoded tiles from `cb_decoded`, issues `noc_async_write` to the destination DRAM tensor address(es), barriers at end.

4. **Host program** (`dfloat11_tt/host/df11_op.cpp`):
   - Allocates DRAM and L1 buffers, builds the `Program`, sets compile-time and runtime args (n, T, k, L, base addresses, per-core gap-array offsets), and launches across the chosen core grid (default: all 140 Tensix cores; a runtime flag selects a subgrid for profiling).
   - Exposes `decompress_df11(device, df11_bundle) -> ttnn::Tensor` returning a tiled BF16 tensor on device.

5. **Build glue:** add a CMake target `df11_kernels` in `dfloat11_tt/CMakeLists.txt`, hooked into the tt-metal build so the kernels compile with the right RISC-V toolchain and the host op links against `tt_metal` and `ttnn`.

### Phase 3 — TT-NN op + PyTorch / HF integration
Path: `dfloat11_tt/op/`

- Register a TT-NN custom op `ttnn.experimental.dfloat11_decompress(bundle) -> Tensor` following the patterns in `~/tt-metal/ttnn/cpp/ttnn/operations/`.
- Implement a `DF11Linear` PyTorch module: holds the on-device DF11 bundle for its weight, in `forward(x)` calls `dfloat11_decompress` to materialize BF16 weights on device, then `ttnn.linear`, then frees the materialized weight. Bias (if any) is stored uncompressed.
- Implement a `DF11TransformerBlock` wrapper that batches the decompression of all `DF11Linear`s inside it into **one** Metalium program launch (matching the paper's Section 2.3.3). This is the throughput-critical path.
- Provide a HuggingFace patcher: `dfloat11_tt.hf.from_pretrained_df11(model_name_or_path, compressed_bundle_path)` which loads an HF model on Tenstorrent device, swapping every targeted Linear with `DF11Linear`. Mirror `~/DFloat11`'s patching scheme but rewritten for `ttnn`.

### Phase 4 — Multi-model testing pipeline
Path: `dfloat11_tt/eval/`

A single CLI orchestrator: `python -m dfloat11_tt.eval --config eval/configs/<model>.yaml`.

Required test categories (each must run end-to-end on the actual Blackhole device, not stubs):

1. **Bit-identity** (`test_bit_identity.py`): for every targeted weight tensor in the model, decompress on device, copy back to host, compare element-wise to the original BF16 tensor. **Zero tolerance — any mismatch fails the test.** Print a summary table.
2. **Output-equivalence** (`test_output_equivalence.py`): run the original (uncompressed) and the DF11 model on a fixed prompt set with greedy decoding and the same seeds. Token sequences and logits must be bit-identical. (If the original BF16 model doesn't fit on the device, run reference on CPU/host and compare logits.)
3. **Standard benchmarks** (`test_benchmarks.py`): MMLU + TruthfulQA accuracy via `lm-evaluation-harness`; WikiText-2 and C4 word-level perplexity. Mirror Table 2 of the paper.
4. **Performance** (`test_perf.py`):
   - Decompression throughput (GB/s and tiles/s) per-tensor across matrix sizes 4096×{2048,4096,8192,16384,32768,65536}.
   - End-to-end token-decode latency and throughput for batch sizes {1,2,4,8,16,32,64,128}.
   - Peak device memory.
   - Generation length until OOM at batch 1.
   - Latency breakdown: token-embedding decompress, per-block decompress, per-block forward, LM-head decompress, LM-head forward (mirror Figure 6).
5. **Memory** (`test_memory.py`): compression ratio per-tensor and aggregate; effective bit width; verify ≤ ~70% of original size.

Configuration for each model lives in `eval/configs/`, one YAML per model, covering at minimum:
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct` (small, for fast CI)
- `mistralai/Mistral-Small-Instruct-2409` (or a Mistral that fits)
- `Qwen/Qwen3-8B`
- `microsoft/Phi-4` (optional; gate on availability)

Each YAML lists: HF model id, target precision, target patterns of layers to compress, eval suite to run, expected compression ratio, expected accuracy parity vs. reference. The runner produces:
- `results/<model>/raw/` — JSON per test
- `results/<model>/summary.json` — aggregated
- `results/<model>/report.md` — human-readable

A top-level `make eval-all` (or `python -m dfloat11_tt.eval --all`) runs every config and produces `results/aggregate.json` for the visualizer.

### Phase 5 — Visualizer (for the presentation)
Path: `dfloat11_tt/viz/`

Build a **standalone Python web app** (FastAPI backend + a single-page frontend, vanilla HTML/CSS/JS or React, your call — keep it simple but presentable). It reads `results/aggregate.json` and renders:

1. **Compression dashboard**: per-model bar chart of original size vs DF11 size; effective bit-width gauge; per-layer-type compression-ratio breakdown (q_proj vs k_proj vs gate_proj …); exponent-frequency histogram and Huffman-tree visualization for one selected layer.
2. **Performance dashboard**: throughput vs batch size for BF16 (CPU offload) vs DF11 (on-device); decompression throughput vs matrix size (this directly mirrors paper Figure 7); latency breakdown stacked bar (mirror Figure 6); generation-length-until-OOM line plot.
3. **Architecture explainer**: an animated SVG showing **side-by-side** how DFloat11 maps to NVIDIA GPU vs Tenstorrent Blackhole — grid of 140 Tensix cores light up as a transformer block is decompressed, with a parallel CUDA-block visualization. This is what sells the talk: visually communicates the architectural mapping you justified in `INVESTIGATION.md`. Use real per-core timings extracted from `tt-metal` profiler output where available; otherwise, synthesize plausible animation timing from the measured per-block latency.
4. **Live model comparison**: dropdown to pick two models from the eval set; renders a side-by-side comparison card.
5. **Export-to-slides mode**: each chart has a "Download PNG" + a "Download standalone HTML" button. The architecture explainer specifically should export as a self-contained HTML file that runs offline in a browser during the presentation.

Deliver: `dfloat11_tt/viz/README.md` with launch instructions; `make viz` should serve it on localhost.

---

## 4. Final project structure

```
dfloat11_tt/
├── README.md                          # quickstart + results table
├── INVESTIGATION.md                   # your DFloat11 + tt-metal study
├── ARCHITECTURE.md                    # the mapping document, with diagrams
├── Makefile                           # build, eval-all, viz, clean
├── CMakeLists.txt                     # ties into tt-metal build
├── pyproject.toml
├── compress/
│   ├── FORMAT.md
│   ├── compressor.py                  # Huffman, LUTs, gaps, bundle writer
│   ├── bundle.py                      # binary format reader/writer
│   └── reference_decoder.py           # pure-Python decoder for bit-identity tests
├── kernels/
│   ├── reader_df11.cpp
│   ├── decompress_df11.cpp
│   ├── writer_df11.cpp
│   └── df11_common.hpp                # constants, bit-pack helpers, LUT struct
├── host/
│   └── df11_op.cpp                    # Metalium Program assembly + launch
├── op/
│   ├── dfloat11_decompress.cpp        # TT-NN op registration
│   ├── dfloat11_decompress.hpp
│   └── pybind.cpp
├── nn/
│   ├── df11_linear.py                 # PyTorch DF11Linear
│   ├── df11_block.py                  # batched-block wrapper
│   └── hf_patch.py                    # HF Transformers integration
├── eval/
│   ├── configs/
│   │   ├── llama-3.1-8b.yaml
│   │   ├── llama-3.2-1b.yaml
│   │   ├── mistral-small.yaml
│   │   └── qwen3-8b.yaml
│   ├── test_bit_identity.py
│   ├── test_output_equivalence.py
│   ├── test_benchmarks.py
│   ├── test_perf.py
│   ├── test_memory.py
│   └── runner.py
├── viz/
│   ├── server.py                      # FastAPI
│   ├── static/
│   ├── templates/
│   └── README.md
└── tests/                             # unit tests for kernels + Python
    ├── test_huffman.py
    ├── test_lut_hierarchy.py
    ├── test_kernel_smoke.py           # tiny synthetic input, on device
    └── test_format_roundtrip.py
```

---

## 5. Critical implementation rules

- **No stubs, no TODOs in delivered code.** If you cannot implement something, say so explicitly in `INVESTIGATION.md` with a justification, and propose a concrete fallback that you do implement.
- **Bit-identity is sacred.** Any test that compares decompressed BF16 to the original must be exact-equal. If you ever find yourself reaching for `torch.allclose`, you've made a mistake.
- **Numerical types.** BFloat16 is `(1 sign, 8 exp, 7 mantissa)`. The exponent occupies bits 7–14, sign is bit 15. Pack/unpack from `uint16_t` carefully — endianness has bitten every previous attempt at this.
- **Tile layout.** Tenstorrent's 32×32 tile, on Blackhole, has a face structure (4 faces of 16×16 in BF16). Read the tt-metal docs and the existing tilize/untilize ops before writing the writer kernel. Get this wrong and matmul output will be silently scrambled.
- **Profiler.** Use `tt-metal`'s tracy-based profiler from day one. Don't optimize blind — every perf claim in the report must come with a profile screenshot or trace.
- **Determinism.** The Huffman tree depends on tie-breaking in the priority queue. Use the same tie-break rule as `~/DFloat11` (typically: lower symbol value wins ties) so a re-compressed bundle is byte-identical to the original's. Add a regression test for this.
- **Testing on device.** Smoke tests must run on real Blackhole hardware (the user has access). Mark CPU-only mock tests clearly and never count them toward correctness claims.
- **Style.** C++17 for kernels and host code, Python ≥3.10 with type hints, `ruff` + `mypy` clean, `clang-format` clean (use the tt-metal style file). All public functions documented.
- **Logging.** Use `loguru` in Python and `fmt`/tt-metal's logging in C++. The eval runner must produce structured logs that the visualizer can ingest.
- **Reproducibility.** Pin all dependencies. The `Makefile` must rebuild kernels, run the full eval suite, and produce the visualizer dataset from a clean checkout in one command (`make all`).

---

## 6. Acceptance criteria

The implementation is accepted only when **all of the following** hold:

1. `make all` from a clean checkout on a Blackhole-equipped machine produces:
   - compiled kernels,
   - a compressed bundle for at least Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct,
   - bit-identity tests pass for every targeted tensor in those models,
   - output-equivalence tests pass on a 50-prompt set,
   - the eval suite produces `results/aggregate.json`,
   - the visualizer launches and renders all five views.
2. Compression ratio matches the paper to within 0.5% per model.
3. The MMLU and TruthfulQA accuracy of the DF11 model exactly equals the BF16 reference (paper Table 2 reproduction).
4. A `RESULTS.md` summarizes performance numbers vs. the paper's NVIDIA numbers and discusses, with profiler data, where Tenstorrent is faster, where it is slower, and why — grounded in the architectural mapping from `ARCHITECTURE.md`.
5. The visualizer's architecture-explainer view has a recorded screencast (`viz/demo.mp4` or `.gif`) suitable for embedding in a presentation.

---

## 7. How to work

- Spend the first session entirely on reading and on `INVESTIGATION.md`. Do not write kernels until that document is done and accurate.
- Build phases 1 → 5 in order. Each phase has its own acceptance test that must pass before moving to the next. **Phase 2 is the hardest; budget accordingly.**
- Use `~/tt-metal/tt_metal/programming_examples/` heavily as templates — there are reader/compute/writer examples for matmul, eltwise, and tilize that map cleanly onto what we need.
- When you hit a tt-metal API question, the canonical reference order is: `METALIUM_GUIDE.md` → relevant header in `tt_metal/include/` or `tt_metal/hw/ckernels/blackhole/metal/llk_api/` → the closest existing op under `ttnn/cpp/ttnn/operations/` → finally the tt-metal Discord / GitHub issues. Cite the source in code comments when you copy a non-trivial pattern.
- Commit early and often with descriptive messages. Tag the end of each phase.

When in doubt, prefer correctness over performance. A correct slow kernel can be optimized; an incorrect fast one wastes the project.

Begin with `INVESTIGATION.md`. Do not skip it.