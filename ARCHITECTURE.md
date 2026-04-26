# ARCHITECTURE.md — DFloat11-TT Mapping Document

## 1. DFloat11 Algorithm Summary

DFloat11 exploits the low entropy of BFloat16 exponents. The 8-bit exponent field has only ~2.6 bits of effective entropy in trained LLMs (most weights cluster around a few exponent values). Huffman coding the exponent stream yields ~70% compression while keeping sign+mantissa raw.

```
BF16: [S(1) | E(8) | M(7)]
        ↓ split
EncodedExponent: Huffman(E)  → ~30% of original size
PackedSignMantissa: raw(SM)  → 50% of original size
Total: ~80% ... but overhead (LUTs, Gaps, OutputPositions) ≈ -10% → ~70%
```

## 2. NVIDIA CUDA Architecture (Original)

```
Grid: N_blocks = ceil(n_bytes / (512 × 8))
Block: 512 SIMT threads

Thread i in Block b:
  ├── Reads bytes [i×8, i×8+11] from EncodedExponent (12 bytes, for cross-boundary codes)
  ├── Reads gap[b*512+i] (5-bit value from gaps array)
  └── Phase 1: count decoded symbols → thread_counter
        └── __syncthreads() + Blelloch prefix-sum on shared accumulators[]
  └── Phase 2: decode + write BF16 to shared write_buffer[]
        └── __syncthreads() + coalesced write to HBM

Shared memory per block:
  accumulators: 512×4 + 4 = 2052 bytes (thread-level output positions)
  write_buffer: max_block_elements × 2 bytes (assembled BF16s)
  
Total: 2052 + ~8KB typical = ~10KB/block
LUT: 4×256 bytes, cached in L1/L2 texture cache
```

## 3. Tenstorrent Blackhole Architecture (This Port)

```
Chip: 13×10 = 130 compute Tensix cores
Each Tensix: 5 RISC-V baby cores + FPU + SFPU + 1.5MB L1 SRAM

Work Distribution:
  Total encoded bytes N → split into 130 chunks of N/130 each
  Core (x,y) handles bytes [start_byte .. end_byte) of EncodedExponent
  and the corresponding sign_mantissa slice and gap entries.

Per-Core (=per-Tensix) execution:

  BRISC (reader / DM0):
    ├── Loads LUTs (≤1280 bytes) from DRAM → L1 scratch (once per tensor)
    ├── Streams encoded exponent chunk (cb_encoded circular buffer)
    ├── Streams sign_mantissa chunk (cb_signmant circular buffer)
    └── Loads gaps slice + block_output_positions entry

  TRISC1 (compute / Math core, plain RISC-V scalar):
    ├── Phase 1: sequential walk through byte range
    │     For each logical thread t (t = 0 .. T-1):
    │       - Load 12 bytes from cb_encoded
    │       - Extract gap[t]
    │       - Walk bit-buffer with LUT chain (identical logic to CUDA Phase 1)
    │       - Store num_elements[t] in L1 scratch
    │     Sequential prefix sum: output_pos[0]=block_start; output_pos[t] = output_pos[t-1]+num_elements[t-1]
    └── Phase 2: sequential walk, emit BF16 values
          For each logical thread t:
            For each decoded symbol:
              - Assemble BF16 from decoded exponent + sign_mantissa byte
              - Compute tiled byte offset (Path A: direct tile layout)
              - Write to L1 write_buffer at tiled offset

  NCRISC (writer / DM1):
    └── Issues noc_async_write from L1 write_buffer → DRAM output tensor
        (one NoC transfer per tile-worth of output)

L1 layout per core:
  [0x0000 - 0x04FF] LUTs (1280 bytes, k=4 rows × 256 + 1 code-len row)
  [0x0500 - 0x14FF] cb_encoded (double-buffered, 2 × 4KB)
  [0x1500 - 0x24FF] cb_signmant (double-buffered, 2 × 4KB)
  [0x2500 - 0x2FFF] gaps slice (~512 × 5 bits / 8 = ~320 bytes, padded to 4KB)
  [0x3000 - 0x3FFF] scratch: num_elements[], output_pos[] arrays (T × 4B each)
  [0x4000 - 0x4FFF] write_buffer (tiles being assembled, 2KB per tile × 2)
  [0x5000 ...     ] reserved / unused (remaining ~1.4 MB free)
```

## 4. Key Mapping Table

| Concept | CUDA (NVIDIA) | Tenstorrent Blackhole |
|---------|--------------|----------------------|
| Parallelism unit | 512 SIMT threads per block | 1 Tensix core (5 sequential RISC-V) |
| Data parallelism | 1000s of blocks across grid | 130 Tensix cores via SPMD |
| Shared memory | 48-100 KB shared between threads | L1 SRAM (1.5 MB, explicitly managed) |
| Inter-thread sync | `__syncthreads()` | Not needed (sequential scalar) |
| Prefix sum | Blelloch tree reduction | Sequential forward scan in L1 |
| LUT access | Texture cache (`__ldg`) | L1-resident array (reader loads once) |
| Output buffer | Shared memory write_buffer | L1 scratch write_buffer |
| Final write | Coalesced global memory write | `noc_async_write` (NoC1 → DRAM) |
| Data fetch | Implicit L1/L2 cache | Explicit `noc_async_read` (NoC0 ← DRAM) |
| Tile layout | Row-major (arbitrary) | Tiled 32×32 (Path A: direct tile write) |
| Block launch | CUDA `<<<N, 512>>>` | Program on CoreRangeSet of 130 cores |

## 5. Transformer-Block Batched Decompression

The paper's Section 2.3.3 batches decompression of an entire transformer block to overlap decompression with data transfer.

**NVIDIA approach**: multiple CUDA kernel launches (or streams) before block forward pass.

**Blackhole approach**: one Metalium `Program` whose `CoreRangeSet` spans all 130 compute cores, partitioned across weight matrices:

```
Cores 0..19  → q_proj decompression
Cores 20..39 → k_proj decompression
Cores 40..59 → v_proj decompression
Cores 60..89 → o_proj decompression
Cores 90..109 → gate_proj decompression
Cores 110..129 → up_proj decompression (down_proj in next batch)
```

All 130 cores work simultaneously. The Program completes when all cores finish. Then the block's matmul operations proceed on the now-decompressed BF16 tensors.

For large matrices (e.g., 4096×14336 in Llama-3.1-8B gate_proj), more cores are allocated proportionally.

## 6. Tile Layout Detail (Path A)

For a decompressed matrix of shape `(R, C)` padded to `(R', C')` where `R' = ceil(R/32)*32`, `C' = ceil(C/32)*32`:

Given linear element index `idx` (where element 0 is matrix row 0, col 0):
```c++
uint32_t row = idx / C;
uint32_t col = idx % C;
uint32_t tile_row = row / 32;
uint32_t tile_col = col / 32;
uint32_t n_tile_cols = C_padded / 32;
uint32_t face = (row % 32) / 16 * 2 + (col % 32) / 16;
uint32_t intra_face = (row % 16) * 16 + (col % 16);
uint32_t byte_offset = (tile_row * n_tile_cols + tile_col) * 2048
                     + face * 512   // 256 elements × 2 bytes
                     + intra_face * 2;
```

Padding elements (row ≥ R or col ≥ C) are written as `0x0000` (BF16 zero).

## 7. Why No SFPU for Huffman Walk

The SFPU executes 32-lane SIMD over float32 or int32 values. Huffman decode requires:
- Variable-length shifts of a 64-bit integer (`long_buffer <<= decoded_length`)
- Conditional branches based on LUT lookup result (0-3 chained if-statements)
- Indexed byte loads from a small table

None of these map efficiently to SFPU 32-lane SIMD. A 32-wide SIMD walk would require all 32 lanes to process independent bit-streams simultaneously, which requires 32× the input bandwidth and 32× the LUT broadcast — all for a unit that is designed for float math, not bit manipulation.

**Conclusion**: SFPU is not used. All Huffman decode logic runs as plain RISC-V scalar C++ on TRISC1. This is correct, maintainable, and performs well because the bottleneck is memory bandwidth (DRAM → L1 → output), not arithmetic throughput.
