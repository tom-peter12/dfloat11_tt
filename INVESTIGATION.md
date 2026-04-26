# INVESTIGATION.md — DFloat11 and TT-Metal Study

## 1. DFloat11 Deep-Dive

### 1.1 BFloat16 Bit Layout

BFloat16 is a 16-bit float with layout (MSB→LSB):

```
bit 15 : sign (S)
bits 14-7 : exponent (E, 8 bits, bias 127)
bits 6-0  : mantissa (M, 7 bits)
```

As a `uint16_t` (little-endian storage on x86/NVIDIA):
- low byte  (bits 7-0):  low bit of exponent (bit 0) + full mantissa (bits 7-1 of bf16 → bits 6-0 of the byte)
- high byte (bits 15-8): sign + upper 7 bits of exponent

The DFloat11 split (from `dfloat11_utils.py`):

```python
W = weight.view(torch.int16)
exponent_8bits = (W >> 7) & 0xFF   # bits 14-7 of bf16 → 8-bit exponent
other_8bits    = (W >> 8) & 0x80   # sign bit
               | (W & 0x7F)        # mantissa (bits 6-0)
```

So:
- `EncodedExponent`: the 8-bit exponent field, Huffman-compressed.
- `PackedSignMantissa` (`sign_mantissa`): packed byte = `(sign << 7) | mantissa`, stored raw (no compression).

Reconstruction in the CUDA kernel (decode.cu lines 207-210):

```c
buffer[8] = sign_mantissa[output_idx];
buffer[9] = (buffer[8] & 128) | (decoded >> 1);   // high byte: sign | upper 7 bits of exponent
buffer[8] = (decoded << 7)    | (buffer[8] & 127); // low byte: low bit of exponent | mantissa
write_buffer[output_idx - write_offset] = short_buffer;  // little-endian BF16
```

Key endianness detail: `buffer[8]` is the *low* byte and `buffer[9]` is the *high* byte of the uint16_t stored in `short_buffer`. The `alignas(8)` buffer at bytes 8-9 is cast to a `uint16_t`. On a little-endian machine `buffer[8]` occupies bits 7-0 and `buffer[9]` occupies bits 15-8 of the resulting uint16.

### 1.2 Huffman Tree and LUT Hierarchy

**Histogram**: Count occurrences of each of the 256 possible exponent values across all bf16 elements in the weight matrix.

**Tree construction** (`dfloat11_utils.py::get_codec`):
- Uses `dahuffman.HuffmanCodec.from_frequencies(counter)`.
- If any code length exceeds 32 bits, the rarest symbols are merged (set to frequency 1) iteratively until `max_len ≤ 32`.
- Tie-breaking: `dahuffman` uses heap ordering, which ties by insertion order (symbol value for integer keys). **This must be reproduced exactly** to guarantee bit-identity across independent re-compressions.

**LUT structure** (`dfloat11_utils.py::get_luts`):
- A 2D numpy array of shape `(k+1, 256)` as `uint8`, where:
  - `luts[0..k-1]`: the decode LUTs. Each entry is either a decoded exponent value (0-239) or a pointer `256 - next_lut_idx` (values 240-255 act as overflow pointers).
  - `luts[k]` (last row): the code-lengths LUT. `luts[k][exponent_val] = code_length_in_bits`.
- Stored as a flat `torch.Tensor` of shape `(k+1, 256)` dtype `uint8`.
- Total size: `(k+1) × 256` bytes. With `k ≤ 3` this is at most 4×256 = 1024 bytes — fits trivially in L1.

**LUT lookup chain** (decode.cu lines 104-118, using `long_buffer` = 64-bit MSB-aligned bit buffer):

```c
decoded = luts[long_buffer >> 56];           // top 8 bits → LUT[0]
if (decoded >= 240)
    decoded = luts[256*(256-decoded) + ((long_buffer>>48)&0xff)]; // next 8 bits → LUT[1]
if (decoded >= 240)
    decoded = luts[256*(256-decoded) + ((long_buffer>>40)&0xff)]; // → LUT[2]
if (decoded >= 240)
    decoded = luts[256*(256-decoded) + ((long_buffer>>32)&0xff)]; // → LUT[3]
// decoded is now the raw exponent value (0-255)
decoded_length = luts[256*(n_luts-1) + decoded];  // code length in bits
long_buffer <<= decoded_length;
free_bits     += decoded_length;
```

The LUT chain depth is at most 4 (max 32-bit code, 4 × 8-bit LUT lookups). The `256 - decoded` trick converts the pointer value to a 1-indexed LUT offset.

### 1.3 Gaps Array

`gaps` is a bit-packed array of 5-bit values, one per CUDA thread (= per block of `BYTES_PER_THREAD=8` bytes).

Each 5-bit value encodes the **bit offset within the thread's byte range where the first complete Huffman code begins**. A value of 0 means the first code starts at the first bit of byte 0 of this thread's range; a value up to 31 (2^5-1) means it starts up to 31 bits into the range.

This allows each thread to independently start decoding without needing to see the preceding thread's state.

**Extraction** (decode.cu lines 85-87):

```c
buffer[8] = gaps[global_thread_id * 5 / 8 + 1];
buffer[9] = gaps[global_thread_id * 5 / 8];
const uint8_t gap = (short_buffer >> (11 - (global_thread_id * 5 % 8))) & 0x1f;
```

5-bit values are packed big-endian (MSB first) across bytes. The formula extracts the 5 bits straddling two bytes at `global_thread_id * 5` bits into the stream.

### 1.4 BlockOutputPos (output_positions)

`output_positions` is a `uint32_t` array of length `n_blocks + 1`. Element `i` is the absolute index of the first BF16 element produced by block `i`. The last element equals `n_elements`.

This is the **block-level prefix sum** — each CUDA block knows where in the output array its first element goes, without needing inter-block communication.

In Python: computed during encoding by tracking total element count at each block boundary.

### 1.5 CUDA Kernel Internals — Two-Phase Decode

**Parameters**: `threads_per_block = 512`, `bytes_per_thread = 8`.

**Phase 1 — counting** (lines 83-174):
- Each thread loads 12 bytes (bytes 0-7 from its range; bytes 8-11 which straddle into the next range to handle codes that span the boundary).
- Extracts its `gap` from the `gaps` array.
- Walks the bit buffer, decoding as many Huffman symbols as fit in `32 + extra_bits` bits (the condition `while (free_bits < 32)` means: stop when we've consumed past byte 4 of the 8-byte range, i.e. the first 32 bits worth of full codes in this thread's range).
- Loads the remaining 4 bytes and continues until `4 + free_bits/8 >= BYTES_PER_THREAD`.
- Counts decoded symbols in `thread_counter`.
- Performs **Blelloch prefix sum** on `accumulators[]` in shared memory to compute per-thread output offsets within the block.

**Phase 2 — writing** (lines 182-247):
- Re-walks the same byte range from scratch (using register_buffer again).
- This time, for each decoded exponent:
  - Fetches `sign_mantissa[output_idx]`.
  - Assembles the BF16 `uint16_t`.
  - Writes into `write_buffer[output_idx - write_offset]` (shared memory).
- After both threads finish: coalesced stride-N write from `write_buffer` to global `outputs[]`.

**Shared memory size**: `N_THREADS * 4 + 4 + max_block_output_elements * 2` bytes.
- `N_THREADS * 4 + 4`: accumulators (512 uint32s + 1 for THREAD_ID=0's final accumulator).
- `max_block_output_elements * 2`: write_buffer for BF16 values.

### 1.6 HF Integration

- `compress_model()` traverses the model, matches modules by regex patterns in `pattern_dict`.
- Matching modules: `nn.Linear`, `nn.Embedding`, and composite patterns (e.g., fused QKV).
- Registers buffers: `luts`, `encoded_exponent`, `sign_mantissa`, `output_positions`, `gaps`, `split_positions`.
- Removes the `weight` attribute.
- Registers a `forward_pre_hook` that decompresses on every forward pass.
- `DFloat11Model.from_pretrained()`: re-creates model skeleton, loads compressed safetensors, re-attaches hooks.

---

## 2. TT-Metal / Blackhole Deep-Dive

### 2.1 Hardware

| Feature | Blackhole |
|---------|-----------|
| Total Tensix | 14×10 = 140 |
| Compute Tensix | 13×10 = 130 |
| L1 per Tensix | 1464 KB ≈ 1.5 MB |
| L1 data cache (BH-specific) | 4×16B writeback, disabled by default |
| DRAM banks | 8 banks × ~4 GB = ~32 GB total |
| DRAM read alignment | 64B |
| DRAM write alignment | 16B |
| L1 read/write alignment | 16B |
| Multicast | Rectangular + Strided + L-shaped |
| SFPU lanes | 32 per face (4 faces = 128 lanes per 32×32 tile) |

**Baby RISC-V cores per Tensix (5 total)**:
- `BRISC` (Data Movement 0 / RISCV_0): runs reader kernel; handles NoC0.
- `NCRISC` (Data Movement 1 / RISCV_1): runs writer kernel; handles NoC1.
- `TRISC0` (Unpack): unpacks from L1 CB into FPU src registers.
- `TRISC1` (Math): drives FPU/SFPU computation.
- `TRISC2` (Pack): packs FPU dst registers back to L1 CB.

For our DFloat11 decode kernel, TRISC0/1/2 all run the **same compute kernel source** compiled three times; each compile emits code for only the relevant functional blocks.

### 2.2 Tile Layout on Blackhole

A **32×32 BF16 tile** = 2048 bytes. Internal layout (matching tt-metal's tilize):
- 4 faces of 16×16 elements.
- Face ordering: face[0] = rows 0-15 cols 0-15, face[1] = rows 0-15 cols 16-31, face[2] = rows 16-31 cols 0-15, face[3] = rows 16-31 cols 16-31.
- Within each face: row-major 16×16 elements.

For a matrix of shape `(R, C)`:
- `n_tile_rows = ceil(R/32)`, `n_tile_cols = ceil(C/32)`.
- Tile `(tr, tc)` starts at byte offset `(tr * n_tile_cols + tc) * 2048`.
- Element at row `r`, col `c` within the tile (0-based): face `(r//16)*2 + (c//16)`, then offset `(r%16)*16 + (c%16)` within the face.

For the writer kernel, to place the output element for matrix row `r` and col `c`:
```
tile_row = r / 32, tile_col = c / 32
face = (r%32)/16 * 2 + (c%32)/16
intra_face = (r%16)*16 + (c%16)
byte_offset = (tile_row * n_tile_cols + tile_col) * 2048
            + face * 256 * 2     // 256 elements per face × 2 bytes
            + intra_face * 2
```

**Critical**: the TT-NN matmul expects tiled tensors. We decompress directly into tiled layout (Path A) to avoid a second tilize pass.

### 2.3 Metalium Programming Model

The triplet per Tensix:
1. **Reader** (`RISCV_0`/BRISC): issues `noc_async_read` to fetch data from DRAM into L1 circular buffers; calls `cb_push_back`.
2. **Compute** (TRISC0+1+2 via single source): waits on `cb_wait_front`, calls FPU/SFPU APIs or plain scalar C++ on TRISC1.
3. **Writer** (`RISCV_1`/NCRISC): pops from output CB, issues `noc_async_write` to DRAM.

**Circular buffers** are L1-backed FIFO queues. They have a fixed slot count; `cb_reserve_back`/`cb_push_back` are producer ops; `cb_wait_front`/`cb_pop_front` are consumer ops.

**Synchronization**: within a kernel, use `noc_async_read_barrier()` and `noc_async_write_barrier()` (or `_flushed()` variants). Cross-kernel synchronization happens implicitly through CBs.

**SPMD**: each Tensix core runs the same kernel code but may receive different runtime args (e.g., which byte-range of the encoded stream to process), enabling SPMD data-parallel decode.

### 2.4 No "Shared Memory" Between CUDA Threads

Unlike CUDA where 512 threads within a block share 48–100 KB of shared memory, within a Tensix we have:
- Only 5 separate RISC-V cores (no 512-way parallelism).
- The FPU operates on full 32×32 tiles — not useful for Huffman decode.
- The SFPU has 32 lanes per face and can run SIMD on 128 values simultaneously — potentially useful for the prefix sum step.

The mapping we adopt:
- **One Tensix ↔ one CUDA block** (or a multiple thereof).
- Within a Tensix, the Huffman walk is a sequential scalar loop on TRISC1.
- The "threads" of the original kernel become sequential iterations on TRISC1.
- The Blelloch prefix-sum (which required `__syncthreads()`) is replaced by a simple sequential prefix sum in L1 scratch — no barrier needed since it's single-core.
- The write_buffer in CUDA shared memory → a contiguous L1 region in the compute kernel's assigned address space.

### 2.5 Memory Budget per Tensix

A typical weight tensor (e.g., `q_proj` in Llama-3.1-8B: 4096×4096 bf16 = 32 MB uncompressed, ~22 MB compressed):
- We shard the encoded exponent stream across all 130 compute cores.
- Each core processes ≈22MB/130 ≈ 170 KB of encoded bytes.
- L1 budget per core:
  - CB for encoded exponents (double-buffered, 2×8 KB) = 16 KB
  - CB for sign_mantissa (same, 2×8 KB) = 16 KB
  - CB for gaps + block output positions (~few KB) = 4 KB
  - LUTs + CodeLengths: at most 4×256 + 256 = 1280 bytes ≈ 1.25 KB
  - Write buffer (tiled output, 2×2048-byte tiles) = 4 KB
  - Scratch for prefix sum (T×4 bytes, T=number of logical threads per block ≈ 256) = 1 KB
  - **Total: ~42 KB** out of 1.5 MB — very comfortable.

### 2.6 TT-NN Op Registration Pattern

Following `ttnn/cpp/ttnn/operations/examples/example/`:
1. Define `DeviceOperation` struct with `operation_attributes_t`, `tensor_args_t`, `ProgramFactory`.
2. Implement `create()` (builds the Metalium Program with kernels and CBs) and `override_runtime_arguments()` (updates addresses for repeated calls).
3. Bind to Python via nanobind: `ttnn::bind_function<"dfloat11_decompress">(...)`.

---

## 3. Architectural Mapping Decision

### 3.1 Chosen Mapping

| CUDA (DFloat11 original) | Tenstorrent Blackhole (this port) |
|---|---|
| 1 CUDA block of 512 threads | 1 Tensix core (sequential scalar on TRISC1) |
| 512 threads process 512×8=4096 B/block | TRISC1 processes ~4096 B sequentially |
| `__syncthreads()` Blelloch prefix sum | Sequential scan in L1 scratch (single core, no sync needed) |
| Shared memory write buffer | L1 scratch buffer at fixed address |
| `__ldg` (texture cache) for LUT | L1-resident LUT array (explicit load once by reader) |
| Coalesced global write via shared memory | `noc_async_write` of assembled tiled region |
| CUDA grid of N_blocks | 130 Tensix cores, work distributed by host |

### 3.2 Tile Layout: Path A

We decompress directly into tiled layout to avoid a second pass. The compute kernel calculates the correct byte offset in L1 for each output element.

```
global_element = block_start_element + local_idx
row = global_element / C
col = global_element % C
tile_row = row / 32,  tile_col = col / 32
face = (row%32)/16 * 2 + (col%32)/16
intra = (row%16)*16 + (col%16)
dest_byte = (tile_row * n_tile_cols + tile_col) * 2048 + face * 512 + intra * 2
```

### 3.3 SFPU Tradeoff

The SFPU is optimally used for dense, regular vector operations. Huffman decode is inherently variable-length and branch-heavy (the LUT chain has data-dependent depth 1-4). Running SFPU in a "divergent" way wastes most lanes and adds complexity with little benefit. **Decision: use TRISC1 scalar RISC-V for the Huffman walk, and use no SFPU.** The prefix sum on T=256 items is O(T) scalar work and does not justify SFPU setup overhead.

### 3.4 Known Limitations / Fallbacks

1. **No actual Blelloch needed**: since our "threads" are sequential on a single RISC-V core, the prefix sum is just a sequential forward scan. This is strictly simpler and correct.

2. **Tile padding**: weight matrix dimensions must be padded to multiples of 32. We handle this in the host program by rounding `R` and `C` up to the nearest multiple of 32 and writing zeros into the pad region.

3. **Max code length**: the Python compressor enforces `max_len ≤ 32`, which means at most 4 LUT lookups per symbol. This is hardcoded in the kernel.

4. **Batched block decompression**: `DF11TransformerBlock` launches a single Metalium Program that assigns subsets of the 130 cores to each weight matrix. This provides true overlap of decompression across all weight matrices in a block.

5. **Blackhole-specific NoC flush**: per the Blackhole Bring-Up Guide, kernels must add explicit `noc_async_write_barrier()` / `noc_async_read_barrier()` calls that were implicit on Wormhole. All our kernels do this.
