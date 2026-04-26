# DFloat11-TT Binary Format v1

## Magic and Version

All bundles start with the 8-byte ASCII magic `DF11TT01` followed by the header.

## Per-Tensor Bundle Layout

Each compressed tensor is stored as a contiguous binary blob:

```
[0]   magic:       8 bytes  "DF11TT01"
[8]   dtype:       1 byte   0x00 = bfloat16 (only supported value)
[9]   ndim:        1 byte   number of dimensions (1..4)
[10]  shape:       8*ndim bytes  uint64_t, each dimension
[10+8*ndim] k:     1 byte   number of decode LUT tables (1..4)
[11+8*ndim] n:     1 byte   bytes_per_thread (always 8)
[12+8*ndim] T:     4 bytes  uint32_t, threads_per_block (CUDA; used for gap array packing)
[16+8*ndim] B:     4 bytes  uint32_t, total blocks (= ceil(n_bytes / (n*T)))
[20+8*ndim] R_pad: 4 bytes  uint32_t, padded row count (for tile alignment)
[24+8*ndim] C_pad: 4 bytes  uint32_t, padded col count (for tile alignment)
[28+8*ndim] n_elements: 8 bytes uint64_t, total number of BF16 elements
[36+8*ndim] n_bytes:    8 bytes uint64_t, total encoded exponent bytes
--- variable-length sections ---
luts:               (k+1) * 256 bytes  uint8_t[k+1][256]
                    rows 0..k-1: decode LUTs; row k: code-lengths LUT
encoded_exponent:   n_bytes bytes      uint8_t[]
sign_mantissa:      n_elements bytes   uint8_t[]
gaps:               ceil(T*B*5/8) bytes uint8_t[] (5-bit-packed, big-endian)
output_positions:   (B+1) * 4 bytes   uint32_t[] (block start element indices)
```

## Notes

- `k+1` rows of LUT: first k rows are the hierarchical decode tables; last row maps exponent value → code length.
- `output_positions[B]` = `n_elements` (sentinel).
- Gaps are 5-bit values packed MSB-first across bytes, exactly as in the original DFloat11 CUDA kernel.
- Multiple tensors can be stored in one file by concatenating bundles; each begins with `DF11TT01`.
- The format is little-endian for all multi-byte integers.
