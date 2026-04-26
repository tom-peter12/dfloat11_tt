# DFloat11-TT Pipeline Notes

This document explains what the reproduction scripts run.

## Runtime Flow

For each model config:

```text
CPU compress bundle if missing
open TT device
upload compressed tensors to TT memory
load Hugging Face model structure
replace token embedding with DF11Embedding
replace attention/MLP/lm_head nn.Linear modules with DF11Linear
generate tokens
clear cached decompressed weights
write results
```

The `*-full.yaml` configs compress:

```text
model.embed_tokens
all self_attn q/k/v/o projections
all MLP gate/up/down projections
lm_head
```

LayerNorm and other small non-Linear parameters remain BF16 in the base Hugging
Face model.

## Device Work

The `.df11tt` file is loaded by Python, then the compressed arrays are uploaded
to TT memory as `ttnn.Tensor`s. Tensix does not read the file directly.

For each compressed Linear:

```text
compressed tensor on TT
-> TT decompression kernel
-> tiled/transposed TT weight cache
-> ttnn.linear
```

For embeddings:

```text
compressed embedding table on TT
-> TT decompression kernel
-> cached BF16 embedding table
-> embedding lookup
```

The current integration is hybrid:

```text
Hugging Face/PyTorch control flow on CPU
DF11 decompression on TT device
TTNN matmul on TT device
layer output copied back to CPU for the next HF operation
```

This proves the end-to-end DFloat11 device implementation. It is not yet an
optimized all-on-device LLM runtime.

## Useful Toggles

Show per-layer progress:

```bash
export DFLOAT11_TRACE_LINEAR=1
```

Disable decompressed weight caching:

```bash
export DFLOAT11_CACHE_WEIGHTS=0
```

Enable slow CPU bit-identity verification while compressing:

```yaml
compression_check: true
```

## Troubleshooting

If `ttnn` says no devices are detected:

```bash
tt-smi -ls
echo "$TT_METAL_VISIBLE_DEVICES"
echo "$DFLOAT11_TT_DEVICE_ID"
```

If fabric mapping fails, set:

```bash
export TT_MESH_GRAPH_DESC_PATH="$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
```

If you see `k must be in [1,4]`, rebuild the extension. The current code accepts
`k <= 5`:

```bash
make build TT_METAL_HOME="$TT_METAL_HOME" PYTHON="$TT_METAL_HOME/python_env/bin/python3"
```

If generation reaches the final layer and starts again at layer 0, that is
normal. `generate()` runs all transformer layers once per generated token.
