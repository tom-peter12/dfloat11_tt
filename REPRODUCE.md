# DFloat11-TT Reproduction Runbook

This is the short path to reproduce the working DFloat11 pipeline on the
Tenstorrent Blackhole machine.

The tested setup on `tt-blackhole-02` is:

- repo: `/home/tomasissac/dfloat11_tt`
- tt-metal: `/home/tomasissac/tt-metal`
- Python: `/home/tomasissac/tt-metal/python_env/bin/python3`
- hardware: Blackhole P150 with 4 chips visible in `tt-smi`
- pipeline device selection: expose PCI device `0`, then open TT-Metal device id `3`

## 1. Enter The Repo

```bash
cd /home/tomasissac/dfloat11_tt
```

If your checkout is somewhere else, `cd` there instead. The scripts derive
`PYTHONPATH` from the repo location.

## 2. Build The C++/TTNN Extension

```bash
export TT_METAL_HOME=/home/tomasissac/tt-metal
make build TT_METAL_HOME="$TT_METAL_HOME" PYTHON="$TT_METAL_HOME/python_env/bin/python3"
```

This installs `dfloat11_tt_cpp.cpython-310-x86_64-linux-gnu.so` into the repo root.

## 3. Use The Known-Good Device Environment

The scripts set this automatically, but this is the explicit environment:

```bash
export TT_METAL_HOME=/home/tomasissac/tt-metal
export TT_METAL_VISIBLE_DEVICES=0
export TT_MESH_GRAPH_DESC_PATH="$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
export DFLOAT11_TT_DEVICE_ID=3
export DFLOAT11_CACHE_WEIGHTS=1
export DFLOAT11_TRACE_LINEAR=1
```

On this machine, `TT_METAL_VISIBLE_DEVICES=0` maps to `DFLOAT11_TT_DEVICE_ID=3`.
That mismatch is expected for this box.

## 4. Optional Sanity Tests

CPU-only tests:

```bash
PYTHONPATH="$(dirname "$PWD")" \
  "$TT_METAL_HOME/python_env/bin/python3" -m pytest tests/ -q \
  --ignore=tests/test_kernel_smoke.py
```

Tiny hardware kernel smoke:

```bash
env \
  TT_METAL_HOME="$TT_METAL_HOME" \
  TT_METAL_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH="$TT_MESH_GRAPH_DESC_PATH" \
  DFLOAT11_TT_DEVICE_ID=3 \
  PYTHONPATH="$(dirname "$PWD")" \
  "$TT_METAL_HOME/python_env/bin/python3" -m pytest \
  tests/test_kernel_smoke.py::test_smoke_tiny -vv --tb=short -s
```

Expected result:

```text
PASSED
```

## 5. Run The Fast LLM Smoke

This runs one compressed Linear layer from SmolLM2 on the TT device:

```bash
./scripts/run_llm_smoke.sh
```

Expected signs of success:

```text
Bundle contains 1 compressed tensors.
Replaced 1 modules with DF11Linear.
[df11] ... cache=cold
[df11] ... cache=cached
[output_equiv] ALL PASS
```

## 6. Run The Small Model Pipelines

Run these separately. The first run may compress the model on CPU and create a
`.df11tt` bundle under `compressed/`. Later runs reuse that bundle.

Smallest full-model pipeline:

```bash
./scripts/run_smollm2_135m_pipeline.sh
```

Qwen under 1B full-model run:

```bash
./scripts/run_qwen2_5_0_5b_pipeline.sh
```

Llama 3.2 1B full-model run:

```bash
./scripts/run_llama3_2_1b_pipeline.sh
```

Llama models may require Hugging Face access:

```bash
huggingface-cli login
```

## What The Pipeline Does

For each config, the runner does:

```text
CPU compress bundle if missing
open TT device
upload compressed tensors to TT memory
load Hugging Face model structure
replace token embedding with DF11Embedding
replace attention/MLP/lm_head nn.Linear modules with DF11Linear
generate tokens
clear cached decompressed TT weights
write results
```

The professor-facing scripts use the `*-full.yaml` configs. These compress:

```text
model.embed_tokens
all self_attn q/k/v/o projections
all MLP gate/up/down projections
lm_head
```

LayerNorm and other small non-Linear parameters remain in the base BF16
Hugging Face model.

Inside each `DF11Linear`:

```text
first use:
  compressed tensor on TT -> TT decompression -> tiled/transposed TT weight cache
  ttnn.linear

later token steps:
  reuse cached TT weight
  ttnn.linear

after eval:
  discard cached TT weights
```

Inside `DF11Embedding`:

```text
first use:
  compressed embedding table on TT -> TT decompression -> cached BF16 table
  embedding lookup

later token steps:
  reuse cached embedding table

after eval:
  discard cached embedding table
```

The current path is a hybrid integration:

```text
Hugging Face/PyTorch control flow on CPU
DF11 decompression on TT device
TTNN matmul on TT device
layer output copied back to CPU for the next HF operation
```

This is enough to prove the end-to-end DFloat11 device implementation, but it is
not yet an optimized all-on-device LLM runtime.

## Useful Environment Toggles

Show per-layer progress:

```bash
export DFLOAT11_TRACE_LINEAR=1
```

Disable per-run decompressed weight caching:

```bash
export DFLOAT11_CACHE_WEIGHTS=0
```

Enable slow CPU bit-identity verification during compression:

```yaml
compression_check: true
```

Add that to an eval config only when auditing the compressor. The default
pipeline skips this check so bundle creation is much faster.

## Results

Per-model results are written under:

```text
results/<model-name>/summary.json
results/<model-name>/report.md
results/<model-name>/raw/output_equivalence.json
```

The latest aggregate is:

```text
results/aggregate.json
```

## Troubleshooting

If `ttnn` says no devices are detected, check:

```bash
tt-smi -ls
echo "$TT_METAL_VISIBLE_DEVICES"
echo "$DFLOAT11_TT_DEVICE_ID"
```

For this machine, use:

```bash
export TT_METAL_VISIBLE_DEVICES=0
export DFLOAT11_TT_DEVICE_ID=3
```

If fabric mapping fails, make sure this descriptor is set:

```bash
export TT_MESH_GRAPH_DESC_PATH="$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
```

If you see `k must be in [1,4]`, rebuild the extension. The current code accepts
`k <= 5`, which is required by some real Llama/Qwen tensors:

```bash
make build TT_METAL_HOME="$TT_METAL_HOME" PYTHON="$TT_METAL_HOME/python_env/bin/python3"
```

If generation looks like it restarted at layer 0 after reaching the final layer,
that is normal. `generate()` loops through all transformer layers once per token.
For example, with `max_new_tokens: 4`, the model does four layer sweeps.

If the first token is slow, that is also expected. The first pass fills the
decompressed TT weight cache. Later token passes should show `cache=cached` in
the logs and run faster.
