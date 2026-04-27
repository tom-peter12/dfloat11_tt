# Reproduce DFloat11-TT

Use this after cloning the repo on a Tenstorrent Blackhole machine with
`tt-metal` already built.

## 1. Clone

```bash
git clone <repo-url> dfloat11_tt
cd dfloat11_tt
```

## 2. Point To tt-metal

If `tt-metal` is next to this repo:

```bash
export TT_METAL_HOME="$(cd ../tt-metal && pwd)"
```

Otherwise:

```bash
export TT_METAL_HOME=/path/to/tt-metal
```

## 3. Build

```bash
make build TT_METAL_HOME="$TT_METAL_HOME" PYTHON="$TT_METAL_HOME/python_env/bin/python3"
```

## 4. Set Device Env

```bash
export TT_METAL_VISIBLE_DEVICES=0
export TT_MESH_GRAPH_DESC_PATH="$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
export DFLOAT11_TT_DEVICE_ID=3
export DFLOAT11_CACHE_WEIGHTS=1
export DFLOAT11_TRACE_LINEAR=1
```

On the tested Blackhole box, visible device `0` maps to TT device id `3`.

## 5. Quick Check

```bash
./scripts/run_llm_smoke.sh
```

Expected result:

```text
[output_equiv] ALL PASS
```

## 6. Run A Model

Smallest:

```bash
./scripts/run_smollm2_135m_pipeline.sh
```

Qwen under 1B:

```bash
./scripts/run_qwen2_5_0_5b_pipeline.sh
```

Llama 3.2 1B:

```bash
./scripts/run_llama3_2_1b_pipeline.sh
```

The first run creates a compressed bundle under `compressed/`. Later runs reuse
that bundle.

To keep comparison runs separate, set a results suffix before running:

```bash
DFLOAT11_MAX_CORES=1 DFLOAT11_RESULTS_SUFFIX=cores1 ./scripts/run_llama3_2_1b_pipeline.sh
DFLOAT11_MAX_CORES=30 DFLOAT11_RESULTS_SUFFIX=cores30 ./scripts/run_llama3_2_1b_pipeline.sh
```

Those runs write to `results/llama-3.2-1b-full-cores1/`,
`results/llama-3.2-1b-full-cores30/`, `results/aggregate_cores1.json`,
and `results/aggregate_cores30.json`.

If Llama access fails, run:

```bash
huggingface-cli login
```

## 7. Open The Visualization Notebook

After at least one model run finishes:

```bash
make viz
```

Then open:

```text
viz/dfloat11_tt_presentation.ipynb
```

Run all cells. In VS Code, select the same Python environment used for the
pipeline, usually:

```text
$TT_METAL_HOME/python_env/bin/python3
```

The notebook uses `pandas`, `matplotlib`, and `seaborn`. These are included in
the tested environment; install them in the selected Python env if needed.

## Output

Results are written to:

```text
results/aggregate.json
results/<model-name>/
```

With `DFLOAT11_RESULTS_SUFFIX=<suffix>`, results are written to:

```text
results/aggregate_<suffix>.json
results/<model-name>-<suffix>/
```

More details on what the pipeline does are in [PIPELINE.md](PIPELINE.md).
