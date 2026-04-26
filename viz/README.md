# DFloat11-TT Visualizer

A FastAPI + vanilla JS web dashboard for exploring compression and performance results.

## Launch

```bash
make viz
# or directly:
python -m uvicorn dfloat11_tt.viz.server:app --host 0.0.0.0 --port 8080
```

Opens at http://localhost:8080

## Views

1. **Compression**: bar charts of original vs compressed size per model; per-layer-type ratio breakdown.
2. **Performance**: decompression throughput (GB/s) vs matrix size; token decode latency vs batch size.
3. **Architecture**: animated side-by-side visualization of NVIDIA GPU (CUDA blocks) vs Tenstorrent Blackhole (Tensix cores) executing a transformer block decompression. Exportable as standalone HTML.
4. **Compare Models**: side-by-side model comparison card.

## Data source

The dashboard reads `results/aggregate.json` produced by `make eval-all`.
If no results exist, the dashboard shows empty charts.

## Export

Each chart has a "Download PNG" button. The Architecture view has "Export Standalone HTML" which produces a self-contained `df11tt_architecture.html` suitable for offline presentation.
