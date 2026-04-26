"""DFloat11-TT visualization server (FastAPI + vanilla HTML/JS)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent.parent
RESULTS_PATH = ROOT / "results" / "aggregate.json"
STATIC_PATH  = Path(__file__).parent / "static"
TEMPLATES_PATH = Path(__file__).parent / "templates"

app = FastAPI(title="DFloat11-TT Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")


def _load_results() -> List[Dict]:
    if not RESULTS_PATH.exists():
        return []
    with open(RESULTS_PATH) as f:
        return json.load(f)


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (TEMPLATES_PATH / "index.html").read_text()
    return HTMLResponse(html)


@app.get("/api/results")
async def get_results():
    return JSONResponse(_load_results())


@app.get("/api/summary")
async def get_summary():
    """Return a flat summary suitable for the compression dashboard."""
    results = _load_results()
    summary = []
    for r in results:
        model_id = r.get("model_id", "unknown")
        mem = r.get("memory", {})
        perf = r.get("performance", {})
        bi = r.get("bit_identity", {})
        entry = {
            "model_id": model_id,
            "model_slug": model_id.split("/")[-1],
            "compression_ratio": mem.get("aggregate_ratio"),
            "orig_gb": (mem.get("total_orig_bytes") or 0) / 1e9,
            "comp_gb": (mem.get("total_comp_bytes") or 0) / 1e9,
            "eff_bits": mem.get("aggregate_eff_bits"),
            "bit_identity_pass": bi.get("all_pass"),
            "decode_latency": perf.get("decode_latency", []),
            "decompress_throughput": perf.get("decompress_throughput", []),
            "tensor_compression": mem.get("tensors", []),
        }
        summary.append(entry)
    return JSONResponse(summary)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
