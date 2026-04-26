# DFloat11-TT Visualization

The presentation visuals live in one notebook:

```text
viz/dfloat11_tt_presentation.ipynb
```

Open it in Jupyter or VS Code after running one of the model pipelines, then
run all cells. It uses `pandas`, `matplotlib`, and `seaborn`.

The notebook reads:

```text
results/aggregate.json
compressed/*.df11tt
eval/configs/*.yaml
```

It shows compression size, per-module compression breakdown, BF16 vs DF11
generation timing when timing fields are available, output completions, LUT
depth distribution, and the largest compressed tensors.
