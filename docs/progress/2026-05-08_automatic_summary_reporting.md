# Automatic Summary Reporting For Llama/Qwen Bridge Results

## Goal

Make the completed Llama/Qwen transfer run easier to inspect, reproduce, and cite later.

The immediate purpose was not to change the SetConCA architecture. The purpose was to turn the finished `transfer_steering_results.csv` into stable, human-readable summary tables and visual figures so we can decide the next scientific step without manually reading thousands of pairwise rows.

## Context

The active result directory is:

```text
results/llama_qwen_set_vs_pointwise_linear_seed0
```

This run compares:

- `setconca`
- `pointwise_topk`

over:

- Llama 3 and Qwen 3 only,
- set sizes `2,4,6,8,10,12,14,16`,
- layer depths near `20%`, `60%`, and `90%`,
- bridge types `identity`, `procrustes`, and `ridge`,
- concept-transfer rows at `steering_alpha=0`.

Gemma 3 remains paused because earlier shuffled-anchor diagnostics showed weak controlled bridge evidence for the current phase.

## What Changed

The reporting layer in `scripts/run_transfer_steering_grid.py` now writes grouped summaries and figures automatically after each run.

Generated summary tables:

| File | Purpose |
| --- | --- |
| `summaries/bridge_method_summary.csv` | Overall method x bridge comparison. |
| `summaries/method_relation_summary.csv` | Within-family vs cross-family comparison. |
| `summaries/family_pair_summary.csv` | Source-family to target-family bridge behavior. |
| `summaries/depth_pair_summary.csv` | Source-depth to target-depth bridge behavior. |
| `summaries/set_size_summary.csv` | How bridge signal changes with semantic set size. |
| `summaries/family_depth_pair_summary.csv` | Family-pair and depth-pair interaction. |

Generated figures:

| Figure | Purpose |
| --- | --- |
| `figures/summary_method_bridge_adjusted.png` | Fastest view of SetConCA vs pointwise TopK by bridge. |
| `figures/summary_relation_bridge_adjusted.png` | Within-family vs cross-family controlled transfer. |
| `figures/summary_set_size_adjusted.png` | Set-size scaling using the controlled bridge metric. |
| `figures/summary_depth_heatmap_setconca_ridge.png` | Best first-pass depth-pair map for SetConCA with a linear ridge bridge. |
| `figures/summary_family_heatmap_setconca_ridge.png` | Best first-pass family-pair map for SetConCA with a linear ridge bridge. |

## Main Metric

The primary first-pass bridge score remains:

```text
real_minus_shuffled_topk = raw_topk - shuffled_topk
```

Reason: raw TopK overlap can be inflated when unrelated examples still activate similar high-frequency dimensions. The shuffled-anchor control estimates this background overlap. The adjusted score is the part of the bridge overlap that is more specific to matched semantic anchors.

## Reproducibility

The completed run was regenerated with `--resume`, so trained model artifacts were reused and only the reports, summaries, and figures were refreshed.

```powershell
python scripts\run_transfer_steering_grid.py `
  --activation-root data\activations\model_grid_s16_min16_4A100 `
  --out-dir results\llama_qwen_set_vs_pointwise_linear_seed0 `
  --only-family llama3,qwen3 `
  --set-sizes 2,4,6,8,10,12,14,16 `
  --max-sets 300 `
  --epochs 25 `
  --batch-size 128 `
  --steering-alphas 0 `
  --bridges identity,procrustes,ridge `
  --methods setconca,pointwise_topk `
  --device cuda `
  --seed 0 `
  --resume
```

Observed regenerated run summary:

```json
{
  "elapsed_s": 90.10924363136292,
  "n_banks": 18,
  "n_trained": 288,
  "n_result_rows": 14688
}
```

## Verification

The code passed the current test suite:

```text
14 passed
```

There was one pytest cache warning from the sandbox path, but it did not affect the tests.

## Figure Cleanup

After inspecting the first generated figures, the initial charts were not clean enough for reporting:

- long internal labels such as `pointwise_topk:cross_family:ridge` collided on the x-axis,
- some figures exposed raw metric column names instead of readable labels,
- the most useful message was hidden by identity-control bars near zero,
- subtitles collided with titles in the saved PNGs.

The plotting code was updated to make the visual story clearer:

- use readable labels such as `SetConCA`, `Pointwise TopK`, `Ridge`, and `Procrustes`,
- group bars by bridge or relation instead of using long one-piece labels,
- annotate bars with exact values,
- omit identity from the high-level method/set-size figures because it stays near zero and is not the useful linear bridge,
- keep identity behavior available in the CSV summaries and detailed result files,
- rename heatmap colorbars to `Controlled TopK overlap`,
- use family labels such as `Llama 3` and `Qwen 3` in heatmaps.

The figures were regenerated from the existing `transfer_steering_results.json`; no training or model evaluation was rerun for this visualization cleanup.

Regeneration command:

```powershell
python -c "import json; from pathlib import Path; import scripts.run_transfer_steering_grid as r; run_dir=Path('results/llama_qwen_set_vs_pointwise_linear_seed0'); rows=json.loads((run_dir/'transfer_steering_results.json').read_text(encoding='utf-8')); summaries=r.write_summary_artifacts(run_dir, rows); r.plot_summary_artifacts(run_dir, summaries); print('regenerated', len(rows), 'rows')"
```

Observed output:

```text
regenerated 14688 rows
```

## What The New Summaries Show

The automatic summaries preserve the previous interpretation:

- SetConCA has lower raw overlap than pointwise TopK.
- Pointwise TopK keeps much of its raw overlap under shuffled anchors.
- SetConCA has stronger shuffled-controlled bridge evidence.
- Ridge and Procrustes are the strongest bridge families for the current linear-transfer question.
- Within-family transfer is stronger than cross-family transfer, but cross-family Llama/Qwen transfer remains meaningfully above the shuffled control.
- Mid-depth target layers, especially `60%`, are currently more promising than target `90%` for the next causal-steering preparation.

Representative controlled scores from `method_relation_summary.csv`:

| Method | Relation | Bridge | Real - shuffled |
| --- | --- | --- | ---: |
| `setconca` | `cross_family` | `ridge` | 0.1391 |
| `setconca` | `within_family` | `ridge` | 0.1519 |
| `pointwise_topk` | `cross_family` | `ridge` | 0.0530 |
| `pointwise_topk` | `within_family` | `ridge` | 0.0652 |

## Why This Matters For The Project Goal

The original ConCA motivation is a linear, log-posterior concept representation. The current result does not yet prove monosemantic transfer in the strong behavioral sense, but it gives us a clearer candidate set for that test:

- use SetConCA rather than pointwise TopK as the main representation,
- keep linear bridges first (`ridge`, `procrustes`),
- prioritize Llama/Qwen cross-family directions,
- start with the stronger depth pairs before testing behavioral causal steering.

This keeps the next step close to the original theory instead of jumping to a more expressive MLP bridge too early.

## Next Step

Move from aggregate transfer evidence to real concept inspection:

1. Choose candidate trained SetConCA banks from the strongest summary cells.
2. Extract top concept dimensions for matched semantic anchors.
3. Decode or inspect the source and target examples that activate each concept.
4. Select a small set of honest candidate concepts.
5. Only then run causal steering with explicit before/after behavioral checks.

Recommended first candidates:

- `qwen3 -> llama3`
- `llama3 -> qwen3`
- `llama3 -> llama3`
- SetConCA, `S=16`
- ridge/procrustes bridges
- target depth around `60%`
