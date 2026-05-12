# Epoch Sweep Comparison: 25 vs 50 vs 100

## Goal

Compare the completed Llama/Qwen transfer runs trained for different epoch counts and decide whether longer training improves the scientific metric we care about.

The target metric is not training loss alone. The main first-pass transfer metric is:

```text
real_minus_shuffled_topk = raw_topk - shuffled_topk
```

This is the controlled bridge score: how much more top-k concept overlap we get for matched semantic anchors than for shuffled anchors.

## Compared Runs

| Label | Result directory | Trained models | Result rows |
| --- | --- | ---: | ---: |
| `e25` | `results/llama_qwen_set_vs_pointwise_linear_seed0` | 288 | 14688 |
| `e50` | `results/llama_qwen_set_vs_pointwise_linear_seed0_e50` | 288 | 14688 |
| `e100` | `results/llama_qwen_set_vs_pointwise_linear_seed0_e100` | 288 | 14688 |

## New Comparison Tool

Added:

```text
scripts/compare_transfer_runs.py
```

This script compares already-completed transfer runs. It does not train models. It reads each run's existing `summaries/*.csv` files and writes:

- `bridge_epoch_comparison.csv`
- `set_size_epoch_comparison.csv`
- `REPORT.md`
- `figures/epoch_bridge_controlled.png`
- `figures/epoch_setconca_ridge_set_size.png`

Command used:

```powershell
python scripts\compare_transfer_runs.py `
  --run e25=results\llama_qwen_set_vs_pointwise_linear_seed0 `
  --run e50=results\llama_qwen_set_vs_pointwise_linear_seed0_e50 `
  --run e100=results\llama_qwen_set_vs_pointwise_linear_seed0_e100 `
  --out-dir results\llama_qwen_epoch_comparison
```

Output:

```text
Wrote epoch comparison to results\llama_qwen_epoch_comparison
```

## Main Result

| Run | SetConCA + Procrustes | SetConCA + Ridge | Pointwise TopK + Procrustes | Pointwise TopK + Ridge |
| --- | ---: | ---: | ---: | ---: |
| `e25` | 0.1335 | 0.1452 | 0.0466 | 0.0587 |
| `e50` | 0.1303 | 0.1416 | 0.0591 | 0.0738 |
| `e100` | 0.1263 | 0.1368 | 0.0732 | 0.0886 |

## Interpretation

Longer training did not improve SetConCA's controlled linear bridge signal in this sweep.

SetConCA:

- best at `25` epochs for both Procrustes and ridge,
- slowly decreases at `50` and `100` epochs,
- still remains stronger than pointwise TopK at all tested epochs.

Pointwise TopK:

- improves from `25` to `50` to `100` epochs,
- remains below SetConCA after shuffled-anchor correction,
- may need its own hyperparameter tuning if we want a maximally strong baseline.

## Why This Matters

This is a useful result for the original SetConCA goal. It shows that more reconstruction optimization does not necessarily produce more transferable concept coordinates.

For the next concept extraction and causal steering phase, the current best candidate remains:

- SetConCA,
- `25` epochs,
- `S=16`,
- ridge and Procrustes bridges,
- Llama/Qwen active families,
- target depth around `60%`.

## Artifacts

Result directory:

```text
results/llama_qwen_epoch_comparison
```

Key figures:

- `figures/epoch_bridge_controlled.png`
- `figures/epoch_setconca_ridge_set_size.png`

## Verification

The comparison script compiled successfully:

```text
python -m py_compile scripts\compare_transfer_runs.py
```
