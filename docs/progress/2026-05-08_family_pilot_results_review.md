# 2026-05-08 Family Pilot Results Review

## Goal

Review the first family pilot results:

```text
results/pilot_gemma3_all_sizes_seed0
results/pilot_llama3_all_sizes_seed0
results/pilot_qwen3_all_sizes_seed0
```

These pilots test within-family SetConCA V2 transfer across all available sizes and layers for each family.

## Run Scope

Each family pilot used:

```text
activation_root: data/activations/model_grid_s16_min16_4A100
set_sizes: 2,4,6,8,10,12,14,16
max_sets: 300
epochs: 10
batch_size: 32
concept_dim: 128
topk: 32
bridges: identity,procrustes,ridge,mlp
steering_alphas: 0,0.5,1,2,5
seed: 0
include_self_pairs: false
```

Each run selected 9 activation banks and trained:

```text
9 banks x 8 set sizes = 72 SetConCA models
```

Each run wrote:

```text
11520 transfer/bridge/steering rows
```

## Family-Level Bridge Summary

Transfer metric: mean TopK overlap at `steering_alpha = 0`.

| Family | Identity | Procrustes | Ridge | MLP |
| --- | ---: | ---: | ---: | ---: |
| Gemma 3 | 0.2477 | 0.9519 | 0.9650 | 0.9645 |
| Llama 3 | 0.2487 | 0.5806 | 0.5842 | 0.6642 |
| Qwen 3 | 0.2493 | 0.7279 | 0.8055 | 0.8127 |

## Family-Level Steering Proxy Summary

Metric: mean structured direction similarity minus random direction similarity.

| Family | alpha 0.5 | alpha 1 | alpha 2 | alpha 5 |
| --- | ---: | ---: | ---: | ---: |
| Gemma 3 | 0.0137 | 0.0231 | 0.0325 | 0.0411 |
| Llama 3 | 0.1774 | 0.2942 | 0.4101 | 0.5077 |
| Qwen 3 | 0.0777 | 0.1293 | 0.1809 | 0.2263 |

## Set-Size Pattern

### Llama 3

Llama shows the cleanest set-size scaling in bridge transfer. Procrustes and ridge improve as set size increases:

| S | Procrustes | Ridge | MLP |
| ---: | ---: | ---: | ---: |
| 2 | 0.4909 | 0.4957 | 0.5625 |
| 4 | 0.5448 | 0.5474 | 0.6241 |
| 6 | 0.5696 | 0.5709 | 0.6537 |
| 8 | 0.5898 | 0.5921 | 0.6744 |
| 10 | 0.5908 | 0.5952 | 0.6792 |
| 12 | 0.6090 | 0.6124 | 0.6961 |
| 14 | 0.6167 | 0.6229 | 0.7053 |
| 16 | 0.6334 | 0.6372 | 0.7183 |

This is promising for the set-size hypothesis.

### Qwen 3

Qwen also improves with larger sets, though it starts higher and saturates earlier:

| S | Procrustes | Ridge | MLP |
| ---: | ---: | ---: | ---: |
| 2 | 0.6805 | 0.7633 | 0.7787 |
| 4 | 0.7044 | 0.7821 | 0.7926 |
| 6 | 0.7253 | 0.7971 | 0.8076 |
| 8 | 0.7279 | 0.8070 | 0.8139 |
| 10 | 0.7313 | 0.8132 | 0.8173 |
| 12 | 0.7468 | 0.8208 | 0.8247 |
| 14 | 0.7508 | 0.8290 | 0.8326 |
| 16 | 0.7559 | 0.8319 | 0.8337 |

### Gemma 3

Gemma is suspiciously high across all set sizes:

| S | Procrustes | Ridge | MLP |
| ---: | ---: | ---: | ---: |
| 2 | 0.9556 | 0.9668 | 0.9673 |
| 4 | 0.9385 | 0.9560 | 0.9559 |
| 6 | 0.9388 | 0.9554 | 0.9556 |
| 8 | 0.9501 | 0.9632 | 0.9628 |
| 10 | 0.9527 | 0.9667 | 0.9658 |
| 12 | 0.9561 | 0.9686 | 0.9669 |
| 14 | 0.9639 | 0.9732 | 0.9729 |
| 16 | 0.9592 | 0.9701 | 0.9688 |

This may be real, but it is high enough that it needs an audit before being treated as evidence.

## Training-Loss Caveat

Gemma reconstruction losses were extremely large, especially deeper and larger models. Example late-layer losses reached millions in `training_summary.csv`.

This does not automatically invalidate transfer overlap, but it means:

- Gemma results need normalization/activation-scale audit.
- Bridge overlap may be dominated by code support patterns rather than good reconstruction.
- We should not report Gemma as a major win until the diagnostic run confirms the codes are meaningful and not degenerate.

Qwen also has large late-layer losses for some banks, but not at Gemma's scale. Llama losses are much better behaved and show the cleanest reconstruction improvement with larger set size.

## Initial Interpretation

Status: `Preliminary`

The first pilots support three early observations:

1. Identity stays near 0.25, matching the TopK chance reference `k / C = 32 / 128`. This is good: unbridged concept spaces are not trivially aligned.
2. Linear bridges matter. Procrustes and ridge substantially improve transfer for all families.
3. Set size appears to help most clearly for Llama 3 and Qwen 3.

However, Gemma requires audit before claims:

- bridge overlap is near-perfect,
- steering proxy is weak,
- reconstruction losses are huge.

This combination is a warning sign.

## Artifact Note

The Gemma manifest records `out_dir` as:

```text
results/pilot_gemma_all_sizes_seed0
```

while the inspected folder is:

```text
results/pilot_gemma3_all_sizes_seed0
```

The result files are present in the inspected folder, but this path mismatch should be cleaned up in future runs to avoid provenance confusion.

## Recommended Next Step

Before the full all-family grid, run an audit/diagnostic pass:

1. Add or run diagnostics for code degeneracy:
   - active support frequency per concept,
   - mean number of unique active supports,
   - train vs test bridge gap,
   - random-label bridge control,
   - shuffled-anchor bridge control,
   - reconstruction-loss normalization by input activation norm.
2. Rerun a focused Gemma diagnostic on one small slice.
3. If Gemma remains valid after controls, proceed to full grid.
4. If Gemma fails controls, document the failure and fix evaluation/training before scaling.

## Status

Status: `Succeeded`

The first result folders were inspected and summarized. The results are promising for linear bridging and set-size scaling, but not yet final scientific evidence.
