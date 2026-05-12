# 2026-05-08 Bridge Diagnostics

## Goal

Add diagnostic controls before running the full all-family grid.

The first family pilots showed promising linear bridge behavior, but Gemma 3 produced near-perfect bridge overlap while also showing very large reconstruction-loss scale. That combination needs an audit before we treat the result as scientific evidence.

## Diagnostic Questions

1. Are the learned TopK supports degenerate?
2. Does a bridge perform similarly on train and test, or does it overfit?
3. Does a bridge still look strong if target anchors are shuffled?
4. Are reconstruction losses large only because activation norms are large?
5. Do the diagnostics preserve the original ConCA/Set-ConCA goal of linear concept alignability?

## Planned Additions

Add diagnostics to:

```text
scripts/run_transfer_steering_grid.py
```

without changing the SetConCA architecture.

### Per-model code diagnostics

For every trained model:

- mean number of active concepts,
- number of concepts ever active,
- fraction of concepts ever active,
- mean active frequency,
- max active frequency,
- support entropy,
- mean code norm,
- mean dense-code norm.

These help detect collapse such as the same concepts being active for almost every example.

### Per-pair bridge controls

For every source-target pair and bridge:

- train TopK overlap,
- test TopK overlap,
- train-test gap,
- shuffled-target test overlap,
- real-minus-shuffled overlap.

These help distinguish real aligned anchors from a bridge that creates generally similar supports regardless of pairing.

### Normalized reconstruction losses

Add input energy and normalized reconstruction terms during model evaluation:

```text
shared_recon_norm = shared_recon / input_energy
full_recon_norm = full_recon / input_energy
```

This is important for comparing layers/families whose raw activation scales differ.

## Implementation

Implemented in:

```text
scripts/run_transfer_steering_grid.py
```

No SetConCA architecture change was made.

The script now records:

- `train_topk_overlap`
- `topk_overlap`
- `train_test_topk_gap`
- `shuffled_topk_overlap`
- `real_minus_shuffled_topk`
- normalized reconstruction terms
- `code_diagnostics.csv`

The generated run report now treats `real_minus_shuffled_topk` as the main first-pass controlled bridge score. Raw TopK overlap remains useful, but it is no longer enough by itself.

New figures added for future runs:

- `figures/bridge_adjusted_topk_overlap.png`
- `figures/set_size_adjusted_topk_overlap.png`

## Diagnostic Runs

Small diagnostic slices were run for the three families:

```text
results/diagnostic_gemma3_small_slice_seed0
results/diagnostic_llama3_small_slice_seed0
results/diagnostic_qwen3_small_slice_seed0
```

Shared settings:

- small model only,
- three layer banks,
- set sizes `2,16`,
- `max_sets=120`,
- `epochs=3`,
- bridges `identity,procrustes,ridge,mlp`,
- CPU diagnostic run,
- seed `0`.

## Results

### Gemma 3 small slice

| Bridge | Raw TopK | Shuffled TopK | Real - shuffled | Train-test gap |
| --- | ---: | ---: | ---: | ---: |
| `identity` | 0.2333 | 0.2298 | 0.0035 | 0.0006 |
| `mlp` | 0.9057 | 0.8915 | 0.0142 | 0.0282 |
| `procrustes` | 0.9198 | 0.8611 | 0.0587 | 0.0430 |
| `ridge` | 0.9367 | 0.8670 | 0.0698 | 0.0404 |

Code/reconstruction diagnostics:

- average fraction of concepts ever active: `0.4401`
- average max active frequency: `1.0000`
- average support entropy: `0.7558`
- average input energy: `238730.9261`
- average shared reconstruction normalized loss: `0.9985`
- average full reconstruction normalized loss: `0.9560`

Interpretation: the raw bridge scores are very high, but most of that score survives shuffled target anchors. This is a warning that the bridge may be aligning broad support patterns rather than anchor-specific semantic concepts.

### Llama 3 small slice

| Bridge | Raw TopK | Shuffled TopK | Real - shuffled | Train-test gap |
| --- | ---: | ---: | ---: | ---: |
| `identity` | 0.2626 | 0.2553 | 0.0073 | -0.0130 |
| `mlp` | 0.6258 | 0.5867 | 0.0391 | 0.0489 |
| `procrustes` | 0.5757 | 0.4728 | 0.1030 | 0.2914 |
| `ridge` | 0.5605 | 0.4838 | 0.0767 | 0.4359 |

Code/reconstruction diagnostics:

- average fraction of concepts ever active: `0.8698`
- average max active frequency: `0.8389`
- average support entropy: `0.9009`
- average input energy: `0.0467`
- average shared reconstruction normalized loss: `4.7884`
- average full reconstruction normalized loss: `4.8402`

Interpretation: Llama gives the clearest controlled bridge signal in this small slice, especially for Procrustes. However, train-test gaps are large for the linear bridges, so the signal must be checked with more data and stronger train/test reporting.

### Qwen 3 small slice

| Bridge | Raw TopK | Shuffled TopK | Real - shuffled | Train-test gap |
| --- | ---: | ---: | ---: | ---: |
| `identity` | 0.2628 | 0.2614 | 0.0014 | 0.0023 |
| `mlp` | 0.8484 | 0.8467 | 0.0017 | 0.0057 |
| `procrustes` | 0.7839 | 0.7629 | 0.0209 | 0.0759 |
| `ridge` | 0.8503 | 0.8169 | 0.0333 | 0.0997 |

Code/reconstruction diagnostics:

- average fraction of concepts ever active: `0.5742`
- average max active frequency: `0.9944`
- average support entropy: `0.7839`
- average input energy: `68.6568`
- average shared reconstruction normalized loss: `0.8710`
- average full reconstruction normalized loss: `0.8427`

Interpretation: Qwen has strong raw bridge overlap, but the shuffled control is also strong. The controlled signal is small in this diagnostic slice.

## Conclusion

The next step should not be a blind full-scale bridge claim. The pipeline works, but the metric audit changed the interpretation:

- Raw TopK overlap is optimistic.
- Shuffled-anchor controls are necessary.
- Linear bridges remain central because they are the ConCA-motivated test, but they must be judged by controlled overlap, not raw overlap alone.
- Gemma and Qwen need caution because their raw overlap is heavily explained by shuffled controls in the small diagnostic slice.
- Llama is the most promising diagnostic slice, but train-test bridge gaps need more careful reporting.

## Status

Status: `Completed for first diagnostic pass`

Next recommended step: rerun the family pilots or a larger subset with the updated report so `real_minus_shuffled_topk` and train-test gap are visible in every run report.
