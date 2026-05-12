# 2026-05-08 Layer-Pair Breakdown

## Goal

Check whether the Llama/Qwen bridge runs covered different source-target depth combinations:

```text
20% -> 20%
20% -> 60%
20% -> 90%
60% -> 20%
60% -> 60%
60% -> 90%
90% -> 20%
90% -> 60%
90% -> 90%
```

## Run Checked

```text
results/llama_qwen_set_vs_pointwise_linear_seed0
```

The result table includes source and target keys with depth labels:

```text
layer_..._20pct
layer_..._60pct
layer_..._90pct
```

Therefore the run did cover all start/mid/late combinations.

## SetConCA Depth-Pair Results

Metric:

```text
real_minus_shuffled_topk
```

| Source depth | Target depth | Procrustes | Ridge |
| ---: | ---: | ---: | ---: |
| 20 | 20 | 0.1644 | 0.1541 |
| 20 | 60 | 0.1477 | 0.1452 |
| 20 | 90 | 0.0868 | 0.0951 |
| 60 | 20 | 0.1632 | 0.1555 |
| 60 | 60 | 0.1743 | 0.1696 |
| 60 | 90 | 0.1098 | 0.1179 |
| 90 | 20 | 0.1152 | 0.1513 |
| 90 | 60 | 0.1386 | 0.1744 |
| 90 | 90 | 0.1074 | 0.1467 |

## Pointwise TopK Comparison

| Source depth | Target depth | Procrustes | Ridge |
| ---: | ---: | ---: | ---: |
| 20 | 20 | 0.0361 | 0.0564 |
| 20 | 60 | 0.0499 | 0.0624 |
| 20 | 90 | 0.0235 | 0.0314 |
| 60 | 20 | 0.0451 | 0.0596 |
| 60 | 60 | 0.0785 | 0.0838 |
| 60 | 90 | 0.0421 | 0.0479 |
| 90 | 20 | 0.0357 | 0.0521 |
| 90 | 60 | 0.0689 | 0.0844 |
| 90 | 90 | 0.0410 | 0.0526 |

SetConCA is stronger than pointwise TopK in every depth-pair bucket under the shuffled-controlled metric.

## Interpretation

The strongest overall SetConCA depth pair is:

```text
60% -> 60%
```

with Procrustes `0.1743` and ridge `0.1696`.

Ridge also performs strongly for:

```text
90% -> 60%: 0.1744
```

Transfers into 90% target layers are generally weaker, especially:

```text
20% -> 90%
60% -> 90%
```

This suggests the latest layers may contain more family/model-specific or output-adjacent structure, making sparse concept support less directly bridgeable from earlier/mid layers.

## Family-Specific Notes

Llama 3 within-family transfer is strong across depth, especially:

- `90% -> 90%`
- `60% -> 60%`
- `60% -> 90%`
- `90% -> 60%`

Qwen 3 to Llama 3 is the strongest cross-family direction. Ridge is especially strong for:

- `90% -> 90%`: `0.2290`
- `90% -> 60%`: `0.2170`
- `90% -> 20%`: `0.1827`

Llama 3 to Qwen 3 becomes weak when the target is Qwen 90%, especially for:

- `20% -> 90%`
- `60% -> 90%`
- `90% -> 90%`

Qwen within-family also weakens when targeting Qwen 90%.

## Status

Status: `Succeeded`

Next recommended action: add automatic depth-pair summary CSV/figures to `scripts/run_transfer_steering_grid.py`, because this breakdown is too important to leave only in ad hoc shell summaries.
