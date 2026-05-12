# 2026-05-08 SetConCA Vs Pointwise TopK Baseline Results

## Goal

Evaluate whether SetConCA's set-level training adds value beyond a pointwise sparse TopK encoder whose codes are mean-pooled only after training.

This directly tests an important V1 concern:

```text
Is the bridgeability coming from SetConCA's set objective, or would pointwise TopK codes plus post-hoc pooling do the same thing?
```

## Run

```text
results/llama_qwen_set_vs_pointwise_linear_seed0
```

Run summary:

```json
{
  "n_banks": 18,
  "n_trained": 288,
  "n_result_rows": 14688
}
```

Scope:

- families: Llama 3 and Qwen 3,
- Gemma 3 excluded,
- set sizes: `2,4,6,8,10,12,14,16`,
- max sets: `300`,
- epochs: `25`,
- batch size: `128`,
- methods: `setconca`, `pointwise_topk`,
- bridges: `identity`, `procrustes`, `ridge`,
- steering alpha: `0`.

## Main Bridge Results

Values are averaged over all Llama/Qwen banks and set sizes at `alpha=0`.

| Method | Bridge | Raw TopK | Shuffled TopK | Real - shuffled | Train-test gap |
| --- | --- | ---: | ---: | ---: | ---: |
| `pointwise_topk` | `identity` | 0.2506 | 0.2506 | -0.0000 | -0.0003 |
| `pointwise_topk` | `procrustes` | 0.8251 | 0.7787 | 0.0465 | 0.0400 |
| `pointwise_topk` | `ridge` | 0.8693 | 0.8106 | 0.0586 | 0.0542 |
| `setconca` | `identity` | 0.2492 | 0.2495 | -0.0003 | 0.0006 |
| `setconca` | `procrustes` | 0.5432 | 0.4099 | 0.1333 | 0.1056 |
| `setconca` | `ridge` | 0.6107 | 0.4658 | 0.1449 | 0.2185 |

## Interpretation

Pointwise TopK has very high raw overlap, but it also has very high shuffled-anchor overlap. This means much of its apparent bridge success is not anchor-specific.

SetConCA has lower raw overlap, but much better shuffled-controlled overlap:

```text
Procrustes: 0.1333 vs 0.0465
Ridge:      0.1449 vs 0.0586
```

This is strong evidence that SetConCA's set-level training is doing something useful for anchor-specific concept transfer.

## Within-Family Vs Cross-Family

| Method | Relation | Bridge | Real - shuffled |
| --- | --- | --- | ---: |
| `pointwise_topk` | cross | `procrustes` | 0.0394 |
| `pointwise_topk` | cross | `ridge` | 0.0529 |
| `pointwise_topk` | within | `procrustes` | 0.0544 |
| `pointwise_topk` | within | `ridge` | 0.0651 |
| `setconca` | cross | `procrustes` | 0.1214 |
| `setconca` | cross | `ridge` | 0.1390 |
| `setconca` | within | `procrustes` | 0.1466 |
| `setconca` | within | `ridge` | 0.1514 |

SetConCA beats pointwise TopK for both within-family and cross-family bridge transfer under the controlled metric.

## Directed Family-Pair Results

| Method | Source | Target | Bridge | Real - shuffled |
| --- | --- | --- | --- | ---: |
| `pointwise_topk` | llama3 | llama3 | `procrustes` | 0.0892 |
| `pointwise_topk` | llama3 | llama3 | `ridge` | 0.1072 |
| `pointwise_topk` | llama3 | qwen3 | `procrustes` | 0.0190 |
| `pointwise_topk` | llama3 | qwen3 | `ridge` | 0.0220 |
| `pointwise_topk` | qwen3 | llama3 | `procrustes` | 0.0597 |
| `pointwise_topk` | qwen3 | llama3 | `ridge` | 0.0838 |
| `pointwise_topk` | qwen3 | qwen3 | `procrustes` | 0.0196 |
| `pointwise_topk` | qwen3 | qwen3 | `ridge` | 0.0229 |
| `setconca` | llama3 | llama3 | `procrustes` | 0.1876 |
| `setconca` | llama3 | llama3 | `ridge` | 0.1784 |
| `setconca` | llama3 | qwen3 | `procrustes` | 0.0906 |
| `setconca` | llama3 | qwen3 | `ridge` | 0.1096 |
| `setconca` | qwen3 | llama3 | `procrustes` | 0.1522 |
| `setconca` | qwen3 | llama3 | `ridge` | 0.1685 |
| `setconca` | qwen3 | qwen3 | `procrustes` | 0.1057 |
| `setconca` | qwen3 | qwen3 | `ridge` | 0.1245 |

The strongest cross-family direction remains:

```text
qwen3 -> llama3
```

## Set-Size Pattern

SetConCA improves strongly with set size:

| Set size | SetConCA Procrustes | SetConCA Ridge |
| ---: | ---: | ---: |
| 2 | 0.0974 | 0.1004 |
| 4 | 0.1138 | 0.1184 |
| 6 | 0.1244 | 0.1343 |
| 8 | 0.1349 | 0.1472 |
| 10 | 0.1426 | 0.1551 |
| 12 | 0.1446 | 0.1618 |
| 14 | 0.1518 | 0.1673 |
| 16 | 0.1568 | 0.1745 |

Pointwise TopK is much flatter and weaker under the controlled metric:

| Set size | Pointwise Procrustes | Pointwise Ridge |
| ---: | ---: | ---: |
| 2 | 0.0458 | 0.0533 |
| 4 | 0.0376 | 0.0449 |
| 6 | 0.0429 | 0.0534 |
| 8 | 0.0453 | 0.0578 |
| 10 | 0.0471 | 0.0612 |
| 12 | 0.0481 | 0.0631 |
| 14 | 0.0506 | 0.0662 |
| 16 | 0.0542 | 0.0694 |

This is exactly the expected pattern if set-level training is improving the anchor-specific semantic signal.

## Reconstruction Caveat

Pointwise TopK has lower pointwise reconstruction-normalized loss:

```text
pointwise_topk point_recon_norm: 0.3618
```

SetConCA has higher shared reconstruction-normalized loss:

```text
setconca shared_recon_norm: 1.1886
```

This is not a contradiction. The pointwise baseline is optimized for individual-view reconstruction. SetConCA is optimized to separate shared set-level concept structure from residual view-specific detail. For the bridge objective, the controlled transfer metric is the more relevant result.

## Scientific Takeaway

This is one of the strongest V2 results so far.

It supports the claim that:

```text
Set-level training learns sparse codes with stronger anchor-specific linear bridgeability than pointwise TopK codes pooled after training.
```

This does not yet prove monosemanticity or behavioral causal steering, but it directly supports the central SetConCA design choice.

## Status

Status: `Succeeded`

Next recommended step: run interpretability and steering checks on the strongest directions:

- qwen3 -> llama3,
- llama3 -> qwen3,
- llama3 -> llama3.
