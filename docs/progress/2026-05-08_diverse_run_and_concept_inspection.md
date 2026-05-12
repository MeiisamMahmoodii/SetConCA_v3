# Diverse Run And Concept Inspection

## Goal

Run the Llama/Qwen SetConCA-vs-pointwise bridge experiment on the balanced 300-set slice and inspect the strongest bridged concept candidates before causal steering.

This step exists because the earlier first-300 experiment was label-skewed. The balanced slice should give us a better chance of finding steering candidates beyond business/technology.

## Inputs

Dataset:

```text
data/generated/server_4gpu_2000/diverse_s16_300/sets_min16_diverse300.jsonl
```

Activation banks:

```text
data/activations/model_grid_s16_min16_diverse300_4A100
```

Transfer run:

```text
results/llama_qwen_diverse300_set_vs_pointwise_linear_seed0
```

Concept inspection:

```text
results/concept_inspection_llama_qwen_diverse300_e25
```

## What Ran

The completed transfer run used:

- Llama 3 and Qwen 3 only,
- set sizes `2,4,6,8,10,12,14,16`,
- 25 epochs,
- batch size 128,
- bridges `identity`, `procrustes`, and `ridge`,
- methods `setconca` and `pointwise_topk`,
- steering alpha `0` for bridge evaluation only.

The run completed:

| Metric | Value |
| --- | ---: |
| Activation banks | 18 |
| Trained models | 288 |
| Result rows | 14,688 |
| Runtime | 297.3 seconds |

## Main Result

The balanced run strengthened the controlled SetConCA bridge signal.

| Method | Bridge | Controlled score |
| --- | --- | ---: |
| SetConCA | ridge | 0.1827 |
| SetConCA | procrustes | 0.1674 |
| pointwise TopK | ridge | 0.0664 |
| pointwise TopK | procrustes | 0.0509 |

The controlled score is `real_minus_shuffled_topk`: raw TopK overlap minus shuffled-anchor overlap.

This is the current honest interpretation:

- SetConCA is still much stronger than pointwise TopK after shuffled controls.
- Ridge remains the best linear bridge.
- Cross-family transfer is weaker than within-family, but still above the shuffled control.
- The balanced slice improved the SetConCA score relative to the earlier skewed first-300 run.

## Relation Breakdown

| Method | Relation | Bridge | Controlled score |
| --- | --- | --- | ---: |
| SetConCA | within-family | ridge | 0.1901 |
| SetConCA | cross-family | ridge | 0.1760 |
| SetConCA | within-family | procrustes | 0.1816 |
| SetConCA | cross-family | procrustes | 0.1547 |

This supports continuing with cross-model linear bridges, but it does not yet prove monosemantic steering.

## Concept Review Outputs

The concept inspection produced:

- `concept_summary.csv`
- `concept_examples.csv`
- `REPORT.md`
- `review_pages/concept_review_table.md`
- `review_pages/first_pass_labels.md`
- `review_pages/first_pass_labels.csv`

The first-pass keep list is:

| Rank | Candidate | Label | Steering status |
| ---: | --- | --- | --- |
| 2 | Google IPO / stock offering | clean semantic | yes |
| 9 | Windows / software security updates | clean semantic | yes |
| 17 | stock market / earnings / prices | clean semantic | yes |
| 19 | Google IPO / stock offering | clean semantic | yes |
| 30 | corporate earnings / company performance | clean semantic | yes/maybe |

Broad sports/Olympics candidates were also found, but they are better treated as broad-topic controls for now, not as strong monosemantic proof.

## Caveat

The balanced dataset improved diversity and bridgeability, but the cleanest discovered concepts are still concentrated in business and technology. That means the next steering step should be modest:

1. steer only the clean concepts first,
2. include broad sports concepts as controls,
3. report failures and mixed concepts explicitly,
4. avoid claiming full monosemantic transfer until behavioral steering confirms the directions.

## Status

Succeeded.
