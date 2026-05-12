# 2026-05-08 Cross-Family Pair Verification

## Goal

Verify whether the combined all-family transfer run actually covered cross-model cases.

The question was not only whether the script finished, but whether the result table includes source-target pairs across:

- different families,
- different model sizes,
- different model/layer banks.

## Run Checked

```text
results/pilot_all_families_cross_family_seed0_diagnostic
```

Run summary:

```json
{
  "n_banks": 27,
  "n_trained": 216,
  "n_result_rows": 112320
}
```

This means:

```text
27 activation banks x 8 set sizes = 216 SetConCA models
```

## Coverage Check

Rows at `steering_alpha=0`:

| Coverage type | Count |
| --- | ---: |
| total alpha-0 rows | 22,464 |
| within-family rows | 6,912 |
| cross-family rows | 15,552 |
| same-size rows | 3,456 |
| cross-size rows | 19,008 |

Families present as both source and target:

```text
gemma3, llama3, qwen3
```

## Family Pair Counts

| Source family | Target family | Rows |
| --- | --- | ---: |
| gemma3 | gemma3 | 2,304 |
| gemma3 | llama3 | 2,592 |
| gemma3 | qwen3 | 2,592 |
| llama3 | gemma3 | 2,592 |
| llama3 | llama3 | 2,304 |
| llama3 | qwen3 | 2,592 |
| qwen3 | gemma3 | 2,592 |
| qwen3 | llama3 | 2,592 |
| qwen3 | qwen3 | 2,304 |

Conclusion: yes, cross-family model transfer was covered.

## Controlled Bridge Summary

Values below are averaged at `steering_alpha=0`.

| Relation | Bridge | Raw TopK | Shuffled TopK | Real - shuffled | Train-test gap |
| --- | --- | ---: | ---: | ---: | ---: |
| cross-family | `identity` | 0.2494 | 0.2495 | -0.0001 | 0.0000 |
| cross-family | `mlp` | 0.6934 | 0.6347 | 0.0586 | 0.0721 |
| cross-family | `procrustes` | 0.5365 | 0.4941 | 0.0424 | 0.0482 |
| cross-family | `ridge` | 0.6745 | 0.6226 | 0.0519 | 0.1096 |
| within-family | `identity` | 0.2502 | 0.2500 | 0.0002 | -0.0001 |
| within-family | `mlp` | 0.7670 | 0.6335 | 0.1335 | 0.1104 |
| within-family | `procrustes` | 0.7012 | 0.5933 | 0.1079 | 0.0731 |
| within-family | `ridge` | 0.7343 | 0.6223 | 0.1119 | 0.1491 |

Within-family transfer is stronger than cross-family transfer under the shuffled-controlled metric.

## Directed Family-Pair Linear Bridge Results

| Source | Target | Bridge | Real - shuffled |
| --- | --- | --- | ---: |
| gemma3 | gemma3 | `procrustes` | 0.0296 |
| gemma3 | gemma3 | `ridge` | 0.0321 |
| gemma3 | llama3 | `procrustes` | 0.0053 |
| gemma3 | llama3 | `ridge` | 0.0155 |
| gemma3 | qwen3 | `procrustes` | 0.0043 |
| gemma3 | qwen3 | `ridge` | 0.0105 |
| llama3 | gemma3 | `procrustes` | 0.0009 |
| llama3 | gemma3 | `ridge` | 0.0041 |
| llama3 | llama3 | `procrustes` | 0.1874 |
| llama3 | llama3 | `ridge` | 0.1787 |
| llama3 | qwen3 | `procrustes` | 0.0921 |
| llama3 | qwen3 | `ridge` | 0.1104 |
| qwen3 | gemma3 | `procrustes` | 0.0015 |
| qwen3 | gemma3 | `ridge` | 0.0032 |
| qwen3 | llama3 | `procrustes` | 0.1504 |
| qwen3 | llama3 | `ridge` | 0.1677 |
| qwen3 | qwen3 | `procrustes` | 0.1067 |
| qwen3 | qwen3 | `ridge` | 0.1250 |

## Interpretation

The run does cover cross models.

The strongest controlled linear signals are:

- Llama 3 within-family,
- Qwen 3 within-family,
- Qwen 3 to Llama 3,
- Llama 3 to Qwen 3.

Pairs involving Gemma 3 have very weak controlled transfer even when raw overlap is high. This is consistent with previous diagnostics showing Gemma 3 has high shuffled-anchor overlap.

## Status

Status: `Succeeded`

Next recommended step: add an explicit cross-family summary artifact to the script so future all-family runs automatically produce these relation and directed-pair tables.
