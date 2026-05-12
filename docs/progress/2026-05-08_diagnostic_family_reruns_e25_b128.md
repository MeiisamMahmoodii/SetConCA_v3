# 2026-05-08 Diagnostic Family Reruns, 25 Epochs, Batch 128

## Goal

Review the updated family pilot runs after adding shuffled-anchor bridge controls.

The scientific question is still the ConCA-motivated one: can a concept space learned from log-posterior-inspired SetConCA codes support a mostly linear bridge between models?

For this pass, the primary bridge metric is no longer raw TopK overlap. The main first-pass evidence is:

```text
real_minus_shuffled_topk = real TopK overlap - shuffled-anchor TopK overlap
```

This is necessary because raw bridge overlap can be high even when target examples are shuffled.

## Runs Reviewed

| Family | Result folder |
| --- | --- |
| Qwen 3 | `results/pilot_qwen3_all_sizes_seed0_diagnostic` |
| Llama 3 | `results/pilot_llama3_all_sizes_seed0_diagnostic` |
| Gemma 3 | `results/pilot_gemma3_all_sizes_seed0_diagnostic` |

Shared settings from the manifests:

- set sizes: `2,4,6,8,10,12,14,16`
- max sets: `300`
- epochs: `25`
- batch size: `128`
- seed: `0`
- MLP bridge epochs: `100`
- bridges: `identity, procrustes, ridge, mlp`

Each family run produced:

- 9 activation banks,
- 72 trained SetConCA models,
- 11,520 transfer/steering rows.

## Bridge Summary

Values below are averaged over `steering_alpha=0`.

| Family | Bridge | Raw TopK | Shuffled TopK | Real - shuffled | Train-test gap |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen 3 | `identity` | 0.2484 | 0.2483 | 0.0001 | -0.0001 |
| Qwen 3 | `mlp` | 0.7464 | 0.6164 | 0.1300 | 0.0775 |
| Qwen 3 | `procrustes` | 0.6410 | 0.5345 | 0.1065 | 0.0575 |
| Qwen 3 | `ridge` | 0.7266 | 0.6014 | 0.1252 | 0.1321 |
| Llama 3 | `identity` | 0.2500 | 0.2505 | -0.0005 | 0.0000 |
| Llama 3 | `mlp` | 0.5905 | 0.3504 | 0.2402 | 0.2580 |
| Llama 3 | `procrustes` | 0.5107 | 0.3229 | 0.1878 | 0.1588 |
| Llama 3 | `ridge` | 0.5130 | 0.3328 | 0.1802 | 0.3108 |
| Gemma 3 | `identity` | 0.2483 | 0.2483 | -0.0000 | 0.0007 |
| Gemma 3 | `mlp` | 0.9643 | 0.9353 | 0.0291 | -0.0045 |
| Gemma 3 | `procrustes` | 0.9520 | 0.9225 | 0.0294 | -0.0008 |
| Gemma 3 | `ridge` | 0.9646 | 0.9323 | 0.0323 | 0.0017 |

## Interpretation

Identity bridge is at chance for all three families. This is good because it shows the trained concept axes are not trivially identical across independently trained models.

Llama 3 has the strongest controlled transfer signal. Procrustes gives `0.1878` real-minus-shuffled TopK and ridge gives `0.1802`. This is the cleanest support so far for the linear bridge goal.

Qwen 3 gives a moderate controlled signal. Procrustes gives `0.1065` and ridge gives `0.1252`.

Gemma 3 still has extremely high raw TopK overlap, but almost all of it is present under shuffled target anchors. Its controlled signal is only about `0.03`. This should not be interpreted as strong semantic concept transfer.

MLP bridge is best or competitive in raw controlled score, especially for Llama. That is useful as a nonlinear upper comparison, but it is not the main ConCA claim. The main theory-relevant comparison is Procrustes/ridge.

## Set-Size Pattern

For the linear bridges, controlled overlap generally rises with set size in Llama 3 and Qwen 3.

### Llama 3

| Set size | Procrustes adjusted | Ridge adjusted |
| ---: | ---: | ---: |
| 2 | 0.1305 | 0.1212 |
| 4 | 0.1603 | 0.1486 |
| 6 | 0.1772 | 0.1689 |
| 8 | 0.1878 | 0.1792 |
| 10 | 0.1978 | 0.1923 |
| 12 | 0.2117 | 0.2053 |
| 14 | 0.2148 | 0.2083 |
| 16 | 0.2223 | 0.2180 |

### Qwen 3

| Set size | Procrustes adjusted | Ridge adjusted |
| ---: | ---: | ---: |
| 2 | 0.0859 | 0.0923 |
| 4 | 0.0916 | 0.1051 |
| 6 | 0.1034 | 0.1177 |
| 8 | 0.1029 | 0.1219 |
| 10 | 0.1117 | 0.1319 |
| 12 | 0.1169 | 0.1395 |
| 14 | 0.1249 | 0.1507 |
| 16 | 0.1145 | 0.1423 |

### Gemma 3

| Set size | Procrustes adjusted | Ridge adjusted |
| ---: | ---: | ---: |
| 2 | 0.0876 | 0.0959 |
| 4 | 0.0303 | 0.0359 |
| 6 | 0.0262 | 0.0269 |
| 8 | 0.0275 | 0.0300 |
| 10 | 0.0183 | 0.0200 |
| 12 | 0.0171 | 0.0188 |
| 14 | 0.0156 | 0.0170 |
| 16 | 0.0129 | 0.0137 |

Gemma 3 does not show the same set-size improvement under the controlled metric. Larger sets mostly reduce the already small controlled signal.

## Reconstruction Diagnostics

Average final test reconstruction diagnostics by set size showed:

- Qwen 3 normalized full reconstruction is around `0.71-0.73`.
- Llama 3 normalized full reconstruction is around `1.54-1.62`.
- Gemma 3 normalized full reconstruction is around `0.86-0.90`, but raw losses are extremely large because Gemma activation energy is very large.

This means raw loss should not be compared across families. Normalized reconstruction is the safer cross-family diagnostic.

## Scientific Takeaway

The updated result supports a cautious version of the SetConCA hypothesis:

- There is evidence for linear bridgeability in Llama 3 and Qwen 3.
- The evidence gets stronger with larger semantic sets for Llama 3, and mostly for Qwen 3.
- Gemma 3 raw bridge overlap is not trustworthy as evidence because the shuffled control is also very high.
- The strongest current linear bridge evidence is Llama 3 Procrustes at `S=16`, with adjusted TopK overlap `0.2223`.

## Next Step

Run a targeted all-family pair evaluation instead of only within-family pilots.

The next run should include cross-family source-target pairs and should report:

- within-family vs out-family,
- same-size vs cross-size,
- same-depth vs cross-depth,
- Procrustes and ridge as primary linear bridges,
- MLP as nonlinear upper comparison,
- shuffled-adjusted TopK as the primary bridge metric.

Status: `Succeeded`
