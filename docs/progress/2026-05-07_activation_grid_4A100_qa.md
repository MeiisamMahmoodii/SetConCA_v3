# 2026-05-07 Activation Grid 4A100 QA

## Goal

Check the copied activation outputs in:

```text
data/activations/model_grid_s16_min16_4A100
```

The purpose was to confirm that all model-family, model-size, and layer activation banks are present and usable before starting SetConCA V2 training sweeps over set sizes.

## Context

The activation grid was extracted on the server from:

```text
data/generated/server_4gpu_2000/merged/sets_min16.jsonl
```

This dataset contains 805 semantic sets with at least 16 accepted views per original sentence. The extraction target is therefore one activation bank per model/layer, each shaped:

```text
[805, 16, hidden_dim]
```

## Checks Run

1. Listed the copied activation directory tree.
2. Loaded every `activation_bank.pt` with `torch.load(..., map_location="cpu")`.
3. Checked each payload contains a `hidden` tensor.
4. Checked each tensor has shape `[805, 16, hidden_dim]`.
5. Checked sampled tensor values are finite.
6. Compared actual banks against the configured grid in `configs/activation_model_grid.json`.
7. Scanned all `extract.log` files for obvious failure words such as `Traceback`, `ERROR`, `OutOfMemoryError`, `failed`, and `Killed`.

## Findings

Status: `Succeeded`

- Expected activation banks: 27
- Actual activation banks: 27
- Missing banks: 0
- Extra banks: 0
- Logs checked: 27
- Logs with failure patterns: 0
- Total activation size: about 3.60 GB

The root `activation_grid_manifest.json` only records the last rerun subset, not the full 27-job grid. This is not a data problem because all 27 activation banks are present. It means the manifest should not be treated as the authoritative full-run record for this copied folder.

## Activation Summary

| Family | Size | Model | Layers | Shape Pattern |
| --- | --- | --- | --- | --- |
| Gemma 3 | 1B | `google/gemma-3-1b-pt` | 5, 16, 23 | `[805, 16, 1152]` |
| Gemma 3 | 4B | `google/gemma-3-4b-pt` | 7, 20, 31 | `[805, 16, 2560]` |
| Gemma 3 | 12B | `google/gemma-3-12b-pt` | 10, 29, 43 | `[805, 16, 3840]` |
| Llama 3 | 1B | `meta-llama/Llama-3.2-1B` | 3, 10, 14 | `[805, 16, 2048]` |
| Llama 3 | 3B | `meta-llama/Llama-3.2-3B` | 6, 17, 25 | `[805, 16, 3072]` |
| Llama 3 | 8B | `meta-llama/Llama-3.1-8B` | 6, 19, 29 | `[805, 16, 4096]` |
| Qwen 3 | 0.6B | `Qwen/Qwen3-0.6B` | 6, 17, 25 | `[805, 16, 1024]` |
| Qwen 3 | 4B | `Qwen/Qwen3-4B` | 7, 22, 32 | `[805, 16, 2560]` |
| Qwen 3 | 8B | `Qwen/Qwen3-8B` | 7, 22, 32 | `[805, 16, 4096]` |

## Interpretation

The activation grid is complete enough to begin SetConCA V2 training experiments. The next research step is to train on each activation bank with set sizes:

```text
S = 2, 4, 6, 8, 10, 12, 14, 16
```

That creates the planned comparison of how semantic set size affects the learned concept representation across family, model size, and depth.

## Notes

The layer indices are produced by `scripts/run_activation_grid.py` using:

```text
round(num_hidden_layers * layer_fraction)
```

with fractions `0.2`, `0.6`, and `0.9`. The indices are therefore the launcher's intended layer numbers, not a zero-based `round((num_hidden_layers - 1) * fraction)` conversion.
