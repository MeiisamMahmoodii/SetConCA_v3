# SetConCA V2 Experiment Report

## Current Objective

SetConCA V2 is now moving from dataset construction to representation experiments.

The current objective is to test whether semantic set size improves learned concept representations, and whether that behavior changes across:

- model family,
- model size,
- model depth,
- and set size.

## Dataset Under Study

The controlled dataset for the first serious sweep is:

```text
data/generated/server_4gpu_2000/merged/sets_min16.jsonl
```

Why this file:

- It has 805 semantic sets.
- Every set has at least 16 accepted rewrites.
- It supports fair set-size comparisons from `S=2` through `S=16`.
- All set-size conditions can use the same original sentences.

## Model Grid

| Family | Small | Mid | Big |
| --- | --- | --- | --- |
| Llama 3 | `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-3B` | `meta-llama/Llama-3.1-8B` |
| Gemma 3 | `google/gemma-3-1b-pt` | `google/gemma-3-4b-pt` | `google/gemma-3-12b-pt` |
| Qwen 3 | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-4B` | `Qwen/Qwen3-8B` |

These are nearest official family sizes. Some families do not have exact 1B/4B/7B releases.

## Layer Grid

For each model, extract hidden states from three depth points:

```text
20%, 60%, 90%
```

The script reads each model config and converts these fractions to actual layer indices.

## Activation Extraction Plan

For every model/layer pair:

1. Load `sets_min16.jsonl`.
2. Select 16 views per set.
3. Run the representation model.
4. Extract hidden states from the chosen layer.
5. Pool the last non-padding token by default.
6. Save an activation bank shaped:

```text
[805, 16, hidden_dim]
```

## Training Plan

After extraction, train SetConCA V2 with set sizes:

```text
S = 2, 4, 6, 8, 10, 12, 14, 16
```

This produces:

```text
27 activation banks x 8 set sizes = 216 training runs
```

## Why This Is Scientifically Clean

The activation extraction uses one fixed dataset source. Smaller set sizes are created by slicing the same activation bank, not by switching to a different dataset.

Therefore, changes across `S` are less confounded by changes in original sentence coverage.

## Main Scripts

| Script | Purpose |
| --- | --- |
| `scripts/run_activation_grid.py` | Run all model-family/size/layer extraction jobs. |
| `scripts/extract_activation_bank.py` | Extract one activation bank. |
| `scripts/train_setconca_v2.py` | Train SetConCA V2 on one bank and one set size. |
| `scripts/summarize_and_filter_sets.py` | Create dataset stats and filtered set files. |

## Recommended Server Command

```bash
uv run python scripts/run_activation_grid.py \
  --config configs/activation_model_grid.json \
  --out-root data/activations/model_grid_s16_min16 \
  --gpus 4
```

## Recommended Pilot First

```bash
uv run python scripts/run_activation_grid.py \
  --config configs/activation_model_grid.json \
  --out-root data/activations/pilot_qwen3_small_s16 \
  --only-family qwen3 \
  --only-size small \
  --max-sets 25 \
  --gpus 1
```

## Current Status

- Large semantic-set dataset exists.
- `sets_min16.jsonl` is selected as the controlled sweep dataset.
- Activation extraction script exists.
- Training script exists.
- Model/layer grid config exists.
- Activation-grid launcher exists.
- Next action is a server pilot extraction.
