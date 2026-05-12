# Diverse Steering Slice

## Goal

Create a more diverse steering candidate set before causal steering.

The first clean concept candidates were mostly finance/market concepts. That suggested the candidate set was biased. We audited the dataset and found the cause: the previous transfer run used `--max-sets 300`, which selected the first 300 rows from `sets_min16.jsonl`.

## Dataset Audit

Full `sets_min16.jsonl` label counts:

| Label | Count |
| --- | ---: |
| `business` | 163 |
| `science_technology` | 356 |
| `sports` | 87 |
| `world` | 199 |

First 300 rows used by the previous run:

| Label | Count |
| --- | ---: |
| `business` | 45 |
| `science_technology` | 194 |
| `sports` | 17 |
| `world` | 44 |

This explains why the first steering candidates were not diverse.

## New Script

Added:

```text
scripts/build_diverse_dataset_slice.py
```

The script:

1. Selects a balanced subset of dataset rows.
2. Writes a new JSONL dataset.
3. Writes selected index metadata.
4. Creates matching activation banks with the same selected row order.

The activation-bank step is important because SetConCA training reads `activation_bank.pt`, not the JSONL text directly.

## Command Run

```powershell
python scripts\build_diverse_dataset_slice.py `
  --dataset data\generated\server_4gpu_2000\merged\sets_min16.jsonl `
  --activation-root data\activations\model_grid_s16_min16_4A100 `
  --out-dataset-dir data\generated\server_4gpu_2000\diverse_s16_300 `
  --out-activation-root data\activations\model_grid_s16_min16_diverse300_4A100 `
  --total 300 `
  --seed 0
```

Output label counts:

| Label | Count |
| --- | ---: |
| `business` | 75 |
| `science_technology` | 75 |
| `sports` | 75 |
| `world` | 75 |

Activation banks written:

```text
27
```

Verified one new bank shape:

```text
[300, 16, hidden_dim]
```

## New Artifacts

Dataset:

```text
data/generated/server_4gpu_2000/diverse_s16_300/sets_min16_diverse300.jsonl
```

Dataset metadata:

```text
data/generated/server_4gpu_2000/diverse_s16_300/README.md
data/generated/server_4gpu_2000/diverse_s16_300/diverse_slice_manifest.json
data/generated/server_4gpu_2000/diverse_s16_300/selected_indices.csv
data/generated/server_4gpu_2000/diverse_s16_300/selected_indices.json
```

Activation root:

```text
data/activations/model_grid_s16_min16_diverse300_4A100
```

## Recommended Next Run

Run the same Llama/Qwen SetConCA vs pointwise comparison on the diverse activation root:

```powershell
uv run python scripts\run_transfer_steering_grid.py `
  --activation-root data\activations\model_grid_s16_min16_diverse300_4A100 `
  --out-dir results\llama_qwen_diverse300_set_vs_pointwise_linear_seed0 `
  --only-family llama3,qwen3 `
  --set-sizes 2,4,6,8,10,12,14,16 `
  --epochs 25 `
  --batch-size 128 `
  --steering-alphas 0 `
  --bridges identity,procrustes,ridge `
  --methods setconca,pointwise_topk `
  --device cuda `
  --seed 0
```

Then inspect concepts from this diverse run. The goal is to find clean candidates across more themes, not only finance.
