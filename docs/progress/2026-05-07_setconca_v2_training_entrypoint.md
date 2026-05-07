# Task: SetConCA V2 Training Entrypoint

Tags: #progress #training #implementation #phase-3 #smoke-test

Related notes: [[2026-05-07_activation_extraction_pipeline]] [[2026-05-07_server_4gpu_2000_dataset_qa]] [[PROJECT_GRAPH]]

## 1. Goal

Add a reproducible training CLI for SetConCA V2 activation banks, then verify it end to end on the dry-run activation bank.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-07 |
| Training script | `scripts/train_setconca_v2.py` |
| Smoke activation bank | `data/activations/smoke_fake_min8_s8.pt` |
| Smoke output folder | `results/smoke_train_fake_min8_s8` |
| Model | `model/setconca_v2.py` |
| Losses | `training/losses.py` |

## 3. Rationale

The repository already had the V2 model and loss functions, but it did not have a CLI that loads an activation bank, trains the model, records metrics, and saves a checkpoint. The missing piece made Phase 3 hard to reproduce.

The smoke run uses fake activations only to validate the mechanics. It does not provide a scientific result. It proves the training path can consume the activation-bank format produced by Phase 2.

## 4. Actions

| Step | Action | Why | Result | Status |
| --- | --- | --- | --- | --- |
| 1 | Added `scripts/train_setconca_v2.py`. | Need reproducible training from activation `.pt` files. | Script loads banks, builds `SetConCAV2`, trains with `compute_v2_loss`, and saves checkpoint/metrics. | Succeeded |
| 2 | Fixed `training/losses.py` import. | It referenced `setconca_v2.model`, but the actual module is `model/setconca_v2.py`. | Training losses import correctly. | Succeeded |
| 3 | Fixed `training/__init__.py`. | It imported a non-existent `training.trainer`. | Package import no longer fails. | Succeeded |
| 4 | Ran smoke training. | Needed end-to-end proof. | 3 epochs completed and saved outputs. | Succeeded |
| 5 | Ran tests. | Needed regression check. | 14 tests passed. | Succeeded |

## 5. Smoke Command

```powershell
python scripts\train_setconca_v2.py `
  --activations data\activations\smoke_fake_min8_s8.pt `
  --out-dir results\smoke_train_fake_min8_s8 `
  --epochs 3 `
  --batch-size 2 `
  --concept-dim 16 `
  --topk 4 `
  --device cpu
```

## 6. Smoke Output

```text
epoch=1 train_total=1.699815 test_total=1.419380
epoch=2 train_total=1.650428 test_total=1.418474
epoch=3 train_total=1.632497 test_total=1.417421
Saved checkpoint to results/smoke_train_fake_min8_s8/model.pt
Saved metrics to results/smoke_train_fake_min8_s8/metrics.json
```

## 7. Produced Files

| File | Purpose |
| --- | --- |
| `scripts/train_setconca_v2.py` | Train SetConCA V2 from activation banks. |
| `results/smoke_train_fake_min8_s8/model.pt` | Smoke-test checkpoint. |
| `results/smoke_train_fake_min8_s8/metrics.json` | Smoke-test metrics and manifest. |

## 8. Interpretation

The training path is now wired. Because the input activations were fake, the decreasing train loss only proves optimizer/model/loss compatibility. It is not evidence about SetConCA quality.

The next scientific step is:

1. Extract real activation banks from `sets_min8.jsonl`.
2. Train with the same script on real activations.
3. Add evaluation tests: semantic set vs shuffled set, duplicated set, and hard negatives.

## 9. Real Training Example

```bash
uv run python scripts/train_setconca_v2.py \
  --activations data/activations/gemma_2_2b_layer_-1_s8.pt \
  --out-dir results/train_gemma_2_2b_layer_-1_s8 \
  --epochs 50 \
  --batch-size 64 \
  --concept-dim 128 \
  --topk 32
```
