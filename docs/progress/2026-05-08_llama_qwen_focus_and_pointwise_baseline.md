# 2026-05-08 Llama/Qwen Focus And Pointwise Baseline

## Goal

Narrow the next SetConCA V2 phase to Llama 3 and Qwen 3, and add a same-data pointwise sparse baseline.

Reason:

- Gemma 3 raw bridge overlap was high, but shuffled-anchor overlap was also high.
- After shuffled correction, Gemma 3 contributed weak evidence for the linear bridge objective.
- Llama 3 and Qwen 3 showed the strongest controlled linear bridge signals, especially Qwen 3 to Llama 3 and Llama 3 to Qwen 3.

## Decision

Gemma 3 is disabled for future activation extraction in:

```text
configs/activation_model_grid.json
```

The entries are preserved for provenance, but `enabled` is now `false`.

Existing Gemma activation banks are not deleted. They remain available for audit or later investigation.

## Code Change

Updated:

```text
scripts/run_transfer_steering_grid.py
```

Added:

- `--exclude-family`, so existing activation roots can skip Gemma without moving files.
- `--methods`, so the same run can compare:
  - `setconca`
  - `pointwise_topk`
- `--include-cross-method-pairs`, off by default.

## Pointwise TopK Baseline

The new pointwise baseline trains on individual views rather than sets:

```text
x_i -> linear encoder -> LayerNorm -> TopK -> linear decoder -> x_i_hat
```

For bridge evaluation, the per-view dense codes are averaged only after pointwise encoding:

```text
z_set = TopK(mean_i z_dense_i)
```

This gives a fair baseline for the key question:

```text
Does set-level training add value beyond pointwise sparse codes pooled after the fact?
```

The baseline uses the same:

- activation banks,
- train/test split,
- concept dimension,
- TopK,
- bridge methods,
- shuffled-anchor controls,
- reports and figures.

## Recommended Next Run

Linear-only first:

```powershell
uv run python scripts\run_transfer_steering_grid.py `
  --activation-root data\activations\model_grid_s16_min16_4A100 `
  --out-dir results\llama_qwen_set_vs_pointwise_linear_seed0 `
  --only-family llama3,qwen3 `
  --set-sizes 2,4,6,8,10,12,14,16 `
  --max-sets 300 `
  --epochs 25 `
  --batch-size 128 `
  --steering-alphas 0 `
  --bridges identity,procrustes,ridge `
  --methods setconca,pointwise_topk `
  --device cuda `
  --seed 0
```

This run trains:

```text
18 Llama/Qwen activation banks x 8 set sizes x 2 methods = 288 models
```

## Expected Interpretation

If SetConCA beats `pointwise_topk` on shuffled-controlled linear bridge transfer, then we have evidence that set-level training is doing more than simple post-hoc pooling.

If `pointwise_topk` matches or beats SetConCA, then the current SetConCA architecture may still be useful, but the main claim must shift: the bridgeability may come from sparse linear coding plus paraphrase pooling, not necessarily from the SetConCA training objective.

## Verification

Commands run:

```text
python -m py_compile scripts\run_transfer_steering_grid.py
python scripts\run_transfer_steering_grid.py --activation-root data\activations\model_grid_s16_min16_4A100 --out-dir results\dry_run_llama_qwen_methods --only-family llama3,qwen3 --set-sizes 2 --max-banks 2 --methods setconca,pointwise_topk --dry-run
```

Status:

```text
Succeeded
```

## Runtime Fix

The first full baseline attempt failed at job 2:

```text
NameError: name 'offdiag_decorrelation' is not defined
```

Cause:

```text
scripts/run_transfer_steering_grid.py
```

used the shared off-diagonal diagnostic in the new `pointwise_topk` path but did not import it from `training.losses`.

Fix:

- imported `offdiag_decorrelation`,
- added `--resume` so completed model artifacts can be reused after interrupted long runs,
- smoke-tested `pointwise_topk` end to end.

Smoke command:

```text
python scripts\run_transfer_steering_grid.py --activation-root data\activations\model_grid_s16_min16_4A100 --out-dir results\smoke_pointwise_topk_fix --only-family llama3 --set-sizes 2 --max-banks 1 --max-sets 12 --epochs 1 --batch-size 8 --steering-alphas 0 --bridges identity,procrustes,ridge --methods pointwise_topk --include-self-pairs --device cpu --seed 0
```

Smoke result:

```json
{
  "n_banks": 1,
  "n_trained": 1,
  "n_result_rows": 3
}
```

## Status

Status: `Ready for resumed baseline run`
