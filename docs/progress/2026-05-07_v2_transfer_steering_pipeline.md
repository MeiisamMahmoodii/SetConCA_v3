# 2026-05-07 V2 Transfer and Steering Pipeline

## Goal

Build the V2 experiment pipeline step by step, with documentation before and after each implementation step.

The pipeline should use the activation banks under:

```text
data/activations/model_grid_s16_min16_4A100
```

and evaluate SetConCA V2 across:

- set sizes: `2, 4, 6, 8, 10, 12, 14, 16`
- model families: Llama 3, Gemma 3, Qwen 3
- model sizes: small, mid, big
- layer depths: 20%, 60%, 90%
- transfer directions: within-family and cross-family, all available paired activation banks
- bridge types: linear, ridge, MLP, and other methods only when justified
- steering-style tests: structured concept directions compared against random directions

## Design Discipline

The architecture should stay close to the ConCA -> Set-ConCA -> V1 lineage unless a real problem is found.

Core references for implementation decisions:

- Original ConCA: log-posterior interpretation of linear concept representations.
- Initial Set-ConCA proposal: set-level concept estimation from multiple related views.
- V1 results: TopK sparsity, `u_bar` vs `z_hat`, dual decoder, linear bridge strength, pointwise TopK warning, consistency-loss caveat.

## Implementation Rule

No code change should be treated as complete unless this note records:

1. What changed.
2. Why it changed.
3. Which existing project idea it follows.
4. How it was tested.
5. Where outputs/logs/results are saved.

## Step 1: Pre-Implementation State

Status: `Succeeded`

Before adding new experiment code, inspect existing V2 files for:

- current SetConCA model implementation,
- current activation-bank loader,
- current training script,
- current docs and report structure,
- current activation metadata and `.pt` shape assumptions.

No architecture changes are planned in this first step.

Findings:

- `model/setconca_v2.py` already implements the current V2 architecture with shared encoder, mean aggregation, LayerNorm, TopK sparse code, shared decoder, residual decoder, and explicit `z_dense` / `z` outputs.
- `training/losses.py` already implements shared reconstruction, full reconstruction, contrastive terms, support consistency, off-diagonal decorrelation, and residual-energy penalty.
- `data/activation_sets.py` already loads `.pt` activation banks into `ActivationSetBank` with `hidden`, `texts`, and `meta`.
- `scripts/train_setconca_v2.py` trains one SetConCA V2 model on one activation bank.

Decision:

Keep the architecture unchanged. Build the next layer as experiment orchestration and evaluation around existing V2 modules.

## Step 2: Experiment Pipeline Design

Status: `Succeeded`

The first implementation target is a reusable script that can:

1. Discover activation banks under an activation root.
2. Train one SetConCA V2 model per selected bank and set size.
3. Extract test concept codes from trained models.
4. Evaluate concept transfer for all paired trained models over shared test anchors.
5. Compare bridge types:
   - `identity` when concept spaces have the same dimension,
   - `procrustes` for orthogonal linear alignment,
   - `ridge` for regularized linear regression,
   - `mlp` for a small nonlinear bridge.
6. Evaluate a steering-style proxy:
   - choose source concept directions,
   - bridge them into the target concept space,
   - add scaled directions to target codes,
   - measure whether target similarity moves toward the matched concept direction more than random directions.
7. Save raw metrics, logs, tables, and figures under one organized run directory.

The smoke version should support filters such as `--only-family`, `--only-size`, `--max-banks`, `--set-sizes`, `--epochs`, and `--max-sets` so we can verify correctness before the full sweep.

This is an evaluation/orchestration addition, not a change to the SetConCA architecture.

## Step 3: Initial Implementation

Status: `Succeeded`

Added:

```text
scripts/run_transfer_steering_grid.py
```

Purpose:

- Discover `activation_bank.pt` files.
- Train SetConCA V2 models for selected banks and set sizes.
- Save checkpoints and concept codes.
- Evaluate paired concept transfer with multiple bridge types.
- Save raw transfer/steering metrics as CSV and JSON.
- Generate simple figures:
  - bridge TopK overlap,
  - steering proxy gain.
- Write a local `REPORT.md` inside each run directory.

Bridge methods implemented:

- `identity`: direct comparison when concept dimensions match.
- `procrustes`: orthogonal linear bridge.
- `ridge`: regularized linear bridge.
- `mlp`: small nonlinear bridge.

Steering metric implemented:

- Concept-code proxy only.
- It compares adding bridged source concept directions to target concept codes against adding randomly permuted directions.
- This is not a full language-model behavioral steering result, and reports must say that clearly.

Architecture changes:

- None.

Validation so far:

```text
python -m py_compile scripts\run_transfer_steering_grid.py
```

passed.

Dry run:

```text
python scripts\run_transfer_steering_grid.py --activation-root data\activations\model_grid_s16_min16_4A100 --out-dir results\_dry_transfer_steering --only-family qwen3 --only-size small --only-layer-pct 20 --set-sizes 2 --max-banks 1 --max-sets 20 --dry-run
```

selected the expected Qwen3 small 20% layer bank.

## Step 4: Smoke Run

Status: `Succeeded`

Command:

```text
python scripts\run_transfer_steering_grid.py --activation-root data\activations\model_grid_s16_min16_4A100 --out-dir results\smoke_transfer_steering_qwen3_small --only-family qwen3 --only-size small --set-sizes 2,4 --max-banks 2 --max-sets 24 --epochs 1 --batch-size 8 --mlp-epochs 5 --steering-alphas 0,1 --bridges identity,procrustes,ridge,mlp --include-self-pairs --device cpu
```

Why this command:

- Uses only two Qwen3 small activation banks.
- Uses two set sizes, `S=2` and `S=4`.
- Uses only 24 sets.
- Trains for one epoch.
- Includes self-pairs to make sure bridge logic works in both self and cross-layer cases.
- Runs on CPU to avoid depending on local GPU availability for the smoke test.

Output directory:

```text
results/smoke_transfer_steering_qwen3_small
```

Observed output:

```text
n_banks: 2
n_trained: 4
n_result_rows: 64
```

Generated artifacts:

```text
results/smoke_transfer_steering_qwen3_small/run_manifest.json
results/smoke_transfer_steering_qwen3_small/run_summary.json
results/smoke_transfer_steering_qwen3_small/training_summary.csv
results/smoke_transfer_steering_qwen3_small/transfer_steering_results.csv
results/smoke_transfer_steering_qwen3_small/transfer_steering_results.json
results/smoke_transfer_steering_qwen3_small/REPORT.md
results/smoke_transfer_steering_qwen3_small/figures/bridge_topk_overlap.png
results/smoke_transfer_steering_qwen3_small/figures/steering_proxy_gain.png
```

Smoke report summary:

| Bridge | Mean TopK overlap at alpha=0 |
| --- | ---: |
| identity | 0.6281 |
| mlp | 0.5555 |
| procrustes | 0.6148 |
| ridge | 0.6984 |

Steering proxy:

| Alpha | Mean structured-random similarity |
| ---: | ---: |
| 0.00 | 0.0000 |
| 1.00 | 0.1101 |

Interpretation:

This smoke result only verifies that the pipeline works. It is not a scientific result because it used one epoch, 24 sets, two banks, and self-pairs. The values should not be cited as evidence for or against SetConCA.

## Step 5: Verification

Status: `Succeeded`

Commands:

```text
python -m py_compile scripts\run_transfer_steering_grid.py
python -m pytest -q
```

Results:

```text
14 passed
```

Pytest emitted a cache warning because Windows could not create a `.pytest_cache` path, but the tests passed.

## Next Full-Run Direction

The full local/server run should use the same script without the smoke-test caps.

Example pilot without self-pairs:

```text
python scripts\run_transfer_steering_grid.py --activation-root data\activations\model_grid_s16_min16_4A100 --out-dir results\pilot_transfer_steering_qwen3_small --only-family qwen3 --only-size small --set-sizes 2,4,8,16 --max-sets 200 --epochs 5 --batch-size 32 --mlp-epochs 50
```

## Step 6: Small Pilot Plan

Status: `Succeeded`

Run a small but nontrivial pilot:

```text
python scripts\run_transfer_steering_grid.py --activation-root data\activations\model_grid_s16_min16_4A100 --out-dir results\pilot_transfer_steering_qwen3_small --only-family qwen3 --only-size small --set-sizes 2,4,8,16 --max-sets 100 --epochs 3 --batch-size 16 --mlp-epochs 20 --steering-alphas 0,1,2 --bridges identity,procrustes,ridge,mlp --device cpu
```

Purpose:

- Train all three Qwen3 0.6B layer banks.
- Compare set sizes `2, 4, 8, 16`.
- Exclude self-pairs by default, so transfer summaries are not inflated by model-to-itself rows.
- Keep CPU runtime acceptable while producing real artifacts and figures.

This is still a pilot, not the final scientific run.

Pilot completed:

```text
n_banks: 3
n_trained: 12
n_result_rows: 288
```

Output directory:

```text
results/pilot_transfer_steering_qwen3_small
```

Pilot bridge summary:

| Bridge | Mean TopK overlap at alpha=0 |
| --- | ---: |
| identity | 0.2479 |
| mlp | 0.8365 |
| procrustes | 0.7740 |
| ridge | 0.8338 |

Pilot steering proxy:

| Alpha | Mean structured-random similarity |
| ---: | ---: |
| 0.00 | 0.0000 |
| 1.00 | 0.0720 |
| 2.00 | 0.0960 |

Important pilot caveats:

- This used only Qwen3 0.6B.
- It used only 100 sets.
- It trained for only 3 epochs.
- It should be treated as a script/pipeline pilot, not as a final scientific claim.
- Late-layer Qwen3 small activations had much larger reconstruction-loss scale than early/mid layers, so final reports must stratify by layer and avoid mixing raw loss values without normalization/context.

## Step 7: Figure Improvement

Status: `Succeeded`

Added two more automatic figures to `scripts/run_transfer_steering_grid.py`:

- `figures/set_size_bridge_topk_overlap.png`
- `figures/training_loss_by_layer_set_size.png`

Reason:

The first smoke/pilot reports had bridge and steering charts, but did not directly visualize the two key V2 axes: set size and layer depth. These plots make later reports easier to audit and reuse.

Also added:

```text
--seed
```

Reason:

The first regenerated pilot shifted slightly because model initialization, batch sampling, MLP bridge training, and random steering directions were not tied to a command-line seed. The script now calls `torch.manual_seed(args.seed)` at run start and records the seed in `run_manifest.json`.

Seeded pilot command:

```text
python scripts\run_transfer_steering_grid.py --activation-root data\activations\model_grid_s16_min16_4A100 --out-dir results\pilot_transfer_steering_qwen3_small_seed0 --only-family qwen3 --only-size small --set-sizes 2,4,8,16 --max-sets 100 --epochs 3 --batch-size 16 --mlp-epochs 20 --steering-alphas 0,1,2 --bridges identity,procrustes,ridge,mlp --device cpu --seed 0
```

Seeded pilot output:

```text
n_banks: 3
n_trained: 12
n_result_rows: 288
```

Seeded pilot bridge summary:

| Bridge | Mean TopK overlap at alpha=0 |
| --- | ---: |
| identity | 0.2357 |
| mlp | 0.8307 |
| procrustes | 0.7648 |
| ridge | 0.8256 |

Seeded pilot set-size / bridge summary:

| S | identity | mlp | procrustes | ridge |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.2750 | 0.7997 | 0.7247 | 0.7818 |
| 4 | 0.2422 | 0.8432 | 0.7750 | 0.8365 |
| 8 | 0.1927 | 0.8479 | 0.7911 | 0.8401 |
| 16 | 0.2328 | 0.8318 | 0.7682 | 0.8440 |

Seeded pilot steering proxy:

| Alpha | Mean structured-random similarity |
| ---: | ---: |
| 0.00 | 0.0000 |
| 1.00 | 0.0751 |
| 2.00 | 0.1001 |

Seeded pilot training-loss snapshot:

| Layer | S=2 | S=4 | S=8 | S=16 |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 0.7814 | 0.7128 | 0.7322 | 0.5534 |
| 17 | 4.5179 | 4.3415 | 4.2484 | 4.1648 |
| 25 | 205.9135 | 201.8225 | 200.6408 | 200.9428 |

Interpretation:

This is still a pilot, not a paper result. It does show that the pipeline can train across layers and set sizes, compare bridges, emit honest raw tables, and generate reusable figures.

The late-layer loss scale is much larger than early/mid layers for this Qwen3 0.6B pilot. Later reports should keep reconstruction loss stratified by model/layer and should emphasize transfer metrics separately from raw reconstruction scale.

Example full grid:

```text
python scripts\run_transfer_steering_grid.py --activation-root data\activations\model_grid_s16_min16_4A100 --out-dir results\full_transfer_steering_grid --set-sizes 2,4,6,8,10,12,14,16 --epochs 10 --batch-size 32 --mlp-epochs 100
```

The full grid trains:

```text
27 activation banks x 8 set sizes = 216 SetConCA models
```

and then evaluates all directed trained-model pairs with matching set size. This is a large run and should be done on the server or in staged family/size slices.
