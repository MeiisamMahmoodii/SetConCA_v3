# SetConCA V2

Clean restart for Set-ConCA with stricter semantic-set construction.

The first goal is to build better paraphrase sets from scratch:

1. Choose original sentences.
2. Extract likely copied keywords and ban them.
3. Load rewrite models one by one.
4. Ask each model to rewrite each sentence under several word-count bands.
5. Validate word count and banned-word avoidance.
6. Save every attempt, accepted or rejected, with reasons.

This is separate from the original project and previous results.

## Step 1: Generate Constrained Paraphrase Sets

### Download Fresh News Data For V2

This does not use any V1 files.

From inside `SetConCA_V2`:

```powershell
python scripts\download_news_dataset.py `
  --dataset ag_news `
  --split train `
  --limit 1000 `
  --out data\raw\ag_news_train.jsonl
```

From the parent repo folder, the old style also works:

```powershell
python SetConCA_V2\scripts\download_news_dataset.py --dataset ag_news --split train --limit 1000 --out SetConCA_V2\data\raw\ag_news_train.jsonl
```

Pilot download for only 10 rows:

```powershell
python scripts\download_news_dataset.py `
  --dataset ag_news `
  --split train `
  --limit 10 `
  --out data\raw\ag_news_train_10.jsonl
```

Edit:

- `configs/rewrite_models.example.json`
- `data/raw/ag_news_train_full.jsonl` or a smaller downloaded pilot JSONL

Then run:

```powershell
python scripts\generate_constrained_sets.py `
  --models-config configs\rewrite_models.example.json `
  --input data\raw\ag_news_train_10.jsonl `
  --out-dir data\generated `
  --max-originals 5
```

The script uses CUDA automatically when available and loads models one at a time.

### Server And Multi-GPU Generation

For Linux/WSL2 CUDA servers, install the optional server dependency group:

```bash
uv sync --extra server
```

Run a single-GPU vLLM pilot:

```bash
python scripts/launch_dataset_generation.py \
  --models-config configs/rewrite_models.example.json \
  --input data/raw/ag_news_train_full.jsonl \
  --out-dir data/generated/server_single_pilot \
  --backend vllm \
  --gpus 1 \
  --max-originals 10
```

Run one shard process per GPU, then merge automatically:

```bash
python scripts/launch_dataset_generation.py \
  --models-config configs/rewrite_models.example.json \
  --input data/raw/ag_news_train_full.jsonl \
  --out-dir data/generated/server_vllm_4gpu \
  --backend vllm \
  --gpus 4 \
  --max-originals 1000
```

Detailed guide:

```text
docs/progress/guides/server/multi_gpu_server_usage_guide.md
```

The current multi-GPU path is for dataset generation. It launches one rewrite-generation process per GPU with model shards and merges outputs after successful completion.

## Outputs

- `attempts.jsonl`: every generation attempt with validation status.
- `accepted.jsonl`: only validated paraphrases.
- `sets.jsonl`: grouped semantic sets by original sentence.
- `run_manifest.json`: config, device, counts, and validation summary.
- `review_table.md`: human-readable review file.
- `logs/shard_*.log`: per-GPU logs for multi-GPU launcher runs.

## Step 2: Prepare Sets For Activation Extraction

After a merged generation run, summarize and filter the grouped sets:

```powershell
python scripts\summarize_and_filter_sets.py `
  --input data\generated\server_4gpu_2000\merged\sets.jsonl `
  --out-dir data\generated\server_4gpu_2000\merged `
  --min-rewrites 8 `
  --filtered-name sets_min8.jsonl
```

The first large V2 dataset uses:

```text
data/generated/server_4gpu_2000/merged/sets_min8.jsonl
```

This keeps 1928 semantic sets with at least 8 accepted rewrites.

## Step 3: Extract Activation Banks

Dry-run smoke test:

```powershell
python scripts\extract_activation_bank.py `
  --sets data\generated\server_4gpu_2000\merged\sets_min8.jsonl `
  --out data\activations\smoke_fake_min8_s8.pt `
  --model-id dry-run/mock `
  --layer -1 `
  --views 8 `
  --max-sets 3 `
  --dry-run `
  --fake-hidden-dim 32
```

Real extraction example:

```bash
uv run python scripts/extract_activation_bank.py \
  --sets data/generated/server_4gpu_2000/merged/sets_min8.jsonl \
  --out data/activations/gemma_2_2b_layer_-1_s8.pt \
  --model-id google/gemma-2-2b \
  --layer -1 \
  --views 8 \
  --batch-size 8 \
  --max-length 256 \
  --dtype bfloat16
```

### Model-Family Layer Grid

The first full representation sweep uses:

```text
data/generated/server_4gpu_2000/merged/sets_min16.jsonl
```

It extracts 16-view activation banks for Llama 3, Gemma 3, and Qwen 3 across small/mid/big sizes and 20/60/90 percent depth layers.

Print the planned jobs:

```powershell
python scripts\run_activation_grid.py `
  --config configs\activation_model_grid.json `
  --out-root data\activations\model_grid_s16_min16 `
  --print-only
```

Run a server pilot:

```bash
uv run python scripts/run_activation_grid.py \
  --config configs/activation_model_grid.json \
  --out-root data/activations/pilot_qwen3_small_s16 \
  --only-family qwen3 \
  --only-size small \
  --max-sets 25 \
  --gpus 1
```

Run the full 4-GPU extraction grid:

```bash
uv run python scripts/run_activation_grid.py \
  --config configs/activation_model_grid.json \
  --out-root data/activations/model_grid_s16_min16 \
  --gpus 4
```

## Step 4: Train SetConCA V2

Smoke training on fake activations:

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

Real training example:

```bash
uv run python scripts/train_setconca_v2.py \
  --activations data/activations/gemma_2_2b_layer_-1_s8.pt \
  --out-dir results/train_gemma_2_2b_layer_-1_s8 \
  --epochs 50 \
  --batch-size 64 \
  --concept-dim 128 \
  --topk 32
```

## Tests

From inside `SetConCA_V2`:

```powershell
python -m pytest tests -q
```

From the parent repo folder:

```powershell
python -m pytest SetConCA_V2\tests -q
```
