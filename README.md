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
- `data/raw/original_sentences.example.jsonl`

Then run:

```powershell
python scripts\generate_constrained_sets.py `
  --models-config configs\rewrite_models.example.json `
  --input data\raw\ag_news_train_10.jsonl `
  --out-dir data\generated `
  --max-originals 5
```

The script uses CUDA automatically when available and loads models one at a time.

## Outputs

- `attempts.jsonl`: every generation attempt with validation status.
- `accepted.jsonl`: only validated paraphrases.
- `sets.jsonl`: grouped semantic sets by original sentence.
- `run_manifest.json`: config, device, counts, and validation summary.

## Tests

```powershell
python -m pytest SetConCA_V2\tests -q
```
