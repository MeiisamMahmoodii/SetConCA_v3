# Task: Model Family And Layer Grid Plan

Tags: #progress #experiment-plan #activation-extraction #model-grid #layer-sweep #set-size-sweep

Related notes: [[2026-05-07_server_4gpu_2000_dataset_qa]] [[2026-05-07_activation_extraction_pipeline]] [[2026-05-07_setconca_v2_training_entrypoint]] [[PROJECT_GRAPH]]

## 1. Goal

Define and implement the first serious SetConCA V2 representation experiment:

1. Use only the controlled `sets_min16.jsonl` dataset.
2. Extract hidden states from three model families.
3. Use three size tiers per family.
4. Extract from early, middle-late, and late layers.
5. Later train SetConCA with set sizes `S = 2, 4, 6, 8, 10, 12, 14, 16` from the same activation banks.

## 2. Why This Design

The key scientific question is:

```text
Does SetConCA benefit from larger semantic sets, and does that behavior change across model family, model scale, and network depth?
```

Using `sets_min16.jsonl` makes the set-size sweep fair. Every run can use the same 805 originals, because every original has at least 16 accepted rewrites. We can extract one `S=16` activation bank per model/layer and then train smaller set-size conditions by slicing views from that same bank.

This avoids the confound where `S=16` is trained on a smaller or different dataset than `S=8`.

## 3. Dataset

| Field | Value |
| --- | --- |
| Dataset | `data/generated/server_4gpu_2000/merged/sets_min16.jsonl` |
| Sets | 805 |
| Rewrites | 14834 |
| Min rewrites per set | 16 |
| Mean rewrites per set | 18.4273 |
| SHA256 | `a367c5b6bba43d9996eb0adcae7faa41605941394b996e7350126e268dc7b74e` |

## 4. Model Grid

Sizes are nearest official sizes. The requested 1B/4B/7B pattern does not exist exactly in every family.

| Family | Small | Mid | Big |
| --- | --- | --- | --- |
| Llama 3 | `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-3B` | `meta-llama/Llama-3.1-8B` |
| Gemma 3 | `google/gemma-3-1b-pt` | `google/gemma-3-4b-pt` | `google/gemma-3-12b-pt` |
| Qwen 3 | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-4B` | `Qwen/Qwen3-8B` |

Interpretation:

- Llama 3 uses 1B, 3B, 8B because there is no official 4B or 7B Llama 3 text model.
- Gemma 3 uses 1B, 4B, 12B because there is no 7B Gemma 3.
- Qwen 3 uses 0.6B, 4B, 8B because those are clean official tiers near the requested sizes.

## 5. Layer Policy

For every model, the extraction script reads the model config and computes:

```text
early = round(num_hidden_layers * 0.20)
middle_late = round(num_hidden_layers * 0.60)
late = round(num_hidden_layers * 0.90)
```

The exact layer indices are written into:

```text
data/activations/model_grid_s16_min16/activation_grid_manifest.json
```

This is better than hardcoding layers because different model families have different depths.

The grid config also records fallback `num_hidden_layers` values. These support offline planning and dry runs when the local machine cannot reach Hugging Face. For real server extraction, the launcher can resolve live model configs unless `--use-config-layers` is supplied.

## 6. Folder Layout

Each extraction job writes one activation bank and one log:

```text
data/activations/model_grid_s16_min16/
  activation_grid_manifest.json
  llama3/
    small_1b/
      meta-llama__llama-3.2-1b/
        layer_XX_20pct/
          activation_bank.pt
          extract.log
        layer_YY_60pct/
        layer_ZZ_90pct/
    mid_3b/
    big_8b/
  gemma3/
    small_1b/
    mid_4b/
    big_12b/
  qwen3/
    small_0.6b/
    mid_4b/
    big_8b/
```

The activation bank payload is compatible with `data/activation_sets.py`.

## 7. Script

| File | Purpose |
| --- | --- |
| `configs/activation_model_grid.json` | Declares dataset, set-size sweep, layer fractions, extraction settings, and model grid. |
| `scripts/run_activation_grid.py` | Resolves layer indices and launches extraction jobs locally or across GPUs. |

## 8. Commands

### Print Planned Commands

```powershell
python scripts\run_activation_grid.py `
  --config configs\activation_model_grid.json `
  --out-root data\activations\model_grid_s16_min16 `
  --print-only
```

### Local One-GPU Or CPU Run

Use this for a small pilot, not the full 27-job grid:

```powershell
python scripts\run_activation_grid.py `
  --config configs\activation_model_grid.json `
  --out-root data\activations\pilot_qwen3_small_s16 `
  --only-family qwen3 `
  --only-size small `
  --max-sets 25
```

### Server 4-GPU Run

Run with `uv` on the server:

```bash
uv run python scripts/run_activation_grid.py \
  --config configs/activation_model_grid.json \
  --out-root data/activations/model_grid_s16_min16 \
  --gpus 4
```

This launches up to four extraction jobs at a time and pins each process with `CUDA_VISIBLE_DEVICES`.

### Server 4-GPU Pilot

```bash
uv run python scripts/run_activation_grid.py \
  --config configs/activation_model_grid.json \
  --out-root data/activations/pilot_qwen3_small_s16 \
  --only-family qwen3 \
  --only-size small \
  --max-sets 25 \
  --gpus 1
```

## 9. Expected Experiment Size

Full activation extraction:

```text
3 families x 3 sizes x 3 layers = 27 activation banks
```

Training sweep after extraction:

```text
27 activation banks x 8 set sizes = 216 SetConCA training runs
```

This is large but cleanly structured. The recommended order is:

1. Qwen3 small pilot.
2. Full Qwen3 family.
3. Full 27-bank extraction.
4. Small training sweep on one family.
5. Full training sweep.

## 10. Current Status

The model grid config and launcher exist. The next required action is a server pilot extraction to confirm model access, memory, and runtime before launching all 27 activation jobs.
