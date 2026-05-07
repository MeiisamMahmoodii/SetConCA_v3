# Task: Activation Extraction Pipeline

Tags: #progress #activation-extraction #implementation #phase-2 #semantic-sets

Related notes: [[2026-05-07_server_4gpu_2000_dataset_qa]] [[2026-05-06_v2_clean_restart]] [[PROJECT_GRAPH]]

## 1. Goal

Implement Phase 2: convert reviewed semantic-set JSONL into activation banks that SetConCA V2 can train on.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-07 |
| Input dataset | `data/generated/server_4gpu_2000/merged/sets_min8.jsonl` |
| First target view count | `S=8` |
| Output format | PyTorch `.pt` dict loadable by `data/activation_sets.py` |
| New CLI | `scripts/extract_activation_bank.py` |
| Core module | `src/setconca_v2/activation_extraction.py` |

## 3. Rationale

SetConCA V2 trains on activation tensors, not text. The dataset generation phase created semantic text sets; the activation phase must map each set view into a representation vector from a downstream model.

The expected bank shape is:

```text
[num_sets, num_views, hidden_dim]
```

For the first real training pass, `num_views=8` is the right target because `sets_min8.jsonl` keeps 1928 semantic sets with at least 8 accepted rewrites each.

## 4. Implementation

The activation extraction pipeline:

```text
read sets_min8.jsonl
for each semantic set:
    keep original text as view 0
    sample 7 rewrites deterministically by original_id and seed
flatten all view texts
run a Hugging Face causal LM with output_hidden_states=True
select one layer
pool either last non-padding token or mean token state
reshape back to [sets, views, hidden_dim]
save torch payload with text, labels, set IDs, rewrite metadata, and provenance
```

The script supports two modes:

| Mode | Purpose |
| --- | --- |
| Real extraction | Loads a Hugging Face model and extracts hidden states. |
| Dry run | Writes deterministic fake activations to test the pipeline locally without model downloads or GPU. |

## 5. Commands

### Local Smoke Test

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

### Verify Loader Compatibility

```powershell
python -c "from data.activation_sets import load_activation_bank; b=load_activation_bank('data/activations/smoke_fake_min8_s8.pt'); print(b.hidden.shape); print(b.meta)"
```

### Real Extraction Example

Run this on the machine that has the representation model available:

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

For a smaller pilot:

```bash
uv run python scripts/extract_activation_bank.py \
  --sets data/generated/server_4gpu_2000/merged/sets_min8.jsonl \
  --out data/activations/pilot_gemma_2_2b_layer_-1_s8_100sets.pt \
  --model-id google/gemma-2-2b \
  --layer -1 \
  --views 8 \
  --max-sets 100 \
  --batch-size 8 \
  --max-length 256 \
  --dtype bfloat16
```

## 6. Results

The local dry-run smoke test succeeded.

| Output | Shape | Meaning |
| --- | --- | --- |
| `data/activations/smoke_fake_min8_s8.pt` | `[3, 8, 32]` | Fake activation bank proving the extraction and loader format. |

Loader verification:

```text
torch.Size([3, 8, 32])
3 8 32
torch.Size([2, 8, 32]) torch.Size([1, 8, 32])
```

## 7. Payload Format

The saved `.pt` object contains:

| Key | Meaning |
| --- | --- |
| `hidden` | Tensor `[num_sets, views, hidden_dim]`. |
| `texts` | Original text for each set. |
| `view_texts` | Full text list for each set, including original and sampled rewrites. |
| `set_ids` | Original IDs. |
| `labels` | AG News labels. |
| `sources` | Source provenance. |
| `rewrite_meta` | Metadata for sampled rewrites. |
| `meta` | Model, layer, token position, source hash, view count, timing, and extraction settings. |

## 8. Interpretation

The pipeline now has a clean handoff from text dataset to model activations. This is the main bridge between Phase 1 and SetConCA training.

The smoke test does not validate model quality because it used fake activations. It validates the file shape, metadata, deterministic view sampling, and compatibility with the existing `ActivationSetBank` loader. The next empirical step is to run a real 100-set pilot on the server or local CUDA machine.

## 9. Files Changed

| File | Change |
| --- | --- |
| `src/setconca_v2/activation_extraction.py` | Added semantic-set loading, deterministic view sampling, HF hidden-state extraction, and activation-bank writing. |
| `scripts/extract_activation_bank.py` | Added CLI for real and dry-run activation extraction. |
| `data/activations/smoke_fake_min8_s8.pt` | Added dry-run smoke-test activation bank. |

## 10. Next Step

Run real activation extraction for a small pilot, then train SetConCA V2 on that pilot bank. If the pilot loss decreases and output shapes are stable, scale to all 1928 `S=8` sets.
