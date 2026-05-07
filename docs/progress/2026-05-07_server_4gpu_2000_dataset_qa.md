# Task: Server 4-GPU 2000-Original Dataset QA

Tags: #progress #dataset #qa #server #vllm #semantic-sets

Related notes: [[2026-05-06_constrained_paraphrase_pipeline]] [[2026-05-07_multi_gpu_server_usage_guide]] [[PROJECT_GRAPH]]

## 1. Goal

Record the completed 4-GPU vLLM generation run after it was moved back into the local V2 workspace, then create a filtered semantic-set file suitable for the first activation extraction pass.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-07 |
| Dataset folder | `data/generated/server_4gpu_2000` |
| Final merged sets | `data/generated/server_4gpu_2000/merged/sets.jsonl` |
| Filtered training candidate | `data/generated/server_4gpu_2000/merged/sets_min8.jsonl` |
| Generation backend | vLLM |
| Server device | 4 x NVIDIA A100-SXM4-40GB |
| Originals requested | 2000 |
| Rewrite models | 10 enabled models, sharded across 4 GPU processes |
| Length bands | 5-7, 10-12, 15-17, 20-22 |

## 3. Rationale

SetConCA training needs semantic sets with enough views per original sentence. The full merged dataset includes all successful rewrites, but some originals have too few accepted rewrites for stable set training. A minimum-view filter gives a clear first training substrate.

I chose `min_rewrites=8` because it keeps 1928 of 2000 sets. This preserves most of the data while guaranteeing enough views for an initial `S=8` training/evaluation sweep.

## 4. Actions

| Step | Action | Why | Result | Status |
| --- | --- | --- | --- | --- |
| 1 | Inspected moved server output folder. | Confirmed all shard and merged artifacts arrived locally. | Found `logs`, `shard_0` through `shard_3`, and `merged`. | Succeeded |
| 2 | Read merged manifest. | Needed provenance and run totals. | Manifest records 4 shards, 2000 sets, 29183 accepted rewrites, and 641146 attempts. | Succeeded |
| 3 | Checked logs for errors. | Needed to catch silent shard failures. | No `Traceback`, `ERROR`, `failed`, or `RuntimeError` entries found in shard logs. | Succeeded |
| 4 | Checked duplicate accepted keys. | There should be at most one accepted row per original/model/length band. | Duplicate count was 0. | Succeeded |
| 5 | Added reusable stats/filter code. | Dataset QA should be repeatable for future runs. | Added `src/setconca_v2/set_dataset.py` and `scripts/summarize_and_filter_sets.py`. | Succeeded |
| 6 | Created `sets_min8.jsonl`. | Need a stable first activation extraction input. | `sets_min8.jsonl` contains 1928 sets and 28764 rewrites. | Succeeded |

## 5. Commands

```powershell
python scripts\summarize_and_filter_sets.py `
  --input data\generated\server_4gpu_2000\merged\sets.jsonl `
  --out-dir data\generated\server_4gpu_2000\merged `
  --min-rewrites 8 `
  --filtered-name sets_min8.jsonl
```

## 6. Results

### Merged Dataset

| Metric | Value |
| --- | ---: |
| Sets | 2000 |
| Accepted rewrites | 29183 |
| Attempts | 641146 |
| Min rewrites per set | 2 |
| Max rewrites per set | 30 |
| Mean rewrites per set | 14.5915 |
| SHA256 | `f95acc0221bf27a29aff1766052a25b17fdb2f6bb5765b317a6bfeb77de33948` |

### Set Coverage By Minimum Rewrite Count

| Minimum rewrites | Sets retained |
| ---: | ---: |
| 2 | 2000 |
| 4 | 1995 |
| 8 | 1928 |
| 16 | 805 |
| 24 | 21 |
| 32 | 0 |
| 40 | 0 |

### Accepted Rewrites By Model

| Model | Accepted rewrites |
| --- | ---: |
| llama-3.2-3b-instruct | 6822 |
| gemma-2-2b-it | 5714 |
| phi-3.5-mini-instruct | 5212 |
| qwen2.5-7b-instruct | 4239 |
| mistral-7b-instruct | 2161 |
| qwen2.5-3b-instruct | 1736 |
| olmo-2-7b-instruct | 1633 |
| llama-3.2-1b-instruct | 1036 |
| qwen2.5-1.5b-instruct | 446 |
| tinyllama-1.1b-chat | 184 |

### Accepted Rewrites By Length Band

| Length band | Accepted rewrites |
| --- | ---: |
| 5-7 | 8151 |
| 10-12 | 7966 |
| 15-17 | 6966 |
| 20-22 | 6100 |

### Filtered Dataset

| File | Sets | Rewrites | Purpose |
| --- | ---: | ---: | --- |
| `data/generated/server_4gpu_2000/merged/sets_min8.jsonl` | 1928 | 28764 | First activation extraction input for `S=8`. |

## 7. Interpretation

The generation run is usable. The large number of attempts is expected because strict word-count and banned-word constraints reject many candidates. The model imbalance is also informative: larger/stronger instruction models produced more valid rewrites under the exact constraints, while TinyLlama contributed very few accepted rows.

The safest first training plan is to use `sets_min8.jsonl` and sample exactly 8 views per original. Using `S=16` is possible later, but it would reduce the dataset to 805 sets, so it should be a second experiment rather than the first baseline.

## 8. Files Added Or Produced

| File | Purpose |
| --- | --- |
| `src/setconca_v2/set_dataset.py` | Reusable stats, hashing, and filtering helpers for grouped set datasets. |
| `scripts/summarize_and_filter_sets.py` | CLI to summarize and filter `sets.jsonl`. |
| `data/generated/server_4gpu_2000/merged/set_stats.json` | Machine-readable stats for full merged set file. |
| `data/generated/server_4gpu_2000/merged/set_stats.md` | Human-readable stats report for full merged set file. |
| `data/generated/server_4gpu_2000/merged/sets_min8.jsonl` | Filtered semantic-set file with at least 8 rewrites per original. |
| `data/generated/server_4gpu_2000/merged/sets_min8_stats.json` | Machine-readable stats for filtered set file. |
| `data/generated/server_4gpu_2000/merged/sets_min8_stats.md` | Human-readable stats report for filtered set file. |

## 9. Next Step

Phase 2 is activation extraction:

1. Read `sets_min8.jsonl`.
2. For each set, build views from original plus sampled rewrites.
3. Run a representation model and extract hidden states.
4. Save an activation bank with shape `[num_sets, num_views, hidden_dim]`.
5. Train SetConCA V2 using the existing model and loss code.
