# Task: Fresh AG News Dataset For V2

Tags: #progress #dataset #ag-news #v2

Related notes: [[README]] [[2026-05-06_v2_clean_restart]] [[2026-05-06_constrained_paraphrase_pipeline]]

## 1. Goal

Download a fresh V2 news dataset independently of V1. The user explicitly requested not to use V1 files.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-06 |
| Dataset | AG News |
| Main script | `scripts/download_news_dataset.py` |
| Output file | `data/raw/ag_news_train.jsonl` |
| Manifest | `data/raw/ag_news_train.manifest.json` |

## 3. Hypothesis Or Rationale

Using a fresh dataset gives V2 clean provenance. It prevents accidental dependence on old V1 preprocessing, old results, or previously selected examples.

## 4. Actions

| Step | Action | Why | Result | Status |
| --- | --- | --- | --- | --- |
| 1 | Added dataset formatting utilities. | Needed a stable V2 row schema. | `DatasetRow` has `id`, `text`, `source`, `label`, optional `title`. | Succeeded |
| 2 | Added AG News label mapping. | Needed readable labels. | Labels map to `world`, `sports`, `business`, `science_technology`. | Succeeded |
| 3 | Added text normalization. | Needed clean single-line sentence-like text. | Whitespace/newlines are normalized and long text can be truncated. | Succeeded |
| 4 | Added download CLI. | Needed reproducible data creation. | `scripts/download_news_dataset.py` downloads from Hugging Face datasets and writes JSONL plus manifest. | Succeeded |
| 5 | Downloaded V2 training rows. | Needed local raw data for generation. | `data/raw/ag_news_train.jsonl` and manifest exist. | Succeeded |
| 6 | Added tests. | Needed schema and normalization checks. | `tests/test_dataset_download.py` covers normalization, labels, and JSONL dict schema. | Succeeded |

## 5. Code And Pseudocode

```text
load_dataset("ag_news", split)
for record in dataset:
    text = normalize_news_text(record["text"])
    label = AG_NEWS_LABELS[record["label"]]
    write row:
        id = f"ag_news_{split}_{idx:06d}"
        text = text
        source = f"hf:ag_news:{split}"
        label = label
write JSONL rows
write manifest with dataset, split, limit, count, output path, and schema
```

## 6. Results

### Local Dataset Artifacts

| File | Observed Purpose |
| --- | --- |
| `data/raw/ag_news_train.jsonl` | Fresh V2 raw AG News training rows. |
| `data/raw/ag_news_train.manifest.json` | Provenance and schema manifest for the raw file. |

### Run Command

From inside `SetConCA_V2`:

```powershell
python scripts\download_news_dataset.py `
  --dataset ag_news `
  --split train `
  --limit 1000 `
  --out data\raw\ag_news_train.jsonl
```

## 7. Interpretation

The V2 dataset source is now explicit and reproducible. The manifest is important because later claims about activation extraction and training should trace back to exact dataset files.

## 8. Successes

The dataset task succeeded because V2 no longer needs V1 raw files. The schema is simple and tested.

## 9. Failures Or Limits

AG News is only the first dataset. It is useful for fast controlled pilots, but it is not enough by itself for all claims about cross-model concept transfer, multilingual transfer, or causal steering.

## 10. External Works And Papers

| Work | Link | Core Objective | How We Used It |
| --- | --- | --- | --- |
| Character-level Convolutional Networks for Text Classification, Zhang et al. 2015 | [arXiv 1509.01626](https://arxiv.org/abs/1509.01626) | Introduced large-scale text classification datasets including AG News experiments. | Used AG News as the initial V2 source of original news sentences. |
| Hugging Face Datasets | [datasets documentation](https://huggingface.co/docs/datasets) | Provide standardized dataset loading. | Used through `datasets.load_dataset("ag_news", split=...)`. |

## 11. Files Changed

| File | Change | Reason |
| --- | --- | --- |
| `src/setconca_v2/dataset_download.py` | Added row schema, normalization, AG News formatting. | Produce clean V2 dataset rows. |
| `scripts/download_news_dataset.py` | Added dataset download CLI. | Make fresh V2 data reproducible. |
| `data/raw/ag_news_train.jsonl` | Added downloaded rows. | Local raw input for paraphrase generation. |
| `data/raw/ag_news_train.manifest.json` | Added manifest. | Preserve provenance and schema. |
| `tests/test_dataset_download.py` | Added dataset tests. | Guard schema and normalization behavior. |

## 12. Follow-Up

- [ ] Add a small `--limit 10` pilot dataset note when real rewrites are generated.
- [ ] Add dataset statistics before training: label counts, length distribution, banned-word distribution.
