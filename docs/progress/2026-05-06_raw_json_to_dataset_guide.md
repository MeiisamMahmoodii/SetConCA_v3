# Task: Raw JSON To Dataset Guide

Tags: #progress #documentation #dataset #guide

Related notes: [[README]] [[raw_json_to_dataset_guide]] [[2026-05-06_fresh_ag_news_dataset]] [[2026-05-06_constrained_paraphrase_pipeline]] [[2026-05-06_project_graph_documentation]]

## 1. Goal

The user asked for a document explaining how raw data is converted into the project dataset. The requested guide needed to explain each step from JSON/JSONL formatting through running the scripts, and it needed to connect to the other progress documents.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-06 |
| Main guide created | `docs/progress/guides/data_pipeline/raw_json_to_dataset_guide.md` |
| Existing related docs | `docs/PROJECT_GRAPH.md`, `docs/progress/2026-05-06_fresh_ag_news_dataset.md`, `docs/progress/2026-05-06_constrained_paraphrase_pipeline.md` |
| Main scripts inspected | `scripts/download_news_dataset.py`, `scripts/generate_constrained_sets.py` |
| Main modules inspected | `dataset_download.py`, `io_utils.py`, `paths.py`, `rewrite_generation.py`, `text_constraints.py`, `semantic_validation.py` |

## 3. Hypothesis Or Rationale

A data-conversion guide should be separate from a task log because it will be reused many times. The task log records that the guide was made. The guide itself explains the durable process.

## 4. Actions

| Step | Action | Why | Result | Status |
| --- | --- | --- | --- | --- |
| 1 | Inspected current data files. | Needed to document real local artifacts. | Found `data/raw/ag_news_train.jsonl` and `data/raw/ag_news_train.manifest.json`. | Succeeded |
| 2 | Read dataset download code. | Needed exact schema and normalization behavior. | Documented `DatasetRow`, label mapping, text normalization, JSONL writing, and manifest writing. | Succeeded |
| 3 | Read generation code. | Needed to explain how raw rows become semantic sets. | Documented prompt creation, banned-word extraction, length bands, validation, grouping, and output artifacts. | Succeeded |
| 4 | Read tests and project graph. | Needed to connect guide to current verification and architecture docs. | Linked the guide to related notes and `PROJECT_GRAPH`. | Succeeded |
| 5 | Created categorized guide folder. | Needed cleaner organization as requested by the user. | Added `docs/progress/guides/README.md` and `docs/progress/guides/data_pipeline/raw_json_to_dataset_guide.md`. | Succeeded |
| 6 | Updated progress index. | Needed Obsidian navigation from the main progress page. | Added the task note and guide link to `docs/progress/README.md`. | Succeeded |

## 5. Code And Pseudocode

The guide records the data conversion as:

```text
raw AG News record
    -> normalize text
    -> map numeric label to readable label
    -> assign stable V2 id and source
    -> write JSONL row
    -> write manifest
    -> generate constrained rewrites
    -> validate candidates
    -> write attempts, accepted rows, grouped sets, review table, run manifest
```

## 6. Results

| Output | Purpose |
| --- | --- |
| `docs/progress/guides/README.md` | Index for reusable progress guides. |
| `docs/progress/guides/data_pipeline/raw_json_to_dataset_guide.md` | Full raw JSON/JSONL to SetConCA V2 dataset guide. |
| `docs/progress/2026-05-06_raw_json_to_dataset_guide.md` | Lab-note record of creating the guide. |
| `docs/progress/README.md` | Updated with the new task and guide link. |

## 7. Interpretation

The project now has a durable data-pipeline guide and a progress record for its creation. The guide is intentionally practical: it includes commands, schemas, pseudocode, artifact tables, manual review checks, paper links, and known risks.

## 8. Successes

The task succeeded because the guide is connected to existing progress notes and explains the actual current code path rather than inventing a generic one.

## 9. Failures Or Limits

No code was changed and no dataset generation run was executed during this task. The guide documents the pipeline from inspection. A future task should run a small generation pilot and add real output counts, screenshots, review-table excerpts, and failure statistics.

## 10. External Works And Papers

| Work | Link | Core Objective | How We Used It |
| --- | --- | --- | --- |
| Character-level Convolutional Networks for Text Classification, Zhang et al. 2015 | [arXiv 1509.01626](https://arxiv.org/abs/1509.01626) | Establish large-scale text classification datasets including AG News experiments. | Cited as the paper source context for AG News. |
| Hugging Face Datasets | [Documentation](https://huggingface.co/docs/datasets) | Provide a standard dataset loading API. | Documented because the project uses `load_dataset`. |
| Sentence-BERT, Reimers and Gurevych 2019 | [arXiv 1908.10084](https://arxiv.org/abs/1908.10084) | Compare sentence meaning using embeddings. | Documented because optional semantic validation uses embedding cosine. |
| MultiNLI, Williams et al. 2018 | [arXiv 1704.05426](https://arxiv.org/abs/1704.05426) | Natural language inference for entailment and contradiction. | Documented because optional semantic validation can use NLI checks. |

## 11. Files Changed

| File | Change | Reason |
| --- | --- | --- |
| `docs/progress/guides/README.md` | Added reusable guide index. | Keep progress docs organized. |
| `docs/progress/guides/data_pipeline/raw_json_to_dataset_guide.md` | Added full data conversion guide. | Explain raw data to dataset process. |
| `docs/progress/2026-05-06_raw_json_to_dataset_guide.md` | Added task log. | Record this documentation action. |
| `docs/progress/README.md` | Added task and guide links. | Connect new document to the progress notebook. |

## 12. Follow-Up

- [ ] Run a 10-row generation pilot and record real artifact counts.
- [ ] Add dataset quality statistics: label counts, text lengths, banned-word counts, accepted/rejected ratio.
- [ ] Add cleaning rule for embedded artifacts such as `\\band` if confirmed across more rows.

