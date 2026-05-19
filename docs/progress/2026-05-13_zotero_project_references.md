# Task: Zotero Project References

Tags: #progress #references #zotero #bibliography

Related notes: [[README]] [[DATASET_PROJECT_PLAN]] [[EXPERIMENT_REPORT]]

## 1. Goal

The user asked to find all references and papers that the SetConCA V2 project uses or builds on, then add them to Zotero for later citation.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-13 |
| Workspace | `C:\Users\MPC\Documents\code\SetConCA\SetConCA_V2` |
| Primary sources scanned | `docs`, `src`, `scripts` |
| Zotero route | Local Zotero connector server at `127.0.0.1:23119` |

## 3. Actions

| Step | Action | Result | Status |
| --- | --- | --- | --- |
| 1 | Searched project docs and code for URLs, model IDs, dataset IDs, citation-like sections, and paper names. | Found explicit references in `docs/DATASET_PROJECT_PLAN.md`, progress guides, experiment report, and server/data-pipeline notes. | Succeeded |
| 2 | Checked whether the requested Zotero plugin exposed callable tools. | No callable Zotero plugin tools were discoverable, but Zotero desktop was running locally. | Partial |
| 3 | Tested Zotero local connector. | `http://127.0.0.1:23119/connector/ping` responded, and `/connector/saveItems` accepted saves with HTTP 201. | Succeeded |
| 4 | Added project references to Zotero. | Added 33 real project references: papers, dataset cards, model cards, benchmarks, and tooling docs. | Succeeded |
| 5 | Wrote repo-side fallback files. | Added an audit JSON and BibTeX import file under `docs/references`. | Succeeded |

## 4. References Added

The Zotero batch included:

- Core papers: AG News / Character-level CNN, MultiNLI, SBERT, GPT-3, vLLM/PagedAttention, PAWS, TruthfulQA, CAA, and RepE.
- Dataset and benchmark references: HotpotQA, FEVER, FEVEROUS, SNLI, MultiNLI, AllNLI, Quora Duplicates, BBQ, SycophancyEval, RealToxicityPrompts, and MTEB.
- Tooling references: Hugging Face Datasets, vLLM installation/scaling docs, and NVIDIA NCCL docs.
- Model cards used in the experiment grid: Llama 3.2 1B/3B, Llama 3.1 8B, Qwen3 0.6B/4B/8B, and Gemma 3 1B/4B/12B.

## 5. Output Files

| File | Purpose |
| --- | --- |
| `docs/references/setconca_project_references_zotero_audit.json` | Machine-readable audit of Zotero save status and all references. |
| `docs/references/setconca_project_references.bib` | BibTeX fallback/import file for Zotero or LaTeX. |

## 6. Notes

One temporary Zotero status-probe webpage item was created while verifying the local connector response code. It is titled `Zotero import status probe - do not keep` and can be deleted manually from Zotero.

The audit file records source project files for each reference in the Zotero `extra` field and in the JSON payload.
