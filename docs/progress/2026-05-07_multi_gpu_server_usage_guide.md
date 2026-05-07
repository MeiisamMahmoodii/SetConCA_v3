# Task: Multi-GPU Server Usage Guide

Tags: #progress #documentation #server #multi-gpu #vllm

Related notes: [[README]] [[PROJECT_GRAPH]] [[multi_gpu_server_usage_guide]] [[raw_json_to_dataset_guide]] [[2026-05-06_progress_update_protocol]]

## 1. Goal

The user asked to update all documents and add a multi-GPU server usage guide.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-07 |
| Workspace | `C:\Users\MPC\Documents\code\SetConCA\SetConCA_V2` |
| New guide | `docs/progress/guides/server/multi_gpu_server_usage_guide.md` |
| Main scripts inspected | `scripts/generate_constrained_sets.py`, `scripts/launch_dataset_generation.py`, `scripts/merge_generated_shards.py` |
| Main config inspected | `configs/rewrite_models.example.json` |
| Main result inspected | `data/generated/pilot_real_50/run_manifest.json` |

## 3. Hypothesis Or Rationale

The earlier documentation described single-process generation and did not fully explain the current server path. The code now supports `vllm`, model sharding with `--model-shard`, launcher-managed one-process-per-GPU runs, and shard merging. A dedicated guide is needed so server runs are reproducible and logged scientifically.

## 4. Actions

| Step | Action | Why | Result | Status |
| --- | --- | --- | --- | --- |
| 1 | Inspected project scripts and config. | Needed to document actual supported server behavior. | Found `--backend vllm`, `--model-shard`, launcher script, merge script, and vLLM config. | Succeeded |
| 2 | Inspected existing generated manifests. | Needed real baseline values. | Found 50-original CUDA pilot with 15702 attempts and 796 accepted rewrites. | Succeeded |
| 3 | Created server guide folder and guide. | Needed a clean categorized location. | Added [[multi_gpu_server_usage_guide]]. | Succeeded |
| 4 | Updated guide indexes. | Needed Obsidian navigation. | Updated `docs/progress/guides/README.md` and `docs/progress/README.md`. | Succeeded |
| 5 | Updated project map. | Needed top-level docs to show server generation and merge path. | Added server folder roles, diagrams, commands, tests/artifacts, and risks. | Succeeded |
| 6 | Updated main README. | Needed visible quick-start commands and accurate local instructions. | Added server/multi-GPU section, linked the detailed guide, corrected the stale input-file reference, and added an inside-`SetConCA_V2` test command. | Succeeded |
| 7 | Updated raw JSON to dataset guide. | Needed the data-pipeline guide to mention vLLM, sharding, and merging. | Added server scaling references and current pilot baseline. | Succeeded |

## 5. Code And Pseudocode

Current multi-GPU launcher logic:

```text
for gpu_idx in range(number_of_gpus):
    shard = f"{gpu_idx}/{number_of_gpus}"
    out_dir = base_out / f"shard_{gpu_idx}"
    CUDA_VISIBLE_DEVICES = gpu_idx
    run generate_constrained_sets.py --backend vllm --model-shard shard

if every shard exits successfully:
    run merge_generated_shards.py into base_out / "merged"
else:
    keep shard logs and do not merge
```

## 6. Results

| File | Change |
| --- | --- |
| `docs/progress/guides/server/README.md` | Added server guide index. |
| `docs/progress/guides/server/multi_gpu_server_usage_guide.md` | Added full multi-GPU server usage guide. |
| `docs/progress/guides/README.md` | Added link to server guide. |
| `docs/progress/guides/data_pipeline/raw_json_to_dataset_guide.md` | Added server scaling and current baseline sections. |
| `docs/PROJECT_GRAPH.md` | Updated project map for launcher, vLLM, sharding, merging, and server risks. |
| `README.md` | Added server quick-start section and corrected stale quick-start references. |
| `docs/progress/README.md` | Added task and guide links. |

## 7. Current Baseline Recorded

| Metric | Value |
| --- | ---: |
| Input | `data/raw/ag_news_train_full.jsonl` |
| Raw rows | 120000 |
| Pilot output | `data/generated/pilot_real_50` |
| Device | CUDA |
| GPU | NVIDIA GeForce RTX 3090 |
| Originals | 50 |
| Models | 10 |
| Attempts | 15702 |
| Accepted | 796 |
| Runtime | 18597.3 seconds, about 5.17 hours |

## 8. Interpretation

The project now has enough server documentation to run controlled paraphrase generation on a multi-GPU server. The recommended mode is one process per GPU with model shards, then merged outputs. Tensor parallelism is documented separately for cases where one model is too large for a single GPU.

## 9. Successes

The documentation succeeded because it is grounded in the actual implementation and observed manifests, not generic cluster advice.

## 10. Failures Or Limits

No server run was executed during this documentation update. The guide explains how to run on a server, but the commands were not validated on the current Windows workspace.

The initial broad file listing hit `.venv` and pytest cache permission issues. The follow-up scan narrowed to project folders and succeeded.

## 11. External Works And Papers

| Work | Link | Core Objective | How We Used It |
| --- | --- | --- | --- |
| vLLM / PagedAttention, Kwon et al. 2023 | [arXiv 2309.06180](https://arxiv.org/abs/2309.06180) | Improve LLM serving throughput using efficient KV-cache management. | Documented as the server rewrite-generation backend. |
| vLLM docs | [Installation](https://docs.vllm.ai/en/v0.6.6/getting_started/installation.html), [parallelism](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html) | Explain vLLM installation and tensor-parallel inference. | Used to explain server requirements and `tensor_parallel_size`. |
| PyTorch Distributed Data Parallel, Li et al. 2020 | [arXiv 2006.15704](https://arxiv.org/abs/2006.15704) | Describe distributed data-parallel training. | Added as future background for multi-GPU training, not current generation. |
| NVIDIA NCCL docs | [NVIDIA NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) | Explain multi-GPU communication primitives. | Added as troubleshooting background for tensor-parallel failures. |

## 12. Files Changed

| File | Change | Reason |
| --- | --- | --- |
| `README.md` | Added server quick-start and corrected stale input/test-command guidance. | Make common server commands visible and keep quick-start accurate. |
| `docs/PROJECT_GRAPH.md` | Updated map for server generation. | Keep top-level project map current. |
| `docs/progress/README.md` | Added guide and task links. | Keep notebook index current. |
| `docs/progress/guides/README.md` | Added server guide entry. | Keep guide index current. |
| `docs/progress/guides/data_pipeline/raw_json_to_dataset_guide.md` | Added server scaling and baseline references. | Connect data guide to server generation. |
| `docs/progress/guides/server/README.md` | Added server guide index. | Categorize server documentation. |
| `docs/progress/guides/server/multi_gpu_server_usage_guide.md` | Added multi-GPU guide. | Explain server usage. |
| `docs/progress/2026-05-07_multi_gpu_server_usage_guide.md` | Added task log. | Record this documentation update. |

## 13. Follow-Up

- [ ] Run a real multi-GPU vLLM pilot on the server.
- [ ] Record per-shard manifests, logs, acceptance rates, and merged manifest.
- [ ] Add a Slurm-specific section if the server uses Slurm.
- [ ] Add native distributed SetConCA training docs when training code supports it.
