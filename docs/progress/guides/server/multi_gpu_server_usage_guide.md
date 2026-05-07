# Guide: Multi-GPU Server Usage For SetConCA V2

Tags: #guide #server #multi-gpu #cuda #vllm #dataset-generation

Related notes: [[README]] [[PROJECT_GRAPH]] [[raw_json_to_dataset_guide]] [[2026-05-07_multi_gpu_server_usage_guide]]

## 1. Purpose

This guide explains how to run SetConCA V2 dataset generation on a CUDA server with one or more GPUs.

The current project uses multi-GPU in two practical ways:

| Mode | What It Does | Best For | Current Project Support |
| --- | --- | --- | --- |
| Model sharding across GPUs | Launches one process per GPU, each process runs a different shard of the model list. | Many rewrite models, each fitting on one GPU. | Supported by `scripts/launch_dataset_generation.py` and `--model-shard`. |
| vLLM tensor parallelism | Splits one large model across multiple GPUs. | One model is too large for a single GPU. | Supported through the `vllm.tensor_parallel_size` config value. |

Important: this is for rewrite dataset generation. It is not yet a native distributed SetConCA training guide. The training-side model code is present, but the current runnable server workflow is the constrained paraphrase generation pipeline.

## 2. Server Requirements

| Requirement | Why |
| --- | --- |
| Linux server or WSL2 with CUDA | vLLM is intended for Linux CUDA environments. |
| NVIDIA driver visible to PyTorch | Needed for `torch.cuda.is_available()`. |
| Python 3.12 if using this project as configured | `pyproject.toml` requires Python `>=3.12`; the optional `server` extra installs `vllm` only on Linux and Python `<3.13`. |
| Enough disk space for Hugging Face models | The enabled config includes multiple 1B-7B class models. |
| Hugging Face access where required | Some model IDs may require authentication or license acceptance. |

Check the server:

```bash
nvidia-smi
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

## 3. Install Server Dependencies

Recommended with `uv`:

```bash
cd SetConCA_V2
uv sync --extra server
```

Fallback with `pip` inside an activated environment:

```bash
pip install -e ".[server]"
```

If vLLM installation fails, check CUDA, Python, and Linux compatibility. vLLM is fast-moving and can be sensitive to CUDA/PyTorch wheel combinations.

## 4. Prepare Data

The full local raw source currently documented in the workspace is:

```text
data/raw/ag_news_train_full.jsonl
```

Its manifest records:

| Field | Value |
| --- | ---: |
| Dataset | `ag_news` |
| Split | `train` |
| Rows | 120000 |

To recreate it:

```bash
python scripts/download_news_dataset.py \
  --dataset ag_news \
  --split train \
  --limit 999999 \
  --out data/raw/ag_news_train_full.jsonl
```

## 5. Configure Models And vLLM

Config file:

```text
configs/rewrite_models.example.json
```

Important server fields:

```json
{
  "generation": {
    "max_new_tokens": 80,
    "temperature": 0.8,
    "top_p": 0.9,
    "do_sample": true,
    "num_return_sequences": 2,
    "max_attempts_per_slot": 5
  },
  "vllm": {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "batch_size": 128
  }
}
```

Meaning:

| Field | Meaning | Change When |
| --- | --- | --- |
| `tensor_parallel_size` | Number of GPUs used by one vLLM model instance. | Increase when one model does not fit on one GPU. |
| `gpu_memory_utilization` | Fraction of GPU memory vLLM may use. | Lower if you hit out-of-memory or share the server. |
| `batch_size` | Number of prompts per vLLM batch. | Lower if memory is tight; raise for throughput if stable. |
| `max_attempts_per_slot` | Retries per original/model/length slot. | Lower for quick pilots, raise for quality. |

## 6. Single-GPU Server Run

Use this first. It confirms the server environment, model access, CUDA, and output schema.

```bash
python scripts/launch_dataset_generation.py \
  --models-config configs/rewrite_models.example.json \
  --input data/raw/ag_news_train_full.jsonl \
  --out-dir data/generated/server_single_pilot \
  --backend vllm \
  --gpus 1 \
  --max-originals 10
```

Outputs:

```text
data/generated/server_single_pilot/
  attempts.jsonl
  accepted.jsonl
  sets.jsonl
  review_table.md
  run_manifest.json
```

## 7. Multi-GPU Model-Shard Run

This is the recommended first multi-GPU mode for this project.

Example for 4 GPUs:

```bash
python scripts/launch_dataset_generation.py \
  --models-config configs/rewrite_models.example.json \
  --input data/raw/ag_news_train_full.jsonl \
  --out-dir data/generated/server_vllm_4gpu \
  --backend vllm \
  --gpus 4 \
  --max-originals 1000
```

What happens:

```text
GPU 0 -> --model-shard 0/4 -> data/generated/server_vllm_4gpu/shard_0
GPU 1 -> --model-shard 1/4 -> data/generated/server_vllm_4gpu/shard_1
GPU 2 -> --model-shard 2/4 -> data/generated/server_vllm_4gpu/shard_2
GPU 3 -> --model-shard 3/4 -> data/generated/server_vllm_4gpu/shard_3
```

After all shards finish successfully, the launcher automatically calls:

```bash
python scripts/merge_generated_shards.py \
  --out-dir data/generated/server_vllm_4gpu/merged \
  data/generated/server_vllm_4gpu/shard_0 \
  data/generated/server_vllm_4gpu/shard_1 \
  data/generated/server_vllm_4gpu/shard_2 \
  data/generated/server_vllm_4gpu/shard_3
```

Merged output:

```text
data/generated/server_vllm_4gpu/merged/
  attempts.jsonl
  accepted.jsonl
  sets.jsonl
  review_table.md
  run_manifest.json
```

## 8. Manual Shard Run

Use this if a job scheduler such as Slurm should launch each GPU separately.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_constrained_sets.py \
  --models-config configs/rewrite_models.example.json \
  --input data/raw/ag_news_train_full.jsonl \
  --out-dir data/generated/server_manual/shard_0 \
  --backend vllm \
  --model-shard 0/4 \
  --max-originals 1000
```

Repeat for shards `1/4`, `2/4`, and `3/4`, changing both `CUDA_VISIBLE_DEVICES` and `--out-dir`.

Then merge:

```bash
python scripts/merge_generated_shards.py \
  --out-dir data/generated/server_manual/merged \
  data/generated/server_manual/shard_0 \
  data/generated/server_manual/shard_1 \
  data/generated/server_manual/shard_2 \
  data/generated/server_manual/shard_3
```

## 9. Tensor Parallel Run For One Large Model

If one model is too large for a single GPU, edit the config:

```json
{
  "vllm": {
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.90,
    "batch_size": 64
  }
}
```

Then run with enough visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/generate_constrained_sets.py \
  --models-config configs/rewrite_models.example.json \
  --input data/raw/ag_news_train_full.jsonl \
  --out-dir data/generated/server_tp2_pilot \
  --backend vllm \
  --max-originals 50
```

Use tensor parallelism carefully:

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| NCCL or process group error | GPU communication/runtime issue | Try `tensor_parallel_size: 1`, then debug server NCCL/CUDA. |
| Out of memory | Batch too large or model too large | Lower `batch_size`, lower `gpu_memory_utilization`, or increase tensor parallel size. |
| Low throughput | Batches too small or model loading dominates | Increase `batch_size` after a stable pilot. |

## 10. Monitoring

In another terminal:

```bash
watch -n 2 nvidia-smi
```

Watch shard logs:

```bash
tail -f data/generated/server_vllm_4gpu/logs/shard_0.log
```

Important manifest fields:

| Field | Meaning |
| --- | --- |
| `backend` | `hf` or `vllm`. |
| `model_shard` | Which model shard ran, for example `0/4`. |
| `n_models` | Number of models handled by this run or shard. |
| `n_models_total` | Number of enabled models before sharding. |
| `n_attempts` | Total generated candidates saved. |
| `n_accepted` | Accepted constrained rewrites. |
| `elapsed_s` | Runtime in seconds. |

## 11. Current Baseline From Local Pilot

The workspace contains a real pilot:

```text
data/generated/pilot_real_50/run_manifest.json
```

Observed result:

| Metric | Value |
| --- | ---: |
| Backend | Hugging Face |
| Device | CUDA |
| GPU | NVIDIA GeForce RTX 3090 |
| Originals | 50 |
| Models | 10 |
| Length bands | 4 |
| Attempts | 15702 |
| Accepted | 796 |
| Runtime | 18597.3 seconds, about 5.17 hours |

Interpretation:

The pilot proves the current schema and generation loop can complete on one GPU. It also shows why server scaling matters: full AG News has 120000 rows, so a naive single-GPU full run would be far too slow.

## 12. Failure Recovery

| Failure | What To Keep | What To Do |
| --- | --- | --- |
| One shard fails | Keep all shard folders and logs. | Fix the failing shard, rerun only that shard, then merge. |
| Merge fails | Keep shard outputs. | Run `merge_generated_shards.py` manually. |
| Bad review quality | Keep attempts and review table. | Tune prompt, banned-word logic, semantic validation, or model list. |
| OOM | Keep manifest and log. | Lower `batch_size`, lower `gpu_memory_utilization`, reduce `max_originals`, or shard fewer/larger models differently. |

## 13. Scientific Logging Rule

Every server run should produce or update a progress note with:

- Exact command.
- Git state if available.
- Server GPU names and count.
- Config file used.
- Input file and manifest.
- Output folder.
- `run_manifest.json` summary.
- Acceptance rate.
- Failure reasons from `attempts.jsonl`.
- Review-table observations.
- Any manual changes to prompt, model list, or validation.

## 14. External Works And Technology Sources

| Work Or Tool | Link | Core Objective | How We Use It |
| --- | --- | --- | --- |
| vLLM / PagedAttention, Kwon et al. 2023 | [arXiv 2309.06180](https://arxiv.org/abs/2309.06180) | Improve LLM serving throughput by reducing KV-cache memory waste with PagedAttention. | Used as the high-throughput backend for batched rewrite generation. |
| vLLM installation documentation | [vLLM docs](https://docs.vllm.ai/en/v0.6.6/getting_started/installation.html) | Explain Linux, Python, CUDA, and install requirements for vLLM. | Used to define server setup expectations and compatibility risks. |
| vLLM parallelism and scaling documentation | [vLLM docs](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html) | Explain tensor parallelism and multi-GPU inference settings. | Used to document `tensor_parallel_size` behavior. |
| PyTorch Distributed Data Parallel, Li et al. 2020 | [arXiv 2006.15704](https://arxiv.org/abs/2006.15704) | Describe PyTorch distributed data parallel training and communication overlap. | Not used directly in the current generation pipeline, but relevant for future SetConCA distributed training. |
| NVIDIA NCCL documentation | [NVIDIA docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) | Provide multi-GPU communication primitives used by distributed GPU libraries. | Used as background for diagnosing tensor-parallel/NCCL failures. |

## 15. What This Guide Does Not Cover Yet

- Native multi-GPU SetConCA training.
- Slurm script templates for a specific cluster.
- Multi-node vLLM/Ray setup.
- Automatic quality scoring beyond current constraints and optional semantic validation.

These should be added as new sections when the code or server environment actually needs them.

