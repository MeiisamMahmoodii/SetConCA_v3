# Visual-Grounded Set Dataset

This folder is intentionally isolated from the main SetConCA V2 code. It builds a
visual-grounded semantic-set dataset where the image is the latent anchor and
text views are generated independently by vision-language models.

## Goal

Create SetConCA-compatible latent sets that are less tied to surface text
artifacts than paraphrase-only sets.

One set should mean:

```text
same image -> many independent descriptions
```

The nuisance variables should vary inside each set:

```text
VLM model, language, prompt perspective, style, length
```

The intended invariant is the visual scene itself: objects, relations, actions,
setting, mood, and contextual facts visible or strongly implied by the image.

## Dataset Choice

Use a stratified image pool rather than XM3600 as the main source.

- Primary: Open Images V7, because it is large and diverse across objects,
  scenes, people, activities, boxes, relationships, and localized narratives.
- Relationship add-on: Visual Genome, because it is dense in objects,
  attributes, relationships, region descriptions, and scene graphs.
- Small baseline: COCO, because it is familiar and useful for sanity checks.
- Optional multilingual benchmark: XM3600, kept for evaluation only, not as the
  main image source.

The v1 target is:

```text
10,000 Open Images anchors
2,000 Visual Genome relationship-heavy anchors
1,000 COCO sanity-check anchors
```

## Pipeline

1. Build an image manifest with topic tags.
2. Render generation jobs across models, languages, and prompt views.
3. Generate VLM descriptions from the image itself. Do not show original captions.
4. Filter wrong-language, refusal, empty, duplicate, and metadata-leaking outputs.
5. Group filtered responses into SetConCA-compatible latent sets.
6. Build matched control sets.
7. Run artifact audits before using the data for training claims.

## Real VLM Backends

The generator supports:

- `mock`: fast pipeline test, no model download.
- `transformers`: real local VLM generation.

Configured real adapters:

- `generic_pipeline`: Hugging Face `pipeline("image-text-to-text")`.
- `qwen_vl`: direct Qwen2.5-VL/Qwen3-VL path with `qwen-vl-utils` when
  available, otherwise generic pipeline fallback.
- `internvl_chat`: InternVL chat API with local image tiling.
- `gemma4_multimodal`: Gemma 4 multimodal API.

Real generation needs accepted Hugging Face model licenses where required and
enough GPU/VRAM. Start one model at a time:

Install VLM extras first:

```powershell
uv pip install -r visual_grounded_dataset\requirements-vlm.txt
```

If Qwen3-VL raises `Qwen3VLVideoProcessor requires the Torchvision library`,
install a torchvision wheel compatible with the active torch build, then rerun.

```powershell
python visual_grounded_dataset\scripts\generate_with_vlm.py `
  --jobs visual_grounded_dataset\data\jobs\pilot_jobs.jsonl `
  --out visual_grounded_dataset\data\responses\pilot_qwen3_4b_raw.jsonl `
  --backend transformers `
  --only-model-source qwen3_vl_4b `
  --limit 20 `
  --continue-on-error
```

Then scale the same command by removing `--limit`.

### Local Single-Model Generation

The local/manual command path is intentionally conservative. If you omit
`--batch-size`, it defaults to `1`, and Qwen uses the original one-image
`qwen-vl-utils` flow. This is the safest path for Windows/local runs and should
continue to work with existing raw response files and `--resume`.

Example for Qwen2.5-VL:

```powershell
uv run python visual_grounded_dataset\scripts\generate_with_vlm.py `
  --jobs visual_grounded_dataset\data\jobs\pilot_qwen_500img_v2views_jobs.jsonl `
  --out visual_grounded_dataset\data\responses\pilot_qwen2_5_500img_v2views_raw.jsonl `
  --backend transformers `
  --only-model-source qwen2_5_vl_7b `
  --continue-on-error `
  --resume
```

Optional speed knobs for machines where you have already verified the model
fits:

```powershell
  --batch-size 2
  --device-map single
```

Use `--device-map single` only when the selected process should keep the model
on one visible GPU. The default remains `auto`.

## Folder Layout

```text
configs/
  datasets.json
  generation.json
  languages.json
  models.json
  views.json
scripts/
  artifact_audit.py
  build_controls.py
  build_manifest.py
  build_sets.py
  convert_sets_for_activation.py
  filter_responses.py
  filter_sets_by_diversity.py
  generate_with_vlm.py
  render_generation_jobs.py
  scan_image_folder.py
  stratified_sample_manifest.py
  vg_common.py
data/
  manifests/
  jobs/
  responses/
  sets/
  controls/
  reports/
docs/
  GOAL_AND_PLAN.md
```

## Minimal Dry Run

Create a tiny CSV:

```csv
image_id,image_path,source_dataset,topic_tags
demo_1,C:\path\to\image1.jpg,local_demo,people|outdoor
demo_2,C:\path\to\image2.jpg,local_demo,animals|indoor
```

Or scan an already downloaded image folder:

```powershell
python visual_grounded_dataset\scripts\scan_image_folder.py `
  --image-dir D:\openimages\train `
  --out visual_grounded_dataset\data\manifests\openimages_train_partial_manifest.jsonl `
  --source-dataset open_images_v7_train_partial `
  --topic-tags open_images_train_partial `
  --limit 10000
```

Then run:

```powershell
python visual_grounded_dataset\scripts\build_manifest.py `
  --input-csv tiny_images.csv `
  --out visual_grounded_dataset\data\manifests\tiny_manifest.jsonl

# Optional for real runs: sample a balanced topic subset from a larger manifest.
python visual_grounded_dataset\scripts\stratified_sample_manifest.py `
  --manifest visual_grounded_dataset\data\manifests\full_manifest.jsonl `
  --out visual_grounded_dataset\data\manifests\balanced_manifest.jsonl `
  --per-topic 500 `
  --max-total 13000

python visual_grounded_dataset\scripts\render_generation_jobs.py `
  --manifest visual_grounded_dataset\data\manifests\tiny_manifest.jsonl `
  --out visual_grounded_dataset\data\jobs\tiny_jobs.jsonl `
  --limit-images 2 `
  --limit-models 1 `
  --limit-languages 2 `
  --limit-views 2

python visual_grounded_dataset\scripts\generate_with_vlm.py `
  --jobs visual_grounded_dataset\data\jobs\tiny_jobs.jsonl `
  --out visual_grounded_dataset\data\responses\tiny_raw.jsonl `
  --backend mock

python visual_grounded_dataset\scripts\filter_responses.py `
  --responses visual_grounded_dataset\data\responses\tiny_raw.jsonl `
  --out visual_grounded_dataset\data\responses\tiny_filtered.jsonl `
  --report visual_grounded_dataset\data\reports\tiny_filter_report.md

python visual_grounded_dataset\scripts\build_sets.py `
  --responses visual_grounded_dataset\data\responses\tiny_filtered.jsonl `
  --out visual_grounded_dataset\data\sets\tiny_sets.jsonl `
  --min-views 2

python visual_grounded_dataset\scripts\build_controls.py `
  --sets visual_grounded_dataset\data\sets\tiny_sets.jsonl `
  --out-dir visual_grounded_dataset\data\controls

python visual_grounded_dataset\scripts\artifact_audit.py `
  --sets visual_grounded_dataset\data\sets\tiny_sets.jsonl `
  --out visual_grounded_dataset\data\reports\tiny_artifact_audit.md
```

## Linux End-To-End Script

For Linux/zsh/bash environments, use the phase-based runner:

```bash
chmod +x visual_grounded_dataset/scripts/run_visual_pipeline.sh
DOWNLOAD_IMAGES=1000 LIMIT_IMAGES=1000 \
  ./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_1000_v2views
```

Passing an empty image directory (`""`) makes the script download a configurable
Open Images subset first. To use an existing image folder instead:

```bash
LIMIT_IMAGES=500 \
  ./visual_grounded_dataset/scripts/run_visual_pipeline.sh /path/to/openimages/train qwen_500_v2views
```

Useful overrides:

```bash
DATA_DIR=visual_grounded_dataset/data_server
VLM_BACKEND=vllm_openai
DOWNLOAD_IMAGES=2000
DOWNLOAD_SPLIT=train
DOWNLOAD_PROCESSES=16
LIMIT_IMAGES=1000
PROGRESS_EVERY=100
VLM_BATCH_SIZE=4
VLM_DEVICE_MAP=single
```

### Runner Phases

The runner can execute the whole pipeline or only selected phases:

```bash
PHASES=images,manifest,jobs,generate,filter,sets,export,controls,activations,train
```

The server default is:

```bash
PHASES=images,manifest,jobs,generate,filter,sets,export
```

That means the server produces one portable dataset file and stops before local
activation extraction or SetConCA training:

```text
visual_grounded_dataset/data_server/export/<run_name>_dataset.jsonl
```

Useful recovery examples:

```bash
# Reuse downloaded images and responses; rebuild filtered files and sets.
DATA_DIR=visual_grounded_dataset/data_server \
PHASES=filter,sets,export \
./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_500_v2views

# Rerun only Qwen3 generation, then rebuild downstream.
DATA_DIR=visual_grounded_dataset/data_server \
VLM_MODELS=qwen3_vl_4b \
PHASES=generate,filter,sets,export \
./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_500_v2views

# Build a one-model Qwen2.5 dataset from already filtered responses.
DATA_DIR=visual_grounded_dataset/data_server \
VLM_MODELS=qwen2_5_vl_7b \
MIN_VIEWS=24 \
PHASES=filter,sets,export \
./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_500_v2views
```

### Server Generation Modes

#### vLLM Backend

vLLM supports the current Qwen2.5-VL and Qwen3-VL model families through its
OpenAI-compatible multimodal API. Start one vLLM server per model, then point
the runner at those servers.

Example, two terminals:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  ./visual_grounded_dataset/scripts/start_vllm_server.sh qwen3_vl_4b 8001 2
```

```bash
CUDA_VISIBLE_DEVICES=2,3 \
  ./visual_grounded_dataset/scripts/start_vllm_server.sh qwen2_5_vl_7b 8002 2
```

The helper currently knows these model source IDs:

```text
qwen3_vl_4b
qwen2_5_vl_7b
internvl3_8b
gemma4_e4b_it
gemma3_4b_it
aya_vision_8b
paligemma2_3b_mix
qwen3_vl_8b
```

Then generate the portable dataset:

```bash
DATA_DIR=visual_grounded_dataset/data_server \
VLM_BACKEND=vllm_openai \
VLM_MODELS=qwen3_vl_4b,qwen2_5_vl_7b \
VLLM_BASE_URL_QWEN3=http://127.0.0.1:8001/v1 \
VLLM_BASE_URL_QWEN2_5=http://127.0.0.1:8002/v1 \
VLM_BATCH_SIZE=16 \
VLLM_REQUEST_CONCURRENCY=16 \
PARALLEL_MODELS=1 \
PHASES=images,manifest,jobs,generate,filter,sets,export \
./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_500_v2views
```

`VLM_BATCH_SIZE` and `VLLM_REQUEST_CONCURRENCY` control how many independent
requests the client sends concurrently. Each generated view is still a separate
request and remains a separate dataset row.

When `VLM_BACKEND=vllm_openai`, `VLM_SHARD=auto` uses one client process per
model because the vLLM server already handles request batching. Set
`VLM_SHARD=1` only if you deliberately want multiple client processes per
model server.

**Default server run:** auto-detect visible GPUs, split each model's jobs across
those GPUs, run Qwen3 first, then Qwen2.5. Each worker keeps one model loaded
for its job shard. Dependency installation is not repeated unless
`INSTALL_VLM_DEPS=1`; dependency checking is controlled by `CHECK_VLM_DEPS`.

```bash
./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_500_v2views
```

**Exclude a busy GPU** (e.g. shared with another user):

```bash
VLM_EXCLUDE_GPUS=3 VLM_GPUS=0,1,2 VLM_BATCH_SIZE=8 \
  ./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_500_v2views
```

**Both models at once** (2 GPUs assigned to each model, data-parallel job
shards):

```bash
PARALLEL_MODELS=1 \
VLM_DEVICE_MAP=single \
VLM_BATCH_SIZE=4 \
VLM_GPUS=0,1,2,3 \
./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_500_v2views
```

`VLM_DEVICE_MAP=single` keeps each worker on one visible GPU (faster when the
model fits). If you hit OOM, lower `VLM_BATCH_SIZE` first, then try
`VLM_DEVICE_MAP=auto` as a slower fallback.

**One process per model, no job sharding:**

```bash
PARALLEL_MODELS=1 VLM_SHARD=0 VLM_GPUS=0,1 \
  ./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_1000_v2views
```

Shard logs: `visual_grounded_dataset/data/reports/<run_name>_qwen3_gpu0.log`,
etc. Merged outputs: `<run_name>_qwen3_raw.jsonl`,
`<run_name>_qwen2_5_raw.jsonl`.

```bash
tail -f visual_grounded_dataset/data/reports/qwen_500_v2views_qwen3_gpu0.log
```

### Generation Troubleshooting

If the log repeatedly prints `Loading model adapter` and `Loading weights`, the
model setup is failing and being retried. Current `generate_with_vlm.py` treats
model-load failure as fatal so the real error is visible immediately. Per-image
generation failures can still be recorded with `--continue-on-error`.

A healthy worker log should show one model-load block, then progress:

```text
Loading model adapter: qwen2_5_vl_7b (...)
Loaded qwen2_5_vl_7b in ...
progress ...
```

### Local Use After Server Generation

Move only the portable dataset JSONL back to the local PC:

```text
visual_grounded_dataset/data_server/export/qwen_500_v2views_dataset.jsonl
```

Place it anywhere convenient, for example:

```text
visual_grounded_dataset/data/sets/qwen_500_v2views_dataset.jsonl
```

Then continue locally with controls, activation conversion, activation
extraction, and SetConCA training. The image files are not needed for those
steps because the dataset file already contains the generated text views.

## Causal/Artifact Claim

This dataset cannot prove causality from data alone. It can support a causal
identification argument under explicit assumptions:

```text
I = image / visual scene
M = VLM model
L = language
P = prompt perspective
T = generated text
H = target LLM activation
Z = SetConCA concept code
```

If `M`, `L`, and `P` vary inside each positive set while `I` is fixed, and
matched controls preserve `M/L/P` distributions while changing `I`, then a
signal that survives those controls is more likely image-concept signal than
surface artifact signal.

Required controls:

- Real same-image sets.
- Shuffled-image sets with matched language/model/prompt distribution.
- Same-language different-image controls.
- Same-prompt different-image controls.
- Same-model different-image controls.
- Ablations for one model, one language, one prompt family.

## Activation Extraction Compatibility

The main SetConCA activation extractor currently expects the older paraphrase
shape with `original_text` and `rewrites`. Convert visual latent sets before
activation extraction:

```powershell
python visual_grounded_dataset\scripts\convert_sets_for_activation.py `
  --sets visual_grounded_dataset\data\sets\pilot_qwen_100img_sets.jsonl `
  --out visual_grounded_dataset\data\sets\pilot_qwen_100img_sets_activation.jsonl
```

Then run the extractor with `--no-original` so all sampled views come from the
visual descriptions:

```powershell
python scripts\extract_activation_bank.py `
  --sets visual_grounded_dataset\data\sets\pilot_qwen_100img_sets_activation.jsonl `
  --out visual_grounded_dataset\data\activations\pilot_qwen_100img_real_layer_-1_s24.pt `
  --model-id meta-llama/Llama-3.2-1B-Instruct `
  --layer -1 `
  --views 24 `
  --no-original
```
