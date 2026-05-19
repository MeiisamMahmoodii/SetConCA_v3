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
- `qwen_vl`: Qwen2.5-VL path with `qwen-vl-utils` when available, otherwise
  generic pipeline fallback.
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

For Linux/zsh/bash environments, use:

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
DOWNLOAD_IMAGES=2000
DOWNLOAD_SPLIT=train
DOWNLOAD_PROCESSES=16
LIMIT_IMAGES=1000
PROGRESS_EVERY=100
```

To run the two Qwen VLM generators in parallel on two GPUs:

```bash
PARALLEL_VLMS=1 \
VLM_GPU_QWEN3=0 \
VLM_GPU_QWEN25=1 \
DOWNLOAD_IMAGES=1000 \
LIMIT_IMAGES=1000 \
./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_1000_v2views
```

This starts one Qwen3-VL process with `CUDA_VISIBLE_DEVICES=0` and one
Qwen2.5-VL process with `CUDA_VISIBLE_DEVICES=1`. Logs are written to:

```text
visual_grounded_dataset/data/reports/<run_name>_qwen3_generation.log
visual_grounded_dataset/data/reports/<run_name>_qwen2_5_generation.log
```

Watch them while the run is active with:

```bash
tail -f visual_grounded_dataset/data/reports/qwen_1000_v2views_qwen3_generation.log
tail -f visual_grounded_dataset/data/reports/qwen_1000_v2views_qwen2_5_generation.log
```

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
