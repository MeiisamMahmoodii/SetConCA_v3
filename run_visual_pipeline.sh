#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR="${1:-}"
RUN_NAME="${2:-pilot_qwen_500img_v2views}"
LIMIT_IMAGES="${LIMIT_IMAGES:-500}"
LIMIT_MODELS="${LIMIT_MODELS:-2}"
LIMIT_LANGUAGES="${LIMIT_LANGUAGES:-4}"
LIMIT_VIEWS="${LIMIT_VIEWS:-8}"
MIN_VIEWS="${MIN_VIEWS:-48}"

if [[ -z "$IMAGE_DIR" ]]; then
  echo "Usage: ./run_visual_pipeline.sh /path/to/openimages/train [run_name]"
  exit 1
fi

mkdir -p \
  visual_grounded_dataset/data/manifests \
  visual_grounded_dataset/data/jobs \
  visual_grounded_dataset/data/responses \
  visual_grounded_dataset/data/sets \
  visual_grounded_dataset/data/controls/"$RUN_NAME" \
  visual_grounded_dataset/data/reports \
  visual_grounded_dataset/data/activations \
  visual_grounded_dataset/results

echo "[1/12] Build manifest"
uv run python visual_grounded_dataset/scripts/scan_image_folder.py \
  --image-dir "$IMAGE_DIR" \
  --out visual_grounded_dataset/data/manifests/${RUN_NAME}_manifest.jsonl \
  --source-dataset open_images_v7_train_partial \
  --topic-tags open_images_train_partial \
  --limit 10000

echo "[2/12] Render jobs"
uv run python visual_grounded_dataset/scripts/render_generation_jobs.py \
  --manifest visual_grounded_dataset/data/manifests/${RUN_NAME}_manifest.jsonl \
  --out visual_grounded_dataset/data/jobs/${RUN_NAME}_jobs.jsonl \
  --limit-images "$LIMIT_IMAGES" \
  --limit-models "$LIMIT_MODELS" \
  --limit-languages "$LIMIT_LANGUAGES" \
  --limit-views "$LIMIT_VIEWS"

echo "[3/12] Generate Qwen3"
uv run python visual_grounded_dataset/scripts/generate_with_vlm.py \
  --jobs visual_grounded_dataset/data/jobs/${RUN_NAME}_jobs.jsonl \
  --out visual_grounded_dataset/data/responses/${RUN_NAME}_qwen3_raw.jsonl \
  --backend transformers \
  --only-model-source qwen3_vl_4b \
  --continue-on-error \
  --resume

echo "[4/12] Generate Qwen2.5"
uv run python visual_grounded_dataset/scripts/generate_with_vlm.py \
  --jobs visual_grounded_dataset/data/jobs/${RUN_NAME}_jobs.jsonl \
  --out visual_grounded_dataset/data/responses/${RUN_NAME}_qwen2_5_raw.jsonl \
  --backend transformers \
  --only-model-source qwen2_5_vl_7b \
  --continue-on-error \
  --resume

echo "[5/12] Filter responses"
uv run python visual_grounded_dataset/scripts/filter_responses.py \
  --responses visual_grounded_dataset/data/responses/${RUN_NAME}_qwen3_raw.jsonl \
  --out visual_grounded_dataset/data/responses/${RUN_NAME}_qwen3_filtered.jsonl \
  --report visual_grounded_dataset/data/reports/${RUN_NAME}_qwen3_filter.md

uv run python visual_grounded_dataset/scripts/filter_responses.py \
  --responses visual_grounded_dataset/data/responses/${RUN_NAME}_qwen2_5_raw.jsonl \
  --out visual_grounded_dataset/data/responses/${RUN_NAME}_qwen2_5_filtered.jsonl \
  --report visual_grounded_dataset/data/reports/${RUN_NAME}_qwen2_5_filter.md

echo "[6/12] Merge filtered responses"
cat \
  visual_grounded_dataset/data/responses/${RUN_NAME}_qwen3_filtered.jsonl \
  visual_grounded_dataset/data/responses/${RUN_NAME}_qwen2_5_filtered.jsonl \
  > visual_grounded_dataset/data/responses/${RUN_NAME}_filtered_merged.jsonl

echo "[7/12] Build sets"
uv run python visual_grounded_dataset/scripts/build_sets.py \
  --responses visual_grounded_dataset/data/responses/${RUN_NAME}_filtered_merged.jsonl \
  --out visual_grounded_dataset/data/sets/${RUN_NAME}_sets.jsonl \
  --min-views "$MIN_VIEWS"

echo "[8/12] Audit and diversity filter"
uv run python visual_grounded_dataset/scripts/artifact_audit.py \
  --sets visual_grounded_dataset/data/sets/${RUN_NAME}_sets.jsonl \
  --out visual_grounded_dataset/data/reports/${RUN_NAME}_artifact_audit.md

uv run python visual_grounded_dataset/scripts/filter_sets_by_diversity.py \
  --sets visual_grounded_dataset/data/sets/${RUN_NAME}_sets.jsonl \
  --out visual_grounded_dataset/data/sets/${RUN_NAME}_sets_diverse.jsonl \
  --rejected-out visual_grounded_dataset/data/sets/${RUN_NAME}_sets_diversity_rejected.jsonl \
  --report visual_grounded_dataset/data/reports/${RUN_NAME}_diversity_filter.md

echo "[9/12] Build controls"
uv run python visual_grounded_dataset/scripts/build_controls.py \
  --sets visual_grounded_dataset/data/sets/${RUN_NAME}_sets_diverse.jsonl \
  --out-dir visual_grounded_dataset/data/controls/${RUN_NAME}

echo "[10/12] Convert real and controls for activation"
uv run python visual_grounded_dataset/scripts/convert_sets_for_activation.py \
  --sets visual_grounded_dataset/data/sets/${RUN_NAME}_sets_diverse.jsonl \
  --out visual_grounded_dataset/data/sets/${RUN_NAME}_sets_diverse_activation.jsonl

uv run python visual_grounded_dataset/scripts/convert_sets_for_activation.py \
  --sets visual_grounded_dataset/data/controls/${RUN_NAME}/controls_shuffled_image.jsonl \
  --out visual_grounded_dataset/data/controls/${RUN_NAME}/controls_shuffled_image_activation.jsonl

uv run python visual_grounded_dataset/scripts/convert_sets_for_activation.py \
  --sets visual_grounded_dataset/data/controls/${RUN_NAME}/controls_same_prompt_different_image.jsonl \
  --out visual_grounded_dataset/data/controls/${RUN_NAME}/controls_same_prompt_different_image_activation.jsonl

echo "[11/12] Extract activations"
uv run python scripts/extract_activation_bank.py \
  --sets visual_grounded_dataset/data/sets/${RUN_NAME}_sets_diverse_activation.jsonl \
  --out visual_grounded_dataset/data/activations/${RUN_NAME}_real_llama_layer_-1_s24.pt \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --layer -1 \
  --views 24 \
  --no-original \
  --batch-size 4 \
  --max-length 256 \
  --dtype bfloat16

uv run python scripts/extract_activation_bank.py \
  --sets visual_grounded_dataset/data/controls/${RUN_NAME}/controls_shuffled_image_activation.jsonl \
  --out visual_grounded_dataset/data/activations/${RUN_NAME}_shuffled_llama_layer_-1_s24.pt \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --layer -1 \
  --views 24 \
  --no-original \
  --batch-size 4 \
  --max-length 256 \
  --dtype bfloat16

uv run python scripts/extract_activation_bank.py \
  --sets visual_grounded_dataset/data/controls/${RUN_NAME}/controls_same_prompt_different_image_activation.jsonl \
  --out visual_grounded_dataset/data/activations/${RUN_NAME}_same_prompt_llama_layer_-1_s24.pt \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --layer -1 \
  --views 24 \
  --no-original \
  --batch-size 4 \
  --max-length 256 \
  --dtype bfloat16

echo "[12/12] Train SetConCA"
uv run python scripts/train_setconca_v2.py \
  --activations visual_grounded_dataset/data/activations/${RUN_NAME}_real_llama_layer_-1_s24.pt \
  --out-dir visual_grounded_dataset/results/${RUN_NAME}_real_train \
  --epochs 30 \
  --batch-size 16 \
  --concept-dim 128 \
  --topk 32

uv run python scripts/train_setconca_v2.py \
  --activations visual_grounded_dataset/data/activations/${RUN_NAME}_shuffled_llama_layer_-1_s24.pt \
  --out-dir visual_grounded_dataset/results/${RUN_NAME}_shuffled_train \
  --epochs 30 \
  --batch-size 16 \
  --concept-dim 128 \
  --topk 32

uv run python scripts/train_setconca_v2.py \
  --activations visual_grounded_dataset/data/activations/${RUN_NAME}_same_prompt_llama_layer_-1_s24.pt \
  --out-dir visual_grounded_dataset/results/${RUN_NAME}_same_prompt_train \
  --epochs 30 \
  --batch-size 16 \
  --concept-dim 128 \
  --topk 32

echo "Done."
echo "Reports:"
echo "  visual_grounded_dataset/data/reports/${RUN_NAME}_artifact_audit.md"
echo "  visual_grounded_dataset/data/reports/${RUN_NAME}_diversity_filter.md"
echo "Metrics:"
echo "  visual_grounded_dataset/results/${RUN_NAME}_real_train/metrics.json"
echo "  visual_grounded_dataset/results/${RUN_NAME}_shuffled_train/metrics.json"
echo "  visual_grounded_dataset/results/${RUN_NAME}_same_prompt_train/metrics.json"
