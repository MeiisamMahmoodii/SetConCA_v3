#!/usr/bin/env bash
set -euo pipefail

# End-to-end visual-grounded dataset pipeline.
#
# Usage with existing images:
#   ./visual_grounded_dataset/scripts/run_visual_pipeline.sh /path/to/openimages/train run_name
#
# Usage with automatic Open Images download:
#   ./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" run_name
#
# Common overrides:
#   DOWNLOAD_IMAGES=1000 LIMIT_IMAGES=1000 ./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_1000_v2views
#   LIMIT_IMAGES=50 ./visual_grounded_dataset/scripts/run_visual_pipeline.sh /path/to/images smoke_v2views
#   PARALLEL_VLMS=1 VLM_GPU_QWEN3=0 VLM_GPU_QWEN25=1 ./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_1000_v2views

IMAGE_DIR="${1:-}"
RUN_NAME="${2:-pilot_qwen_1000img_v2views}"

DOWNLOAD_IMAGES="${DOWNLOAD_IMAGES:-1000}"
DOWNLOAD_SPLIT="${DOWNLOAD_SPLIT:-train}"
DOWNLOAD_PROCESSES="${DOWNLOAD_PROCESSES:-8}"
DOWNLOAD_SHUFFLE_SEED="${DOWNLOAD_SHUFFLE_SEED:-19}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-visual_grounded_dataset/data/images/openimages_${DOWNLOAD_SPLIT}_${DOWNLOAD_IMAGES}}"
OPENIMAGES_ID_CSV="${OPENIMAGES_ID_CSV:-visual_grounded_dataset/data/openimages/${DOWNLOAD_SPLIT}_image_ids.csv}"
OPENIMAGES_LIST="${OPENIMAGES_LIST:-visual_grounded_dataset/data/openimages/${DOWNLOAD_SPLIT}_${DOWNLOAD_IMAGES}_image_list.txt}"
OPENIMAGES_DOWNLOADER="${OPENIMAGES_DOWNLOADER:-visual_grounded_dataset/data/openimages/downloader.py}"

LIMIT_IMAGES="${LIMIT_IMAGES:-$DOWNLOAD_IMAGES}"
LIMIT_MODELS="${LIMIT_MODELS:-2}"
LIMIT_LANGUAGES="${LIMIT_LANGUAGES:-4}"
LIMIT_VIEWS="${LIMIT_VIEWS:-8}"
MIN_VIEWS="${MIN_VIEWS:-48}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
PARALLEL_VLMS="${PARALLEL_VLMS:-0}"
VLM_GPU_QWEN3="${VLM_GPU_QWEN3:-0}"
VLM_GPU_QWEN25="${VLM_GPU_QWEN25:-1}"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
LAYER="${LAYER:--1}"
VIEWS_FOR_ACTIVATION="${VIEWS_FOR_ACTIVATION:-24}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DTYPE="${DTYPE:-bfloat16}"
EPOCHS="${EPOCHS:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
CONCEPT_DIM="${CONCEPT_DIM:-128}"
TOPK="${TOPK:-32}"

run_qwen3_generation() {
  echo "[3/12] Generate Qwen3"
  echo "- gpu: ${CUDA_VISIBLE_DEVICES:-all visible}"
  uv run python visual_grounded_dataset/scripts/generate_with_vlm.py \
    --jobs "visual_grounded_dataset/data/jobs/${RUN_NAME}_jobs.jsonl" \
    --out "visual_grounded_dataset/data/responses/${RUN_NAME}_qwen3_raw.jsonl" \
    --backend transformers \
    --only-model-source qwen3_vl_4b \
    --continue-on-error \
    --resume \
    --progress-every "$PROGRESS_EVERY"
}

run_qwen25_generation() {
  echo "[4/12] Generate Qwen2.5"
  echo "- gpu: ${CUDA_VISIBLE_DEVICES:-all visible}"
  uv run python visual_grounded_dataset/scripts/generate_with_vlm.py \
    --jobs "visual_grounded_dataset/data/jobs/${RUN_NAME}_jobs.jsonl" \
    --out "visual_grounded_dataset/data/responses/${RUN_NAME}_qwen2_5_raw.jsonl" \
    --backend transformers \
    --only-model-source qwen2_5_vl_7b \
    --continue-on-error \
    --resume \
    --progress-every "$PROGRESS_EVERY"
}

mkdir -p \
  visual_grounded_dataset/data/openimages \
  visual_grounded_dataset/data/images \
  visual_grounded_dataset/data/manifests \
  visual_grounded_dataset/data/jobs \
  visual_grounded_dataset/data/responses \
  visual_grounded_dataset/data/sets \
  visual_grounded_dataset/data/controls/"$RUN_NAME" \
  visual_grounded_dataset/data/reports \
  visual_grounded_dataset/data/activations \
  visual_grounded_dataset/results

download_openimages_subset() {
  echo "[0/12] Download Open Images subset"
  echo "- split: $DOWNLOAD_SPLIT"
  echo "- requested images: $DOWNLOAD_IMAGES"
  echo "- download dir: $DOWNLOAD_DIR"
  echo "- processes: $DOWNLOAD_PROCESSES"

  mkdir -p "$(dirname "$OPENIMAGES_ID_CSV")" "$DOWNLOAD_DIR"

  if [[ ! -f "$OPENIMAGES_DOWNLOADER" ]]; then
    echo "- downloading Open Images downloader.py"
    curl -L \
      https://raw.githubusercontent.com/openimages/dataset/master/downloader.py \
      -o "$OPENIMAGES_DOWNLOADER"
  fi

  if [[ ! -f "$OPENIMAGES_ID_CSV" ]]; then
    echo "- downloading image id CSV"
    if [[ "$DOWNLOAD_SPLIT" == "train" ]]; then
      curl -L \
        https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv \
        -o "$OPENIMAGES_ID_CSV"
    elif [[ "$DOWNLOAD_SPLIT" == "validation" ]]; then
      curl -L \
        https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv \
        -o "$OPENIMAGES_ID_CSV"
    elif [[ "$DOWNLOAD_SPLIT" == "test" ]]; then
      curl -L \
        https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv \
        -o "$OPENIMAGES_ID_CSV"
    else
      echo "Unsupported DOWNLOAD_SPLIT=$DOWNLOAD_SPLIT. Use train, validation, or test." >&2
      exit 1
    fi
  fi

  echo "- building image list: $OPENIMAGES_LIST"
  uv run python - "$OPENIMAGES_ID_CSV" "$OPENIMAGES_LIST" "$DOWNLOAD_SPLIT" "$DOWNLOAD_IMAGES" "$DOWNLOAD_SHUFFLE_SEED" <<'PY'
import csv
import random
import sys
from pathlib import Path

csv_path, out_path, split, limit, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
ids = []
with open(csv_path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    field = "ImageID" if "ImageID" in (reader.fieldnames or []) else (reader.fieldnames or [""])[0]
    for row in reader:
        image_id = (row.get(field) or "").strip()
        if image_id:
            ids.append(image_id)
rng = random.Random(seed)
rng.shuffle(ids)
selected = ids[:limit]
target = Path(out_path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("\n".join(f"{split}/{image_id}" for image_id in selected) + "\n", encoding="utf-8")
print(f"selected {len(selected)} ids from {len(ids)} available")
PY

  echo "- installing downloader runtime deps if needed"
  uv pip install boto3 botocore tqdm >/dev/null

  echo "- downloading images"
  uv run python "$OPENIMAGES_DOWNLOADER" \
    "$OPENIMAGES_LIST" \
    --download_folder "$DOWNLOAD_DIR" \
    --num_processes "$DOWNLOAD_PROCESSES"

  IMAGE_DIR="$DOWNLOAD_DIR"
  echo "- image dir ready: $IMAGE_DIR"
}

if [[ -z "$IMAGE_DIR" ]]; then
  download_openimages_subset
elif [[ ! -d "$IMAGE_DIR" ]]; then
  echo "IMAGE_DIR does not exist: $IMAGE_DIR" >&2
  exit 1
else
  echo "[0/12] Using existing image dir: $IMAGE_DIR"
fi

echo "[1/12] Build manifest"
uv run python visual_grounded_dataset/scripts/scan_image_folder.py \
  --image-dir "$IMAGE_DIR" \
  --out "visual_grounded_dataset/data/manifests/${RUN_NAME}_manifest.jsonl" \
  --source-dataset "open_images_v7_${DOWNLOAD_SPLIT}_partial" \
  --topic-tags "open_images_${DOWNLOAD_SPLIT}_partial" \
  --limit "$LIMIT_IMAGES"

echo "[2/12] Render jobs"
uv run python visual_grounded_dataset/scripts/render_generation_jobs.py \
  --manifest "visual_grounded_dataset/data/manifests/${RUN_NAME}_manifest.jsonl" \
  --out "visual_grounded_dataset/data/jobs/${RUN_NAME}_jobs.jsonl" \
  --limit-images "$LIMIT_IMAGES" \
  --limit-models "$LIMIT_MODELS" \
  --limit-languages "$LIMIT_LANGUAGES" \
  --limit-views "$LIMIT_VIEWS"

if [[ "$PARALLEL_VLMS" == "1" ]]; then
  echo "[3-4/12] Generate Qwen3 and Qwen2.5 in parallel"
  echo "- qwen3 gpu: $VLM_GPU_QWEN3"
  echo "- qwen2.5 gpu: $VLM_GPU_QWEN25"
  echo "- qwen3 log: visual_grounded_dataset/data/reports/${RUN_NAME}_qwen3_generation.log"
  echo "- qwen2.5 log: visual_grounded_dataset/data/reports/${RUN_NAME}_qwen2_5_generation.log"
  (
    export CUDA_VISIBLE_DEVICES="$VLM_GPU_QWEN3"
    run_qwen3_generation
  ) > "visual_grounded_dataset/data/reports/${RUN_NAME}_qwen3_generation.log" 2>&1 &
  qwen3_pid=$!
  (
    export CUDA_VISIBLE_DEVICES="$VLM_GPU_QWEN25"
    run_qwen25_generation
  ) > "visual_grounded_dataset/data/reports/${RUN_NAME}_qwen2_5_generation.log" 2>&1 &
  qwen25_pid=$!

  set +e
  wait "$qwen3_pid"
  qwen3_status=$?
  wait "$qwen25_pid"
  qwen25_status=$?
  set -e

  if [[ "$qwen3_status" -ne 0 || "$qwen25_status" -ne 0 ]]; then
    echo "Parallel VLM generation failed: qwen3=$qwen3_status qwen2.5=$qwen25_status" >&2
    exit 1
  fi
else
  run_qwen3_generation
  run_qwen25_generation
fi

echo "[5/12] Filter responses"
uv run python visual_grounded_dataset/scripts/filter_responses.py \
  --responses "visual_grounded_dataset/data/responses/${RUN_NAME}_qwen3_raw.jsonl" \
  --out "visual_grounded_dataset/data/responses/${RUN_NAME}_qwen3_filtered.jsonl" \
  --report "visual_grounded_dataset/data/reports/${RUN_NAME}_qwen3_filter.md"

uv run python visual_grounded_dataset/scripts/filter_responses.py \
  --responses "visual_grounded_dataset/data/responses/${RUN_NAME}_qwen2_5_raw.jsonl" \
  --out "visual_grounded_dataset/data/responses/${RUN_NAME}_qwen2_5_filtered.jsonl" \
  --report "visual_grounded_dataset/data/reports/${RUN_NAME}_qwen2_5_filter.md"

echo "[6/12] Merge filtered responses"
cat \
  "visual_grounded_dataset/data/responses/${RUN_NAME}_qwen3_filtered.jsonl" \
  "visual_grounded_dataset/data/responses/${RUN_NAME}_qwen2_5_filtered.jsonl" \
  > "visual_grounded_dataset/data/responses/${RUN_NAME}_filtered_merged.jsonl"

echo "[7/12] Build sets"
uv run python visual_grounded_dataset/scripts/build_sets.py \
  --responses "visual_grounded_dataset/data/responses/${RUN_NAME}_filtered_merged.jsonl" \
  --out "visual_grounded_dataset/data/sets/${RUN_NAME}_sets.jsonl" \
  --min-views "$MIN_VIEWS"

echo "[8/12] Audit and diversity filter"
uv run python visual_grounded_dataset/scripts/artifact_audit.py \
  --sets "visual_grounded_dataset/data/sets/${RUN_NAME}_sets.jsonl" \
  --out "visual_grounded_dataset/data/reports/${RUN_NAME}_artifact_audit.md"

uv run python visual_grounded_dataset/scripts/filter_sets_by_diversity.py \
  --sets "visual_grounded_dataset/data/sets/${RUN_NAME}_sets.jsonl" \
  --out "visual_grounded_dataset/data/sets/${RUN_NAME}_sets_diverse.jsonl" \
  --rejected-out "visual_grounded_dataset/data/sets/${RUN_NAME}_sets_diversity_rejected.jsonl" \
  --report "visual_grounded_dataset/data/reports/${RUN_NAME}_diversity_filter.md"

echo "[9/12] Build controls"
uv run python visual_grounded_dataset/scripts/build_controls.py \
  --sets "visual_grounded_dataset/data/sets/${RUN_NAME}_sets_diverse.jsonl" \
  --out-dir "visual_grounded_dataset/data/controls/${RUN_NAME}"

echo "[10/12] Convert real and controls for activation"
uv run python visual_grounded_dataset/scripts/convert_sets_for_activation.py \
  --sets "visual_grounded_dataset/data/sets/${RUN_NAME}_sets_diverse.jsonl" \
  --out "visual_grounded_dataset/data/sets/${RUN_NAME}_sets_diverse_activation.jsonl"

uv run python visual_grounded_dataset/scripts/convert_sets_for_activation.py \
  --sets "visual_grounded_dataset/data/controls/${RUN_NAME}/controls_shuffled_image.jsonl" \
  --out "visual_grounded_dataset/data/controls/${RUN_NAME}/controls_shuffled_image_activation.jsonl"

uv run python visual_grounded_dataset/scripts/convert_sets_for_activation.py \
  --sets "visual_grounded_dataset/data/controls/${RUN_NAME}/controls_same_prompt_different_image.jsonl" \
  --out "visual_grounded_dataset/data/controls/${RUN_NAME}/controls_same_prompt_different_image_activation.jsonl"

echo "[11/12] Extract activations"
uv run python scripts/extract_activation_bank.py \
  --sets "visual_grounded_dataset/data/sets/${RUN_NAME}_sets_diverse_activation.jsonl" \
  --out "visual_grounded_dataset/data/activations/${RUN_NAME}_real_llama_layer_${LAYER}_s${VIEWS_FOR_ACTIVATION}.pt" \
  --model-id "$MODEL_ID" \
  --layer "$LAYER" \
  --views "$VIEWS_FOR_ACTIVATION" \
  --no-original \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --dtype "$DTYPE"

uv run python scripts/extract_activation_bank.py \
  --sets "visual_grounded_dataset/data/controls/${RUN_NAME}/controls_shuffled_image_activation.jsonl" \
  --out "visual_grounded_dataset/data/activations/${RUN_NAME}_shuffled_llama_layer_${LAYER}_s${VIEWS_FOR_ACTIVATION}.pt" \
  --model-id "$MODEL_ID" \
  --layer "$LAYER" \
  --views "$VIEWS_FOR_ACTIVATION" \
  --no-original \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --dtype "$DTYPE"

uv run python scripts/extract_activation_bank.py \
  --sets "visual_grounded_dataset/data/controls/${RUN_NAME}/controls_same_prompt_different_image_activation.jsonl" \
  --out "visual_grounded_dataset/data/activations/${RUN_NAME}_same_prompt_llama_layer_${LAYER}_s${VIEWS_FOR_ACTIVATION}.pt" \
  --model-id "$MODEL_ID" \
  --layer "$LAYER" \
  --views "$VIEWS_FOR_ACTIVATION" \
  --no-original \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --dtype "$DTYPE"

echo "[12/12] Train SetConCA"
uv run python scripts/train_setconca_v2.py \
  --activations "visual_grounded_dataset/data/activations/${RUN_NAME}_real_llama_layer_${LAYER}_s${VIEWS_FOR_ACTIVATION}.pt" \
  --out-dir "visual_grounded_dataset/results/${RUN_NAME}_real_train" \
  --epochs "$EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --concept-dim "$CONCEPT_DIM" \
  --topk "$TOPK"

uv run python scripts/train_setconca_v2.py \
  --activations "visual_grounded_dataset/data/activations/${RUN_NAME}_shuffled_llama_layer_${LAYER}_s${VIEWS_FOR_ACTIVATION}.pt" \
  --out-dir "visual_grounded_dataset/results/${RUN_NAME}_shuffled_train" \
  --epochs "$EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --concept-dim "$CONCEPT_DIM" \
  --topk "$TOPK"

uv run python scripts/train_setconca_v2.py \
  --activations "visual_grounded_dataset/data/activations/${RUN_NAME}_same_prompt_llama_layer_${LAYER}_s${VIEWS_FOR_ACTIVATION}.pt" \
  --out-dir "visual_grounded_dataset/results/${RUN_NAME}_same_prompt_train" \
  --epochs "$EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --concept-dim "$CONCEPT_DIM" \
  --topk "$TOPK"

echo "Done."
echo "Reports:"
echo "  visual_grounded_dataset/data/reports/${RUN_NAME}_artifact_audit.md"
echo "  visual_grounded_dataset/data/reports/${RUN_NAME}_diversity_filter.md"
echo "Metrics:"
echo "  visual_grounded_dataset/results/${RUN_NAME}_real_train/metrics.json"
echo "  visual_grounded_dataset/results/${RUN_NAME}_shuffled_train/metrics.json"
echo "  visual_grounded_dataset/results/${RUN_NAME}_same_prompt_train/metrics.json"
