#!/usr/bin/env bash
set -euo pipefail

# Visual-grounded dataset pipeline runner.
#
# Usage:
#   ./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_500_v2views
#   DATA_DIR=visual_grounded_dataset/data_server PHASES=filter,sets,export ./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_500_v2views
#
# The script is intentionally phase based. Generated files are resumable; rerun
# only the phase you need instead of starting from image download again.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IMAGE_DIR="${1:-}"
RUN_NAME="${2:-qwen_500_v2views}"

# Storage and run shape.
DATA_DIR="${DATA_DIR:-visual_grounded_dataset/data}"
PHASES="${PHASES:-images,manifest,jobs,generate,filter,sets,export}"
RUN_TRAIN="${RUN_TRAIN:-0}"

# Open Images download.
DOWNLOAD_IMAGES="${DOWNLOAD_IMAGES:-500}"
DOWNLOAD_SPLIT="${DOWNLOAD_SPLIT:-train}"
DOWNLOAD_PROCESSES="${DOWNLOAD_PROCESSES:-8}"
DOWNLOAD_SHUFFLE_SEED="${DOWNLOAD_SHUFFLE_SEED:-19}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$DATA_DIR/images/openimages_${DOWNLOAD_SPLIT}_${DOWNLOAD_IMAGES}}"
OPENIMAGES_ID_CSV="${OPENIMAGES_ID_CSV:-$DATA_DIR/openimages/${DOWNLOAD_SPLIT}_image_ids.csv}"
OPENIMAGES_LIST="${OPENIMAGES_LIST:-$DATA_DIR/openimages/${DOWNLOAD_SPLIT}_${DOWNLOAD_IMAGES}_image_list.txt}"
OPENIMAGES_DOWNLOADER="${OPENIMAGES_DOWNLOADER:-$DATA_DIR/openimages/downloader.py}"

# Dataset size.
LIMIT_IMAGES="${LIMIT_IMAGES:-$DOWNLOAD_IMAGES}"
LIMIT_MODELS="${LIMIT_MODELS:-2}"
LIMIT_LANGUAGES="${LIMIT_LANGUAGES:-4}"
LIMIT_VIEWS="${LIMIT_VIEWS:-8}"
MIN_VIEWS="${MIN_VIEWS:-48}"

# VLM generation.
VLM_MODELS="${VLM_MODELS:-qwen3_vl_4b,qwen2_5_vl_7b}"
VLM_BACKEND="${VLM_BACKEND:-transformers}"
VLM_GPUS="${VLM_GPUS:-auto}"
VLM_EXCLUDE_GPUS="${VLM_EXCLUDE_GPUS:-}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-4}"
VLM_DTYPE="${VLM_DTYPE:-bfloat16}"
VLM_DEVICE_MAP="${VLM_DEVICE_MAP:-single}"
VLM_SHARD="${VLM_SHARD:-auto}"
PARALLEL_MODELS="${PARALLEL_MODELS:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
INSTALL_VLM_DEPS="${INSTALL_VLM_DEPS:-0}"
CHECK_VLM_DEPS="${CHECK_VLM_DEPS:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-180}"
VLLM_REQUEST_CONCURRENCY="${VLLM_REQUEST_CONCURRENCY:-16}"
VLLM_BASE_URL_QWEN3="${VLLM_BASE_URL_QWEN3:-}"
VLLM_BASE_URL_QWEN2_5="${VLLM_BASE_URL_QWEN2_5:-}"

# Activation and SetConCA training.
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
LAYER="${LAYER:--1}"
VIEWS_FOR_ACTIVATION="${VIEWS_FOR_ACTIVATION:-24}"
ACTIVATION_BATCH_SIZE="${ACTIVATION_BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-256}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-bfloat16}"
EPOCHS="${EPOCHS:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
CONCEPT_DIM="${CONCEPT_DIM:-128}"
TOPK="${TOPK:-32}"

GEN_SCRIPT="visual_grounded_dataset/scripts/generate_with_vlm.py"
VLM_REQUIREMENTS="visual_grounded_dataset/requirements-vlm.txt"

MANIFEST="$DATA_DIR/manifests/${RUN_NAME}_manifest.jsonl"
JOBS="$DATA_DIR/jobs/${RUN_NAME}_jobs.jsonl"
RESPONSES_DIR="$DATA_DIR/responses"
SETS_DIR="$DATA_DIR/sets"
CONTROLS_DIR="$DATA_DIR/controls/$RUN_NAME"
REPORTS_DIR="$DATA_DIR/reports"
ACTIVATIONS_DIR="$DATA_DIR/activations"
EXPORT_DIR="$DATA_DIR/export"
RESULTS_DIR="visual_grounded_dataset/results"
EXPORT_DATASET="${EXPORT_DATASET:-$EXPORT_DIR/${RUN_NAME}_dataset.jsonl}"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

contains_phase() {
  local phase="$1"
  [[ "$PHASES" == "all" || ",$PHASES," == *",$phase,"* ]]
}

ensure_dirs() {
  mkdir -p \
    "$DATA_DIR/openimages" \
    "$DATA_DIR/images" \
    "$DATA_DIR/manifests" \
    "$DATA_DIR/jobs" \
    "$RESPONSES_DIR" \
    "$SETS_DIR" \
    "$CONTROLS_DIR" \
    "$REPORTS_DIR" \
    "$ACTIVATIONS_DIR" \
    "$EXPORT_DIR" \
    "$RESULTS_DIR"
}

csv_items() {
  local csv="$1"
  local -a items=()
  IFS=',' read -r -a items <<<"$csv"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -n "$item" ]] && printf '%s\n' "$item"
  done
}

detect_gpus() {
  local -a gpus=()
  if [[ "$VLM_GPUS" != "auto" && -n "$VLM_GPUS" ]]; then
    while IFS= read -r gpu; do gpus+=("$gpu"); done < <(csv_items "$VLM_GPUS")
  elif command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r gpu; do gpus+=("$gpu"); done < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
  else
    gpus=(0)
  fi

  if [[ -n "$VLM_EXCLUDE_GPUS" ]]; then
    local -a excluded=() kept=()
    while IFS= read -r gpu; do excluded+=("$gpu"); done < <(csv_items "$VLM_EXCLUDE_GPUS")
    local gpu ex skip
    for gpu in "${gpus[@]}"; do
      skip=0
      for ex in "${excluded[@]}"; do
        [[ "$gpu" == "$ex" ]] && skip=1
      done
      [[ "$skip" -eq 0 ]] && kept+=("$gpu")
    done
    gpus=("${kept[@]}")
  fi

  [[ ${#gpus[@]} -gt 0 ]] || die "no GPUs available after VLM_EXCLUDE_GPUS=$VLM_EXCLUDE_GPUS"
  printf '%s\n' "${gpus[@]}"
}

model_log_prefix() {
  local model="$1"
  case "$model" in
    qwen3_vl_4b) echo "qwen3" ;;
    qwen2_5_vl_7b) echo "qwen2_5" ;;
    *) echo "$model" | tr -c 'A-Za-z0-9_' '_' ;;
  esac
}

model_raw_path() {
  local model="$1"
  echo "$RESPONSES_DIR/${RUN_NAME}_$(model_log_prefix "$model")_raw.jsonl"
}

model_filtered_path() {
  local model="$1"
  echo "$RESPONSES_DIR/${RUN_NAME}_$(model_log_prefix "$model")_filtered.jsonl"
}

model_filter_report_path() {
  local model="$1"
  echo "$REPORTS_DIR/${RUN_NAME}_$(model_log_prefix "$model")_filter.md"
}

vllm_base_url_for_model() {
  local model="$1"
  case "$model" in
    qwen3_vl_4b) echo "${VLLM_BASE_URL_QWEN3:-$VLLM_BASE_URL}" ;;
    qwen2_5_vl_7b) echo "${VLLM_BASE_URL_QWEN2_5:-$VLLM_BASE_URL}" ;;
    *) echo "$VLLM_BASE_URL" ;;
  esac
}

print_config() {
  log "Configuration"
  echo "- root: $ROOT"
  echo "- data dir: $DATA_DIR"
  echo "- run name: $RUN_NAME"
  echo "- phases: $PHASES"
  echo "- image dir arg: ${IMAGE_DIR:-<download>}"
  echo "- limit images/languages/views/models: $LIMIT_IMAGES/$LIMIT_LANGUAGES/$LIMIT_VIEWS/$LIMIT_MODELS"
  echo "- min views for sets: $MIN_VIEWS"
  echo "- VLM models: $VLM_MODELS"
  echo "- VLM backend: $VLM_BACKEND"
  echo "- VLM GPUs: $VLM_GPUS"
  echo "- VLM shard: $VLM_SHARD"
  echo "- parallel models: $PARALLEL_MODELS"
  echo "- VLM batch/dtype/device_map: $VLM_BATCH_SIZE/$VLM_DTYPE/$VLM_DEVICE_MAP"
}

install_or_check_vlm_deps() {
  [[ "$INSTALL_VLM_DEPS" == "1" ]] && uv pip install -r "$VLM_REQUIREMENTS"
  [[ "$CHECK_VLM_DEPS" == "1" ]] || return 0

  uv run python - <<'PY'
import sys

try:
    import torch
    import torchvision
except ImportError as exc:
    print(f"[vlm] missing package: {exc}", file=sys.stderr)
    sys.exit(1)

print(f"[vlm] torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
print(f"[vlm] torchvision={torchvision.__version__}")
try:
    import qwen_vl_utils  # noqa: F401
    print("[vlm] qwen-vl-utils=ok")
except ImportError:
    print("[vlm] qwen-vl-utils=missing; Qwen may fall back to generic pipeline", file=sys.stderr)
PY
}

download_images() {
  if [[ -n "$IMAGE_DIR" ]]; then
    [[ -d "$IMAGE_DIR" ]] || die "IMAGE_DIR does not exist: $IMAGE_DIR"
    log "Using existing image dir: $IMAGE_DIR"
    return 0
  fi

  log "Downloading Open Images subset"
  echo "- split: $DOWNLOAD_SPLIT"
  echo "- requested images: $DOWNLOAD_IMAGES"
  echo "- download dir: $DOWNLOAD_DIR"

  mkdir -p "$(dirname "$OPENIMAGES_ID_CSV")" "$DOWNLOAD_DIR"

  if [[ ! -f "$OPENIMAGES_DOWNLOADER" ]]; then
    curl -L https://raw.githubusercontent.com/openimages/dataset/master/downloader.py -o "$OPENIMAGES_DOWNLOADER"
  fi

  if [[ ! -f "$OPENIMAGES_ID_CSV" ]]; then
    case "$DOWNLOAD_SPLIT" in
      train)
        curl -L https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv -o "$OPENIMAGES_ID_CSV"
        ;;
      validation)
        curl -L https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv -o "$OPENIMAGES_ID_CSV"
        ;;
      test)
        curl -L https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv -o "$OPENIMAGES_ID_CSV"
        ;;
      *)
        die "unsupported DOWNLOAD_SPLIT=$DOWNLOAD_SPLIT"
        ;;
    esac
  fi

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
random.Random(seed).shuffle(ids)
selected = ids[:limit]
target = Path(out_path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("\n".join(f"{split}/{image_id}" for image_id in selected) + "\n", encoding="utf-8")
print(f"selected {len(selected)} ids from {len(ids)} available")
PY

  uv pip install boto3 botocore tqdm >/dev/null
  uv run python "$OPENIMAGES_DOWNLOADER" "$OPENIMAGES_LIST" --download_folder "$DOWNLOAD_DIR" --num_processes "$DOWNLOAD_PROCESSES"
  IMAGE_DIR="$DOWNLOAD_DIR"
}

build_manifest() {
  log "Build manifest"
  uv run python visual_grounded_dataset/scripts/scan_image_folder.py \
    --image-dir "$IMAGE_DIR" \
    --out "$MANIFEST" \
    --source-dataset "open_images_v7_${DOWNLOAD_SPLIT}_partial" \
    --topic-tags "open_images_${DOWNLOAD_SPLIT}_partial" \
    --limit "$LIMIT_IMAGES"
}

render_jobs() {
  log "Render jobs"
  uv run python visual_grounded_dataset/scripts/render_generation_jobs.py \
    --manifest "$MANIFEST" \
    --out "$JOBS" \
    --model-sources "$VLM_MODELS" \
    --limit-images "$LIMIT_IMAGES" \
    --limit-models "$LIMIT_MODELS" \
    --limit-languages "$LIMIT_LANGUAGES" \
    --limit-views "$LIMIT_VIEWS"
}

count_model_jobs() {
  local model="$1"
  uv run python - "$JOBS" "$model" <<'PY'
import json
import sys

jobs_path, model_source = sys.argv[1], sys.argv[2]
count = 0
with open(jobs_path, encoding="utf-8") as handle:
    for line in handle:
        job = json.loads(line)
        if job["model"]["source_id"] == model_source:
            count += 1
print(count)
PY
}

generate_worker() {
  local model="$1"
  local gpu="$2"
  local start="$3"
  local limit="$4"
  local out="$5"
  local log_path="$6"

  local -a extra=()
  [[ "$CONTINUE_ON_ERROR" == "1" ]] && extra+=(--continue-on-error)
  [[ -f "$out" ]] && extra+=(--resume)
  if [[ "$VLM_BACKEND" == "vllm_openai" ]]; then
    extra+=(
      --vllm-base-url "$(vllm_base_url_for_model "$model")"
      --vllm-api-key "$VLLM_API_KEY"
      --vllm-timeout "$VLLM_TIMEOUT"
      --vllm-request-concurrency "$VLLM_REQUEST_CONCURRENCY"
    )
  fi

  mkdir -p "$(dirname "$out")" "$(dirname "$log_path")"
  echo "[vlm] model=$model backend=$VLM_BACKEND gpu=$gpu start=$start limit=$limit batch=$VLM_BATCH_SIZE out=$out"
  (
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu"
    uv run python "$GEN_SCRIPT" \
      --jobs "$JOBS" \
      --out "$out" \
      --backend "$VLM_BACKEND" \
      --only-model-source "$model" \
      --start "$start" \
      --limit "$limit" \
      --batch-size "$VLM_BATCH_SIZE" \
      --dtype "$VLM_DTYPE" \
      --device-map "$VLM_DEVICE_MAP" \
      --progress-every "$PROGRESS_EVERY" \
      "${extra[@]}"
  ) >"$log_path" 2>&1
}

merge_files() {
  local out="$1"
  shift
  : >"$out"
  local file
  for file in "$@"; do
    [[ -s "$file" ]] && cat "$file" >>"$out"
  done
  echo "- merged $(wc -l <"$out") lines -> $out"
}

generate_model() {
  local model="$1"
  shift
  local -a gpus=("$@")
  local total_jobs
  total_jobs="$(count_model_jobs "$model")"
  [[ "$total_jobs" -gt 0 ]] || die "no jobs for model $model in $JOBS"

  local prefix
  prefix="$(model_log_prefix "$model")"
  local final_out
  final_out="$(model_raw_path "$model")"

  local shard_mode="$VLM_SHARD"
  if [[ "$shard_mode" == "auto" ]]; then
    if [[ "$VLM_BACKEND" == "vllm_openai" ]]; then
      shard_mode=0
    elif [[ ${#gpus[@]} -gt 1 ]]; then
      shard_mode=1
    else
      shard_mode=0
    fi
  fi

  if [[ "$shard_mode" != "1" || ${#gpus[@]} -eq 1 ]]; then
    generate_worker "$model" "${gpus[0]}" 0 "$total_jobs" "$final_out" "$REPORTS_DIR/${RUN_NAME}_${prefix}_generation.log"
    return 0
  fi

  log "Generate $model across GPUs: ${gpus[*]}"
  local n=${#gpus[@]}
  local base=$((total_jobs / n))
  local rem=$((total_jobs % n))
  local start=0
  local -a pids=() shards=()
  local i
  for ((i = 0; i < n; i++)); do
    local limit=$base
    [[ "$i" -lt "$rem" ]] && limit=$((limit + 1))
    [[ "$limit" -eq 0 ]] && continue
    local shard="$RESPONSES_DIR/${RUN_NAME}_${prefix}_gpu${gpus[$i]}.jsonl"
    local log_path="$REPORTS_DIR/${RUN_NAME}_${prefix}_gpu${gpus[$i]}.log"
    shards+=("$shard")
    generate_worker "$model" "${gpus[$i]}" "$start" "$limit" "$shard" "$log_path" &
    pids+=("$!")
    start=$((start + limit))
  done

  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  [[ "$fail" -eq 0 ]] || die "one or more $model workers failed; see $REPORTS_DIR/${RUN_NAME}_${prefix}_gpu*.log"
  merge_files "$final_out" "${shards[@]}"
}

generate_vlms() {
  log "Generate VLM responses"
  install_or_check_vlm_deps

  local -a all_gpus=()
  while IFS= read -r gpu; do all_gpus+=("$gpu"); done < <(detect_gpus)
  local -a models=()
  while IFS= read -r model; do models+=("$model"); done < <(csv_items "$VLM_MODELS")

  echo "- models: ${models[*]}"
  echo "- GPUs: ${all_gpus[*]}"
  echo "- shard: $VLM_SHARD"
  echo "- parallel models: $PARALLEL_MODELS"

  if [[ "$PARALLEL_MODELS" == "1" && ${#models[@]} -gt 1 ]]; then
    local half=$(((${#all_gpus[@]} + 1) / 2))
    local -a first=("${all_gpus[@]:0:$half}")
    local -a second=("${all_gpus[@]:$half}")
    [[ ${#second[@]} -gt 0 ]] || second=("${first[@]}")

    generate_model "${models[0]}" "${first[@]}" &
    local p1=$!
    generate_model "${models[1]}" "${second[@]}" &
    local p2=$!
    set +e
    wait "$p1"
    local s1=$?
    wait "$p2"
    local s2=$?
    set -e
    [[ "$s1" -eq 0 && "$s2" -eq 0 ]] || die "parallel model generation failed: ${models[0]}=$s1 ${models[1]}=$s2"
  else
    local model
    for model in "${models[@]}"; do
      generate_model "$model" "${all_gpus[@]}"
    done
  fi
}

filter_responses() {
  log "Filter responses"
  local -a models=()
  while IFS= read -r model; do models+=("$model"); done < <(csv_items "$VLM_MODELS")

  local -a filtered=()
  local model
  for model in "${models[@]}"; do
    local raw filtered_path report
    raw="$(model_raw_path "$model")"
    filtered_path="$(model_filtered_path "$model")"
    report="$(model_filter_report_path "$model")"
    [[ -f "$raw" ]] || die "missing raw response file: $raw"
    uv run python visual_grounded_dataset/scripts/filter_responses.py \
      --responses "$raw" \
      --out "$filtered_path" \
      --report "$report"
    [[ -s "$filtered_path" ]] && filtered+=("$filtered_path")
  done

  [[ ${#filtered[@]} -gt 0 ]] || die "no non-empty filtered response files"
  merge_files "$RESPONSES_DIR/${RUN_NAME}_filtered_merged.jsonl" "${filtered[@]}"
}

build_sets_and_reports() {
  log "Build sets, audit, and diversity filter"
  uv run python visual_grounded_dataset/scripts/build_sets.py \
    --responses "$RESPONSES_DIR/${RUN_NAME}_filtered_merged.jsonl" \
    --out "$SETS_DIR/${RUN_NAME}_sets.jsonl" \
    --min-views "$MIN_VIEWS"

  uv run python visual_grounded_dataset/scripts/artifact_audit.py \
    --sets "$SETS_DIR/${RUN_NAME}_sets.jsonl" \
    --out "$REPORTS_DIR/${RUN_NAME}_artifact_audit.md"

  uv run python visual_grounded_dataset/scripts/filter_sets_by_diversity.py \
    --sets "$SETS_DIR/${RUN_NAME}_sets.jsonl" \
    --out "$SETS_DIR/${RUN_NAME}_sets_diverse.jsonl" \
    --rejected-out "$SETS_DIR/${RUN_NAME}_sets_diversity_rejected.jsonl" \
    --report "$REPORTS_DIR/${RUN_NAME}_diversity_filter.md"
}

export_dataset() {
  log "Export portable dataset"
  local source="$SETS_DIR/${RUN_NAME}_sets_diverse.jsonl"
  [[ -s "$source" ]] || die "missing or empty dataset sets file: $source"
  mkdir -p "$(dirname "$EXPORT_DATASET")"
  cp "$source" "$EXPORT_DATASET"
  echo "- portable dataset: $EXPORT_DATASET"
  echo "- sets: $(wc -l <"$EXPORT_DATASET")"
  echo "- move this file to the local PC for activation extraction and SetConCA training"
}

build_controls_and_activation_inputs() {
  log "Build controls and activation JSONL files"
  uv run python visual_grounded_dataset/scripts/build_controls.py \
    --sets "$SETS_DIR/${RUN_NAME}_sets_diverse.jsonl" \
    --out-dir "$CONTROLS_DIR"

  uv run python visual_grounded_dataset/scripts/convert_sets_for_activation.py \
    --sets "$SETS_DIR/${RUN_NAME}_sets_diverse.jsonl" \
    --out "$SETS_DIR/${RUN_NAME}_sets_diverse_activation.jsonl"

  uv run python visual_grounded_dataset/scripts/convert_sets_for_activation.py \
    --sets "$CONTROLS_DIR/controls_shuffled_image.jsonl" \
    --out "$CONTROLS_DIR/controls_shuffled_image_activation.jsonl"

  uv run python visual_grounded_dataset/scripts/convert_sets_for_activation.py \
    --sets "$CONTROLS_DIR/controls_same_prompt_different_image.jsonl" \
    --out "$CONTROLS_DIR/controls_same_prompt_different_image_activation.jsonl"
}

extract_activations() {
  log "Extract activations"
  uv run python scripts/extract_activation_bank.py \
    --sets "$SETS_DIR/${RUN_NAME}_sets_diverse_activation.jsonl" \
    --out "$ACTIVATIONS_DIR/${RUN_NAME}_real_llama_layer_${LAYER}_s${VIEWS_FOR_ACTIVATION}.pt" \
    --model-id "$MODEL_ID" \
    --layer "$LAYER" \
    --views "$VIEWS_FOR_ACTIVATION" \
    --no-original \
    --batch-size "$ACTIVATION_BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --dtype "$ACTIVATION_DTYPE"

  uv run python scripts/extract_activation_bank.py \
    --sets "$CONTROLS_DIR/controls_shuffled_image_activation.jsonl" \
    --out "$ACTIVATIONS_DIR/${RUN_NAME}_shuffled_llama_layer_${LAYER}_s${VIEWS_FOR_ACTIVATION}.pt" \
    --model-id "$MODEL_ID" \
    --layer "$LAYER" \
    --views "$VIEWS_FOR_ACTIVATION" \
    --no-original \
    --batch-size "$ACTIVATION_BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --dtype "$ACTIVATION_DTYPE"

  uv run python scripts/extract_activation_bank.py \
    --sets "$CONTROLS_DIR/controls_same_prompt_different_image_activation.jsonl" \
    --out "$ACTIVATIONS_DIR/${RUN_NAME}_same_prompt_llama_layer_${LAYER}_s${VIEWS_FOR_ACTIVATION}.pt" \
    --model-id "$MODEL_ID" \
    --layer "$LAYER" \
    --views "$VIEWS_FOR_ACTIVATION" \
    --no-original \
    --batch-size "$ACTIVATION_BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --dtype "$ACTIVATION_DTYPE"
}

train_setconca() {
  [[ "$RUN_TRAIN" == "1" ]] || return 0
  log "Train SetConCA"
  uv run python scripts/train_setconca_v2.py \
    --activations "$ACTIVATIONS_DIR/${RUN_NAME}_real_llama_layer_${LAYER}_s${VIEWS_FOR_ACTIVATION}.pt" \
    --out-dir "$RESULTS_DIR/${RUN_NAME}_real_train" \
    --epochs "$EPOCHS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --concept-dim "$CONCEPT_DIM" \
    --topk "$TOPK"

  uv run python scripts/train_setconca_v2.py \
    --activations "$ACTIVATIONS_DIR/${RUN_NAME}_shuffled_llama_layer_${LAYER}_s${VIEWS_FOR_ACTIVATION}.pt" \
    --out-dir "$RESULTS_DIR/${RUN_NAME}_shuffled_train" \
    --epochs "$EPOCHS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --concept-dim "$CONCEPT_DIM" \
    --topk "$TOPK"

  uv run python scripts/train_setconca_v2.py \
    --activations "$ACTIVATIONS_DIR/${RUN_NAME}_same_prompt_llama_layer_${LAYER}_s${VIEWS_FOR_ACTIVATION}.pt" \
    --out-dir "$RESULTS_DIR/${RUN_NAME}_same_prompt_train" \
    --epochs "$EPOCHS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --concept-dim "$CONCEPT_DIM" \
    --topk "$TOPK"
}

finish() {
  log "Done"
  echo "- merged responses: $RESPONSES_DIR/${RUN_NAME}_filtered_merged.jsonl"
  echo "- sets: $SETS_DIR/${RUN_NAME}_sets_diverse.jsonl"
  echo "- portable dataset: $EXPORT_DATASET"
  echo "- report: $REPORTS_DIR/${RUN_NAME}_artifact_audit.md"
  echo "- diversity report: $REPORTS_DIR/${RUN_NAME}_diversity_filter.md"
  echo "- activations: $ACTIVATIONS_DIR"
  echo "- results: $RESULTS_DIR/${RUN_NAME}_*_train"
}

main() {
  ensure_dirs
  print_config

  if contains_phase images; then
    download_images
  elif [[ -z "$IMAGE_DIR" ]]; then
    IMAGE_DIR="$DOWNLOAD_DIR"
  fi

  contains_phase manifest && build_manifest
  contains_phase jobs && render_jobs
  contains_phase generate && generate_vlms
  contains_phase filter && filter_responses
  contains_phase sets && build_sets_and_reports
  contains_phase export && export_dataset
  contains_phase controls && build_controls_and_activation_inputs
  contains_phase activations && extract_activations
  contains_phase train && train_setconca

  finish
}

main "$@"
