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
#
# VLM generation (4 GPUs 0-3, shard Qwen3 then Qwen2.5 across all cards):
#   VLM_GPUS=0,1,2,3 ./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" qwen_500_v2views
#
# Both models at once (2 GPUs each):
#   PARALLEL_VLMS=1 VLM_GPU_QWEN3=0,1 VLM_GPU_QWEN25=2,3 ./visual_grounded_dataset/scripts/run_visual_pipeline.sh "" run_name
#
# Legacy: one model per GPU, no sharding:
#   VLM_SHARD=0 PARALLEL_VLMS=1 VLM_GPU_QWEN3=0 VLM_GPU_QWEN25=1 ...

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

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
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-4}"
VLM_DTYPE="${VLM_DTYPE:-bfloat16}"
VLM_DEVICE_MAP="${VLM_DEVICE_MAP:-single}"

# VLM generation layout
# VLM_GPUS: comma-separated device ids (default: all GPUs from nvidia-smi)
# VLM_EXCLUDE_GPUS: skip busy/shared GPUs (e.g. VLM_EXCLUDE_GPUS=3)
# VLM_SHARD: auto|0|1 — auto enables sharding when 2+ GPUs are visible
# PARALLEL_VLMS: 1 = run Qwen3 and Qwen2.5 worker groups concurrently
# VLM_GPU_QWEN3 / VLM_GPU_QWEN25: one id or comma list per model (for PARALLEL_VLMS=1)
VLM_GPUS="${VLM_GPUS:-}"
VLM_EXCLUDE_GPUS="${VLM_EXCLUDE_GPUS:-}"
VLM_SHARD="${VLM_SHARD:-auto}"
PARALLEL_VLMS="${PARALLEL_VLMS:-0}"
VLM_GPU_QWEN3="${VLM_GPU_QWEN3:-0}"
VLM_GPU_QWEN25="${VLM_GPU_QWEN25:-1}"

JOBS_FILE="visual_grounded_dataset/data/jobs/${RUN_NAME}_jobs.jsonl"
GEN_SCRIPT="visual_grounded_dataset/scripts/generate_with_vlm.py"
VLM_REQUIREMENTS="visual_grounded_dataset/requirements-vlm.txt"

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

VLM_GPU_ARRAY=()

parse_gpu_list() {
  # Usage: parse_gpu_list "0,1,2"  -> sets reply array via stdout (caller reads)
  local csv="$1"
  local -a out=()
  IFS=',' read -r -a out <<<"$csv"
  for g in "${out[@]}"; do
    g="${g//[[:space:]]/}"
    [[ -n "$g" ]] || continue
    printf '%s\n' "$g"
  done
}

detect_vlm_gpus() {
  VLM_GPU_ARRAY=()
  if [[ -n "$VLM_GPUS" ]]; then
    while IFS= read -r g; do
      VLM_GPU_ARRAY+=("$g")
    done < <(parse_gpu_list "$VLM_GPUS")
  elif command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r g; do
      VLM_GPU_ARRAY+=("$g")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
  fi
  if [[ ${#VLM_GPU_ARRAY[@]} -eq 0 ]]; then
    VLM_GPU_ARRAY=(0)
  fi
  if [[ -n "$VLM_EXCLUDE_GPUS" ]]; then
    local -a excluded=()
    while IFS= read -r g; do
      excluded+=("$g")
    done < <(parse_gpu_list "$VLM_EXCLUDE_GPUS")
    local -a kept=()
    local gpu ex skip
    for gpu in "${VLM_GPU_ARRAY[@]}"; do
      skip=0
      for ex in "${excluded[@]}"; do
        if [[ "$gpu" == "$ex" ]]; then
          skip=1
          break
        fi
      done
      if [[ "$skip" -eq 0 ]]; then
        kept+=("$gpu")
      fi
    done
    VLM_GPU_ARRAY=("${kept[@]}")
  fi
  if [[ ${#VLM_GPU_ARRAY[@]} -eq 0 ]]; then
    echo "[vlm] no GPUs left after VLM_EXCLUDE_GPUS=$VLM_EXCLUDE_GPUS" >&2
    exit 1
  fi
}

resolve_vlm_shard_mode() {
  if [[ "$VLM_SHARD" == "auto" ]]; then
    if [[ ${#VLM_GPU_ARRAY[@]} -ge 2 ]]; then
      VLM_SHARD=1
    else
      VLM_SHARD=0
    fi
  fi
}

ensure_vlm_deps() {
  echo "[vlm] Installing / verifying VLM dependencies (torchvision must match torch+cu130)"
  if [[ -f "$VLM_REQUIREMENTS" ]]; then
    uv pip install -r "$VLM_REQUIREMENTS" >/dev/null
  else
    uv sync --extra vlm >/dev/null
  fi
  uv run python - <<'PY'
import sys

try:
    import torch
    import torchvision
except ImportError as exc:
    print(f"[vlm] missing package: {exc}", file=sys.stderr)
    print(
        "[vlm] run: uv pip install -r visual_grounded_dataset/requirements-vlm.txt",
        file=sys.stderr,
    )
    sys.exit(1)

tv = torchvision.__version__
th = torch.__version__
if "+cu" in th and "+cu" not in tv:
    print(
        f"[vlm] torchvision ({tv}) is not a CUDA-matched wheel for torch ({th}).",
        file=sys.stderr,
    )
    print(
        "[vlm] fix: uv pip install 'torchvision>=0.26.0' "
        "--index-url https://download.pytorch.org/whl/cu130",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import qwen_vl_utils  # noqa: F401
    qwen_utils = "ok"
except ImportError:
    qwen_utils = "missing (Qwen2.5 will use slower generic pipeline fallback)"

print(f"[vlm] torch {th} cuda={torch.cuda.is_available()}")
print(f"[vlm] torchvision {tv}")
print(f"[vlm] qwen-vl-utils {qwen_utils}")
PY
}

count_model_jobs() {
  local model_source="$1"
  uv run python - "$JOBS_FILE" "$model_source" <<'PY'
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

run_generation_worker() {
  local gpu="$1"
  local model_source="$2"
  local out="$3"
  local start="$4"
  local limit="$5"
  local log="$6"

  local resume_args=()
  if [[ -f "$out" ]]; then
    resume_args=(--resume)
  fi

  mkdir -p "$(dirname "$out")" "$(dirname "$log")"
  echo "[vlm] gpu=$gpu model=$model_source start=$start limit=$limit batch=$VLM_BATCH_SIZE dtype=$VLM_DTYPE device_map=$VLM_DEVICE_MAP -> $out"
  (
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu"
    uv run python "$GEN_SCRIPT" \
      --jobs "$JOBS_FILE" \
      --out "$out" \
      --backend transformers \
      --only-model-source "$model_source" \
      --start "$start" \
      --limit "$limit" \
      --batch-size "$VLM_BATCH_SIZE" \
      --dtype "$VLM_DTYPE" \
      --device-map "$VLM_DEVICE_MAP" \
      --continue-on-error \
      --progress-every "$PROGRESS_EVERY" \
      "${resume_args[@]}"
  ) >"$log" 2>&1
}

merge_shard_outputs() {
  local final_out="$1"
  shift
  local -a shards=("$@")
  mkdir -p "$(dirname "$final_out")"
  : >"$final_out"
  for shard in "${shards[@]}"; do
    if [[ -f "$shard" ]]; then
      cat "$shard" >>"$final_out"
    fi
  done
  echo "[vlm] merged ${#shards[@]} shard(s) -> $final_out ($(wc -l <"$final_out") lines)"
}

run_sharded_model_generation() {
  local model_source="$1"
  local final_out="$2"
  local log_prefix="$3"
  shift 3
  local -a gpus=("$@")

  local total_jobs
  total_jobs="$(count_model_jobs "$model_source")"
  if [[ "$total_jobs" -eq 0 ]]; then
    echo "[vlm] no jobs for $model_source in $JOBS_FILE" >&2
    exit 1
  fi

  local n_gpus=${#gpus[@]}
  local base_chunk=$((total_jobs / n_gpus))
  local remainder=$((total_jobs % n_gpus))
  local -a shard_files=()
  local -a pids=()
  local start=0
  local i

  echo "[vlm] $model_source: $total_jobs jobs across ${n_gpus} GPU(s): ${gpus[*]}"

  for ((i = 0; i < n_gpus; i++)); do
    local limit=$base_chunk
    if [[ "$i" -lt "$remainder" ]]; then
      limit=$((limit + 1))
    fi
    if [[ "$limit" -eq 0 ]]; then
      continue
    fi
    local shard_out="visual_grounded_dataset/data/responses/${RUN_NAME}_${log_prefix}_gpu${gpus[$i]}.jsonl"
    local shard_log="visual_grounded_dataset/data/reports/${RUN_NAME}_${log_prefix}_gpu${gpus[$i]}.log"
    shard_files+=("$shard_out")
    run_generation_worker "${gpus[$i]}" "$model_source" "$shard_out" "$start" "$limit" "$shard_log" &
    pids+=("$!")
    start=$((start + limit))
  done

  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "[vlm] one or more $model_source shard workers failed; see ${RUN_NAME}_${log_prefix}_gpu*.log" >&2
    return 1
  fi

  merge_shard_outputs "$final_out" "${shard_files[@]}"
}

run_single_model_generation() {
  local model_source="$1"
  local final_out="$2"
  local log_path="$3"
  local gpu="${4:-${VLM_GPU_ARRAY[0]}}"

  local total_jobs
  total_jobs="$(count_model_jobs "$model_source")"
  mkdir -p "$(dirname "$final_out")" "$(dirname "$log_path")"
  echo "[vlm] $model_source: $total_jobs jobs on gpu $gpu"
  (
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu"
    local resume_args=()
    if [[ -f "$final_out" ]]; then
      resume_args=(--resume)
    fi
    uv run python "$GEN_SCRIPT" \
      --jobs "$JOBS_FILE" \
      --out "$final_out" \
      --backend transformers \
      --only-model-source "$model_source" \
      --batch-size "$VLM_BATCH_SIZE" \
      --dtype "$VLM_DTYPE" \
      --device-map "$VLM_DEVICE_MAP" \
      --continue-on-error \
      --progress-every "$PROGRESS_EVERY" \
      "${resume_args[@]}"
  ) >"$log_path" 2>&1
}

run_model_on_gpus() {
  local model_source="$1"
  local final_out="$2"
  local log_prefix="$3"
  local gpu_csv="$4"

  local -a gpus=()
  while IFS= read -r g; do
    gpus+=("$g")
  done < <(parse_gpu_list "$gpu_csv")

  if [[ ${#gpus[@]} -eq 0 ]]; then
    echo "[vlm] empty GPU list for $model_source" >&2
    exit 1
  fi

  if [[ "$VLM_SHARD" == "1" && ${#gpus[@]} -ge 2 ]]; then
    run_sharded_model_generation "$model_source" "$final_out" "$log_prefix" "${gpus[@]}"
  else
    run_single_model_generation "$model_source" "$final_out" \
      "visual_grounded_dataset/data/reports/${RUN_NAME}_${log_prefix}_generation.log" \
      "${gpus[0]}"
  fi
}

run_vlm_generation_phase() {
  detect_vlm_gpus
  resolve_vlm_shard_mode
  ensure_vlm_deps

  echo "[3-4/12] VLM generation"
  echo "- repo: $ROOT"
  echo "- jobs: $JOBS_FILE"
  echo "- visible GPUs: ${VLM_GPU_ARRAY[*]}"
  if [[ -n "$VLM_EXCLUDE_GPUS" ]]; then
    echo "- excluded GPUs: $VLM_EXCLUDE_GPUS"
  fi
  echo "- VLM_SHARD: $VLM_SHARD"
  echo "- PARALLEL_VLMS: $PARALLEL_VLMS"
  echo "- VLM_BATCH_SIZE: $VLM_BATCH_SIZE (raise to use more VRAM; lower if OOM)"
  echo "- VLM_DTYPE: $VLM_DTYPE"
  echo "- VLM_DEVICE_MAP: $VLM_DEVICE_MAP (single pins one visible GPU per worker)"

  local qwen3_out="visual_grounded_dataset/data/responses/${RUN_NAME}_qwen3_raw.jsonl"
  local qwen25_out="visual_grounded_dataset/data/responses/${RUN_NAME}_qwen2_5_raw.jsonl"

  if [[ "$PARALLEL_VLMS" == "1" ]]; then
    local -a qwen3_gpus=() qwen25_gpus=()
    while IFS= read -r g; do qwen3_gpus+=("$g"); done < <(parse_gpu_list "$VLM_GPU_QWEN3")
    while IFS= read -r g; do qwen25_gpus+=("$g"); done < <(parse_gpu_list "$VLM_GPU_QWEN25")

    if [[ ${#qwen3_gpus[@]} -eq 0 || ${#qwen25_gpus[@]} -eq 0 ]]; then
      echo "PARALLEL_VLMS=1 requires non-empty VLM_GPU_QWEN3 and VLM_GPU_QWEN25" >&2
      exit 1
    fi

    echo "- qwen3 GPUs: ${qwen3_gpus[*]}"
    echo "- qwen2.5 GPUs: ${qwen25_gpus[*]}"

    local q3_csv q25_csv
    q3_csv=$(IFS=,; echo "${qwen3_gpus[*]}")
    q25_csv=$(IFS=,; echo "${qwen25_gpus[*]}")

    (
      run_model_on_gpus "qwen3_vl_4b" "$qwen3_out" "qwen3" "$q3_csv"
    ) &
    local qwen3_pid=$!
    (
      run_model_on_gpus "qwen2_5_vl_7b" "$qwen25_out" "qwen2_5" "$q25_csv"
    ) &
    local qwen25_pid=$!

    set +e
    wait "$qwen3_pid"
    local qwen3_status=$?
    wait "$qwen25_pid"
    local qwen25_status=$?
    set -e

    if [[ "$qwen3_status" -ne 0 || "$qwen25_status" -ne 0 ]]; then
      echo "Parallel VLM generation failed: qwen3=$qwen3_status qwen2.5=$qwen25_status" >&2
      exit 1
    fi
  else
    local all_csv
    all_csv=$(IFS=,; echo "${VLM_GPU_ARRAY[*]}")
    echo "[3/12] Generate Qwen3"
    run_model_on_gpus "qwen3_vl_4b" "$qwen3_out" "qwen3" "$all_csv"
    echo "[4/12] Generate Qwen2.5"
    run_model_on_gpus "qwen2_5_vl_7b" "$qwen25_out" "qwen2_5" "$all_csv"
  fi
}

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
  --out "$JOBS_FILE" \
  --limit-images "$LIMIT_IMAGES" \
  --limit-models "$LIMIT_MODELS" \
  --limit-languages "$LIMIT_LANGUAGES" \
  --limit-views "$LIMIT_VIEWS"

run_vlm_generation_phase

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
