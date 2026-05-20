#!/usr/bin/env bash
set -euo pipefail

# Start one OpenAI-compatible vLLM server for a configured VLM source id.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0,1 ./visual_grounded_dataset/scripts/start_vllm_server.sh qwen3_vl_4b 8001 2
#   CUDA_VISIBLE_DEVICES=2,3 ./visual_grounded_dataset/scripts/start_vllm_server.sh qwen2_5_vl_7b 8002 2

MODEL_SOURCE="${1:-qwen3_vl_4b}"
PORT="${2:-8001}"
TP_SIZE="${3:-1}"
HOST="${HOST:-127.0.0.1}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT:-{\"image\":1}}"
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
if [[ -n "${VLLM_CMD:-}" ]]; then
  VLLM_LAUNCH=($VLLM_CMD)
elif command -v vllm >/dev/null 2>&1; then
  VLLM_LAUNCH=(vllm)
else
  VLLM_LAUNCH=(uv run vllm)
fi

case "$MODEL_SOURCE" in
  qwen3_vl_4b)
    MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
    ;;
  qwen2_5_vl_7b)
    MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
    ;;
  internvl3_8b)
    MODEL_ID="OpenGVLab/InternVL3-8B-Instruct"
    ;;
  gemma4_e4b_it)
    MODEL_ID="google/gemma-4-E4B-it"
    ;;
  gemma3_4b_it)
    MODEL_ID="google/gemma-3-4b-it"
    ;;
  aya_vision_8b)
    MODEL_ID="CohereLabs/aya-vision-8b"
    ;;
  paligemma2_3b_mix)
    MODEL_ID="google/paligemma2-3b-mix-448"
    ;;
  qwen3_vl_8b)
    MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
    ;;
  *)
    echo "Unknown MODEL_SOURCE=$MODEL_SOURCE" >&2
    echo "Known: qwen3_vl_4b qwen2_5_vl_7b internvl3_8b gemma4_e4b_it gemma3_4b_it aya_vision_8b paligemma2_3b_mix qwen3_vl_8b" >&2
    exit 1
    ;;
esac

echo "Starting vLLM server"
echo "- source: $MODEL_SOURCE"
echo "- model: $MODEL_ID"
echo "- host: $HOST"
echo "- port: $PORT"
echo "- tensor parallel size: $TP_SIZE"
echo "- CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "- launcher: ${VLLM_LAUNCH[*]}"

exec "${VLLM_LAUNCH[@]}" serve "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
  $EXTRA_ARGS
