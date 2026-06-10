#!/usr/bin/env bash
# Start the vLLM server (localhost) in the background, then run the gateway in
# the foreground. The gateway is PID 1, so it receives Cloud Run's SIGTERM and
# can flush telemetry on shutdown. Cloud Run's startup probe hits the gateway's
# /health/live, so traffic flows within seconds while the model is still
# loading; the gateway answers 503 (model_loading) until vLLM is up and warmed,
# which the UI shows as a wake-up screen.
set -uo pipefail

MODEL_PATH="${MODEL_PATH:-/models/qwen-awq}"
PORT="${PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"
BOOT_T0="$(date +%s)"

# Cloud Run streams the container image lazily from the registry, so the first
# read of every file goes over the network. Pull the model files into the page
# cache now, in parallel with vLLM's ~1 minute of Python startup, so the
# weight loader hits warm cache instead of the registry.
(
  find "${MODEL_PATH}" -type f -print0 | xargs -0 -n1 -P4 cat > /dev/null 2>&1
  echo "Model files prefetched in $(( $(date +%s) - BOOT_T0 ))s"
) &

echo "Starting vLLM on 127.0.0.1:${VLLM_PORT} (model: ${MODEL_PATH})"
python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME:-Qwen2.5-7B-Instruct}" \
  --quantization "${QUANTIZATION:-awq_marlin}" \
  --host 127.0.0.1 \
  --port "${VLLM_PORT}" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-num-seqs "${MAX_NUM_SEQS:-64}" \
  ${ENFORCE_EAGER:+--enforce-eager} &

echo "Starting gateway on 0.0.0.0:${PORT}"
exec /opt/gw/bin/uvicorn app.main:app --app-dir /app --host 0.0.0.0 --port "${PORT}"
