#!/usr/bin/env bash
# Start the vLLM server (localhost) in the background, then run the gateway in
# the foreground. The gateway is PID 1, so it receives Cloud Run's SIGTERM and
# can flush telemetry on shutdown. Cloud Run's startup probe hits the gateway's
# /health/ready, which only reports ready once vLLM has loaded the model.
set -uo pipefail

MODEL_PATH="${MODEL_PATH:-/models/qwen-awq}"
PORT="${PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"

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
