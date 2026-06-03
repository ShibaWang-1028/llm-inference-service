# Multi-stage build for the LLM inference service.
#
# Base: vLLM's official OpenAI-server image (CUDA 12.9, vLLM 0.22.0). We run that
# server on localhost inside the container and put our FastAPI gateway in front.
# The gateway's deps live in an isolated venv so they can't clash with vLLM's,
# and the AWQ weights are baked in for a fast, reproducible cold start.

ARG VLLM_TAG=v0.22.0-cu129-ubuntu2404
FROM vllm/vllm-openai:${VLLM_TAG} AS base

# ---- stage: gateway dependencies in an isolated venv ----
FROM base AS gwdeps
COPY requirements-gateway.txt /tmp/requirements-gateway.txt
RUN python3 -m venv /opt/gw \
 && /opt/gw/bin/pip install --no-cache-dir -U pip \
 && /opt/gw/bin/pip install --no-cache-dir -r /tmp/requirements-gateway.txt

# ---- stage: bake the AWQ weights into the image ----
FROM base AS weights
ARG MODEL_REPO=Qwen/Qwen2.5-7B-Instruct-AWQ
ARG HF_TOKEN=""
RUN HF_TOKEN="${HF_TOKEN}" huggingface-cli download "${MODEL_REPO}" --local-dir /models/qwen-awq

# ---- final image ----
FROM base

ENV PORT=8080 \
    VLLM_PORT=8000 \
    UPSTREAM_BASE_URL=http://127.0.0.1:8000 \
    MODEL_PATH=/models/qwen-awq \
    MODEL_NAME=Qwen2.5-7B-Instruct \
    QUANTIZATION=awq_marlin \
    MAX_MODEL_LEN=16384 \
    GPU_MEMORY_UTILIZATION=0.90 \
    MAX_NUM_SEQS=64 \
    HF_HUB_OFFLINE=1

COPY --from=gwdeps /opt/gw /opt/gw
COPY --from=weights /models /models

WORKDIR /app
COPY app/ /app/app/
COPY ui/ /app/ui/
COPY deploy/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
