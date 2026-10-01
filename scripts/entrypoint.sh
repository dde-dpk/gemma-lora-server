#!/usr/bin/env bash
# Starts the vLLM OpenAI server (inference, :8000) and the LoRA manager (:8001).
# If either process exits, the container shuts down so the orchestrator restarts it.
set -euo pipefail

# ---- Tunables (override via environment) -------------------------------------
: "${MODEL_ID:=google/gemma-3-4b-it}"
: "${VLLM_PORT:=8000}"
: "${MANAGER_PORT:=8001}"
: "${DTYPE:=bfloat16}"               # A10G (Ampere) runs bf16 well
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_LORAS:=4}"                  # max adapters resident on GPU at once
: "${MAX_LORA_RANK:=32}"             # must be >= the rank of any adapter you load
: "${MAX_CPU_LORAS:=16}"             # adapters cached in CPU RAM, swapped in on demand
: "${VLLM_EXTRA_ARGS:=}"             # e.g. "--quantization fp8"

# Enable runtime load/unload of adapters.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# Hugging Face auth for the gated Gemma weights (either var name works).
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

echo "[entrypoint] vLLM: model=${MODEL_ID} port=${VLLM_PORT} dtype=${DTYPE}"
python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enable-lora \
  --max-loras "${MAX_LORAS}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --max-cpu-loras "${MAX_CPU_LORAS}" \
  ${VLLM_EXTRA_ARGS} &
VLLM_PID=$!

echo "[entrypoint] LoRA manager: port=${MANAGER_PORT}"
uvicorn app.manager:app --host 0.0.0.0 --port "${MANAGER_PORT}" &
MANAGER_PID=$!

# Shut everything down if either child dies.
trap 'kill -TERM "${VLLM_PID}" "${MANAGER_PID}" 2>/dev/null || true' TERM INT
wait -n
echo "[entrypoint] a child process exited; shutting down."
kill -TERM "${VLLM_PID}" "${MANAGER_PID}" 2>/dev/null || true
wait || true
exit 1
