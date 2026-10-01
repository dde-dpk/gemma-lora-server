# Base image ships vLLM + CUDA runtime. Pin to a version you've tested.
# Browse tags at https://hub.docker.com/r/vllm/vllm-openai/tags
ARG VLLM_TAG=latest
FROM vllm/vllm-openai:${VLLM_TAG}

WORKDIR /workspace

# Add only the sidecar's extra deps; vLLM is already present in the base image.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/entrypoint.sh ./scripts/entrypoint.sh
RUN chmod +x ./scripts/entrypoint.sh && mkdir -p /loras

ENV LORA_LOCAL_DIR=/loras \
    VLLM_BASE_URL=http://localhost:8000

EXPOSE 8000 8001

# Override the base image's default entrypoint (it only starts vLLM).
ENTRYPOINT ["./scripts/entrypoint.sh"]
