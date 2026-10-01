# Gemma 3 + dynamic LoRA inference server

A containerized inference server for **g5.xlarge** (1× NVIDIA A10G, 24 GB) that
hosts a base **Gemma 3 4B** model and lets you **load/unload LoRA adapters at
runtime** — adapters are pulled from **Amazon S3** on demand.

## How it works

Two processes run in one container:

| Port | Process | Purpose |
|------|---------|---------|
| 8000 | **vLLM** OpenAI server | Inference. Call it with `model: "<lora_name>"` (or the base model id). |
| 8001 | **LoRA manager** (FastAPI) | `load`/`unload` adapters. Pulls them from S3, then registers them with vLLM. |

vLLM's native runtime-LoRA endpoints only accept a *local* path, so the manager
is the piece that bridges S3 → local disk → vLLM.

```
                      :8000 (OpenAI API)
client ───inference──────────────────────────►  vLLM  ◄── load/unload ── LoRA manager :8001
                                                              (localhost)        │ boto3
                                                                                 ▼
                                                                           s3://.../loras/<name>/
```

## Prerequisites on the g5.xlarge

Use the **AWS Deep Learning Base AMI** (ships with NVIDIA driver + Docker +
NVIDIA Container Toolkit), or on a plain Ubuntu AMI install the driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Verify the GPU is visible to Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**Credentials & permissions**
- Attach an **EC2 instance role** to the g5.xlarge with *read-only* access to the
  adapter bucket — see [`iam-policy.json`](iam-policy.json) (least privilege:
  `s3:GetObject` + `s3:ListBucket` scoped to your prefix). boto3 picks this up
  automatically via instance metadata; no keys on disk.
- Gemma 3 is **gated** on Hugging Face. Request access, create a token, and set
  `HF_TOKEN`. (Alternatively, pre-stage the weights in S3/EFS and point
  `MODEL_ID` at a local path to avoid HF at runtime.)

## Build & run

```bash
cp .env.example .env        # set HF_TOKEN, LORA_S3_PREFIX, VLLM_TAG, AWS_REGION
docker compose up --build
```

First boot downloads the base model (cached in the `hf-cache` volume). Wait for
vLLM to report ready:

```bash
curl localhost:8000/health
curl localhost:8001/health      # {"manager":"ok","vllm_ready":true}
```

> On the g5.xlarge, delete the `${HOME}/.aws` volume mount in
> `docker-compose.yml` — it's for local dev only; the instance role is used in prod.

## Using LoRA adapters

Your S3 prefix should contain one folder per adapter, each holding the standard
PEFT output (`adapter_config.json`, `adapter_model.safetensors`, ...):

```
s3://my-bucket/loras/
  customer-support/
    adapter_config.json
    adapter_model.safetensors
  legal-tone/
    ...
```

**Load** (resolves to `${LORA_S3_PREFIX}/customer-support` when `s3_uri` is omitted):

```bash
curl -X POST localhost:8001/v1/loras/load \
  -H 'Content-Type: application/json' \
  -d '{"lora_name": "customer-support"}'
```

Or point at an explicit location:

```bash
curl -X POST localhost:8001/v1/loras/load \
  -H 'Content-Type: application/json' \
  -d '{"lora_name": "legal-tone", "s3_uri": "s3://other-bucket/adapters/legal-tone"}'
```

**List** what's currently served:

```bash
curl localhost:8001/v1/loras            # base model + loaded adapters
```

**Run inference against an adapter** — standard OpenAI API, `model` = the lora_name:

```bash
curl -X POST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "customer-support",
    "messages": [{"role": "user", "content": "My order is late, help!"}]
  }'
```

Use `"model": "google/gemma-3-4b-it"` to hit the base model with no adapter.

**Unload** (optionally delete the local copy):

```bash
curl -X POST localhost:8001/v1/loras/unload \
  -H 'Content-Type: application/json' \
  -d '{"lora_name": "customer-support", "delete_local": true}'
```

## Tuning (see `.env` / `docker-compose.yml`)

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_ID` | `google/gemma-3-4b-it` | Base model (HF id or local path). |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of the 24 GB vLLM may use. |
| `MAX_MODEL_LEN` | `8192` | Context length; raise if you have KV-cache room. |
| `MAX_LORAS` | `4` | Adapters resident on GPU simultaneously. |
| `MAX_LORA_RANK` | `32` | **Must be ≥ the rank of every adapter you load**, or the load fails. |
| `MAX_CPU_LORAS` | `16` | Adapters kept in CPU RAM and swapped onto GPU on demand. |
| `VLLM_EXTRA_ARGS` | — | Passthrough, e.g. `--quantization fp8`. |

### Capacity notes for the A10G (24 GB)
- Gemma 3 4B in bf16 ≈ 9 GB, leaving ~13 GB for KV cache + adapters — comfortable.
- Each LoRA adds a small amount of VRAM; `MAX_CPU_LORAS` lets you register many
  more than fit on the GPU and swap them in per request.
- Moving to **Gemma 3 12B** on this card means bf16 is tight — add
  `VLLM_EXTRA_ARGS=--quantization fp8` (or an AWQ/GPTQ checkpoint) and expect
  less concurrency headroom.

## Production hardening (next steps)
- Put an ALB / auth proxy in front — neither port is authenticated by itself.
- Health checks: `GET :8000/health`; restart policy via ECS/EKS or
  `restart: unless-stopped`.
- Metrics: vLLM exposes Prometheus metrics at `:8000/metrics`.
- For multi-host / autoscaling, bake adapters you know you need into a warm set,
  and keep the S3-dynamic path for the long tail.
```
