# Architecture

A two-part platform: an **offline training path** that creates LoRA adapters and
an **online serving path** that hosts a base Gemma 3 model and attaches adapters
at inference time. **S3** is the handoff between them.

```mermaid
flowchart LR
    subgraph OFFLINE["Offline — create adapters (venv on GPU box)"]
        DATA["Chat JSONL dataset"]
        TRAIN["train_lora.py<br/>QLoRA on gemma-3-4b-it"]
        ADPT["Adapter files<br/>adapter_config.json<br/>adapter_model.safetensors"]
        DATA --> TRAIN --> ADPT
    end

    S3[("S3<br/>s3://loras-root/&lt;name&gt;/")]
    HF[("Hugging Face Hub")]

    ADPT -->|"aws s3 cp"| S3

    subgraph HOST["g5.xlarge — NVIDIA A10G 24GB"]
        subgraph CT["Docker container (entrypoint.sh)"]
            VLLM["vLLM OpenAI server :8000<br/>Gemma 3 4B bf16 + LoRAs<br/>(data plane)"]
            MGR["LoRA manager :8001<br/>FastAPI<br/>(control plane)"]
            MGR -->|"load_lora_adapter(path)"| VLLM
        end
    end

    CLIENT(["Client"]) -->|"POST :8000 /v1/chat/completions<br/>model = base id or lora_name"| VLLM
    ADMIN(["Admin"]) -->|"POST :8001 /v1/loras/load"| MGR
    MGR -->|"boto3 (instance role)"| S3
    S3 -.->|"download adapter dir"| MGR
    HF -.->|"base weights + HF_TOKEN"| VLLM
```

## Legend / flow

- **Offline:** your chat-format JSONL → `train_lora.py` (4-bit QLoRA, rank-32 on
  the language layers) → PEFT adapter files → uploaded to S3.
- **S3 (`s3://loras-root/<name>/`):** durable source of truth; one folder per adapter.
- **vLLM `:8000` (data plane):** loads Gemma 3 4B once; serves the OpenAI API.
  A request selects weights via the `model` field (base id or a loaded `lora_name`).
- **LoRA manager `:8001` (control plane):** on a load request, pulls the adapter
  from S3 to local disk (boto3 / EC2 instance role), then calls vLLM's runtime
  `load_lora_adapter` with the local path — bridging S3 to vLLM's local-path-only API.
- **Hugging Face Hub:** source of the gated base weights at startup (needs `HF_TOKEN`).

## Notes

- Solid arrows = request/data flow; dashed = pull-from-remote.
- Both ports are unauthenticated on their own — front them with an ALB/auth proxy
  or keep them private (the `:8001` control plane especially).
- Loaded-adapter set is in-memory vLLM state (cleared on restart); S3 is durable.
