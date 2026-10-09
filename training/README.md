# Training custom LoRA adapters

Fine-tune your own LoRA on `google/gemma-3-4b-it`, then serve it on the running
gemma-lora-server. Output plugs straight into the dynamic-load pipeline.

## 1. One-time setup (on the GPU box)

Training needs a different dependency set than the inference container, so use a
venv — **don't** install these into the server image.

```bash
cd training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt
export HF_TOKEN=<your-token>   # gated Gemma weights
```

> **GPU contention:** training and the inference container both want the full
> 24 GB. Stop the server while you train: `docker stop gemma-lora` (restart with
> `docker start gemma-lora` after).

## 2. Prepare your data

One JSON object per line, each a chat conversation. See [data/sample.jsonl](data/sample.jsonl)
(a "talks like a pirate" demo — a deliberately obvious style so you can *see* the
fine-tune worked):

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Even 10–100 good examples produce a visibly different model. More, higher-quality
examples = better results.

## 3. Train

```bash
python train_lora.py \
  --dataset data/sample.jsonl \
  --output-dir ./out/pirate \
  --epochs 3 \
  --s3-uri s3://loras-root/pirate
```

Defaults: rank-32 LoRA, 4-bit QLoRA (fits 24 GB). Useful flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--lora-rank` | 32 | Must be ≤ the server's `MAX_LORA_RANK`. |
| `--epochs` | 3 | Raise for small datasets. |
| `--max-seq-len` | 1024 | Raise for long conversations (more VRAM). |
| `--lr` | 2e-4 | Standard LoRA learning rate. |
| `--bf16` | off | Full bf16 instead of QLoRA — more VRAM, slightly higher quality. |
| `--s3-uri` | — | Auto-upload the finished adapter here. |

With `--s3-uri`, the adapter lands in your bucket and the script prints the exact
`load` command. Without it, upload manually:

```bash
aws s3 cp ./out/pirate s3://loras-root/pirate --recursive --exclude "checkpoint-*/*"
```

## 4. Serve it

Restart the server if you stopped it, then load and test:

```bash
docker start gemma-lora
curl -X POST localhost:8001/v1/loras/load -H 'Content-Type: application/json' -d '{"lora_name":"pirate"}'
curl -X POST localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"pirate","messages":[{"role":"user","content":"How do I make coffee?"}]}'
```

If the fine-tune took, the reply comes back in pirate-speak.

## Notes
- The adapter targets the language-model layers only; the vision tower is left
  alone (matches how vLLM serves Gemma 3 LoRA).
- `base_model_name_or_path` in the output `adapter_config.json` is set to the
  `--model-id`, so it's self-describing and the server validates it on load.
- First run downloads the base model if it isn't cached; subsequent runs are fast.
