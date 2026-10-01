"""LoRA manager API.

Pulls adapters from S3 to local disk, then registers/deregisters them with the
local vLLM server at runtime. Inference itself goes directly to vLLM's
OpenAI-compatible API on port 8000 using the lora_name as the ``model`` field.
"""

import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .s3_sync import download_adapter
from .vllm_client import VLLMClient, VLLMError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lora_manager")

app = FastAPI(title="Gemma 3 LoRA Manager", version="1.0.0")
vllm = VLLMClient(settings.vllm_base_url)


class LoadRequest(BaseModel):
    lora_name: str = Field(..., description="Name used to reference the adapter at inference time.")
    s3_uri: str | None = Field(
        None,
        description="s3:// location of the adapter dir. Defaults to {LORA_S3_PREFIX}/{lora_name}.",
    )


class UnloadRequest(BaseModel):
    lora_name: str
    delete_local: bool = Field(False, description="Also remove the downloaded files from disk.")


def _resolve_s3_uri(req: LoadRequest) -> str:
    if req.s3_uri:
        return req.s3_uri
    if not settings.lora_s3_prefix:
        raise HTTPException(
            status_code=400,
            detail="No s3_uri given and LORA_S3_PREFIX is not configured.",
        )
    return f"{settings.lora_s3_prefix.rstrip('/')}/{req.lora_name}"


@app.get("/health")
async def health():
    return {"manager": "ok", "vllm_ready": await vllm.is_ready()}


@app.get("/v1/loras")
async def list_loras():
    """List everything vLLM currently serves (base model + loaded adapters)."""
    try:
        return await vllm.list_models()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Could not reach vLLM: {exc}") from exc


@app.post("/v1/loras/load")
async def load_lora(req: LoadRequest):
    s3_uri = _resolve_s3_uri(req)
    logger.info("loading adapter %s from %s", req.lora_name, s3_uri)
    try:
        local_path = download_adapter(s3_uri, req.lora_name, settings.lora_local_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001  (boto/network/credentials errors)
        raise HTTPException(status_code=502, detail=f"S3 download failed: {exc}") from exc

    try:
        await vllm.load_lora(req.lora_name, local_path)
    except VLLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"status": "loaded", "lora_name": req.lora_name, "source": s3_uri}


@app.post("/v1/loras/unload")
async def unload_lora(req: UnloadRequest):
    try:
        await vllm.unload_lora(req.lora_name)
    except VLLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if req.delete_local:
        target = Path(settings.lora_local_dir) / req.lora_name
        shutil.rmtree(target, ignore_errors=True)

    return {"status": "unloaded", "lora_name": req.lora_name}
