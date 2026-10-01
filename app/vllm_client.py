"""Thin async client for vLLM's runtime LoRA endpoints and health/model listing."""

import httpx


class VLLMError(RuntimeError):
    """Raised when vLLM returns a non-2xx response."""


class VLLMClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def is_ready(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> dict:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/v1/models")
            resp.raise_for_status()
            return resp.json()

    async def load_lora(self, lora_name: str, lora_path: str) -> None:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{self.base_url}/v1/load_lora_adapter",
                json={"lora_name": lora_name, "lora_path": lora_path},
            )
            if resp.status_code != 200:
                raise VLLMError(f"load_lora_adapter failed ({resp.status_code}): {resp.text}")

    async def unload_lora(self, lora_name: str) -> None:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/v1/unload_lora_adapter",
                json={"lora_name": lora_name},
            )
            if resp.status_code != 200:
                raise VLLMError(f"unload_lora_adapter failed ({resp.status_code}): {resp.text}")
