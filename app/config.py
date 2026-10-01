"""Runtime configuration for the LoRA manager, sourced from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Where the local vLLM OpenAI server is reachable from inside the container.
    vllm_base_url: str = "http://localhost:8000"

    # Default S3 location holding adapters, e.g. "s3://my-bucket/loras".
    # A load request without an explicit s3_uri resolves to "{lora_s3_prefix}/{lora_name}".
    lora_s3_prefix: str = ""

    # Local directory where adapters are downloaded before being handed to vLLM.
    lora_local_dir: str = "/loras"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


settings = Settings()
