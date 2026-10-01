"""Download LoRA adapters from S3 to local disk so vLLM can load them by path."""

import logging
import os
from urllib.parse import urlparse

import boto3

logger = logging.getLogger("lora_manager.s3")

# Files that make up a PEFT/LoRA adapter. We only require the config + weights.
_REQUIRED = ("adapter_config.json",)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Not a valid s3:// URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_adapter(s3_uri: str, lora_name: str, local_root: str) -> str:
    """Sync every object under ``s3_uri`` into ``{local_root}/{lora_name}``.

    Returns the local directory path. Raises FileNotFoundError if the prefix is
    empty or ValueError if the adapter is missing required files.
    """
    bucket, prefix = _parse_s3_uri(s3_uri)
    local_dir = os.path.join(local_root, lora_name)
    os.makedirs(local_dir, exist_ok=True)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):  # skip "directory" placeholder keys
                continue
            rel = os.path.relpath(key, prefix)
            dest = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dest) or local_dir, exist_ok=True)
            logger.info("downloading s3://%s/%s -> %s", bucket, key, dest)
            s3.download_file(bucket, key, dest)
            downloaded += 1

    if downloaded == 0:
        raise FileNotFoundError(f"No objects found under s3://{bucket}/{prefix}")

    missing = [f for f in _REQUIRED if not os.path.exists(os.path.join(local_dir, f))]
    if missing:
        raise ValueError(
            f"Adapter at {s3_uri} is missing required file(s): {', '.join(missing)}"
        )

    return local_dir
