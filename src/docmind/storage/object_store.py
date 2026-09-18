from pathlib import Path
from typing import Protocol

from docmind.core.config import Settings


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes, content_type: str | None = None) -> str: ...
    def get(self, key: str) -> bytes: ...


class LocalObjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes, content_type: str | None = None) -> str:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()


class S3ObjectStore:
    def __init__(self, settings: Settings) -> None:
        import boto3
        if not settings.s3_bucket:
            raise ValueError("S3_BUCKET is required when OBJECT_STORAGE_BACKEND=s3")
        self.bucket = settings.s3_bucket
        self.client = boto3.client("s3", endpoint_url=settings.s3_endpoint_url)

    def put(self, key: str, data: bytes, content_type: str | None = None) -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type or "application/octet-stream")
        return key

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()


def get_object_store(settings: Settings) -> ObjectStore:
    return S3ObjectStore(settings) if settings.object_storage_backend == "s3" else LocalObjectStore(settings.local_object_dir)
