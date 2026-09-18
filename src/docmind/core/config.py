from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "sqlite:///./data/docmind.db"
    hf_token: str | None = None
    hf_provider: str = "auto"
    qwen_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    text_embedding_model: str = "BAAI/bge-m3"
    visual_retrieval_model: str = "vidore/colqwen2-v1.0"
    visual_endpoint_url: str | None = None
    asr_model: str = "openai/whisper-large-v3-turbo"
    object_storage_backend: str = "local"
    local_object_dir: Path = Path("data/objects")
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    max_upload_mb: int = Field(default=200, gt=0)
    video_frame_interval_seconds: int = Field(default=30, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
