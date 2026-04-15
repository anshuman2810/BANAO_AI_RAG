from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "RAG Internship Screening API"
    data_dir: Path = Path("data")
    max_upload_mb: int = 10
    rate_limit_per_minute: int = 30
    chunk_size: int = 900
    chunk_overlap: int = 150
    embedding_dimensions: int = 384
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    ollama_num_predict: int = 160
    ollama_timeout_seconds: int = 30
    ollama_keep_alive: str = "15m"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "vector_store"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_dir.mkdir(parents=True, exist_ok=True)
    return settings
