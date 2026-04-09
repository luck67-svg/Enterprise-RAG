from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:0.5b"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_rag"

    embedding_model: str = "bge-m3"
    embedding_device: str = "cpu"

    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 5

    upload_dir: Path = Path("./data/uploads")


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
