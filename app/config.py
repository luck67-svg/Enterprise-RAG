from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:0.8b"


    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_rag"

    embedding_model: str = "bge-m3"
    embedding_device: str = "cpu"

    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 5

    parent_chunk_size: int = 1000
    child_chunk_size: int = 400
    child_chunk_overlap: int = 60
    child_retrieval_k: int = 25

    reranker_base_url: str = "http://localhost:8001"

    upload_dir: Path = Path("./data/uploads")


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
