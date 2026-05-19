from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:35b"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_rag"

    embedding_model: str = "bge-m3"

    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 5

    parent_chunk_size: int = 1000
    child_chunk_size: int = 400
    child_chunk_overlap: int = 60
    child_retrieval_k: int = 12
    bm25_top_k: int = 6              # 从 12 降到 6，减少 BM25 噪音候选数
    rrf_k: int = 30                  # 从 60 降到 30，让排名差异更重要
    reranker_score_threshold: float = 0.3  # 从 0.0 提高，让 Reranker 真正起过滤作用
    rerank_candidate_limit: int = 8        # 送入 Reranker 的最大候选数

    reranker_base_url: str = "http://localhost:8001"

    upload_dir: Path = Path("./data/uploads")


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
