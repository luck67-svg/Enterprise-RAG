from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.config import settings
from app.rag.embeddings import get_embeddings

_client: QdrantClient | None = None
_store: QdrantVectorStore | None = None

# BGE-M3 dense dim
EMBED_DIM = 1024


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


def ensure_collection() -> None:
    client = get_client()
    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def get_vectorstore() -> QdrantVectorStore:
    global _store
    if _store is None:
        ensure_collection()
        _store = QdrantVectorStore(
            client=get_client(),
            collection_name=settings.qdrant_collection,
            embedding=get_embeddings(),
        )
    return _store
