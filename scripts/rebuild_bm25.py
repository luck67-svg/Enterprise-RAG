"""重建 BM25 索引：从 Qdrant 中读取所有子块，写入 BM25Store。

运行方式：
    uv run python scripts/rebuild_bm25.py

注意：
    - 需要 Qdrant 正在运行
    - 运行后会覆盖现有的 data/bm25_index.pkl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from app.rag.bm25_store import BM25Store, get_bm25_store
from app.rag.vectorstore import get_client, ensure_collection
from app.config import settings
from langchain_core.documents import Document


def rebuild():
    """从 Qdrant 读取所有已有文档，重建 BM25 索引。"""
    ensure_collection()
    client = get_client()

    all_docs: list[Document] = []
    offset = None
    batch_size = 100

    logger.info("Scanning Qdrant for existing chunks...")

    while True:
        results, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            payload = point.payload or {}
            content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            if content:
                all_docs.append(Document(page_content=content, metadata=metadata))

        logger.debug(f"Scanned {len(all_docs)} chunks so far...")
        if next_offset is None:
            break
        offset = next_offset

    if not all_docs:
        logger.warning("No documents found in Qdrant. Upload documents first.")
        return

    # 使用全局单例追加（内部会覆盖现有索引）
    store = get_bm25_store()
    store._documents = []  # 清空现有索引
    store.add_documents(all_docs)  # 重建

    logger.info(f"BM25 index rebuilt: {len(all_docs)} chunks → data/bm25_index.pkl")


if __name__ == "__main__":
    rebuild()