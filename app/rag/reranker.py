import httpx
from langchain_core.documents import Document
from loguru import logger

from app.config import settings


def rerank_documents(query: str, documents: list[Document]) -> list[Document]:
    """通过远程 reranker 服务对父块重排，过滤低相关块后返回降序列表。

    分数低于 reranker_score_threshold 的块被丢弃，但至少保留得分最高的 1 个。
    """
    if not documents:
        return documents

    texts = [doc.page_content for doc in documents]
    try:
        resp = httpx.post(
            f"{settings.reranker_base_url}/rerank",
            json={"query": query, "documents": texts},
            timeout=httpx.Timeout(connect=2.0, read=8.0, write=2.0, pool=2.0),
        )
        resp.raise_for_status()
        scores = resp.json()["scores"]
    except Exception as e:
        logger.warning(f"Reranker service unavailable ({e}), falling back to original order")
        return documents

    if len(scores) != len(documents):
        logger.warning(f"Reranker returned {len(scores)} scores for {len(documents)} docs, falling back")
        return documents

    scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    logger.debug(f"Reranker scores: {[round(float(s), 3) for s, _ in scored]}")

    threshold = settings.reranker_score_threshold
    filtered = [(s, doc) for s, doc in scored if s >= threshold]
    if not filtered:
        filtered = [scored[0]]  # 兜底：至少保留最高分块
        logger.debug(f"All scores below threshold {threshold}, keeping top-1 as fallback")
    else:
        logger.debug(f"Reranker threshold {threshold}: kept {len(filtered)}/{len(scored)} docs")

    return [doc for _, doc in filtered]
