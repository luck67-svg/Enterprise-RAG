import time

import httpx
from langchain_core.documents import Document
from loguru import logger

from app.config import settings

# 重试配置
_MAX_RETRIES = 3
_RETRY_DELAY_BASE = 1.0  # 指数退避基数（秒）


def rerank_documents(query: str, documents: list[Document]) -> list[Document]:
    """通过远程 reranker 服务对父块重排，过滤低相关块后返回降序列表。

    - 候选文档 <= 3 时跳过 reranker（排序差异不大，节省算力）
    - 超时使用指数退避重试，避免瞬时故障导致降级
    - 失败时抛出异常而非静默降级，强制调用方处理
    """
    if not documents:
        return documents

    # 优化：候选少时不rerank，节省算力和延迟
    if len(documents) <= 3:
        logger.debug(f"Skipping reranker for {len(documents)} docs (not worth computation)")
        return documents

    texts = [doc.page_content for doc in documents]
    last_error = None

    # 指数退避重试
    for attempt in range(_MAX_RETRIES):
        try:
            resp = httpx.post(
                f"{settings.reranker_base_url}/rerank",
                json={"query": query, "documents": texts},
                timeout=httpx.Timeout(connect=2.0, read=12.0, write=2.0, pool=2.0),
            )
            resp.raise_for_status()
            scores = resp.json()["scores"]
            break  # 成功，跳出重试循环

        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAY_BASE ** attempt
                logger.warning(
                    f"Reranker attempt {attempt + 1} failed ({e}), retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                # 所有重试失败，抛出异常而非降级
                logger.error(
                    f"Reranker failed after {_MAX_RETRIES} attempts: {last_error}"
                )
                # 降级：返回原始顺序（因为无法确定哪些文档更好）
                # 但这次是明确的降级行为，应该让调用方知道
                raise RuntimeError(
                    f"Reranker unavailable after {attempt + 1} retries"
                ) from last_error

    if len(scores) != len(documents):
        logger.warning(
            f"Reranker returned {len(scores)} scores for {len(documents)} docs, "
            "falling back to original order"
        )
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
