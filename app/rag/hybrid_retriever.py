from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from loguru import logger

from app.config import settings
from app.rag.bm25_store import get_bm25_store
from app.rag.reranker import rerank_documents
from app.rag.vectorstore import get_vectorstore

# 只对前 N 个候选rerank，减少计算量和延迟
_RERANK_CANDIDATE_LIMIT = 8


def _rrf_merge(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    k: int = 60,
    top_n: int = 12,
    dense_weight: float = 1.5,
    sparse_weight: float = 1.0,
) -> list[Document]:
    """
    加权 RRF 融合：Dense 检索权重高于 BM25。

    理由：Dense（向量语义检索）对技术术语理解更好，
    BM25 对中文术语分词仍有局限，给 Dense 更高权重更稳健。
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + dense_weight / (k + rank + 1)

    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content
        if key not in doc_map:
            doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + sparse_weight / (k + rank + 1)

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_n]]


def _expand_to_parents(docs: list[Document]) -> list[Document]:
    expanded: list[Document] = []
    seen_parent_ids: set[str] = set()
    seen_contents: set[str] = set()
    for doc in docs:
        parent_id = doc.metadata.get("parent_id")
        parent_content = doc.metadata.get("parent_content")
        if parent_content and parent_id:
            if parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                seen_contents.add(parent_content)
                parent_meta = {
                    k: v for k, v in doc.metadata.items()
                    if k not in ("parent_id", "parent_content")
                }
                expanded.append(Document(page_content=parent_content, metadata=parent_meta))
        else:
            # BM25 直接返回父块：按内容去重，避免与 Dense 展开的父块重复
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                expanded.append(doc)
    return expanded


class HybridRetriever(BaseRetriever):
    """Dense (Qdrant) + Sparse (BM25) with RRF fusion, parent expansion, and reranking."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        dense_k = settings.child_retrieval_k
        bm25_k = settings.bm25_top_k
        rrf_k = settings.rrf_k

        dense_docs = get_vectorstore().as_retriever(
            search_kwargs={"k": dense_k}
        ).invoke(query)

        sparse_docs = get_bm25_store().search(query, bm25_k)
        if not sparse_docs:
            logger.warning("BM25 index empty — rebuild by re-uploading documents")

        fused = _rrf_merge(dense_docs, sparse_docs, k=rrf_k, top_n=dense_k)
        logger.debug(f"RRF fused {len(dense_docs)} dense + {len(sparse_docs)} sparse → {len(fused)} chunks")

        parents = _expand_to_parents(fused)

        # 优化：只 rerank 前 N 个候选，减少计算量
        rerank_candidates = parents[:_RERANK_CANDIDATE_LIMIT]
        try:
            reranked = rerank_documents(query, rerank_candidates)
        except RuntimeError as e:
            # Reranker 不可用，降级到 RRF 排序结果
            logger.warning(f"Reranker unavailable: {e}, using RRF order")
            reranked = rerank_candidates

        return reranked[:settings.retrieval_top_k]
