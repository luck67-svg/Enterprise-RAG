from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from loguru import logger

from app.config import settings
from app.rag.bm25_store import get_bm25_store
from app.rag.reranker import rerank_documents
from app.rag.vectorstore import get_vectorstore


def _rrf_merge(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    k: int = 60,
    top_n: int = 12,
) -> list[Document]:
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content
        if key not in doc_map:
            doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_n]]


def _expand_to_parents(docs: list[Document]) -> list[Document]:
    expanded: list[Document] = []
    seen_parents: set[str] = set()
    for doc in docs:
        parent_id = doc.metadata.get("parent_id")
        parent_content = doc.metadata.get("parent_content")
        if parent_content and parent_id:
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_meta = {
                    k: v for k, v in doc.metadata.items()
                    if k not in ("parent_id", "parent_content")
                }
                expanded.append(Document(page_content=parent_content, metadata=parent_meta))
        else:
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
        reranked = rerank_documents(query, parents)
        return reranked[:settings.retrieval_top_k]
