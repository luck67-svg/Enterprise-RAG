import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from app.rag.hybrid_retriever import _rrf_merge, _expand_to_parents


def doc(content: str, source: str = "f.pdf") -> Document:
    return Document(page_content=content, metadata={"source": source})


def test_rrf_merge_deduplicates():
    d1 = doc("chunk A")
    d2 = doc("chunk B")
    d3 = doc("chunk C")
    result = _rrf_merge([d1, d2], [d2, d3], k=60, top_n=3)
    contents = [d.page_content for d in result]
    assert contents.count("chunk B") == 1


def test_rrf_merge_top_n():
    docs_dense = [doc(f"dense {i}") for i in range(5)]
    docs_sparse = [doc(f"sparse {i}") for i in range(5)]
    result = _rrf_merge(docs_dense, docs_sparse, k=60, top_n=3)
    assert len(result) == 3


def test_rrf_merge_prefers_double_hit():
    shared = doc("both lists")
    dense_only = doc("dense only")
    sparse_only = doc("sparse only")
    result = _rrf_merge([shared, dense_only], [shared, sparse_only], k=60, top_n=3)
    assert result[0].page_content == "both lists"


def test_rrf_merge_empty_sparse():
    d1 = doc("only dense")
    result = _rrf_merge([d1], [], k=60, top_n=5)
    assert len(result) == 1
    assert result[0].page_content == "only dense"


def test_expand_to_parents_deduplicates():
    child1 = Document(
        page_content="子块1",
        metadata={"parent_id": "p1", "parent_content": "父块内容", "source": "f.pdf"},
    )
    child2 = Document(
        page_content="子块2",
        metadata={"parent_id": "p1", "parent_content": "父块内容", "source": "f.pdf"},
    )
    parents = _expand_to_parents([child1, child2])
    assert len(parents) == 1
    assert parents[0].page_content == "父块内容"


def test_expand_to_parents_no_parent_content():
    plain = Document(page_content="普通块", metadata={"source": "f.pdf"})
    result = _expand_to_parents([plain])
    assert len(result) == 1
    assert result[0].page_content == "普通块"
