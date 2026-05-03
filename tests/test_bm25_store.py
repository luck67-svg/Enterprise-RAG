import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from langchain_core.documents import Document
from app.rag.bm25_store import _tokenize, BM25Store


def test_tokenize_chinese():
    tokens = _tokenize("最大功率输出")
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_tokenize_english():
    tokens = _tokenize("Maximum Power Output")
    assert "maximum" in tokens
    assert "power" in tokens


def test_tokenize_mixed():
    tokens = _tokenize("型号 XYZ-2000 最大功率 500W")
    assert len(tokens) > 0


def make_doc(content: str, source: str, parent_id: str = "p1") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "parent_id": parent_id, "parent_content": content},
    )


def test_add_and_search(tmp_path):
    store = BM25Store(index_path=tmp_path / "bm25.pkl")
    store.add_documents([
        make_doc("最大功率输出为500W", "manual.pdf", "p1"),
        make_doc("故障恢复流程说明", "manual.pdf", "p2"),
        make_doc("Maximum power output 500W", "spec.pdf", "p3"),
    ])
    results = store.search("最大功率", k=2)
    assert len(results) == 2
    assert results[0].page_content == "最大功率输出为500W"


def test_search_empty_store(tmp_path):
    store = BM25Store(index_path=tmp_path / "bm25.pkl")
    results = store.search("任意查询", k=5)
    assert results == []


def test_remove_by_source(tmp_path):
    store = BM25Store(index_path=tmp_path / "bm25.pkl")
    store.add_documents([
        make_doc("内容A", "fileA.pdf", "p1"),
        make_doc("内容B", "fileB.pdf", "p2"),
    ])
    store.remove_by_source("fileA.pdf")
    results = store.search("内容A", k=5)
    assert all(d.metadata["source"] != "fileA.pdf" for d in results)


def test_persistence_roundtrip(tmp_path):
    idx = tmp_path / "bm25.pkl"
    store1 = BM25Store(index_path=idx)
    store1.add_documents([make_doc("持久化测试内容", "test.pdf", "p1")])

    store2 = BM25Store(index_path=idx)
    results = store2.search("持久化测试", k=1)
    assert len(results) == 1
    assert results[0].metadata["source"] == "test.pdf"
