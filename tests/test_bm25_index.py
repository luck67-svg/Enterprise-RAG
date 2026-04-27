from pathlib import Path
import sys
from pathlib import Path as _Path
import pytest

ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document

from app.rag.bm25_index import BM25Index, tokenize


def test_tokenize_handles_ascii_words():
    assert tokenize("expense BM25 tokens 2026") == ["expense", "bm25", "tokens", "2026"]


def test_bm25_index_persists_and_searches(tmp_path: Path):
    path = tmp_path / "bm25_chunks.json"
    index = BM25Index(path)
    docs = [
        Document(
            page_content="expense reimbursement workflow",
            metadata={
                "source": "a.txt",
                "page": 1,
                "parent_id": "p1",
                "parent_content": "expense reimbursement workflow",
                "chunk_id": "c1",
            },
        ),
        Document(
            page_content="leave approval policy",
            metadata={
                "source": "b.txt",
                "page": 2,
                "parent_id": "p2",
                "parent_content": "leave approval policy",
                "chunk_id": "c2",
            },
        ),
    ]

    index.replace_source("a.txt", [docs[0]])
    index.replace_source("b.txt", [docs[1]])

    reloaded = BM25Index(path)
    hits = reloaded.search("reimbursement", k=2)

    assert len(hits) == 1
    assert hits[0].metadata == {
        "chunk_id": "c1",
        "source": "a.txt",
        "page": 1,
        "parent_id": "p1",
        "parent_content": "expense reimbursement workflow",
    }


def test_failed_replace_source_keeps_existing_index_usable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    index = BM25Index(tmp_path / "bm25_chunks.json")
    original = Document(
        page_content="expense reimbursement workflow",
        metadata={
            "source": "a.txt",
            "page": 1,
            "parent_id": "p1",
            "parent_content": "expense reimbursement workflow",
            "chunk_id": "c1",
        },
    )
    replacement = Document(
        page_content="travel booking process",
        metadata={
            "source": "a.txt",
            "page": 9,
            "parent_id": "p9",
            "parent_content": "travel booking process",
            "chunk_id": "c9",
        },
    )

    index.replace_source("a.txt", [original])

    def fail_dumps(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.rag.bm25_index.json.dumps", fail_dumps)

    with pytest.raises(RuntimeError, match="boom"):
        index.replace_source("a.txt", [replacement])

    hits = index.search("reimbursement", k=3)

    assert len(hits) == 1
    assert hits[0].metadata["chunk_id"] == "c1"
    assert hits[0].metadata["source"] == "a.txt"


def test_delete_source_removes_chunks_from_search(tmp_path: Path):
    index = BM25Index(tmp_path / "bm25_chunks.json")
    doc = Document(
        page_content="expense reimbursement workflow",
        metadata={
            "source": "a.txt",
            "page": 1,
            "parent_id": "p1",
            "parent_content": "expense reimbursement workflow",
            "chunk_id": "c1",
        },
    )

    index.replace_source("a.txt", [doc])
    index.delete_source("a.txt")

    assert index.search("reimbursement", k=3) == []
