import sys
from pathlib import Path

from langchain_core.documents import Document

# Ensure this test runs standalone without relying on other tests mutating sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag import splitter as splitter_module
from app.rag.splitter import split_parent_child


def test_split_parent_child_assigns_unique_chunk_ids(monkeypatch):
    docs = [
        Document(
            page_content=("expense policy." * 200),
            metadata={"source": "hr.pdf", "page": 1},
        )
    ]

    monkeypatch.setattr(splitter_module.settings, "parent_chunk_size", 10000)
    monkeypatch.setattr(splitter_module.settings, "child_chunk_size", 50)
    monkeypatch.setattr(splitter_module.settings, "child_chunk_overlap", 0)
    chunks = split_parent_child(docs)

    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    parent_ids = [chunk.metadata["parent_id"] for chunk in chunks]
    expected_parent_content = docs[0].page_content

    assert chunk_ids
    assert len(chunks) > 1
    assert len(set(parent_ids)) == 1
    assert len(chunk_ids) == len(set(chunk_ids))
    assert all(chunk.metadata["parent_id"] == parent_ids[0] for chunk in chunks)
    assert all(chunk.metadata["parent_content"] == expected_parent_content for chunk in chunks)
    assert all(chunk.metadata["source"] == "hr.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 1 for chunk in chunks)
