from langchain_core.documents import Document

from app.rag.splitter import split_parent_child


def test_split_parent_child_assigns_unique_chunk_ids():
    docs = [
        Document(
            page_content=("expense policy " * 200),
            metadata={"source": "hr.pdf", "page": 1},
        )
    ]

    chunks = split_parent_child(docs)

    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    assert chunk_ids
    assert len(chunk_ids) == len(set(chunk_ids))
    assert all(chunk.metadata["parent_id"] for chunk in chunks)
    assert all(chunk.metadata["parent_content"] for chunk in chunks)
