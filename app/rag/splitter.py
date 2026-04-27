import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import settings


def split_documents(docs: list[Document]) -> list[Document]:
    """原有单级切分逻辑，保留不变（向后兼容）。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def split_parent_child(docs: list[Document]) -> list[Document]:
    """
    两级切分：父块(parent_chunk_size) → 子块(child_chunk_size)。

    子块存入 Qdrant，metadata 中携带父块内容和 parent_id。
    检索时命中子块后展开为父块传给 LLM，兼顾检索精度和上下文完整性。
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=200,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
    )

    children: list[Document] = []
    for doc in docs:
        parents = parent_splitter.split_documents([doc])
        for parent in parents:
            parent_id = str(uuid.uuid4())
            child_docs = child_splitter.split_documents([parent])
            for child in child_docs:
                child.metadata["parent_id"] = parent_id
                child.metadata["parent_content"] = parent.page_content
                child.metadata["chunk_id"] = str(uuid.uuid4())
            children.extend(child_docs)

    return children
