from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from loguru import logger

from app.config import settings
from app.llm.ollama_client import get_llm
from app.rag.vectorstore import get_vectorstore

SYSTEM_PROMPT = """你是一个企业知识库问答助手。请严格依据下面提供的【上下文】回答用户问题。
如果上下文中没有答案，请直接回答"根据已有资料无法回答"，不要编造。
回答末尾用 [来源: 文件名 p.页码] 的形式列出引用。

【上下文】
{context}
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# 模块级单例缓存，避免每次请求重建连接
_chain_cache: dict[float, object] = {}


def _format_docs(docs: list[Document]) -> str:
    logger.info(f"retrieval done: {len(docs)} chunks")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        logger.debug(f"  [{i}] {src} p.{page}: {d.page_content[:80]}")
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"[{i}] (source={src} page={page})\n{d.page_content}")
    return "\n\n".join(parts)


def _expand_to_parents(docs: list[Document]) -> list[Document]:
    """将子块列表展开为去重后的父块列表。
    无 parent_content 的旧格式 doc 原样返回（向后兼容）。
    """
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


def get_retriever():
    """返回检索器：检索子块(child_retrieval_k) → 去重展开为父块(retrieval_top_k)。
    供评估脚本和 RAG chain 使用。
    """
    base_retriever = get_vectorstore().as_retriever(
        search_kwargs={"k": settings.child_retrieval_k}
    )
    final_top_k = settings.retrieval_top_k

    class ParentExpandingRetriever(BaseRetriever):
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> list[Document]:
            child_docs = base_retriever.invoke(query)
            parents = _expand_to_parents(child_docs)
            return parents[:final_top_k]

    return ParentExpandingRetriever()


def get_rag_chain(temperature: float = 0.2):
    """返回 RAG chain 单例，相同 temperature 复用同一实例。"""
    if temperature not in _chain_cache:
        retriever = get_retriever()
        llm = get_llm(temperature=temperature)
        _chain_cache[temperature] = (
            {
                "context": RunnableLambda(lambda x: x["question"]) | retriever | _format_docs,
                "question": RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: x["chat_history"]),
            }
            | PROMPT
            | llm
            | StrOutputParser()
        ).with_config({"run_name": "rag_chain"})
        logger.info(f"RAG chain initialized (temperature={temperature})")
    return _chain_cache[temperature]
