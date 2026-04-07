from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.config import settings
from app.llm.ollama_client import get_llm
from app.rag.vectorstore import get_vectorstore

SYSTEM_PROMPT = """你是一个企业知识库问答助手。请严格依据下面提供的【上下文】回答用户问题。
如果上下文中没有答案，请直接回答"根据已有资料无法回答"，不要编造。
回答末尾用 [来源: 文件名 p.页码] 的形式列出引用。

【上下文】
{context}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"[{i}] (source={src} page={page})\n{d.page_content}")
    return "\n\n".join(parts)


def build_rag_chain():
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": settings.retrieval_top_k})
    llm = get_llm()
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
