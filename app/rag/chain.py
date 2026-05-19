from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from loguru import logger

from app.config import settings
from app.llm.ollama_client import get_llm

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

# 多轮问题改写提示词：将依赖上文的问题改为可独立检索的独立问题
_CONTEXTUALIZE_Q_SYSTEM = (
    "给定聊天历史和最新的用户问题，将问题改写为一个无需上下文即可独立理解的问题。"
    "若问题中含有代词或指代（如：它、这个、上述、该），替换为具体名词。"
    "若问题已可独立理解，原样返回。不要回答问题，只改写或原样返回问题。"
)

_CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _CONTEXTUALIZE_Q_SYSTEM),
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


def _log_rewritten_query(q: str) -> str:
    logger.info(f"query rewritten → {q[:100]}")
    return q


def get_retriever():
    """返回混合检索器：BM25 + Dense → RRF → Parent Expand → Rerank。"""
    from app.rag.hybrid_retriever import HybridRetriever
    return HybridRetriever()


def get_rag_chain(temperature: float = 0.2):
    """返回 RAG chain 单例，相同 temperature 复用同一实例。"""
    if temperature not in _chain_cache:
        retriever = get_retriever()
        llm = get_llm(temperature=temperature)

        # 有历史时：LLM 先将依赖上文的问题改写为独立查询，再检索
        # 无历史时：直接用原问题检索，零额外 LLM 调用
        contextualize_q_chain = (
            _CONTEXTUALIZE_Q_PROMPT
            | llm
            | StrOutputParser()
            | RunnableLambda(_log_rewritten_query)
        )
        history_aware_retriever = RunnableBranch(
            (
                lambda x: bool(x.get("chat_history")),
                contextualize_q_chain | retriever,
            ),
            RunnableLambda(lambda x: x["question"]) | retriever,
        )

        _chain_cache[temperature] = (
            {
                "context": history_aware_retriever | _format_docs,
                "question": RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: x["chat_history"]),
            }
            | PROMPT
            | llm
            | StrOutputParser()
        ).with_config({"run_name": "rag_chain"})
        logger.info(f"RAG chain initialized (temperature={temperature})")
    return _chain_cache[temperature]
