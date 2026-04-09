from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger
from app.api import kb, openai_compat
from app.config import settings
from app.llm.ollama_client import list_ollama_models
from app.rag.vectorstore import get_vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 验证 Ollama
    models = list_ollama_models()
    if not models["chat"] and not models["embedding"]:
        logger.warning("Ollama unreachable at startup — chat and embedding may fail")
    else:
        logger.info(f"Ollama ready: chat={models['chat']}, embedding={models['embedding']}")

    # 预热 vectorstore（初始化 Qdrant 连接 + embedding 维度验证）
    try:
        get_vectorstore()
        logger.info("Vectorstore initialized")
    except Exception as e:
        logger.warning(f"Vectorstore init failed: {e}")

    yield


app = FastAPI(title="Enterprise RAG", version="0.1.0", lifespan=lifespan)
app.include_router(kb.router)
app.include_router(openai_compat.router)


@app.get("/health")
def health():
    models = list_ollama_models()
    chat_models = models["chat"]
    embed_models = models["embedding"]

    if not chat_models and not embed_models:
        return {"status": "degraded", "ollama": "unreachable", "models": models}

    active_chat = settings.ollama_model if settings.ollama_model in chat_models else (chat_models[0] if chat_models else None)
    active_embed = settings.embedding_model if settings.embedding_model in embed_models else (embed_models[0] if embed_models else None)

    return {
        "status": "ok",
        "ollama": "reachable",
        "active": {"chat": active_chat, "embedding": active_embed},
        "models": models,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True)
