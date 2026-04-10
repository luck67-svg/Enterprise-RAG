import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from app.api import kb, openai_compat
from app.config import settings
from app.llm.ollama_client import list_ollama_models
from app.rag.vectorstore import get_vectorstore


# 强制 loguru 输出彩色（通过 tee 管道时也生效）
logger.remove()
logger.add(sys.stderr, colorize=True)


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


# ========== 全局异常处理：友好错误提示 ==========
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    msg = _friendly_error(exc)
    logger.error(f"{request.url.path} | {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=503, content={"detail": msg})


def _friendly_error(exc: Exception) -> str:
    name = type(exc).__name__
    text = str(exc).lower()

    # Ollama 相关
    if "connect" in text and "11434" in text:
        return "Ollama 服务不可达，请检查 Ollama 是否已启动（端口 11434）"
    if "ollama" in text or ("connect" in text and ("refused" in text or "timeout" in text)):
        return "Ollama 服务连接失败，请检查 Ollama 是否正在运行"

    # Qdrant 相关
    if "qdrant" in text or ("connect" in text and "6333" in text):
        return "Qdrant 向量数据库不可达，请检查 Docker 容器是否已启动（端口 6333）"

    # Embedding 相关
    if "embed" in text and ("error" in text or "fail" in text):
        return "Embedding 模型调用失败，请检查 Ollama 中 bge-m3 模型是否已拉取"

    # 文档解析
    if name == "ValueError" and "不支持" in str(exc):
        return str(exc)

    # 兜底
    return f"服务内部错误: {name} — {exc}"


app.include_router(kb.router)
app.include_router(openai_compat.router)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


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
