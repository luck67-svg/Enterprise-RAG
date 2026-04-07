from fastapi import FastAPI
from app.api import kb, openai_compat
from app.config import settings

app = FastAPI(title="Enterprise RAG", version="0.1.0")
app.include_router(kb.router)
app.include_router(openai_compat.router)


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.ollama_model}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True)
