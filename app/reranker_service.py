import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

app = FastAPI()

_model: CrossEncoder | None = None

# 模型路径：优先读环境变量，默认指向远程服务器上的本地部署路径
_MODEL_PATH = os.environ.get(
    "RERANKER_MODEL_PATH",
    "/mnt/mydisk/home/veridian/LJY/bge-reranker-v2-m3",
)


def get_model() -> CrossEncoder:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading reranker from {_MODEL_PATH!r} on {device}...")
        _model = CrossEncoder(_MODEL_PATH, device=device)
        print("Reranker model loaded.")
    return _model


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


@app.post("/rerank")
def rerank(req: RerankRequest):
    model = get_model()
    pairs = [[req.query, doc] for doc in req.documents]
    scores = model.predict(pairs)
    return {"scores": scores.tolist()}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("RERANKER_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
