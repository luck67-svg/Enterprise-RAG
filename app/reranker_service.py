import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

app = FastAPI()

_model: CrossEncoder | None = None


def get_model() -> CrossEncoder:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading bge-reranker-v2-m3 on {device}...")
        _model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
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
    uvicorn.run(app, host="0.0.0.0", port=8001)