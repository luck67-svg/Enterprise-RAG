import httpx
from langchain_ollama import ChatOllama
from app.config import settings


# Known embedding model name keywords
_EMBED_KEYWORDS = ("embed", "bge", "minilm", "e5", "gte", "dmeta")


def list_ollama_models() -> dict[str, list[str]]:
    """Return models categorized into 'chat' and 'embedding'."""
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return {"chat": [], "embedding": []}

    chat, embedding = [], []
    for name in names:
        lower = name.lower()
        if any(kw in lower for kw in _EMBED_KEYWORDS):
            embedding.append(name)
        else:
            chat.append(name)
    return {"chat": chat, "embedding": embedding}


def get_llm(temperature: float = 0.2) -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=temperature,
    )
