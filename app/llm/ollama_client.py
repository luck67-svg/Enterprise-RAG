from langchain_ollama import ChatOllama
from app.config import settings


def get_llm(temperature: float = 0.2) -> ChatOllama:
    """Return a ChatOllama pointed at the (tunneled) remote Ollama."""
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=temperature,
    )
