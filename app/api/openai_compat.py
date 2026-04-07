"""Minimal OpenAI-compatible /v1 endpoints so Open WebUI can talk to us as a 'model'."""
import time
import uuid
from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.rag.chain import build_rag_chain

router = APIRouter(prefix="/v1", tags=["openai"])

MODEL_ID = "enterprise-rag"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None


@router.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "local"}
        ],
    }


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # Take the last user message as the question (stage 1: no multi-turn memory).
    question = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    chain = build_rag_chain()
    answer = await chain.ainvoke(question)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
