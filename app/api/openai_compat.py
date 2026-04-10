"""Minimal OpenAI-compatible /v1 endpoints so Open WebUI can talk to us as a 'model'."""
import asyncio
import json
import time
import uuid
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from app.config import settings
from app.rag.chain import build_rag_chain

router = APIRouter(prefix="/v1", tags=["openai"])

MODEL_ID = "enterprise-rag"
REQUEST_TIMEOUT = 120  # 秒


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
async def chat_completions(req: ChatCompletionRequest, request: Request):
    question = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    chain = build_rag_chain()

    if req.stream:
        return StreamingResponse(
            _stream_response(chain, question, request),
            media_type="text/event-stream",
        )

    t0 = time.perf_counter()
    try:
        logger.info(f"invoking chain | question: {question[:50]}")
        answer = await asyncio.wait_for(chain.ainvoke(question), timeout=REQUEST_TIMEOUT)
        logger.info(f"chain total: {time.perf_counter() - t0:.2f}s")
    except asyncio.TimeoutError:
        logger.warning(f"chain timeout after {REQUEST_TIMEOUT}s")
        return _wrap_response("请求超时，请稍后重试或缩短问题。")
    except asyncio.CancelledError:
        logger.info("request cancelled by client")
        raise
    except Exception as e:
        logger.error(f"chain error after {time.perf_counter() - t0:.2f}s: {e!r}")
        raise

    return _wrap_response(answer)


async def _stream_response(chain, question: str, request: Request):
    """SSE streaming with client disconnect detection."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()
    logger.info(f"streaming chain | question: {question[:50]}")

    try:
        async for token in chain.astream(question):
            # 检测客户端是否已断开
            if await request.is_disconnected():
                logger.info("client disconnected, stopping stream")
                return

            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    except asyncio.CancelledError:
        logger.info("stream cancelled by client")
        return

    done_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

    logger.info(f"stream total: {time.perf_counter() - t0:.2f}s")


def _wrap_response(answer: str) -> dict:
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
