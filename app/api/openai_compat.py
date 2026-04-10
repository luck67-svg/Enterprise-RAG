"""Minimal OpenAI-compatible /v1 endpoints so Open WebUI can talk to us as a 'model'."""
import asyncio
import json
import time
import uuid
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger
from pydantic import BaseModel

from app.rag.chain import get_rag_chain

router = APIRouter(prefix="/v1", tags=["openai"])

MODEL_ID = "enterprise-rag"
REQUEST_TIMEOUT = 120       # 秒，非流式总超时
STREAM_TOKEN_TIMEOUT = 30   # 秒，流式单 token 超时


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None

    @property
    def effective_temperature(self) -> float:
        return self.temperature if self.temperature is not None else 0.2


MAX_HISTORY_TURNS = 10  # 最多保留的历史轮数


@router.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "local"}
        ],
    }


def _extract_history_and_question(
    messages: list[ChatMessage],
) -> tuple[list[HumanMessage | AIMessage], str]:
    """从 OpenAI 格式的 messages 中提取历史对话和当前问题。

    返回:
        history: LangChain 消息对象列表（HumanMessage / AIMessage）
        question: 当前用户问题字符串
    """
    question = ""
    history: list[HumanMessage | AIMessage] = []

    for i, m in enumerate(messages):
        if m.role == "system":
            continue
        if m.role == "user" and i == len(messages) - 1:
            question = m.content
        elif m.role == "user":
            history.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            history.append(AIMessage(content=m.content))

    # 限制历史轮数（每轮 = user + assistant 两条）
    history = history[-(MAX_HISTORY_TURNS * 2):]
    return history, question


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    history, question = _extract_history_and_question(req.messages)

    if not question.strip():
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "最后一条消息必须是非空的 user 消息", "type": "invalid_request_error"}},
        )

    chain = get_rag_chain(temperature=req.effective_temperature)
    chain_input = {"question": question, "chat_history": history}

    if history:
        logger.info(f"chat history: {len(history)} messages")

    if req.stream:
        return StreamingResponse(
            _stream_response(chain, chain_input, request),
            media_type="text/event-stream",
        )

    t0 = time.perf_counter()
    try:
        logger.info(f"invoking chain | question: {question[:50]}")
        answer = await asyncio.wait_for(chain.ainvoke(chain_input), timeout=REQUEST_TIMEOUT)
        logger.info(f"chain total: {time.perf_counter() - t0:.2f}s")
    except asyncio.TimeoutError:
        logger.warning(f"chain timeout after {REQUEST_TIMEOUT}s")
        return _wrap_response("请求超时，请稍后重试或缩短问题。")
    except asyncio.CancelledError:
        logger.info("request cancelled by client")
        raise
    except Exception as e:
        logger.error(f"chain error after {time.perf_counter() - t0:.2f}s: {e!r}")
        return _wrap_response(_chain_error_message(e))

    return _wrap_response(answer)


async def _stream_response(chain, chain_input: dict, request: Request):
    """SSE streaming with client disconnect detection and per-token timeout."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()
    logger.info(f"streaming chain | question: {chain_input['question'][:50]}")

    try:
        async for token in chain.astream(chain_input):
            if await request.is_disconnected():
                logger.info("client disconnected, stopping stream")
                return

            # 用 wait_for 包单次 yield 防止 Ollama 卡死
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    except asyncio.TimeoutError:
        logger.warning(f"stream token timeout after {STREAM_TOKEN_TIMEOUT}s")
        err_chunk = _make_error_chunk(chat_id, "流式响应超时，Ollama 可能已卡死，请检查服务状态。")
        yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return
    except asyncio.CancelledError:
        logger.info("stream cancelled by client")
        return
    except Exception as e:
        logger.error(f"stream error: {e!r}")
        err_chunk = _make_error_chunk(chat_id, _chain_error_message(e))
        yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
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


def _make_error_chunk(chat_id: str, message: str) -> dict:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {"content": message}, "finish_reason": "stop"}],
    }


def _chain_error_message(exc: Exception) -> str:
    """将 chain 异常转为用户可读的回答。"""
    text = str(exc).lower()
    if "connect" in text and "11434" in text:
        return "[错误] Ollama 服务不可达，请检查是否已启动。"
    if "connect" in text and "6333" in text:
        return "[错误] Qdrant 不可达，请检查 Docker 容器是否已启动。"
    if "ollama" in text or "refused" in text:
        return "[错误] Ollama 连接失败，请检查服务状态。"
    return f"[错误] 处理请求时出现异常: {type(exc).__name__}"


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
