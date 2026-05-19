#!/usr/bin/env bash
# 项目一键启动脚本：Docker 容器 + Ollama + Reranker + FastAPI
# 使用方式：./scripts/start_tunnel.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_USER="veridian"
REMOTE_HOST="192.168.144.129"
REMOTE_OLLAMA_PORT=11434
REMOTE_RERANKER_PORT=8001
LOCAL_OLLAMA_PORT=11434
LOCAL_RERANKER_PORT=8001
SSH_KEY="$HOME/.ssh/veridian"
SSH_OPTS=(-i "$SSH_KEY" -o BatchMode=yes)
REMOTE_OLLAMA="/mnt/mydisk/home/veridian/ollama/bin/ollama"
REMOTE_VENV_PY="/mnt/mydisk/home/veridian/LJY/.venv/bin/python"
REMOTE_MODEL_PATH="/mnt/mydisk/home/veridian/LJY/bge-reranker-v2-m3"

mkdir -p "${PROJECT_DIR}/logs"
OLLAMA_LOG="${PROJECT_DIR}/logs/ollama.log"
FASTAPI_LOG="${PROJECT_DIR}/logs/fastapi.log"

# ---------- 1. 启动 Docker 容器 ----------
echo "=== 启动 Docker 容器..."
docker compose -f "${SCRIPT_DIR}/docker-compose.yml" up -d
echo "    Qdrant:     http://localhost:6333"
echo "    Open WebUI: http://localhost:3000"

# ---------- 2. 启动远程 Ollama ----------
echo "=== 检查远程 Ollama 状态..."
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" bash -c "
    if curl -sf http://localhost:${REMOTE_OLLAMA_PORT}/api/tags >/dev/null 2>&1; then
        echo 'Ollama 已在运行'
    else
        echo '正在启动 Ollama...'
        nohup ${REMOTE_OLLAMA} serve </dev/null >>/tmp/ollama.log 2>&1 & disown
        sleep 3
        if curl -sf http://localhost:${REMOTE_OLLAMA_PORT}/api/tags >/dev/null 2>&1; then
            echo 'Ollama 启动成功'
        else
            echo 'Ollama 启动失败，请检查 /tmp/ollama.log'
            exit 1
        fi
    fi
"

# SSH 隧道（Ollama）
echo "=== 建立 Ollama SSH 隧道: localhost:${LOCAL_OLLAMA_PORT} -> ${REMOTE_HOST}:${REMOTE_OLLAMA_PORT}"
if ss -tln 2>/dev/null | grep -q ":${LOCAL_OLLAMA_PORT} "; then
    echo "    Ollama 隧道端口 ${LOCAL_OLLAMA_PORT} 已在监听，跳过"
else
    ssh "${SSH_OPTS[@]}" -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -fNL "${LOCAL_OLLAMA_PORT}:localhost:${REMOTE_OLLAMA_PORT}" "${REMOTE_USER}@${REMOTE_HOST}"
    echo "    隧道已在后台运行"
fi

# ---------- 3. 检查/启动远程 Reranker ----------
echo "=== 检查远程 Reranker 服务..."
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "
    if curl -sf http://localhost:${REMOTE_RERANKER_PORT}/health >/dev/null 2>&1; then
        echo 'Reranker 已在运行'
    elif [ ! -f \"\$HOME/reranker_service.py\" ]; then
        echo '[ERROR] ~/reranker_service.py not found on remote. Run deploy_reranker_remote.sh first.'
    else
        echo '正在启动 Reranker...'
        cd \"\$HOME\" && RERANKER_MODEL_PATH=${REMOTE_MODEL_PATH} nohup ${REMOTE_VENV_PY} -m uvicorn reranker_service:app --host 0.0.0.0 --port ${REMOTE_RERANKER_PORT} </dev/null >>\"\$HOME/reranker.log\" 2>&1 & disown
        sleep 8
        if curl -sf http://localhost:${REMOTE_RERANKER_PORT}/health >/dev/null 2>&1; then
            echo 'Reranker 启动成功'
        else
            echo '[WARN] Reranker 启动失败，请检查远程日志 ~/reranker.log'
        fi
    fi
"

# SSH 隧道（Reranker）
echo "=== 建立 Reranker SSH 隧道: localhost:${LOCAL_RERANKER_PORT} -> ${REMOTE_HOST}:${REMOTE_RERANKER_PORT}"
if ss -tln 2>/dev/null | grep -q ":${LOCAL_RERANKER_PORT} "; then
    echo "    Reranker 隧道端口 ${LOCAL_RERANKER_PORT} 已在监听，跳过"
else
    ssh "${SSH_OPTS[@]}" -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -fNL "${LOCAL_RERANKER_PORT}:localhost:${REMOTE_RERANKER_PORT}" "${REMOTE_USER}@${REMOTE_HOST}"
    echo "    Reranker 隧道已在后台运行"
fi

# ---------- 验证服务 + 预热模型 ----------
echo "=== 验证服务..."
sleep 2
curl -sf http://localhost:${LOCAL_OLLAMA_PORT}/api/tags >/dev/null 2>&1 && echo "    Ollama tunnel OK" || echo "    [WARN] Ollama tunnel not ready"
curl -sf http://localhost:${LOCAL_RERANKER_PORT}/health >/dev/null 2>&1 && echo "    Reranker OK" || echo "    [WARN] Reranker not ready"

# 从 .env 读取模型名，后台预热让模型加载进显存
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:35b}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-bge-m3}"
if [ -f "${PROJECT_DIR}/.env" ]; then
    _m=$(grep -E '^OLLAMA_MODEL=' "${PROJECT_DIR}/.env" | cut -d= -f2)
    _e=$(grep -E '^EMBEDDING_MODEL=' "${PROJECT_DIR}/.env" | cut -d= -f2)
    [ -n "$_m" ] && OLLAMA_MODEL="$_m"
    [ -n "$_e" ] && EMBEDDING_MODEL="$_e"
fi
echo "=== 预热 Ollama 模型（后台）: ${EMBEDDING_MODEL}, ${OLLAMA_MODEL}"
(sleep 5 && curl -s -X POST "http://localhost:${LOCAL_OLLAMA_PORT}/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${EMBEDDING_MODEL}\",\"prompt\":\"\",\"stream\":false,\"keep_alive\":\"30m\"}" \
    >/dev/null 2>&1 && echo "    [warm] ${EMBEDDING_MODEL} ready") &
(sleep 5 && curl -s -X POST "http://localhost:${LOCAL_OLLAMA_PORT}/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${OLLAMA_MODEL}\",\"prompt\":\"\",\"stream\":false,\"keep_alive\":\"30m\"}" \
    >/dev/null 2>&1 && echo "    [warm] ${OLLAMA_MODEL} ready") &

# ---------- 4. 启动 FastAPI ----------
echo "=== 启动 FastAPI 服务..."
echo "    API:  http://localhost:8000"
echo "    Docs: http://localhost:8000/docs"
cd "${PROJECT_DIR}"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --use-colors 2>&1 | tee "${FASTAPI_LOG}"