#!/usr/bin/env bash
# 项目一键启动脚本：Docker 容器 + 远程 Ollama + SSH 隧道 + FastAPI
# Usage: ./scripts/start_tunnel.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_USER="${REMOTE_SSH_USER:-veridian}"
REMOTE_HOST="${REMOTE_SSH_HOST:-192.168.144.129}"
REMOTE_PORT="${REMOTE_OLLAMA_PORT:-11434}"
LOCAL_PORT="${LOCAL_TUNNEL_PORT:-11434}"
SSH_KEY="${SSH_KEY_PATH:-$HOME/.ssh/veridian}"
SSH_OPTS=(-i "$SSH_KEY" -o BatchMode=yes)
REMOTE_OLLAMA="/mnt/mydisk/home/veridian/ollama/bin/ollama"

# ---------- 1. 启动 Docker 容器（Qdrant + Open WebUI） ----------
echo ">>> 启动 Docker 容器..."
docker compose -f "${SCRIPT_DIR}/docker-compose.yml" up -d
echo "    Qdrant:     http://localhost:6333"
echo "    Open WebUI: http://localhost:3000"

# ---------- 2. 在远程服务器上启动 Ollama（如果尚未运行） ----------
echo ">>> 检查远程 Ollama 状态..."
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" bash -c "'
if curl -sf http://localhost:${REMOTE_PORT}/api/tags >/dev/null 2>&1; then
    echo \"Ollama 已在运行\"
else
    echo \"正在启动 Ollama...\"
    nohup ${REMOTE_OLLAMA} serve > /tmp/ollama.log 2>&1 &
    sleep 3
    if curl -sf http://localhost:${REMOTE_PORT}/api/tags >/dev/null 2>&1; then
        echo \"Ollama 启动成功\"
    else
        echo \"Ollama 启动失败，请检查远程日志 /tmp/ollama.log\" >&2
        exit 1
    fi
fi
'"

# ---------- 3. 建立 SSH 端口转发（后台） ----------
echo ">>> 建立 SSH 隧道: localhost:${LOCAL_PORT} -> ${REMOTE_HOST}:${REMOTE_PORT}"
# 如果隧道已存在则跳过
if lsof -ti :"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "    端口 ${LOCAL_PORT} 已被占用，跳过隧道建立"
else
    ssh "${SSH_OPTS[@]}" -fNL "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}"
    echo "    隧道已在后台运行"
fi

# ---------- 4. 同步远程 Ollama 日志到本地（后台） ----------
OLLAMA_LOG="${PROJECT_DIR}/logs/ollama.log"
mkdir -p "${PROJECT_DIR}/logs"
echo ">>> 同步远程 Ollama 日志 -> ${OLLAMA_LOG}"
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "tail -f /tmp/ollama.log" > "${OLLAMA_LOG}" 2>&1 &
LOG_SYNC_PID=$!

# ---------- 5. 启动 FastAPI 服务 ----------
# Ctrl+C 时关闭 SSH 隧道和日志同步
cleanup() {
    echo ""
    echo ">>> 正在清理后台进程..."
    # 关闭日志同步
    kill "$LOG_SYNC_PID" 2>/dev/null && echo "    日志同步已关闭"
    # 关闭 SSH 隧道
    TUNNEL_PID=$(lsof -ti :"${LOCAL_PORT}" -sTCP:LISTEN 2>/dev/null)
    if [[ -n "$TUNNEL_PID" ]]; then
        kill "$TUNNEL_PID" 2>/dev/null && echo "    SSH 隧道已关闭"
    fi
}
trap cleanup EXIT

FASTAPI_LOG="${PROJECT_DIR}/logs/fastapi.log"

echo ">>> 启动 FastAPI 服务..."
echo "    API:  http://localhost:8000"
echo "    Docs: http://localhost:8000/docs"
echo "    Ollama 日志: tail -f ${OLLAMA_LOG}"
echo "    FastAPI 日志: tail -f ${FASTAPI_LOG}"
cd "${PROJECT_DIR}"
# tee 到终端保留颜色，写入文件时用 sed 去除 ANSI 转义码
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --use-colors 2>&1 | tee >(sed 's/\x1b\[[0-9;]*m//g' > "${FASTAPI_LOG}")
