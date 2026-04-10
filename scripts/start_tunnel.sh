#!/usr/bin/env bash
# 项目一键启动脚本：Docker 容器 + Ollama + FastAPI
# Usage:
#   ./scripts/start_tunnel.sh          # 默认使用远程 Ollama
#   ./scripts/start_tunnel.sh --local  # 使用本地 Ollama
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-remote}"

REMOTE_USER="${REMOTE_SSH_USER:-veridian}"
REMOTE_HOST="${REMOTE_SSH_HOST:-192.168.144.129}"
REMOTE_PORT="${REMOTE_OLLAMA_PORT:-11434}"
LOCAL_PORT="${LOCAL_TUNNEL_PORT:-11434}"
SSH_KEY="${SSH_KEY_PATH:-$HOME/.ssh/veridian}"
SSH_OPTS=(-i "$SSH_KEY" -o BatchMode=yes)
REMOTE_OLLAMA="/mnt/mydisk/home/veridian/ollama/bin/ollama"

LOG_SYNC_PID=""
mkdir -p "${PROJECT_DIR}/logs"
OLLAMA_LOG="${PROJECT_DIR}/logs/ollama.log"
FASTAPI_LOG="${PROJECT_DIR}/logs/fastapi.log"

# ---------- 1. 启动 Docker 容器（Qdrant + Open WebUI） ----------
echo ">>> 启动 Docker 容器..."
docker compose -f "${SCRIPT_DIR}/docker-compose.yml" up -d
echo "    Qdrant:     http://localhost:6333"
echo "    Open WebUI: http://localhost:3000"

# ---------- 2. 启动 Ollama ----------
if [[ "$MODE" == "--local" ]]; then
    echo ">>> [本地模式] 检查本地 Ollama 状态..."
    if curl -sf http://localhost:${LOCAL_PORT}/api/tags >/dev/null 2>&1; then
        echo "    Ollama 已在运行"
    else
        echo "    正在启动本地 Ollama..."
        nohup ollama serve > "${OLLAMA_LOG}" 2>&1 &
        sleep 3
        if curl -sf http://localhost:${LOCAL_PORT}/api/tags >/dev/null 2>&1; then
            echo "    Ollama 启动成功"
        else
            echo "    Ollama 启动失败，请检查日志 ${OLLAMA_LOG}" >&2
            exit 1
        fi
    fi
else
    # ----- 远程模式：SSH 启动 + 隧道 + 日志同步 -----
    echo ">>> [远程模式] 检查远程 Ollama 状态..."
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

    # SSH 端口转发
    echo ">>> 建立 SSH 隧道: localhost:${LOCAL_PORT} -> ${REMOTE_HOST}:${REMOTE_PORT}"
    if lsof -ti :"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "    端口 ${LOCAL_PORT} 已被占用，跳过隧道建立"
    else
        ssh "${SSH_OPTS[@]}" -fNL "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}"
        echo "    隧道已在后台运行"
    fi

    # 同步远程日志
    echo ">>> 同步远程 Ollama 日志 -> ${OLLAMA_LOG}"
    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "tail -f /tmp/ollama.log" > "${OLLAMA_LOG}" 2>&1 &
    LOG_SYNC_PID=$!
fi

# ---------- 3. 启动 FastAPI 服务 ----------
cleanup() {
    echo ""
    echo ">>> 正在清理后台进程..."
    # 关闭日志同步（远程模式）
    if [[ -n "$LOG_SYNC_PID" ]]; then
        kill "$LOG_SYNC_PID" 2>/dev/null && echo "    日志同步已关闭"
    fi
    # 关闭 SSH 隧道（远程模式）
    if [[ "$MODE" != "--local" ]]; then
        TUNNEL_PID=$(lsof -ti :"${LOCAL_PORT}" -sTCP:LISTEN 2>/dev/null)
        if [[ -n "$TUNNEL_PID" ]]; then
            kill "$TUNNEL_PID" 2>/dev/null && echo "    SSH 隧道已关闭"
        fi
    fi
}
trap cleanup EXIT

echo ">>> 启动 FastAPI 服务..."
echo "    API:  http://localhost:8000"
echo "    Docs: http://localhost:8000/docs"
echo "    Ollama 日志: tail -f ${OLLAMA_LOG}"
echo "    FastAPI 日志: tail -f ${FASTAPI_LOG}"
cd "${PROJECT_DIR}"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --use-colors 2>&1 | tee >(sed 's/\x1b\[[0-9;]*m//g' > "${FASTAPI_LOG}")
