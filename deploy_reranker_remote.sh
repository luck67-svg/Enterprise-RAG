#!/bin/bash
# 将本地 app/reranker_service.py 部署到远程服务器并启动
# Usage: bash deploy_reranker_remote.sh

set -e

REMOTE_USER=veridian
REMOTE_HOST=192.168.144.129
REMOTE_PORT=8001
SSH_KEY="$HOME/.ssh/veridian"
SSH_OPTS=(-i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_SERVICE="${SCRIPT_DIR}/app/reranker_service.py"

REMOTE_VENV_PY="/mnt/mydisk/home/veridian/LJY/.venv/bin/python"
REMOTE_MODEL_PATH="/mnt/mydisk/home/veridian/LJY/bge-reranker-v2-m3"

echo "=== [1/3] 检查远程模型目录..."
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "[ -d '${REMOTE_MODEL_PATH}' ] && echo 'Model OK: ${REMOTE_MODEL_PATH}' || echo '[WARN] Model not found at ${REMOTE_MODEL_PATH}'"

echo "=== [2/3] 上传 reranker_service.py..."
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "${LOCAL_SERVICE}" "${REMOTE_USER}@${REMOTE_HOST}:~/reranker_service.py"
echo "      Uploaded to ~/reranker_service.py"

echo "=== [3/3] 重启远程 Reranker 服务..."
# 先停旧进程（按端口杀，避免 pkill 自匹配 bash session）
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "fuser -k ${REMOTE_PORT}/tcp 2>/dev/null; echo 'Port ${REMOTE_PORT} cleared'"

# 启动新进程（disown 让 shell 立即退出，避免 SSH 等待后台进程）
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "cd ~ && RERANKER_MODEL_PATH=${REMOTE_MODEL_PATH} nohup ${REMOTE_VENV_PY} -m uvicorn reranker_service:app --host 0.0.0.0 --port ${REMOTE_PORT} </dev/null >>~/reranker.log 2>&1 & disown && echo 'Reranker process started'"

echo "      Waiting for service to start..."
sleep 8

# 验证
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "curl -sf http://localhost:${REMOTE_PORT}/health && echo '      Reranker OK' || echo '[WARN] Not responding - check: ssh veridian@${REMOTE_HOST} tail ~/reranker.log'"

echo ""
echo "=== 部署完成 ==="
echo "  远程服务: http://${REMOTE_HOST}:${REMOTE_PORT}"
echo "  模型路径: ${REMOTE_MODEL_PATH}"
echo "  日志: ssh veridian@${REMOTE_HOST} tail -f ~/reranker.log"
