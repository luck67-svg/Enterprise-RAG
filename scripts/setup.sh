#!/usr/bin/env bash
# 项目首次安装脚本（只需运行一次）
# Usage: ./scripts/setup.sh [--remote]
#   --remote  模型拉取到远程服务器（默认本地）
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:---local}"

# 远程配置
REMOTE_USER="${REMOTE_SSH_USER:-veridian}"
REMOTE_HOST="${REMOTE_SSH_HOST:-192.168.144.129}"
SSH_KEY="${SSH_KEY_PATH:-$HOME/.ssh/veridian}"
SSH_OPTS=(-i "$SSH_KEY" -o BatchMode=yes)
REMOTE_OLLAMA="/mnt/mydisk/home/veridian/ollama/bin/ollama"

CHAT_MODEL="qwen3.5:0.8b"
EMBED_MODEL="bge-m3"

ok()   { echo "  ✓ $1"; }
fail() { echo "  ✗ $1" >&2; exit 1; }
info() { echo ">>> $1"; }

check_cmd() {
    if command -v "$1" >/dev/null 2>&1; then
        ok "$1 已安装 ($($1 --version 2>/dev/null | head -1))"
    else
        fail "$1 未安装，请先安装: $2"
    fi
}

# ========== 1. 检查前置工具 ==========
info "检查前置工具..."
check_cmd uv "https://docs.astral.sh/uv/getting-started/installation/"
check_cmd docker "https://docs.docker.com/get-docker/"

if [[ "$MODE" == "--remote" ]]; then
    info "检查远程 SSH 连接..."
    if ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "test -x ${REMOTE_OLLAMA}"; then
        ok "远程 Ollama 可达"
    else
        fail "无法连接远程服务器或 Ollama 不存在"
    fi
else
    check_cmd ollama "https://ollama.com/download"
fi

# ========== 2. 初始化项目环境 ==========
info "安装 Python 依赖..."
cd "${PROJECT_DIR}"
uv sync
ok "依赖安装完成"

# ========== 3. Docker 拉取镜像 ==========
info "拉取 Docker 镜像..."
docker compose -f "${SCRIPT_DIR}/docker-compose.yml" pull
ok "镜像拉取完成"

# ========== 4. 启动 Ollama → 拉取模型 → 关闭 Ollama ==========
OLLAMA_STARTED_BY_US=false

if [[ "$MODE" == "--remote" ]]; then
    info "启动远程 Ollama..."
    ALREADY_RUNNING=$(ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
        "curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && echo yes || echo no")
    if [[ "$ALREADY_RUNNING" == "no" ]]; then
        ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
            "nohup ${REMOTE_OLLAMA} serve > /tmp/ollama.log 2>&1 &"
        sleep 3
        OLLAMA_STARTED_BY_US=true
    fi
    ok "远程 Ollama 已就绪"
else
    info "启动本地 Ollama..."
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        ok "Ollama 已在运行"
    else
        ollama serve > /dev/null 2>&1 &
        OLLAMA_PID=$!
        sleep 3
        OLLAMA_STARTED_BY_US=true
        ok "Ollama 已启动 (PID: $OLLAMA_PID)"
    fi
fi

info "拉取 Ollama 模型..."

pull_model() {
    local model="$1"
    echo "    拉取 ${model}..."
    if [[ "$MODE" == "--remote" ]]; then
        ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "${REMOTE_OLLAMA} pull ${model}"
    else
        ollama pull "$model"
    fi
    ok "${model} 就绪"
}

pull_model "$CHAT_MODEL"
pull_model "$EMBED_MODEL"

# 拉完模型，关闭由本脚本启动的 Ollama
if [[ "$OLLAMA_STARTED_BY_US" == true ]]; then
    info "关闭 Ollama..."
    if [[ "$MODE" == "--remote" ]]; then
        ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "pkill -f 'ollama serve'" 2>/dev/null
    else
        kill "$OLLAMA_PID" 2>/dev/null
    fi
    ok "Ollama 已关闭"
fi

# ========== 完成 ==========
echo ""
echo "========================================="
echo "  安装完成！使用以下命令启动项目："
if [[ "$MODE" == "--remote" ]]; then
    echo "  ./scripts/start_tunnel.sh"
else
    echo "  ./scripts/start_tunnel.sh --local"
fi
echo "========================================="
