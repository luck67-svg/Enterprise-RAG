# Enterprise RAG

企业级 RAG（检索增强生成）系统，使用本地 LLM 驱动，无需依赖外部 API。

## 架构概览

```
知识库管理页面（localhost:8000）
       ↓ 上传文档
Open WebUI（前端对话界面）
       ↓ OpenAI-compatible API（支持 Streaming）
RAG Pipeline Server（本项目，FastAPI）
       ↓ 检索          ↓ 推理
  Qdrant（向量库）   Ollama（LLM + Embedding）
       ↑
  知识库文档（PDF / DOCX / TXT）
```

## 依赖说明

| 组件 | 用途 |
|------|------|
| [Ollama](https://ollama.com) | LLM 推理 + Embedding 模型服务 |
| [Open WebUI](https://github.com/open-webui/open-webui) | 对话前端界面 |
| [LangChain](https://python.langchain.com) | RAG 全流程编排（加载、切片、检索、问答） |
| [Qdrant](https://qdrant.tech) | 高性能向量数据库 |
| [FastAPI](https://fastapi.tiangolo.com) | Pipeline Server + 知识库管理页面 |
| pymupdf / python-docx | 知识库文档解析（PDF / DOCX / TXT） |

## 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) 包管理器
- [Ollama](https://ollama.com/download) 已安装
- [Docker](https://www.docker.com) 用于启动 Qdrant 和 Open WebUI

## 快速开始

### 一键安装（首次）

```bash
# 本地 Ollama
./scripts/setup.sh

# 远程 Ollama
./scripts/setup.sh --remote
```

自动完成：检查前置工具 → 安装 Python 依赖 → 拉取 Docker 镜像 → 拉取 Ollama 模型（qwen3.5:0.8b + bge-m3）

### 一键启动（日常开发）

```bash
# 使用远程 Ollama（默认）
./scripts/start_tunnel.sh

# 使用本地 Ollama
./scripts/start_tunnel.sh --local
```

自动完成：启动 Docker 容器 → 启动 Ollama → SSH 隧道（远程模式） → 启动 FastAPI

Windows 用户使用 `scripts\start_tunnel.bat`。

### 服务地址

| 服务 | 地址 |
|------|------|
| 知识库管理页面 | http://localhost:8000 |
| API 文档 (Swagger) | http://localhost:8000/docs |
| Open WebUI | http://localhost:3000 |
| Qdrant Dashboard | http://localhost:6333/dashboard |

### 使用流程

1. 访问 http://localhost:8000 上传知识库文档（PDF / DOCX / TXT）
2. 访问 http://localhost:3000 打开 Open WebUI，选择模型 `enterprise-rag` 对话
3. 或直接调用 API：

```bash
# 健康检查
curl http://localhost:8000/health

# 上传文档
curl -X POST http://localhost:8000/kb/upload -F "file=@your_doc.pdf"

# 问答（非流式）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"enterprise-rag","messages":[{"role":"user","content":"你的问题"}]}'

# 问答（流式）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"enterprise-rag","messages":[{"role":"user","content":"你的问题"}],"stream":true}'
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（Ollama 状态、模型列表） |
| GET | `/` | 知识库管理页面 |
| POST | `/kb/upload` | 上传文档（支持重复检测） |
| GET | `/kb/documents` | 已上传文档列表 |
| DELETE | `/kb/documents/{filename}` | 删除文档及其向量 |
| GET | `/kb/stats` | 向量库统计信息 |
| GET | `/v1/models` | 模型列表（OpenAI 兼容） |
| POST | `/v1/chat/completions` | 对话补全（支持流式 + 多轮对话） |

## 配置

通过 `.env` 文件覆盖默认配置：

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:0.8b
EMBEDDING_MODEL=bge-m3
QDRANT_URL=http://localhost:6333
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RETRIEVAL_TOP_K=5
```

## 项目结构

```
app/
├── api/
│   ├── kb.py              # 知识库管理 API（上传、列表、删除）
│   └── openai_compat.py   # OpenAI 兼容 API（对话、流式、多轮）
├── llm/
│   └── ollama_client.py   # Ollama 客户端（模型分类、LLM 实例）
├── rag/
│   ├── chain.py           # RAG chain 构建（检索 + 生成）
│   ├── embeddings.py      # Embedding 模型
│   ├── loaders.py         # 文档加载（PDF / DOCX / TXT）
│   ├── splitter.py        # 文档切片
│   └── vectorstore.py     # Qdrant 向量库
├── static/
│   └── index.html         # 知识库管理前端页面
├── config.py              # 配置管理
└── main.py                # FastAPI 入口
scripts/
├── setup.sh               # 首次安装脚本
├── start_tunnel.sh        # 一键启动脚本 (macOS/Linux)
├── start_tunnel.bat        # 一键启动脚本 (Windows)
└── docker-compose.yml     # Qdrant + Open WebUI
```

## 开发

```bash
uv sync --dev
uv run ruff check .
uv run pytest
```
