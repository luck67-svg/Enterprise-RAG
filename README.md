# Enterprise RAG

企业级 RAG（检索增强生成）系统，使用本地 LLM 驱动，无需依赖外部 API。

## 架构概览

```
Open WebUI（前端界面）
       ↓ Pipeline API
RAG Pipeline Server（本项目，FastAPI）
       ↓ 检索          ↓ 推理
  Qdrant（向量库）   Ollama（本地 LLM + Embedding）
       ↑
  知识库文档（PDF / Word / ...）
```

> Open WebUI 的内置 RAG 功能**不启用**，所有检索逻辑由本项目的 Pipeline Server 接管。

## 依赖说明

| 组件 | 用途 |
|------|------|
| [Ollama](https://ollama.com) | 本地 LLM 推理 + Embedding 模型服务 |
| [Open WebUI](https://github.com/open-webui/open-webui) | 对话前端界面 |
| [LangChain](https://python.langchain.com) | RAG 全流程编排（加载、切片、检索、问答） |
| [Qdrant](https://qdrant.tech) | 高性能向量数据库 |
| [FastAPI](https://fastapi.tiangolo.com) | Pipeline Server，供 Open WebUI 调用 |
| [RAGAS](https://docs.ragas.io) | RAG 评估（检索质量、答案忠实度、相关性）|
| pymupdf / python-docx | 知识库文档解析 |

## 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) 包管理器
- [Ollama](https://ollama.com/download) 已安装并运行
- [Docker](https://www.docker.com) 用于启动 Qdrant

## 快速开始

### 1. 安装 Ollama 并拉取模型

```bash
brew install ollama
ollama serve
```

拉取 LLM 和 Embedding 模型：

```bash
ollama pull qwen2.5:0.5b   # 对话模型（轻量）
ollama pull bge-m3          # 向量化模型
```

### 2. 启动 Qdrant

```bash
cd scripts
docker compose up -d qdrant
```

数据持久化在 `./data/qdrant/`。

### 3. 安装 Python 依赖

```bash
uv sync
```

### 4. 启动 FastAPI 服务

```bash
uv run python -m app.main
```

服务启动时会自动：
- 检测 Ollama 可用模型并分类（对话 / 嵌入）
- 初始化 Qdrant 向量库连接

### 5. 验证服务

```bash
# 健康检查（显示 Ollama 模型列表）
curl http://localhost:8000/health

# 上传知识库文档
curl -X POST http://localhost:8000/kb/upload -F "file=@your_doc.pdf"

# 问答测试
curl --noproxy localhost -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"enterprise-rag","messages":[{"role":"user","content":"你的问题"}]}'
```

或访问 [http://localhost:8000/docs](http://localhost:8000/docs) 使用 Swagger UI。

### 6. 启动 Open WebUI（可选）

```bash
cd scripts
docker compose up -d open-webui
```

访问 [http://localhost:3000](http://localhost:3000)，选择模型 `enterprise-rag` 即可对话。

## 配置

通过项目根目录的 `.env` 文件覆盖默认配置：

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:0.5b
EMBEDDING_MODEL=bge-m3
QDRANT_URL=http://localhost:6333
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RETRIEVAL_TOP_K=5
```

## 开发

```bash
uv sync --dev
uv run ruff check .
uv run pytest
```
