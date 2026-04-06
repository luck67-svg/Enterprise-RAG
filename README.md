# Enterprise RAG

企业级 RAG（检索增强生成）系统，使用本地 LLM 驱动，无需依赖外部 API。

## 架构概览

```
Open WebUI（前端界面）
       ↓ Pipeline API
RAG Pipeline Server（本项目，FastAPI）
       ↓ 检索          ↓ 推理
  Qdrant（向量库）   Ollama（本地 LLM）
       ↑
  知识库文档（PDF / Word / ...）
```

> Open WebUI 的内置 RAG 功能**不启用**，所有检索逻辑由本项目的 Pipeline Server 接管。

## 依赖说明

| 组件 | 用途 |
|------|------|
| [Ollama](https://ollama.com) | 本地 LLM 模型服务 |
| [Open WebUI](https://github.com/open-webui/open-webui) | 对话前端界面 |
| [LangChain](https://python.langchain.com) | RAG 全流程编排（加载、切片、检索、问答） |
| [Qdrant](https://qdrant.tech) | 高性能向量数据库，支持本地与服务器模式 |
| [sentence-transformers](https://www.sbert.net) | 本地 Embedding 模型，无需外部 API |
| [FastAPI](https://fastapi.tiangolo.com) | Pipeline Server，供 Open WebUI 调用 |
| [RAGAS](https://docs.ragas.io) | RAG 评估（检索质量、答案忠实度、相关性）|
| pypdf / python-docx | 知识库文档解析 |

## 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) 包管理器
- [Ollama](https://ollama.com/download) 已安装并运行

## 快速开始

### 1. 安装 Ollama

访问 [https://ollama.com/download](https://ollama.com/download) 下载并安装，或通过 Homebrew：

```bash
brew install ollama
```

启动 Ollama 服务：

```bash
ollama serve
```

拉取所需模型（以 `qwen3.5:0.8b` 为例，轻量推荐）：

```bash
ollama pull qwen3.5:0.8b
```

> 首次启动 Open WebUI 时会自动从 HuggingFace 下载约 930MB 的内置模型文件，请耐心等待。完成后浏览器访问 `http://localhost:8080`（注意使用 `http://` 而非 `https://`）。

### 2. 安装 Python 依赖

```bash
uv sync
```

### 3. 启动 Open WebUI

```bash
uv run open-webui serve
```

默认访问地址：[http://localhost:8080](http://localhost:8080)

## 开发

安装开发依赖（包含 pytest、ruff）：

```bash
uv sync --dev
```

代码检查：

```bash
uv run ruff check .
```

运行测试：

```bash
uv run pytest
```
