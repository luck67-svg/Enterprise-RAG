# Enterprise RAG

企业知识库问答系统，基于混合检索（BM25 + 向量检索 + RRF 融合 + Reranker）构建，支持通过 Open WebUI 直接对话，并兼容 OpenAI Chat API 格式。

## 功能特性

- **混合检索**：BM25 稀疏检索 + Qdrant 向量检索，RRF 加权融合（Dense 优先），父子块架构兼顾检索精度与上下文完整性
- **Reranker 精排**：BGE-Reranker-v2-m3 交叉编码器对候选块重排，阈值过滤噪音
- **OpenAI 兼容接口**：`/v1/chat/completions` 支持流式与非流式，可直接挂载到 Open WebUI
- **文档管理 API**：上传 / 删除 / 列表，支持 PDF、Word、TXT，自动去重
- **一键启动**：`start_tunnel.bat`（Windows）/ `start_tunnel.sh`（Linux）自动拉起所有依赖服务

## 系统架构

```
用户 (Open WebUI / API Client)
         │
         ▼
   FastAPI  :8000
   ├─ /v1/chat/completions   (OpenAI 兼容，流式/非流式)
   └─ /kb/upload|documents   (知识库文档管理)
         │
         ▼
   HybridRetriever
   ├─ Dense  → Qdrant :6333        (子块向量检索，bge-m3 embedding)
   ├─ Sparse → BM25 In-Memory      (父块关键词检索，CJK bigram + 整词)
   ├─ RRF 加权融合 (dense_weight=1.5)
   ├─ _expand_to_parents           (命中子块展开为父块，内容去重)
   └─ Reranker :8001               (BGE-Reranker-v2-m3，远程 CPU/GPU)
         │
         ▼
   Ollama :11434                   (LLM 推理，qwen3.5:35b)
```

**远程服务拓扑**（通过 SSH 隧道透明转发）：

```
本地 Windows / Mac                   远程 Linux 服务器
──────────────────────               ──────────────────
FastAPI       :8000                  Ollama    :11434
Qdrant        :6333  (Docker)   ←SSH─ Reranker  :8001
Open WebUI    :3000  (Docker)
```

## 快速启动

### 前置条件

| 依赖 | 说明 |
|------|------|
| Docker Desktop | 运行 Qdrant + Open WebUI |
| [uv](https://docs.astral.sh/uv/) | Python 包管理 |
| SSH 密钥 `~/.ssh/veridian` | 访问远程服务器 |
| 远程服务器 | 已安装 Ollama + bge-reranker-v2-m3 模型 |

### 首次部署远程 Reranker

```bash
# 将 app/reranker_service.py 上传到远程并启动
bash deploy_reranker_remote.sh
```

### 日常一键启动

**Windows（PowerShell）：**

```powershell
.\scripts\start_tunnel.bat
```

**Linux / macOS：**

```bash
bash scripts/start_tunnel.sh
```

启动顺序：Docker（Qdrant + Open WebUI）→ 远程 Ollama → SSH 隧道 → 远程 Reranker → FastAPI

启动后访问：

| 服务 | 地址 |
|------|------|
| Open WebUI | http://localhost:3000 |
| FastAPI Docs | http://localhost:8000/docs |
| Qdrant Dashboard | http://localhost:6333/dashboard |

### Open WebUI 接入配置

在 Open WebUI → 管理员设置 → 连接 中添加：

- **URL**：`http://host.docker.internal:8000/v1`
- **API Key**：任意字符串（系统不校验）

## 环境变量

复制 `.env.example` 为 `.env` 并按需修改：

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:35b
EMBEDDING_MODEL=bge-m3
QDRANT_URL=http://localhost:6333
RERANKER_BASE_URL=http://localhost:8001
```

## API 说明

### 知识库管理

```bash
# 上传文档
curl -X POST http://localhost:8000/kb/upload \
  -F "file=@your_doc.pdf"

# 列出已上传文档
curl http://localhost:8000/kb/documents

# 删除文档（同步清理 Qdrant 向量 + BM25 索引）
curl -X DELETE http://localhost:8000/kb/documents/your_doc.pdf

# 向量库统计
curl http://localhost:8000/kb/stats
```

### 问答（OpenAI 兼容）

```bash
# 非流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag",
    "messages": [{"role": "user", "content": "请介绍知识库中的主要内容"}]
  }'

# 流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rag","messages":[{"role":"user","content":"问题"}],"stream":true}'
```

完整接口列表见 http://localhost:8000/docs。

## 检索参数配置

`app/config.py` 中可调整的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `parent_chunk_size` | 1000 | 父块大小（BM25 索引单元） |
| `child_chunk_size` | 400 | 子块大小（向量检索单元） |
| `child_retrieval_k` | 12 | Dense 检索候选数 |
| `bm25_top_k` | 6 | BM25 候选数 |
| `rrf_k` | 30 | RRF 融合参数（越小排名差异越显著） |
| `retrieval_top_k` | 5 | 最终传给 LLM 的块数 |
| `reranker_score_threshold` | 0.3 | Reranker 分数过滤阈值 |

## 项目结构

```
Enterprise-RAG/
├── app/
│   ├── main.py                 # FastAPI 入口 & lifespan
│   ├── config.py               # 全局配置（pydantic-settings，支持 .env）
│   ├── reranker_service.py     # 部署到远程的 Reranker FastAPI 服务
│   ├── api/
│   │   ├── kb.py               # 知识库上传 / 删除 / 列表
│   │   └── openai_compat.py    # /v1/chat/completions 兼容层
│   ├── llm/
│   │   └── ollama_client.py    # Ollama LLM / Embedding 客户端
│   └── rag/
│       ├── chain.py            # RAG Chain 组装
│       ├── hybrid_retriever.py # BM25 + Dense + RRF + Reranker
│       ├── bm25_store.py       # BM25 持久化索引（CJK bigram 分词）
│       ├── reranker.py         # 远程 Reranker HTTP 客户端（指数退避重试）
│       ├── vectorstore.py      # Qdrant 向量库封装
│       ├── embeddings.py       # Ollama Embedding 封装
│       ├── splitter.py         # 父子块切分（parent/child chunk）
│       └── loaders.py          # PDF / Word / TXT 文档加载
├── scripts/
│   ├── start_tunnel.bat        # Windows 一键启动
│   ├── start_tunnel.sh         # Linux/macOS 一键启动
│   └── docker-compose.yml      # Qdrant + Open WebUI
├── deploy_reranker_remote.sh   # 远程 Reranker 部署脚本（SCP + SSH）
└── pyproject.toml
```

## 开发

```bash
# 安装依赖
uv sync --dev

# 代码检查
uv run ruff check .

# 运行测试
uv run pytest

# 单独启动 FastAPI（已有隧道的情况下）
uv run uvicorn app.main:app --reload
```

## 注意事项

- **重新上传文档**：若从旧版本升级（BM25 由子块改为父块索引），已上传的文档需重新上传一次，BM25 索引才会使用新的父块格式
- **模型预热**：首次启动后第一次问答会触发模型加载（约 30-60s），之后响应正常；预热脚本在 `start_tunnel.bat` 步骤 [4/5] 自动执行
- **CUDA 驱动**：若远程服务器 CUDA 驱动版本过旧，Reranker 自动降级到 CPU 运行，功能正常但速度稍慢
