# 变更说明

本文档对比基准提交 `793d938`（ljy，2026-04-08）与当前 `HEAD`，说明每处修改的内容与原因。

基准提交完成了 Stage 1 的基本骨架：PDF 上传、BGE-M3 embedding（本地 HuggingFace）、Qdrant 向量检索、Ollama LLM、OpenWebUI 集成。之后的所有改动均由 zhaihaojie 完成，共 7 个提交。

---

## 1. Embedding 方案切换（`app/rag/embeddings.py`、`app/config.py`）

### 改了什么

| 项目 | 基准（793d938） | 当前 |
|------|----------------|------|
| Embedding 库 | `langchain_huggingface.HuggingFaceEmbeddings` | `langchain_ollama.OllamaEmbeddings` |
| 模型名称 | `BAAI/bge-m3`（HuggingFace 路径） | `bge-m3`（Ollama 模型名） |
| 运行位置 | 本地 CPU/GPU 推理 | 通过 Ollama HTTP API 调用 |

### 为什么改

原方案在本地加载 HuggingFace 模型，需要下载 ~1.5GB 的模型权重到运行机器，且依赖 `torch`/`transformers`，安装体积大、启动慢。切换到 Ollama 统一管理后：

- Embedding 和 LLM 都通过同一个 Ollama 服务管理，部署更简单
- 本机无需安装 PyTorch，大幅减小依赖体积
- 远程服务器已有 Ollama，`bge-m3` 模型可直接 `ollama pull bge-m3` 拉取

---

## 2. LLM 客户端增强（`app/llm/ollama_client.py`）

### 改了什么

- 新增 `list_ollama_models()` 函数：通过 `/api/tags` 接口获取 Ollama 已有模型，并按关键词自动分类为 `chat` 和 `embedding` 两组
- `get_llm()` 新增 `reasoning=False` 参数
- 默认模型从 `qwen2.5:7b` 改为 `qwen3.5:0.8b`

### 为什么改

- `list_ollama_models()` 用于启动时健康检查和 `/health` 接口，能直观显示当前可用模型状态
- Qwen3 系列默认开启思考模式（`<think>` token），会在正式回答前生成大量中间思考内容，导致响应极慢。`reasoning=False` 关闭此行为
- `qwen3.5:0.8b` 是测试环境下资源占用更小的模型，可通过 `.env` 覆盖为更大模型

---

## 3. 多文件格式支持（`app/rag/loaders.py`）

### 改了什么

- 基准只支持 `.pdf`，遇到其他格式直接报错 `"Unsupported file type for stage 1"`
- 新增 `.docx` 支持：使用 `python-docx` 按段落提取文本
- 新增 `.txt` 支持：直接 `read_text()` 读取
- 导出 `SUPPORTED_EXTENSIONS` 常量，供上传接口复用

### 为什么改

企业场景文档格式多样，仅支持 PDF 无法满足实际需求。DOCX 是 Word 文档的主流格式，TXT 是最简单的纯文本格式，这两种覆盖了大部分常见场景。

---

## 4. 多轮对话（`app/rag/chain.py`、`app/api/openai_compat.py`）

### 改了什么

**chain.py：**
- `PROMPT` 中插入 `MessagesPlaceholder("chat_history")`，支持历史消息注入
- chain 的输入从单个 `question` 字符串改为 `{"question": ..., "chat_history": ...}` 字典
- 新增 `_format_docs` 中的调试日志（检索到几条、每条来源和预览）

**openai_compat.py：**
- 新增 `_extract_history_and_question()`：从 OpenAI 格式的 `messages` 列表中分离当前问题和历史对话
- 历史轮数上限 `MAX_HISTORY_TURNS = 10`，超出部分自动截断

### 为什么改

基准实现忽略所有历史消息，只取最后一条 user 消息作为问题，每轮对话完全独立。这在追问、指代等场景下体验很差（比如"那作者是谁"这种依赖上文的问题会失效）。多轮对话是知识库问答的基本需求。

---

## 5. SSE 流式输出（`app/api/openai_compat.py`）

### 改了什么

- 基准 `chat_completions` 只支持同步返回，等待全部生成完再响应
- 新增 `_stream_response()` 异步生成器，按 token 推送 SSE 事件
- `ChatCompletionRequest` 新增 `stream: bool = False` 字段
- 请求携带 `stream: true` 时返回 `StreamingResponse`
- 新增客户端断开检测（`request.is_disconnected()`），提前终止生成
- 新增 `REQUEST_TIMEOUT = 120s` 非流式超时保护

### 为什么改

等待 LLM 生成完整答案再返回，对于长回答会让用户等待十几秒看到空白页面，体验极差。SSE 流式输出让用户可以看到逐字生成的过程，与 ChatGPT 等产品体验一致。Open WebUI 也默认使用流式模式。

---

## 6. 知识库管理 API（`app/api/kb.py`）

### 改了什么

基准只有 `POST /kb/upload` 一个接口，且无任何文件校验。新增和增强：

| 接口 | 说明 |
|------|------|
| `POST /kb/upload` | 增强：支持格式校验、重复检测、向量替换、友好错误信息 |
| `GET /kb/documents` | 新增：列出已上传文档及文件大小 |
| `DELETE /kb/documents/{filename}` | 新增：删除文档文件及其在 Qdrant 中的向量 |
| `GET /kb/stats` | 新增：返回 Qdrant collection 的向量数量统计 |

**重复上传检测：**
- 启动时扫描 `upload_dir` 构建 SHA256 哈希缓存 `_file_hashes`
- 上传时若同名文件哈希相同，跳过处理并返回 `skipped: true`
- 若同名但内容变化，先删除旧向量再重新入库

### 为什么改

基准上传接口存在几个实际问题：
1. 重复上传同一文件会导致向量库中出现重复数据，检索时同一内容命中多次，干扰排序
2. 没有删除功能，知识库只能增不能减
3. 没有列表功能，不知道库里有哪些文档
4. 错误信息是英文技术报错，对非开发人员不友好

---

## 7. 应用启动与健康检查（`app/main.py`）

### 改了什么

- 新增 `lifespan` 异步上下文管理器：启动时验证 Ollama 连通性并预热 Qdrant 连接
- 新增全局异常处理器 `global_exception_handler`：将底层异常转为中文友好提示
- `/health` 接口从简单返回 `{"status": "ok"}` 改为实际探测 Ollama 并列出可用模型
- 挂载 `app/static/` 目录，`GET /` 返回内置管理界面

### 为什么改

- **启动预热**：不做预热的话，第一次请求时才初始化 Qdrant 连接，会有明显的首次延迟，且启动时的配置错误（比如 Ollama 没开）无法及时发现
- **全局异常处理**：LangChain/Qdrant/httpx 抛出的底层异常包含大量技术细节（堆栈、连接地址），直接透传给用户不友好，统一转为"Ollama 服务不可达"等可操作的提示
- **健康检查增强**：运维时需要知道服务实际状态，而不仅仅是"进程活着"

---

## 8. 新增文件

| 文件 | 说明 |
|------|------|
| `app/static/index.html` | 内置知识库管理 Web 界面，支持文档上传、列表、删除、统计 |
| `scripts/setup.sh` | 一键安装脚本，自动安装依赖、启动 Docker Qdrant、拉取 Ollama 模型 |
| `scripts/start_tunnel.sh` | SSH 隧道脚本，将远程 Ollama 服务转发到本地，支持 `--local`/`--remote` 模式 |
| `scripts/start_tunnel.bat` | Windows 版本的 SSH 隧道脚本 |
| `QA.md` | 开发过程中遇到的问题与排查记录 |
| `assets/test.pdf` | 测试用 PDF 文档 |

---

## 9. 代码审查修复（本次会话）

在本次会话中对 `openai_compat.py` 和 `chain.py` 做了额外的代码质量修复：

| 问题 | 修复方式 |
|------|---------|
| 每次请求重建 chain，重复初始化连接 | `get_rag_chain()` 单例缓存，相同 temperature 复用 |
| `question` 为空时仍向量检索 | 加 400 校验，`not question.strip()` 直接返回错误 |
| 流式请求无超时保护 | 新增 `STREAM_TOKEN_TIMEOUT = 30s`，超时返回错误提示 |
| `chat_history` 使用元组格式 | 改为 `HumanMessage`/`AIMessage` 对象，符合 LangChain 规范 |
| `temperature` 参数接收但丢弃 | 透传给 `get_llm()`，Open WebUI 温度调节生效 |
| `build_rag_chain` 含无效参数 | 重命名为 `get_rag_chain(temperature)`，删除死参数 |
