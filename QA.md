# Q&A 问题排查记录

## 2026-04-10: HuggingFace 本地加载模型阻塞 FastAPI

### 现象

- 上传文件时 FastAPI 整个服务卡死，`/health` 也无法响应
- 首次启动自动从 HuggingFace 下载 BGE-M3 模型，约 3GB（PyTorch 权重 2.27GB + 依赖）

### 原因

`sentence-transformers` 的 `model.encode()` 是同步 CPU 密集操作，会阻塞 FastAPI 的异步事件循环。模型下载也发生在 web 进程内。

### 解决方案

将 embedding 从 `HuggingFaceEmbeddings` 切换为 `OllamaEmbeddings`，通过 HTTP 调用 Ollama 服务，不在 FastAPI 进程内加载模型。同时移除 `sentence-transformers`、`langchain-huggingface` 等依赖，减少约 3GB 安装体积。

---

## 2026-04-10: 首个请求返回"根据已有资料无法回答"

### 现象

- 服务重启后第一次问答返回"根据已有资料无法回答"，第二次恢复正常
- Qdrant 中确认有数据（`vectors_count > 0`）

### 原因

`get_vectorstore()` 是懒加载单例，第一个请求触发初始化（包括调用 Ollama 做一次 `embed_documents(["dummy_text"])` 验证维度），初始化未完成时检索结果为空。

### 解决方案

在 `main.py` 中通过 FastAPI `lifespan` 钩子，在服务启动时预热 vectorstore 和检测 Ollama 连通性。

---

## 2026-04-10: Qwen3 系列模型响应极慢（思考模式）

### 现象

- `qwen3.5:0.8b` 对简单问题也需要 3-4 分钟才响应
- Ollama 日志显示请求被客户端超时中断

### 原因

Qwen3 系列默认开启思考模式（thinking mode），模型会先生成大量 `<think>...</think>` 内部推理，再输出最终答案。

### 解决方案

- 在 `ChatOllama` 中设置 `reasoning=False` 关闭思考模式
- 或换用不带思考模式的模型（如 `qwen2.5:0.5b`）

---

## 2026-04-10: qwen3-vl:4b 纯文本 chat 卡死

### 现象

- FastAPI RAG 服务启动正常，Ollama 健康检查通过
- Embedding（bge-m3）正常完成（200ms）
- 调用 `qwen3-vl:4b` 做 chat 时请求无响应，直到超时

### 原因

`qwen3-vl:4b` 是视觉语言模型（Vision-Language），在 Ollama 中处理纯文本 chat 请求时存在兼容性问题。

### 解决方案

纯文本 RAG 场景应选用纯文本对话模型（如 `qwen3.5:4b`、`qwen2.5:0.5b`），不要使用 VL 模型。

---

## 2026-04-10: 小模型回答不稳定

### 现象

- 使用 `qwen2.5:0.5b` 时，问同一个问题多次，有时正确回答，有时返回"根据已有资料无法回答"
- 检索日志显示每次都检索到了相关 chunks

### 原因

0.5B 参数的模型理解能力有限，无法稳定遵循 system prompt 中"依据上下文回答"的指令，有时忽略已检索到的内容直接触发兜底回复。

### 解决方案

使用更大的模型（3B+），回答稳定性显著提升。
