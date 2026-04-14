# 代码审查建议：ljy 的 3 次提交

> 审查范围：`0f303fe`（Phase 2 Ragas）、`11b171a`（Phase 3a Parent-Child）、`cf5c92d`（Phase 3c CrossEncoder Reranker）  
> 审查时间：2026-04-13  
> 审查人：zhaihaojie + Codex Adversarial Review

---

## 总体评价

三次提交完成了 RAG 管线的三个关键升级：评估体系（Ragas）、检索精度优化（Parent-Child 分块）和精排质量（CrossEncoder Reranking）。功能方向正确，但存在若干影响可信度和稳定性的问题，部分问题会导致评估结论失实，需在下一轮迭代前修复。

---

## 一、HIGH：影响系统正确性或评估可信度

### H1. Ragas 评分使用了"幽灵检索"——答案和上下文来自两次独立调用

**文件**：`scripts/eval_ragas.py:49-57`

```python
def run_rag(question: str) -> tuple[str, list[str]]:
    retriever = get_retriever()
    docs = retriever.invoke(question)          # 第一次检索
    contexts = [d.page_content for d in docs]

    chain = get_rag_chain()
    answer = chain.invoke({"question": question, "chat_history": []})  # 内部第二次检索
    return answer, contexts
```

`get_rag_chain()` 内部的 `RunnableLambda` 会再次调用 `get_retriever()`，导致答案与上下文来自**两次独立的向量近似搜索**。由于 HNSW 的随机性，两次结果完全可能不同。传给 Ragas 的 `retrieved_contexts` 不是 LLM 实际看到的内容，faithfulness/context_precision/context_recall 全部失真。

**Phase 2 建立的"基线"和 Phase 3 的"对比改进"因此无法作为客观依据。**

**建议**：重构 chain，让 retrieval 结果可被显式捕获：
```python
retriever = get_retriever()
docs = retriever.invoke(question)
contexts = [d.page_content for d in docs]
formatted = _format_docs(docs)
answer = (PROMPT | get_llm() | StrOutputParser()).invoke({
    "context": formatted, "question": question, "chat_history": []
})
```

---

### H2. 失败样本被静默跳过，评分因局部故障而虚高

**文件**：`scripts/eval_ragas.py:114-135` 和 `162-169`

```python
except Exception as e:
    logger.warning(f"    问题 [{i}] 处理失败，跳过：{e}")
    failed += 1
# ...
results = evaluate(..., raise_exceptions=False, ...)
```

两层容错机制叠加：单样本异常 catch 后 skip，Ragas 内部也 `raise_exceptions=False`。**如果 reranker 对难题不稳定，恰好是难题被跳过，分母变小，均值上升**，但系统实际质量下降。保存的 JSON 只有 `num_samples`（成功数），无法复现"哪些题失败了"。

**建议**：
1. 保存 `total_questions`、`failed_count`、`failed_indices` 到 JSON
2. 失败率超过 10% 时拒绝打印对比，并以非零状态码退出
3. 考虑 `raise_exceptions=True` 作为 CI 环境的默认选项

---

### H3. macOS/Linux 启动脚本完全不启动 Reranker

**文件**：`scripts/start_tunnel.sh`（全文）vs `scripts/start_tunnel.bat:45-50`

Windows 脚本有：
```bat
echo === 检查远程 Reranker 服务状态...
ssh ... "curl -sf http://localhost:8001/health ..."
echo === 建立 Reranker 隧道: localhost:8001 -> ...
start /b ssh ... -NL 8001:localhost:8001 ...
```

`start_tunnel.sh` 完全没有对应逻辑。在 macOS/Linux 上使用默认启动方式时，`reranker_base_url` 指向的 `http://localhost:8001` 不可达，**每次查询都会触发 30 秒超时后降级**，Phase 3c 宣称的精排优化实际上从未生效。

**建议**：在 `start_tunnel.sh` 的远程模式中补充：
```bash
# 启动远程 Reranker
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "curl -sf http://localhost:8001/health || nohup uvicorn reranker_service:app --host 0.0.0.0 --port 8001 > ~/reranker.log 2>&1 &"

# 建立 8001 隧道
ssh "${SSH_OPTS[@]}" -fNL 8001:localhost:8001 "${REMOTE_USER}@${REMOTE_HOST}"
```
或在文档中明确说明：macOS/Linux 用户需手动执行这两步，否则 reranking 不生效。

---

### H4. Reranker 不可用时每次查询阻塞 30 秒

**文件**：`app/rag/reranker.py:14-24`

```python
resp = httpx.post(
    f"{settings.reranker_base_url}/rerank",
    ...,
    timeout=30.0,   # 连接超时和读取超时均为 30 秒
)
```

流式响应路径中，reranker 调用发生在第一个 token 生成之前。SSH 隧道断开或服务挂起时，用户会看到 30 秒无响应——体验与系统崩溃无异。现有的流式 timeout 保护对这个阶段无效。

**建议**：
```python
resp = httpx.post(
    ...,
    timeout=httpx.Timeout(connect=2.0, read=5.0),  # 快速失败
)
```
并可选加入 circuit breaker：连续 3 次失败后，接下来 60 秒内直接跳过 reranker 调用，不等待。

---

## 二、MEDIUM：影响可维护性或准确性

### M1. 测试集生成使用旧切分方式，与生产不一致

**文件**：`scripts/generate_testset.py:67`

```python
chunks = split_documents(all_docs)   # 旧的单级切分
```

而生产上传走的是 `split_parent_child()`（`app/api/kb.py:67`）。测试集的 reference 和上下文来自旧切分结果，但评估时检索的是 parent-child 向量。这导致 context_recall 等指标的参考答案与实际检索空间不匹配，评分噪声增大。

**建议**：测试集生成时也使用 `split_parent_child()`，并只取 `page_content` 作为 Ragas 文档节点，与生产保持一致。

---

### M2. `parent_content` 直接嵌入 metadata，存在 Qdrant payload 大小风险

**文件**：`app/rag/splitter.py:41`

```python
child.metadata["parent_content"] = parent.page_content
```

`parent_chunk_size=1000`，中文约 1000 字，UTF-8 编码约 3KB。每个子块 metadata 携带一个父块文本，文件较大时每次 add_documents 的 payload 总量可能触及 Qdrant 的 payload 大小限制。此处没有任何大小检查，失败时会抛出 Qdrant 的原始错误，被 `kb.py` 包装为 503，难以定位根因。

**建议**：记录 `parent_content` 的平均长度，或改为存 `parent_id` 并将父块内容存入独立 in-memory store（标准 `ParentDocumentRetriever` 的做法），彻底规避 payload 限制。

---

### M3. 重复文件检测只看文件名，不同名同内容会重复向量化

**文件**：`app/api/kb.py:48`

```python
if file.filename in _file_hashes and _file_hashes[file.filename] == file_hash:
    return {"skipped": True, ...}
```

检查的是 `filename → hash` 映射。如果同一份文档以不同文件名上传两次，两份内容会被重复向量化，召回时产生重复结果，top-k 被同一内容的副本占据，降低检索多样性。

**建议**：同时维护 `_content_hashes: set[str]`，对内容 hash 去重，与文件名解耦。

---

### M4. 父块 overlap 硬编码，无法配置

**文件**：`app/rag/splitter.py:28`

```python
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.parent_chunk_size,
    chunk_overlap=200,    # 硬编码，无配置项
)
```

`child_chunk_overlap` 有配置项，父块 overlap 却是硬编码 200，与 `chunk_overlap=120` 的全局默认值风格不一致，调参时容易遗漏。

**建议**：在 `config.py` 加入 `parent_chunk_overlap: int = 200` 并引用。

---

### M5. `_file_hashes` 内存字典无并发保护

**文件**：`app/api/kb.py:11`

```python
_file_hashes: dict[str, str] = {}
```

FastAPI 多 worker 模式下（`--workers N`），每个 worker 独立的内存字典会导致状态不一致：Worker A 上传了文件并更新字典，Worker B 不知道，可能重复处理同一文件。单 worker 下，`asyncio` 中 `await file.read()` 期间的协程切换也存在潜在竞态。

**建议**：改用文件系统持久化（写 `.json` 哈希表），或在 Qdrant 中查询 `metadata.source` 是否存在来判断重复，替代内存字典。

---

### M6. `reranker.py` 未校验返回的 scores 数量

**文件**：`app/rag/reranker.py:21-26`

```python
scores = resp.json()["scores"]
scored = sorted(zip(scores, documents), key=...)
```

如果远程 reranker 服务 bug 导致返回的 `scores` 数量少于 `documents` 数量，`zip` 会**静默截断**，多余的文档被丢弃，不报任何错误，用户得到的是无提示的召回缺失。

**建议**：
```python
if len(scores) != len(documents):
    logger.warning(f"Reranker returned {len(scores)} scores for {len(documents)} docs, falling back")
    return documents
```

---

## 三、LOW：规范性问题

### L1. 注释仍指向 start_tunnel.bat，未更新到 .sh

**文件**：`scripts/generate_testset.py:14`、`scripts/eval_ragas.py:15`

```python
# 需要 Qdrant + Ollama 正在运行（先启动 start_tunnel.bat）
```

两处注释均指向 Windows 脚本，Linux/macOS 用户无法直接遵循，应统一改为提示两个脚本或指向 README。

---

### L2. 硬编码私有 SSH 密钥名称，可移植性差

**文件**：`scripts/start_tunnel.sh:17`、`scripts/start_tunnel.bat:14`

```bash
SSH_KEY="${SSH_KEY_PATH:-$HOME/.ssh/veridian}"
```

`veridian` 是特定机器的密钥名，新开发者不会有此文件，且文档中没有说明需要设置 `SSH_KEY_PATH`。

**建议**：在 `.env.example` 或 README 中列出所有需要配置的环境变量，包括 `SSH_KEY_PATH`、`REMOTE_SSH_USER`、`REMOTE_SSH_HOST`。

---

### L3. 缺少 .env.example，配置项无自文档化入口

**文件**：项目根目录（缺失）

`start_tunnel.sh`、`start_tunnel.bat` 和 `app/config.py` 依赖多个环境变量，但项目没有 `.env.example`。新成员无法通过阅读文件了解需要配置什么，只能靠口口相传。

---

### L4. 无任何自动化测试

整个项目没有 `tests/` 目录或任何单元/集成测试。`eval_ragas.py` 是端到端评估，不是测试，每次运行需要 10-25 分钟，无法在 CI 中快速运行。以下逻辑无测试覆盖：

- `split_parent_child` 的边界行为（空文档、超长段落）
- `_expand_to_parents` 的去重逻辑
- `rerank_documents` 的 fallback 路径（scores 数量不匹配、超时）
- `kb.py` 的重复检测和向量删除

---

## 四、评估体系完整性评估

| 维度 | 现状 | 缺失 |
|------|------|------|
| 指标覆盖 | Faithfulness、Context Precision/Recall | Answer Relevancy 因 Ollama 兼容问题跳过，影响完整判断 |
| 检索-生成一致性 | **错误**：两次独立检索（H1） | 单 pass 检索+生成 |
| 样本完整性 | 失败静默跳过（H2） | 失败率记录 + 完成度门控（建议 ≥90% 才允许对比） |
| 可重现性 | 仅记录模型名和样本数 | commit SHA、KB 快照 hash、testset hash、retriever 配置 |
| 版本对比 | 按文件修改时间找"最新"历史 | 结构化命名（如 `phase2_baseline.json`）并做 exact diff |
| 测试集与生产一致性 | **不一致**（M1） | testset 生成使用相同切分策略 |

**结论**：当前评估体系可用于方向性参考，但数值结论尚不可信，不能作为版本发布的量化门控标准。

---

## 五、优先修复顺序

| 优先级 | 问题 | 原因 |
|--------|------|------|
| 1 | H1 双重检索 | 所有已有评分需重跑，越晚修复积累的"历史对比"越没意义 |
| 2 | H3 .sh 缺 reranker | macOS/Linux 功能完整性问题，影响实际使用 |
| 3 | H4 30 秒阻塞 | 直接影响用户体验，改动小、收益大 |
| 4 | H2 失败跳过 | 评估可信度，改动小 |
| 5 | M1 testset 切分 | 下次重新生成 testset 时修复 |
| 6 | M6 scores 校验 | 防御性编程，2 行代码 |
| 7 | M3 重复检测 | 有文档量增长后会暴露 |
| 8 | 其余 M/L | 可在下一个功能 PR 中顺手处理 |

---

## 六、模型框架与加载方式审查（ECC Python Reviewer 补充）

> 覆盖范围扩展至全部 15 个 `.py` 源文件；重点方向：模型初始化、单例策略、依赖声明、框架使用正确性。

---

### H5. 流式超时常量声明但从未生效——per-token 保护是死代码

**文件**：`app/api/openai_compat.py:18, 124`

```python
STREAM_TOKEN_TIMEOUT = 30   # 秒，流式单 token 超时  ← 从未使用

async for token in chain.astream(chain_input):   # ← 无 wait_for 包裹
    ...
except asyncio.TimeoutError:   # ← 永远不会被触发
```

`STREAM_TOKEN_TIMEOUT` 被注释为"防止 Ollama 卡死"，但 `chain.astream()` 本身是 async generator，不会主动抛出 `TimeoutError`，`try/except asyncio.TimeoutError` 是死代码。**流式路径实际上没有任何超时保护**，Ollama 卡死时用户会永远等待。

**建议**：
```python
async for token in chain.astream(chain_input):
    try:
        token = await asyncio.wait_for(
            asyncio.shield(anext(stream_iter)), 
            timeout=STREAM_TOKEN_TIMEOUT
        )
    except asyncio.TimeoutError:
        ...
```
或改用 `asyncio.timeout()` context manager（Python 3.11+）包裹整个生成循环。

---

### H6. 向量维度硬编码，换 Embedding 模型会静默创建错误 Collection

**文件**：`app/rag/vectorstore.py:12, 27`

```python
EMBED_DIM = 1024   # BGE-M3 dense dim — 硬编码

client.create_collection(
    collection_name=settings.qdrant_collection,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
)
```

若将 `settings.embedding_model` 改为 768 维模型（如 `nomic-embed-text`、`bge-small`），Collection 会以 1024 维创建，首次 `add_documents` 时 Qdrant 抛出维度不匹配错误，被 `kb.py` 包装为 503，难以定位。更危险的是：若 Collection 已存在（旧数据），`ensure_collection()` 不检查维度，新模型写入旧 Collection 会直接报错。

**建议**：启动时用一次测试嵌入探测实际维度：
```python
def _get_embed_dim() -> int:
    sample = get_embeddings().embed_query("test")
    return len(sample)
```
或在 `config.py` 加 `embed_dim: int = 1024` 并在 README 中标注"换模型必须同步修改"。

---

### H7. docx 加载合并为单个 Document，页码元数据恒为 p.1

**文件**：`app/rag/loaders.py:25-37`

```python
full_text = "\n\n".join(texts)
return [Document(page_content=full_text, metadata={"source": path.name, "page": 1})]
```

所有 docx 段落被合并为一个 Document，`page=1` 硬编码。对比 PDF 加载：`PyMuPDFLoader` 返回每页一个 Document。产生的问题：
1. 答案末尾的 `[来源: 文件名 p.1]` 对 docx 永远显示 p.1，用户无法定位原文
2. 大型 docx 文件产生一个超大 chunk，超过 `chunk_size` 后 splitter 会切断段落中间，破坏语义
3. Word 文档的标题、章节结构完全丢失

**建议**：按标题或固定字符数（如每 3000 字）切分 docx，并补充 `page` 字段（用节/段落计数估算）。

---

### M7. `get_llm()` 无单例缓存，每次调用新建对象

**文件**：`app/llm/ollama_client.py:29-35`

```python
def get_llm(temperature: float = 0.2) -> ChatOllama:
    return ChatOllama(...)   # 每次调用都新建
```

对比 `get_embeddings()` 有 `_embeddings` 单例，`get_llm()` 无缓存。虽然 `ChatOllama` 是轻量客户端对象，但 `get_rag_chain()` 内部调用 `get_llm()` 时每次 temperature 不同都会建链，且每次 `get_llm()` 直接调用（如评估脚本）都新建连接。风格不一致，存在资源浪费。

**建议**：参照 `get_embeddings()` 模式加 `_llm_cache: dict[float, ChatOllama] = {}`。

---

### M8. 启动时未验证指定 Embedding 模型是否已拉取

**文件**：`app/main.py:22-26`

```python
models = list_ollama_models()
if not models["chat"] and not models["embedding"]:
    logger.warning("Ollama unreachable at startup ...")
```

`list_ollama_models()` 只判断 Ollama 整体是否可达，不验证 `settings.embedding_model`（bge-m3）和 `settings.ollama_model` 是否已拉取到本地。若用户忘记 `ollama pull bge-m3`，第一次上传文档时才会在 add_documents 阶段报错，错误信息经过 `kb.py` 的多层包装后变为 "向量入库失败"，难以快速定位。

**建议**：在 lifespan 中添加精确检查：
```python
chat_models = models["chat"]
embed_models = models["embedding"]
if settings.ollama_model not in chat_models:
    logger.warning(f"Chat model '{settings.ollama_model}' not found in Ollama. Run: ollama pull {settings.ollama_model}")
if settings.embedding_model not in embed_models:
    logger.warning(f"Embedding model '{settings.embedding_model}' not found. Run: ollama pull {settings.embedding_model}")
```

---

### M9. Reranker 未在 lifespan 中健康检查，启动即隐性降级

**文件**：`app/main.py:19-35`

lifespan 检查了 Ollama（M8）和 Qdrant，唯独没有 reranker。Phase 3c 启用后，reranker 成为关键依赖，但启动日志不会提示它是否可达。运维人员看到"服务启动成功"，实际 reranker 已离线，每次查询静默降级。

**建议**：在 lifespan yield 前加：
```python
try:
    r = httpx.get(f"{settings.reranker_base_url}/health", timeout=3)
    r.raise_for_status()
    logger.info("Reranker ready")
except Exception:
    logger.warning(f"Reranker unreachable at {settings.reranker_base_url} — queries will use original order")
```

---

### M10. `float` 作为 chain cache 的 key 存在 IEEE 754 精度隐患

**文件**：`app/rag/chain.py:29, 94`

```python
_chain_cache: dict[float, object] = {}

if temperature not in _chain_cache:
    ...
_chain_cache[temperature] = ...
```

`0.1 + 0.2 == 0.30000000000000004`（Python）。客户端传入 `temperature=0.30000000000000004` 与 `temperature=0.3` 会创建两个不同的 chain 实例，内存翻倍。虽然当前请求路径 `effective_temperature` 通常是字面量，但仍是不安全的 pattern。

**建议**：将 key 规范化：`cache_key = round(temperature, 4)`

---

### M11. 依赖版本约束过宽，存在 Breaking Change 风险

**文件**：`pyproject.toml`

| 依赖 | 声明 | 风险 |
|------|------|------|
| `langchain>=0.3` | 无上界 | 0.3→0.4 有 API 变更，`RunnableLambda` / `BaseTool` 接口已在演进 |
| `ragas>=0.2` | 无上界 | 0.2.x 和 0.4.x API 完全不同；当前代码用 `SingleTurnSample`、`EvaluationDataset`（0.4.x 风格），若解析到 0.2.x 立即崩溃 |
| `langchain-community>=0.3` | 无上界 | 仅用于 `PyMuPDFLoader`，拉入大量不必要的传递依赖 |

**建议**：
```toml
"langchain>=0.3,<0.4"
"langchain-core>=0.3,<0.4"
"langchain-ollama>=0.2,<0.3"
"langchain-qdrant>=0.2,<0.3"
"ragas>=0.4,<0.5"
```
并考虑用 `langchain-pymupdf` 替换 `langchain-community` 以减少依赖体积。

---

### M12. `langchain-text-splitters` 未显式声明为依赖

**文件**：`app/rag/splitter.py:1`、`pyproject.toml`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

`langchain-text-splitters` 是独立包，未出现在 `pyproject.toml` 的 `dependencies` 中，目前通过 `langchain-community` 传递引入。若 langchain 调整传递依赖（已发生过），此 import 会在生产环境静默失败。

**建议**：在 `pyproject.toml` 中显式声明：
```toml
"langchain-text-splitters>=0.3,<0.4"
```

---

### M13. `/health` 端点每次都发 HTTP 请求，无缓存

**文件**：`app/main.py:88-103`

```python
@app.get("/health")
def health():
    models = list_ollama_models()   # 每次 HTTP call to Ollama
```

`list_ollama_models()` 内部调 `httpx.get(..., timeout=5)`。Open WebUI 默认每 30 秒探活一次，高并发场景下每次 `/health` 都对 Ollama 发 HTTP 请求，浪费连接资源，且 5 秒 timeout 会阻塞请求线程。

**建议**：加 30 秒 TTL 缓存：
```python
import functools, time

@functools.cache  # 或 TTL 缓存装饰器
def _cached_models(ts: int):  # ts = int(time.time() // 30) 作为 cache key
    return list_ollama_models()
```

---

### M14. 全局异常处理将所有错误统一返回 503，语义错误

**文件**：`app/main.py:42-46`

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=503, ...)
```

503 语义是"服务暂时不可用"，而非所有异常的正确状态码。`ValueError`、`KeyError`、业务逻辑错误都被标为 503。Open WebUI 见到 503 可能触发"服务器离线"提示，而不显示 `detail` 中的实际原因。

**建议**：按异常类型区分状态码：
```python
if isinstance(exc, (ValueError, KeyError, TypeError)):
    status = 400
elif "connect" in str(exc).lower() or "timeout" in str(exc).lower():
    status = 503
else:
    status = 500
```

---

## 七、LOW（ECC 补充）

### L5. `_EMBED_KEYWORDS` 启发式分类会误判新模型

**文件**：`app/llm/ollama_client.py:7`

```python
_EMBED_KEYWORDS = ("embed", "bge", "minilm", "e5", "gte", "dmeta")
```

`nomic-embed-text`、`mxbai-embed-large`、`snowflake-arctic-embed` 等常用 embedding 模型均不在关键词列表中，会被误分类为 chat 模型，导致 `/health` 接口报告错误的模型类别，运维人员无法通过 health check 判断 embedding 是否可用。

---

### L6. `reasoning=False` 是 Qwen3 特定参数，切换模型后行为未定义

**文件**：`app/llm/ollama_client.py:34`

```python
ChatOllama(..., reasoning=False)
```

`reasoning=False` 用于关闭 Qwen3 的思维链（CoT）输出，避免 `<think>...</think>` 标签混入答案。其他模型（Llama、Mistral）不识别此参数，会被 Ollama 静默忽略。但如果切换到某个未来支持 reasoning 的模型但需要开启时，此处会漏改。应加注释说明原因和适用范围。

---

### L7. `.txt` 文件硬编码 UTF-8，无 GBK 回退

**文件**：`app/rag/loaders.py:41`

```python
text = path.read_text(encoding="utf-8")
```

Windows 中文环境下生成的 `.txt` 文件默认为 GBK/GB2312 编码，上传后 `UnicodeDecodeError` 会被 `kb.py` 捕获包装为 422 "文档解析失败"，用户无法自助解决。

**建议**：
```python
for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312"):
    try:
        return [Document(page_content=path.read_text(encoding=enc), ...)]
    except UnicodeDecodeError:
        continue
raise ValueError(f"无法识别文件编码: {path.name}")
```

---

### L8. `_format_docs` 混合日志副作用与格式化职责

**文件**：`app/rag/chain.py:32-43`

`_format_docs` 在 chain pipeline 中既记录 debug 日志又返回格式化字符串。这使得函数无法在测试中纯函数化调用（调用即产生日志输出），也无法在评估脚本中单独使用格式化逻辑。

**建议**：将日志提取到调用方或装饰器，保持 `_format_docs` 纯函数。

---

## 八、综合问题统计

| 严重级别 | 数量 | 来源 |
|---------|------|------|
| HIGH | 7（H1-H7） | 4 个 Codex + 3 个 ECC |
| MEDIUM | 14（M1-M14） | 6 个 Codex + 8 个 ECC |
| LOW | 8（L1-L8） | 4 个 Codex + 4 个 ECC |
| **合计** | **29** | |

---

## 九、完整优先修复顺序（含新增项）

| 优先级 | 问题 | 改动量 |
|--------|------|--------|
| 1 | **H1** 双重检索（评估失实） | 中 |
| 2 | **H5** 流式超时死代码（用户永久等待） | 小 |
| 3 | **H3** `.sh` 缺 reranker 逻辑 | 小 |
| 4 | **H4** reranker 30 秒阻塞 | 小 |
| 5 | **H6** EMBED_DIM 硬编码 | 小 |
| 6 | **H2** 评估失败跳过 | 小 |
| 7 | **M9** lifespan 加 reranker 健康检查 | 小 |
| 8 | **M8** 启动验证指定模型已拉取 | 小 |
| 9 | **M11** 依赖版本约束收紧 | 小 |
| 10 | **H7** docx 加载丢失页码结构 | 中 |
| 11 | **M1** testset 切分与生产一致 | 小 |
| 12 | **M14** 503 状态码语义修正 | 小 |
| 13 | 其余 M/L 项 | 各自独立，可按功能 PR 捎带 |

---

*本文档由 Codex Adversarial Review + ECC code-review + 人工全文阅读生成，覆盖 15 个源文件，约 900 行代码。*

