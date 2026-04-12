"""
测试集生成脚本（只需运行一次，后续所有评估阶段复用同一测试集）

运行方式：
    uv run python scripts/generate_testset.py

运行后：
    1. 查看 data/testset.csv
    2. 人工删掉质量明显差的行（问题含乱码、答案无意义等）
    3. 保留 10-15 条质量好的，保存

注意：
    - 需要 Qdrant + Ollama 正在运行（先启动 start_tunnel.bat）
    - 使用 settings.ollama_model（.env 中配置的大模型）生成，质量更高
    - 生成 20 个问题约需 5-15 分钟，取决于模型速度
"""
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径，使 app.* 模块可导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.graph import NodeType
from ragas.testset.transforms import (
    SummaryExtractor, EmbeddingExtractor,
    CustomNodeFilter, CosineSimilarityBuilder,
)
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import pandas as pd
from loguru import logger

from app.config import settings
from app.rag.loaders import load_file, SUPPORTED_EXTENSIONS
from app.rag.splitter import split_documents


def load_all_documents():
    """加载 data/uploads/ 目录下所有支持的文档。"""
    upload_dir = Path(settings.upload_dir)
    if not upload_dir.exists():
        logger.error(f"上传目录不存在：{upload_dir}")
        sys.exit(1)

    all_docs = []
    files = [f for f in upload_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        logger.error(f"上传目录中没有支持的文档（{SUPPORTED_EXTENSIONS}）")
        logger.info("请先通过 http://localhost:8000 上传文档，再运行本脚本")
        sys.exit(1)

    logger.info(f"找到 {len(files)} 个文档：{[f.name for f in files]}")

    for f in files:
        try:
            docs = load_file(f)
            all_docs.extend(docs)
            logger.info(f"  加载 {f.name}：{len(docs)} 页/段")
        except Exception as e:
            logger.warning(f"  跳过 {f.name}：{e}")

    logger.info(f"共加载 {len(all_docs)} 个页面，开始切分为 chunks...")
    chunks = split_documents(all_docs)
    logger.info(f"切分完成：{len(chunks)} 个 chunks")
    return chunks


def main():
    logger.info("=== Ragas 测试集生成 ===")
    logger.info(f"使用模型：{settings.ollama_model}（via {settings.ollama_base_url}）")
    logger.info(f"使用 Embedding：{settings.embedding_model}")

    # 1. 加载文档
    all_docs = load_all_documents()

    # 2. 配置 Ragas 使用 Ollama（替换默认 OpenAI）
    ragas_llm = LangchainLLMWrapper(
        ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.3,
            reasoning=False,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
        )
    )

    # 3. 构建自定义 transforms
    # ragas 的 generate_with_langchain_docs 始终将输入文档转为 NodeType.DOCUMENT，
    # 因此所有 filter 都必须使用 filter_docs，不能用 filter_chunks。
    # 跳过 CosineSimilarityBuilder/OverlapScoreBuilder（需要多节点关系，容易崩溃）。
    def filter_docs(node):
        content = node.properties.get("page_content", "")
        return (
            node.type == NodeType.DOCUMENT
            and bool(content)
            and len(content.strip()) >= 100
        )

    custom_transforms = [
        SummaryExtractor(llm=ragas_llm, filter_nodes=filter_docs),
        CustomNodeFilter(llm=ragas_llm, filter_nodes=filter_docs),
        EmbeddingExtractor(
            embedding_model=ragas_embeddings,
            property_name="summary_embedding",
            embed_property_name="summary",
            filter_nodes=filter_docs,
        ),
        ThemesExtractor(llm=ragas_llm, filter_nodes=filter_docs),
        NERExtractor(llm=ragas_llm, filter_nodes=filter_docs),
        CosineSimilarityBuilder(
            property_name="summary_embedding",
            new_property_name="summary_similarity",
            threshold=0.7,
            filter_nodes=filter_docs,
        ),
    ]

    # 4. 初始化生成器
    logger.info("初始化 TestsetGenerator...")
    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
    )

    # 5. 生成测试集（传入 chunks，用 prechunked transforms）
    logger.info("开始生成测试集（约 5-15 分钟）...")
    testset = generator.generate_with_langchain_docs(
        documents=all_docs,
        testset_size=20,
        transforms=custom_transforms,
        raise_exceptions=False,
    )

    # 6. 转为 DataFrame 并保存
    records = testset.to_list()
    if not records:
        logger.error("生成结果为空，请检查文档质量和模型连接")
        sys.exit(1)

    df = pd.DataFrame(records)

    # 保留关键列，标准化列名方便查阅
    keep_cols = [c for c in ["user_input", "reference", "synthesizer_name"] if c in df.columns]
    df_display = df[keep_cols].copy()

    output_path = Path("data/testset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info(f"\n✓ 生成完成：{len(df)} 条问题，保存到 {output_path}")
    logger.info("\n===== 生成的问题预览 =====")
    for i, row in df_display.iterrows():
        q = str(row.get("user_input", ""))[:80]
        qtype = str(row.get("synthesizer_name", ""))
        logger.info(f"  [{i+1}] [{qtype}] {q}")

    logger.info(f"\n下一步：")
    logger.info(f"  1. 打开 data/testset.csv 检查质量")
    logger.info(f"  2. 删掉问题含乱码或答案无意义的行")
    logger.info(f"  3. 保留 10-15 条后，运行评估：uv run python scripts/eval_ragas.py")


if __name__ == "__main__":
    main()
