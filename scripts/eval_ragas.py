"""
Ragas 评估脚本：对当前 RAG 系统跑四项指标评估

运行方式：
    uv run python scripts/eval_ragas.py                    # 保存为 baseline
    uv run python scripts/eval_ragas.py --output phase3_chunking

使用流程：
    1. 首次运行：先执行 generate_testset.py 生成 data/testset.csv
    2. 每次模块优化后运行本脚本，与 baseline 对比
    3. 结果保存在 data/eval_results/

注意：
    - 需要 Qdrant + Ollama 正在运行（先启动 start_tunnel.bat）
    - 每道题评估需调用 LLM 多次，15 题约需 10-25 分钟
    - 重点关注 Faithfulness 和 Answer Relevancy（不依赖 ground_truth，更可靠）
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re

import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import AIMessage
from loguru import logger
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
import warnings
# ragas 0.4.x 的 collections 指标仅支持 OpenAI InstructorLLM，
# 旧式 metric 实例支持 LangchainLLMWrapper（有 deprecation warning，但功能正常）
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

from app.config import settings
from app.rag.chain import get_retriever, get_llm, PROMPT, _format_docs
from langchain_core.output_parsers import StrOutputParser


def run_rag(question: str) -> tuple[str, list[str]]:
    """对单个问题运行 RAG，返回 (answer, retrieved_contexts)。

    单次检索 pass：先检索文档，再用同一批文档构造 LLM prompt 生成答案，
    确保传给 Ragas 的 retrieved_contexts 与 LLM 实际看到的上下文严格一致。
    """
    retriever = get_retriever()
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]

    formatted_context = _format_docs(docs)
    answer = (PROMPT | get_llm() | StrOutputParser()).invoke({
        "context": formatted_context,
        "question": question,
        "chat_history": [],
    })
    return answer, contexts


def load_testset(path: str) -> pd.DataFrame:
    """加载测试集，验证必要列存在。"""
    p = Path(path)
    if not p.exists():
        logger.error(f"测试集不存在：{p}")
        logger.info("请先运行：uv run python scripts/generate_testset.py")
        sys.exit(1)

    df = pd.read_csv(p, encoding="utf-8-sig")

    if "user_input" not in df.columns:
        logger.error(f"测试集缺少 'user_input' 列，实际列：{list(df.columns)}")
        sys.exit(1)

    # reference 列可选（Context Precision/Recall 需要它，Faithfulness/AnswerRelevancy 不需要）
    if "reference" not in df.columns:
        logger.warning("测试集没有 'reference' 列，Context Precision/Recall 指标将跳过")
        df["reference"] = None

    df = df.dropna(subset=["user_input"])
    logger.info(f"加载测试集：{len(df)} 条问题（来自 {p}）")
    return df


def build_metrics(has_reference: bool, ragas_llm=None, ragas_embeddings=None) -> list:
    """根据是否有 reference 决定启用哪些指标。
    注：AnswerRelevancy 需要在评估阶段调用 embedding，与本地 Ollama 存在兼容性问题，暂时跳过。
    """
    metrics = [faithfulness]
    if has_reference:
        metrics += [context_precision, context_recall]
    return metrics


def _fix_statements_json(text: str) -> str:
    """将模型输出的 [{"statement": "..."}] 格式修正为 Ragas 期望的 ["..."] 格式。"""
    # 去除 markdown 代码块
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(data, dict) and "statements" in data:
        stmts = data["statements"]
        if stmts and isinstance(stmts[0], dict):
            data["statements"] = [
                s.get("statement") or s.get("text") or next(iter(s.values()), "")
                for s in stmts
            ]
        return json.dumps(data, ensure_ascii=False)
    return text


def main():
    parser = argparse.ArgumentParser(description="Ragas RAG 评估脚本")
    parser.add_argument("--output", default="baseline", help="结果文件名（不含 .json）")
    parser.add_argument("--testset", default="data/testset.csv", help="测试集路径")
    args = parser.parse_args()

    logger.info("=== Ragas 评估开始 ===")
    logger.info(f"RAG 模型：{settings.ollama_model} | Embedding：{settings.embedding_model}")

    # 1. 加载测试集
    df = load_testset(args.testset)
    questions = df["user_input"].tolist()
    references = df["reference"].tolist()
    has_reference = df["reference"].notna().any()

    # 2. 对每个问题运行 RAG，收集答案和检索上下文
    total_questions = len(questions)
    logger.info(f"\n开始运行 RAG（共 {total_questions} 个问题）...")
    samples = []
    failed_indices: list[int] = []

    for i, (question, reference) in enumerate(zip(questions, references), 1):
        logger.info(f"  [{i}/{total_questions}] {str(question)[:60]}...")
        try:
            answer, contexts = run_rag(question)
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=reference if pd.notna(reference) else None,
            )
            samples.append(sample)
            logger.debug(f"    contexts: {len(contexts)} 个，answer: {str(answer)[:60]}")
        except Exception as e:
            logger.warning(f"    问题 [{i}] 处理失败，跳过：{e}")
            failed_indices.append(i)

    failed = len(failed_indices)
    fail_rate = failed / total_questions if total_questions else 0

    if not samples:
        logger.error("所有问题处理失败，请检查 Qdrant 和 Ollama 是否正在运行")
        sys.exit(1)

    if failed:
        logger.warning(f"{failed}/{total_questions} 个问题处理失败（索引：{failed_indices}）")

    # 失败率超过 10% 时拒绝继续，避免评分因样本缺失而虚高
    FAIL_RATE_THRESHOLD = 0.10
    if fail_rate > FAIL_RATE_THRESHOLD:
        logger.error(
            f"失败率 {fail_rate:.1%} 超过阈值 {FAIL_RATE_THRESHOLD:.0%}，"
            "评分结果不可信，请检查服务状态后重试"
        )
        sys.exit(1)

    # 3. 构建 EvaluationDataset
    eval_dataset = EvaluationDataset(samples=samples)

    # 4. 配置 Ragas 使用 Ollama（替换默认 OpenAI）
    # 包装 LLM，将 [{"statement": "..."}] 修正为 ["..."]，避免 Ragas OutputParserException
    class _CleanedOllama(ChatOllama):
        def invoke(self, *args, **kwargs):
            msg = super().invoke(*args, **kwargs)
            msg.content = _fix_statements_json(msg.content)
            return msg

        async def ainvoke(self, *args, **kwargs):
            msg = await super().ainvoke(*args, **kwargs)
            msg.content = _fix_statements_json(msg.content)
            return msg

    ragas_llm = LangchainLLMWrapper(
        _CleanedOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0,
            reasoning=False,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
        )
    )

    # 5. 选择指标并运行评估
    metrics = build_metrics(has_reference, ragas_llm, ragas_embeddings)
    metric_names = [m.__class__.__name__ for m in metrics]
    logger.info(f"\n开始 Ragas 评估（指标：{metric_names}）...")
    logger.info("（每道题需多次调用 LLM，预计 10-25 分钟）")

    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        show_progress=True,
        raise_exceptions=False,
        run_config=RunConfig(timeout=180, max_workers=2),
    )

    # 6. 输出并保存结果
    scores_df = results.to_pandas()
    metric_cols = [m.name for m in metrics]
    scores = {col: float(scores_df[col].mean()) for col in metric_cols if col in scores_df.columns}

    print("\n" + "=" * 45)
    print(f"  Ragas 评估结果 — {args.output}")
    print("=" * 45)
    for metric, score in scores.items():
        bar = "█" * int((score or 0) * 20)
        print(f"  {metric:<35} {score:.4f}  {bar}")
    print("=" * 45)

    output_dir = Path("data/eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.output}.json"

    save_data = {
        "run_name": args.output,
        "scores": scores,
        "num_samples": len(samples),
        "total_questions": total_questions,
        "failed_count": failed,
        "failed_indices": failed_indices,
        "model": settings.ollama_model,
        "embedding": settings.embedding_model,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ 结果保存到 {output_path}")

    # 7. 如果有历史结果，打印对比
    compare_targets = [p for p in output_dir.glob("*.json") if p != output_path]
    if compare_targets:
        latest = max(compare_targets, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest) as f:
                prev = json.load(f)
            prev_total = prev.get("total_questions", prev.get("num_samples", 0))
            prev_failed = prev.get("failed_count", 0)
            prev_complete = (prev_total - prev_failed) / prev_total if prev_total else 0
            curr_complete = (total_questions - failed) / total_questions if total_questions else 0
            if prev_complete < (1 - FAIL_RATE_THRESHOLD) or curr_complete < (1 - FAIL_RATE_THRESHOLD):
                logger.warning(
                    f"跳过历史对比：当前完成率 {curr_complete:.1%}，"
                    f"历史完成率 {prev_complete:.1%}，两者须均 ≥{1-FAIL_RATE_THRESHOLD:.0%}"
                )
            else:
                print(f"\n  与上次结果对比（{latest.stem}）：")
                for metric in scores:
                    curr_val = scores.get(metric) or 0
                    prev_val = (prev.get("scores") or {}).get(metric) or 0
                    delta = curr_val - prev_val
                    arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")
                    print(f"  {metric:<35} {arrow} {delta:+.4f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
