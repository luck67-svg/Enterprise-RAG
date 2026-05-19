import os
import pickle
import re
import tempfile
from pathlib import Path

import jieba
from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi

_DEFAULT_INDEX_PATH = Path("data/bm25_index.pkl")

# BM25 得分低于最高分此比例时视为噪音，不进入 RRF 融合
_BM25_SCORE_RATIO = 0.1

# 分词器版本：tokenize 逻辑变更时递增，使旧 corpus 缓存失效
_TOKENIZER_VERSION = 3  # v3: 移除 dict.fromkeys() 去重，恢复 TF；扩展 CJK 范围


def _tokenize(text: str) -> list[str]:
    """
    混合分词器：N-gram + 整词 + jieba fallback

    - 中文bigram：捕获任意相邻两字组合，处理任意新术语
    - 英文/数字/符号整词：保留 CamelCase、全大写缩写、连字符词如 DeepSeek-R1
    - jieba：处理高频通用词（如"的"、"是"、"在"）为停用词
    - 保留重复 token，确保 BM25 词频（TF）计算正确
    """
    text_lower = text.lower()
    tokens = []

    # 中文bigram：覆盖基本 CJK + 扩展 A + 兼容汉字
    cjk_chars = re.findall(r'[一-鿿㐀-䶿豈-﫿]', text)
    for i in range(len(cjk_chars) - 1):
        tokens.append(cjk_chars[i] + cjk_chars[i + 1])

    # 英文/数字/符号整词
    tokens.extend(re.findall(r'[a-z]+(?:-[a-z0-9]+)+|[a-z0-9]{2,}', text_lower))

    # jieba 补充中文通用词
    jieba_tokens = list(jieba.cut(text_lower))
    tokens.extend(re.findall(r'[一-鿿㐀-䶿豈-﫿]', text))
    tokens.extend(t for t in jieba_tokens if re.match(r'[一-鿿㐀-䶿豈-﫿]', t))

    return tokens  # 保留重复，BM25 依赖词频（TF）


class BM25Store:
    def __init__(self, index_path: Path = _DEFAULT_INDEX_PATH):
        self._index_path = index_path
        self._documents: list[Document] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        try:
            data = pickle.loads(self._index_path.read_bytes())
            self._documents = data["documents"]
            if self._documents:
                cached_corpus = data.get("corpus") if data.get("tokenizer_version") == _TOKENIZER_VERSION else None
                if cached_corpus:
                    self._corpus = cached_corpus
                    logger.info(f"BM25 index loaded (corpus cache hit): {len(self._documents)} chunks")
                else:
                    self._corpus = [_tokenize(d.page_content) for d in self._documents]
                    logger.info(f"BM25 index loaded (corpus rebuilt): {len(self._documents)} chunks")
                self._bm25 = BM25Okapi(self._corpus)
            logger.info(f"BM25 ready: {len(self._documents)} chunks from {self._index_path}")
        except Exception as e:
            logger.warning(f"BM25 index load failed ({e}), starting fresh")
            self._documents = []
            self._corpus = []
            self._bm25 = None

    def _save(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        data = pickle.dumps({
            "documents": self._documents,
            "corpus": self._corpus,
            "tokenizer_version": _TOKENIZER_VERSION,
        })
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=self._index_path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path_str, str(self._index_path))
        except Exception:
            try:
                os.unlink(tmp_path_str)
            except OSError:
                pass
            raise

    def add_documents(self, docs: list[Document]) -> None:
        new_corpus = [_tokenize(d.page_content) for d in docs]
        self._documents.extend(docs)
        self._corpus.extend(new_corpus)
        self._bm25 = BM25Okapi(self._corpus)
        self._save()
        logger.debug(f"BM25 index updated: {len(self._documents)} total chunks")

    def remove_by_source(self, source: str) -> None:
        before = len(self._documents)
        pairs = [
            (d, c) for d, c in zip(self._documents, self._corpus)
            if d.metadata.get("source") != source
        ]
        if pairs:
            self._documents, self._corpus = map(list, zip(*pairs))
        else:
            self._documents, self._corpus = [], []
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None
        self._save()
        logger.info(f"BM25 index: removed {before - len(self._documents)} chunks for source={source!r}")

    def search(self, query: str, k: int) -> list[Document]:
        if self._bm25 is None or not self._documents:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        max_score = float(scores.max()) if len(scores) > 0 else 0.0
        if max_score == 0.0:
            # 完全无词汇匹配，不注入噪音
            return []
        threshold = max_score * _BM25_SCORE_RATIO
        top_indices = [
            i for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            if scores[i] >= threshold
        ]
        return [self._documents[i] for i in top_indices]


_bm25_store: BM25Store | None = None


def get_bm25_store() -> BM25Store:
    global _bm25_store
    if _bm25_store is None:
        _bm25_store = BM25Store()
    return _bm25_store
