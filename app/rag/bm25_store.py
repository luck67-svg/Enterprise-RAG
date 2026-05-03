import pickle
from pathlib import Path

import jieba
from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi

_DEFAULT_INDEX_PATH = Path("data/bm25_index.pkl")


def _tokenize(text: str) -> list[str]:
    return list(jieba.cut(text.lower()))


class BM25Store:
    def __init__(self, index_path: Path = _DEFAULT_INDEX_PATH):
        self._index_path = index_path
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        try:
            data = pickle.loads(self._index_path.read_bytes())
            self._documents = data["documents"]
            if self._documents:
                corpus = [_tokenize(d.page_content) for d in self._documents]
                self._bm25 = BM25Okapi(corpus)
            logger.info(f"BM25 index loaded: {len(self._documents)} chunks from {self._index_path}")
        except Exception as e:
            logger.warning(f"BM25 index load failed ({e}), starting fresh")
            self._documents = []
            self._bm25 = None

    def _save(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_bytes(pickle.dumps({"documents": self._documents}))

    def _rebuild(self) -> None:
        if self._documents:
            corpus = [_tokenize(d.page_content) for d in self._documents]
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None

    def add_documents(self, docs: list[Document]) -> None:
        self._documents.extend(docs)
        self._rebuild()
        self._save()
        logger.debug(f"BM25 index updated: {len(self._documents)} total chunks")

    def remove_by_source(self, source: str) -> None:
        before = len(self._documents)
        self._documents = [d for d in self._documents if d.metadata.get("source") != source]
        self._rebuild()
        self._save()
        logger.info(f"BM25 index: removed {before - len(self._documents)} chunks for source={source!r}")

    def search(self, query: str, k: int) -> list[Document]:
        if self._bm25 is None or not self._documents:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._documents[i] for i in top_indices]


_bm25_store: BM25Store | None = None


def get_bm25_store() -> BM25Store:
    global _bm25_store
    if _bm25_store is None:
        _bm25_store = BM25Store()
    return _bm25_store
