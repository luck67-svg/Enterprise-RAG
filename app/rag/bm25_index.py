import json
import re
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Plus

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


class BM25Index:
    def __init__(self, path: Path):
        self.path = path
        self.records: list[dict] = []
        self._bm25: BM25Plus | None = None
        self._tokenized_corpus: list[list[str]] = []
        self.load()

    def load(self) -> None:
        if self.path.exists():
            self.records = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.records = []
        self.rebuild()

    def rebuild(self) -> None:
        self._tokenized_corpus = [tokenize(record["page_content"]) for record in self.records]
        self._bm25 = BM25Plus(self._tokenized_corpus) if self._tokenized_corpus else None

    def replace_source(self, source: str, docs: list[Document]) -> None:
        self.records = [record for record in self.records if record["source"] != source]
        self.records.extend(
            {
                "chunk_id": doc.metadata["chunk_id"],
                "source": source,
                "page": doc.metadata.get("page"),
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "page_content": doc.page_content,
            }
            for doc in docs
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")
        self.rebuild()

    def delete_source(self, source: str) -> None:
        self.records = [record for record in self.records if record["source"] != source]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")
        self.rebuild()

    def search(self, query: str, k: int) -> list[Document]:
        if not self._bm25:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[Document] = []
        for index in ranked[:k]:
            if scores[index] <= 0:
                continue
            if not set(query_tokens).intersection(self._tokenized_corpus[index]):
                continue
            record = self.records[index]
            hits.append(
                Document(
                    page_content=record["page_content"],
                    metadata={
                        "chunk_id": record["chunk_id"],
                        "source": record["source"],
                        "page": record.get("page"),
                        "parent_id": record["parent_id"],
                        "parent_content": record["parent_content"],
                    },
                )
            )
        return hits
