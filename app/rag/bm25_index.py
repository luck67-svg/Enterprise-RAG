import json
import re
import tempfile
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
            records = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            records = []
        tokenized_corpus, bm25 = self._build_state(records)
        self.records = records
        self._tokenized_corpus = tokenized_corpus
        self._bm25 = bm25

    def rebuild(self) -> None:
        self._tokenized_corpus, self._bm25 = self._build_state(self.records)

    def replace_source(self, source: str, docs: list[Document]) -> None:
        next_records = [record for record in self.records if record["source"] != source]
        next_records.extend(self._record_from_doc(source, doc) for doc in docs)
        self._persist_records(next_records)
        tokenized_corpus, bm25 = self._build_state(next_records)
        self.records = next_records
        self._tokenized_corpus = tokenized_corpus
        self._bm25 = bm25

    def delete_source(self, source: str) -> None:
        next_records = [record for record in self.records if record["source"] != source]
        self._persist_records(next_records)
        tokenized_corpus, bm25 = self._build_state(next_records)
        self.records = next_records
        self._tokenized_corpus = tokenized_corpus
        self._bm25 = bm25

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

    def _build_state(self, records: list[dict]) -> tuple[list[list[str]], BM25Plus | None]:
        tokenized_corpus = [tokenize(record["page_content"]) for record in records]
        bm25 = BM25Plus(tokenized_corpus) if tokenized_corpus else None
        return tokenized_corpus, bm25

    def _record_from_doc(self, source: str, doc: Document) -> dict:
        doc_source = doc.metadata.get("source")
        if doc_source is not None and doc_source != source:
            raise ValueError(f"Document source {doc_source!r} does not match {source!r}")
        return {
            "chunk_id": doc.metadata["chunk_id"],
            "source": source,
            "page": doc.metadata.get("page"),
            "parent_id": doc.metadata["parent_id"],
            "parent_content": doc.metadata["parent_content"],
            "page_content": doc.page_content,
        }

    def _persist_records(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(records, ensure_ascii=False, indent=2)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f"{self.path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_file.write(payload)
                temp_path = Path(temp_file.name)
            temp_path.replace(self.path)
        except Exception:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            raise
